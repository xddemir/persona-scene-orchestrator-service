"""FastAPI app.

    uvicorn scene_orchestrator.api.app:app --reload --reload-dir src --host 127.0.0.1 --port 8001

Run it in WSL: skyboxes are Slurm jobs submitted over a shared SSH connection
to Pegasus, which Windows' ssh cannot provide. Port 8001 because image-gen's
own dev server uses 8000.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from .. import __version__
from ..adapters import ChatbotV2Adapter, PersonaAdapterError
from ..clients.image_gen import SkyboxRequest, SlurmImageGenClient
from ..config import AppConfig, load_config
from ..manifest import read_manifest
from ..models import SceneSpec
from ..outputs import relative_to_out, scene_dir, scene_spec_path, seconds_since
from ..pipeline import UNFINISHED, ScenePipeline, scene_id_for
from .jobs import JobRunner, SkyboxJob
from .models import (
    HealthResponse,
    SceneAccepted,
    SceneBody,
    SceneResponse,
    SceneStatus,
    SkyboxBody,
    SkyboxFiles,
    SkyboxJobResponse,
)
from .scenes import SceneInProgress, SceneJob, SceneRunner

# uvicorn's own logger, so these lines show up in the server console.
log = logging.getLogger("uvicorn.error")

# How long the GET that starts a scene waits for it. A procedural scene takes
# milliseconds, so it is answered "ready" at once instead of sending Unity off
# to ask again later. A panorama takes minutes and isn't waited for.
AUTO_CREATE_WAIT_S = 2.0

# A scene that fell back to the procedural sky gets its 360 image tried again
# when it is asked for and Pegasus can be reached, but not this soon after the
# last attempt. Unity asks every few seconds while a scene is generating, and
# the question that follows a failed attempt has to be answered with the
# fallback, not with yet another attempt.
FALLBACK_RETRY_AFTER_S = 60.0


def create_app(
    config: AppConfig | None = None, *, image_gen: SlurmImageGenClient | None = None
) -> FastAPI:
    if config is None:
        config = load_config()

    # Injectable so tests can use a simulated Pegasus.
    client = image_gen or SlurmImageGenClient(config.image_gen)
    runner = JobRunner(client, config.out_dir)
    adapter = ChatbotV2Adapter()
    scenes = SceneRunner(ScenePipeline(client, config.out_dir, adapter=adapter))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Only a check of the connection you opened; the server starts either
        # way, and /health repeats this whenever you ask.
        status = await asyncio.to_thread(client.status)
        if status.connected:
            log.info("Pegasus: shared SSH connection to %s is open", status.login)
        else:
            log.warning("Pegasus: %s", status.error)
        runner.start()
        scenes.start()
        try:
            yield
        finally:
            scenes.stop()
            runner.stop()

    app = FastAPI(
        version=__version__,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.runner = runner
    app.state.scenes = scenes

    def to_response(job: SkyboxJob) -> SkyboxJobResponse:
        result = job.result
        files = None
        if result is not None and result.ok:
            files = SkyboxFiles(
                png=relative_to_out(config.out_dir, result.png),
                sidecar=relative_to_out(config.out_dir, result.sidecar),
            )
        return SkyboxJobResponse(
            job_id=job.job_id,
            status=job.status,
            scene_id=job.request.scene_id,
            seed=job.request.seed,
            prompt=job.request.prompt,
            detail=job.detail,
            files=files,
            error=result.error if result else None,
            attempts=result.attempts if result else None,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    # Plain def, not async: the upstream check blocks for up to a few seconds,
    # and FastAPI runs sync endpoints in a thread pool, off the event loop.
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(version=__version__, image_gen=client.status())

    # -- /scenes: persona in, Unity's files out (one scene per participant)

    @app.post("/scenes", response_model=SceneAccepted, status_code=202)
    async def create_scene(body: SceneBody) -> SceneAccepted:
        # Only enough here to name the scene. Everything after (a persona that
        # fails to convert included) runs in the background and ends in the
        # manifest; without a scene_id there is nothing to put in it, so these
        # few problems are answered with 422 instead.
        raw = body.persona if body.persona is not None else _read_persona(body.persona_file)
        try:
            scene_id = scene_id_for(adapter.participant_id(raw))
        except (PersonaAdapterError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        try:
            job = scenes.submit(SceneJob(scene_id, body.seed, raw, body.sky_mode))
        except SceneInProgress:
            raise HTTPException(status_code=409, detail=f"{scene_id} is already being generated")
        return SceneAccepted(scene_id=job.scene_id, status=job.status)

    def why_build(scene_id: str, entry: dict[str, Any] | None) -> str | None:
        """Why a scene nobody is generating has to be built before it is
        answered. None when what the manifest has for it stands."""
        if entry is None:
            return "no scene yet"
        if entry.get("error") == UNFINISHED:
            return "its generation was cut short"
        if entry["status"] == "failed":
            return None  # the same persona would fail the same way
        if _files_missing(config.out_dir, scene_id, entry):
            return "its files are gone"
        if (
            entry["status"] == "fallback"
            and seconds_since(entry.get("updated_at")) >= FALLBACK_RETRY_AFTER_S
            and client.status().connected
        ):
            return "no 360 image yet and Pegasus can be reached"
        return None

    def build_if_needed(scene_id: str, entry: dict[str, Any] | None) -> SceneJob | None:
        """Start a scene nobody is generating, from the participant's persona
        file, when there is none to answer with or its 360 image can be made
        after all. None when nothing was started: there is no need, building
        on request is switched off, or there is no such file."""
        auto = config.auto_create
        if auto is None:
            return None
        try:
            path = auto.personas_dir / f"{scene_id_for(scene_id)}.json"
        except ValueError:
            return None
        if not path.is_file():
            return None
        why = why_build(scene_id, entry)
        if why is None:
            return None
        raw = _read_persona(str(path))

        sky_mode, seed = auto.sky_mode, None
        if entry is not None:
            # The scene it was: the sky that was asked for (a fallback was
            # asked for as a panorama), and the seed. A Slurm job still out
            # there is only picked up by a run with its seed.
            fell_back = entry["status"] == "fallback"
            sky_mode = "panorama" if fell_back else entry.get("sky_mode", "panorama")
            seed = entry.get("seed")
            if seed is None:
                spec = _read_spec(config.out_dir, scene_id)
                seed = spec.seed if spec is not None else None

        log.info("%s: %s, building it from %s", scene_id, why, path)
        try:
            scenes.submit(SceneJob(scene_id, seed, raw, sky_mode))
        except SceneInProgress:
            pass  # another request started it in the meantime
        return scenes.wait(scene_id, AUTO_CREATE_WAIT_S)

    @app.get("/scenes/{scene_id}", response_model=SceneResponse)
    def get_scene(scene_id: str) -> SceneResponse:
        """Everything for one participant: status, Slurm progress, persona, spec, files.

        With `auto_create` configured, the scene is built now, from that
        participant's persona file, when there is none to answer with: nobody
        created it, its files are gone, or its generation was cut short. A
        scene that fell back to the procedural sky is built again once Pegasus
        can be reached, so it gets its 360 image whenever one can be made.
        """
        job = scenes.get(scene_id)
        if job is None or not job.active:
            entry = read_manifest(config.out_dir).get(scene_id)
            job = build_if_needed(scene_id, entry) or job
            if job is None:
                if entry is None:
                    raise HTTPException(status_code=404, detail=f"unknown scene {scene_id!r}")
                # Not in memory (e.g. after a restart): answer from the files on disk.
                return SceneResponse(
                    scene_id=scene_id,
                    status=(
                        SceneStatus.READY
                        if entry["status"] in ("ok", "fallback")
                        else SceneStatus.FAILED
                    ),
                    spec=_read_spec(config.out_dir, scene_id),
                    files=entry.get("files", []),
                    error=entry.get("error"),
                )
        return SceneResponse(
            scene_id=job.scene_id, status=job.status, detail=job.detail,
            persona=job.persona, spec=job.spec, files=job.files, error=job.error,
        )

    @app.get("/scenes/{scene_id}/files/{name}")
    def get_scene_file(scene_id: str, name: str) -> FileResponse:
        """Download one of a scene's files, e.g. its skybox PNG.

        Only names listed in the scene's manifest entry are served. Since that
        list is written by the orchestrator, nothing else on disk is reachable,
        and a scene still being generated has nothing to download yet.
        """
        entry = read_manifest(config.out_dir).get(scene_id)
        if entry is None or name not in entry.get("files", []):
            raise HTTPException(status_code=404, detail=f"no file {name!r} for scene {scene_id!r}")
        try:
            folder = scene_dir(config.out_dir, scene_id).resolve()
        except ValueError:
            raise HTTPException(status_code=404, detail=f"unknown scene {scene_id!r}")
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file():
            raise HTTPException(status_code=404, detail=f"no file {name!r} for scene {scene_id!r}")
        return FileResponse(path)

    # -- /skyboxes: one image from a hand-written prompt ------------------

    @app.post("/skyboxes", response_model=SkyboxJobResponse, status_code=202)
    async def create_skybox(body: SkyboxBody) -> SkyboxJobResponse:
        job = runner.submit(SkyboxRequest(**body.model_dump()))
        return to_response(job)

    @app.get("/skyboxes/{job_id}", response_model=SkyboxJobResponse)
    async def get_skybox(job_id: str) -> SkyboxJobResponse:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"unknown job {job_id!r}. Job state is in memory and is lost "
                    "on restart; the files in out/ are the durable record."
                ),
            )
        return to_response(job)

    return app


def _read_persona(persona_file: str) -> dict[str, Any]:
    path = Path(persona_file)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise HTTPException(status_code=422, detail=f"persona file not found: {persona_file}")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot read {persona_file}: {exc}")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail=f"{persona_file} is not a JSON object")
    return raw


def _files_missing(out_dir: Path, scene_id: str, entry: dict[str, Any]) -> bool:
    """Whether a file the manifest lists for the scene is not on disk."""
    try:
        folder = scene_dir(out_dir, scene_id)
    except ValueError:
        return False
    return not all((folder / name).is_file() for name in entry.get("files", []))


def _read_spec(out_dir: Path, scene_id: str) -> SceneSpec | None:
    try:
        path = scene_spec_path(out_dir, scene_id)
        return SceneSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# Module-level app for `uvicorn scene_orchestrator.api.app:app`.
# Built lazily via __getattr__ so importing this module never reads config --
# tests build their own app with create_app().
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
