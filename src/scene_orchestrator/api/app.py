"""FastAPI app.

    uvicorn scene_orchestrator.api.app:app --reload --reload-dir src --host 127.0.0.1 --port 8001

Run it in WSL: skyboxes are Slurm jobs submitted over a shared SSH connection
to Pegasus, which Windows' ssh cannot provide. Port 8001 because image-gen's
own dev server uses 8000.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .. import __version__
from ..clients.image_gen import SkyboxRequest, SlurmImageGenClient
from ..config import AppConfig, load_config
from ..outputs import relative_to_out
from .jobs import JobRunner, SkyboxJob
from .models import HealthResponse, SkyboxBody, SkyboxFiles, SkyboxJobResponse

# uvicorn's own logger, so these lines show up in the server console.
log = logging.getLogger("uvicorn.error")


def create_app(
    config: AppConfig | None = None, *, image_gen: SlurmImageGenClient | None = None
) -> FastAPI:
    if config is None:
        config = load_config()

    # Injectable so tests can use a simulated Pegasus.
    client = image_gen or SlurmImageGenClient(config.image_gen)
    runner = JobRunner(client, config.out_dir)

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
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(
        version=__version__,
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.runner = runner

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


# Module-level app for `uvicorn scene_orchestrator.api.app:app`.
# Built lazily via __getattr__ so importing this module never reads config --
# tests build their own app with create_app().
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)
