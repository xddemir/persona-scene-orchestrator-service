"""A single-worker background queue for /scenes.

Same shape as the skybox queue in jobs.py: POST returns at once, one worker
thread runs the pipeline (minutes, mostly the Slurm job), and job state lives
in memory. On restart, GET /scenes falls back to the manifest and the files
in out/, which are the durable record.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

from ..models import PersonaProfile, SceneSpec, SkyMode
from ..pipeline import ScenePipeline
from ..outputs import utc_now_iso
from .models import SceneStatus

_SHUTDOWN = object()
_ACTIVE = (SceneStatus.QUEUED, SceneStatus.BUILDING_SPEC, SceneStatus.GENERATING_IMAGE)


class SceneInProgress(Exception):
    """The same scene_id is already queued or running."""


@dataclass
class SceneJob:
    scene_id: str  # = the participant id: one scene per participant
    seed: int | None
    raw_persona: dict[str, Any]
    sky_mode: SkyMode = "panorama"
    status: SceneStatus = SceneStatus.QUEUED
    detail: str | None = None
    persona: PersonaProfile | None = None
    spec: SceneSpec | None = None
    files: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None


class SceneRunner:
    def __init__(self, pipeline: ScenePipeline) -> None:
        self._pipeline = pipeline
        self._queue: Queue = Queue()
        self._jobs: dict[str, SceneJob] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._work, name="scene-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._queue.put(_SHUTDOWN)
            self._thread.join(timeout=timeout)
            self._thread = None

    def submit(self, job: SceneJob, *, replace: bool = True) -> SceneJob:
        """With replace=False, a scene that already has a job keeps it: that
        job is returned and nothing is queued."""
        with self._lock:
            current = self._jobs.get(job.scene_id)
            if current is not None and not replace:
                return dataclasses.replace(current)
            if current is not None and current.status in _ACTIVE:
                # Two runs would write the same folder and manifest entry.
                raise SceneInProgress(job.scene_id)
            self._jobs[job.scene_id] = job
            snapshot = dataclasses.replace(job)
        self._queue.put(job)
        return snapshot

    def get(self, scene_id: str) -> SceneJob | None:
        with self._lock:
            job = self._jobs.get(scene_id)
            return dataclasses.replace(job) if job else None

    def wait(self, scene_id: str, timeout: float) -> SceneJob | None:
        """The scene's job once it has finished, or as it stands after `timeout` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(scene_id)
            if job is None or job.status not in _ACTIVE or time.monotonic() >= deadline:
                return job
            time.sleep(0.02)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until no scene is queued or running. For tests and scripts."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not any(j.status in _ACTIVE for j in self._jobs.values()):
                    return True
            time.sleep(0.02)
        return False

    def _work(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if item is _SHUTDOWN:
                return
            self._run_one(item)

    def _run_one(self, job: SceneJob) -> None:
        def on_stage(stage: str) -> None:
            with self._lock:
                job.status = SceneStatus(stage)

        def on_progress(message: str) -> None:
            with self._lock:
                job.detail = message

        def on_built(persona: PersonaProfile, spec: SceneSpec) -> None:
            with self._lock:
                job.persona, job.spec = persona, spec

        # run() never raises; this guard only keeps a bug from killing the
        # worker thread.
        try:
            outcome = self._pipeline.run(
                job.raw_persona, job.scene_id, job.seed, job.sky_mode,
                on_stage, on_progress, on_built,
            )
            # A fallback to the procedural sky is ready too, with the reason
            # for it in error.
            ok, files, error = outcome.usable, outcome.files, outcome.error
            persona, spec = outcome.persona, outcome.spec
        except Exception as exc:  # noqa: BLE001
            ok, files, error = False, [], f"{type(exc).__name__}: {exc}"
            persona, spec = job.persona, job.spec

        with self._lock:
            job.status = SceneStatus.READY if ok else SceneStatus.FAILED
            job.persona, job.spec, job.files, job.error = persona, spec, files, error
            job.finished_at = utc_now_iso()
