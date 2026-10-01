"""A single-worker background queue for skybox jobs.

POST returns at once and the worker does the slow part. One worker thread,
not an asyncio task: a skybox blocks for as long as its Slurm job takes (queue
wait, model load, rendering), which is minutes, not milliseconds.

Job state is in memory and is lost on restart. The files in out/ are the
durable record; the API is a view over them.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue

from ..clients.image_gen import SkyboxRequest, SkyboxResult, SlurmImageGenClient
from ..outputs import utc_now_iso
from .models import JobStatus

_SHUTDOWN = object()


@dataclass
class SkyboxJob:
    job_id: str
    request: SkyboxRequest
    status: JobStatus = JobStatus.QUEUED
    # Latest progress from the client, e.g. "Slurm job 123456: PENDING".
    detail: str | None = None
    result: SkyboxResult | None = None
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None


class JobRunner:
    """Owns the worker thread and the job registry.

    get() and submit() hand out copies taken under the lock, so a caller never
    sees a job halfway through an update.
    """

    def __init__(self, client: SlurmImageGenClient, out_dir: Path) -> None:
        self._client = client
        self._out_dir = Path(out_dir)
        self._queue: Queue = Queue()
        self._jobs: dict[str, SkyboxJob] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._work, name="skybox-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=timeout)
        self._thread = None

    # -- api -------------------------------------------------------------

    def submit(self, request: SkyboxRequest) -> SkyboxJob:
        job = SkyboxJob(job_id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self._jobs[job.job_id] = job
            snapshot = dataclasses.replace(job)
        self._queue.put(job)
        return snapshot

    def get(self, job_id: str) -> SkyboxJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dataclasses.replace(job) if job else None

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until no job is queued or running. For tests and scripts."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                busy = any(
                    j.status in (JobStatus.QUEUED, JobStatus.GENERATING_IMAGE)
                    for j in self._jobs.values()
                )
            if not busy:
                return True
            time.sleep(0.02)
        return False

    # -- worker ----------------------------------------------------------

    def _work(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if item is _SHUTDOWN:
                return
            self._run_one(item)

    def _run_one(self, job: SkyboxJob) -> None:
        with self._lock:
            job.status = JobStatus.GENERATING_IMAGE
            job.started_at = utc_now_iso()

        def progress(message: str) -> None:
            with self._lock:
                job.detail = message

        # The client reports upstream failures as a failed result; anything
        # that escapes is a bug, and belongs on the job rather than killing the
        # worker thread.
        try:
            result = self._client.generate_skybox(job.request, self._out_dir, progress)
        except Exception as exc:  # noqa: BLE001
            result = SkyboxResult(ok=False, attempts=1, error=f"{type(exc).__name__}: {exc}")

        with self._lock:
            job.result = result
            job.status = JobStatus.READY if result.ok else JobStatus.FAILED
            job.finished_at = utc_now_iso()
