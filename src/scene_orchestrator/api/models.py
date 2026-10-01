"""Request and response bodies."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..clients.image_gen import ImageGenStatus
from ..outputs import SCENE_ID_PATTERN


class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING_IMAGE = "generating_image"
    READY = "ready"
    FAILED = "failed"


class SkyboxBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, examples=["a calm misty forest clearing at dawn"])
    # Required: an unseeded skybox cannot be reproduced.
    seed: int = Field(..., ge=0, examples=[1234])
    # Names the folder under out/, so it is validated like a path segment.
    scene_id: str = Field(..., pattern=SCENE_ID_PATTERN, examples=["p07"])


class SkyboxFiles(BaseModel):
    """Paths relative to out/, with forward slashes."""

    png: str
    sidecar: str


class SkyboxJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    scene_id: str
    seed: int
    prompt: str
    # Latest progress, e.g. "Slurm job 123456: PENDING" while in the queue.
    detail: str | None = None
    files: SkyboxFiles | None = None
    error: str | None = None
    attempts: int | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    image_gen: ImageGenStatus
