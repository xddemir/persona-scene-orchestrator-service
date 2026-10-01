"""Request and response bodies."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..clients.image_gen import ImageGenStatus
from ..models import PersonaProfile, SceneSpec
from ..outputs import SCENE_ID_PATTERN

# -- /scenes ----------------------------------------------------------------


class SceneStatus(str, Enum):
    QUEUED = "queued"
    BUILDING_SPEC = "building_spec"
    GENERATING_IMAGE = "generating_image"
    READY = "ready"
    FAILED = "failed"


class SceneBody(BaseModel):
    """Exactly one of persona_file or persona."""

    # The example Swagger pre-fills. Without it, Swagger invents one that sets
    # persona_file and persona (and seed 0) at once, which is then rejected.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"persona_file": "fixtures/personas/chatbot/P01.json"}]},
    )

    # A chatbot_v2 persona file, relative to the project root or absolute.
    persona_file: str | None = Field(
        default=None, examples=["fixtures/personas/chatbot/P01.json"]
    )
    # Or the same JSON inline.
    persona: dict[str, Any] | None = None
    # Omitted: derived from the participant id, so it is stable.
    seed: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _exactly_one_persona(self) -> SceneBody:
        if (self.persona_file is None) == (self.persona is None):
            raise ValueError("give exactly one of persona_file or persona")
        return self


class SceneAccepted(BaseModel):
    scene_id: str
    status: SceneStatus


class SceneResponse(BaseModel):
    """Everything about one participant's scene, in one call."""

    scene_id: str  # = participant id
    status: SceneStatus
    # Latest progress, e.g. "Slurm job 123456: RUNNING".
    detail: str | None = None
    # As converted. Null if it could not be converted, or if the scene ran
    # before the server last started (only the files on disk are kept).
    persona: PersonaProfile | None = None
    spec: SceneSpec | None = None
    # Relative to out/<scene_id>/.
    files: list[str] = Field(default_factory=list)
    error: str | None = None


# -- /skyboxes --------------------------------------------------------------


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
