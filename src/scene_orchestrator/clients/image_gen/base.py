"""Types shared by the image-gen client and its callers.

generate_skybox() does not raise for upstream trouble. No SSH connection, a
failed Slurm job and a timeout all come back as a failed SkyboxResult, because
every pipeline step has to end in a recorded outcome, never in an unhandled
exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ...outputs import SCENE_ID_PATTERN


class ImageGenError(Exception):
    """One failed attempt.

    `retryable` is False when sending the same request again cannot help,
    e.g. image-gen rejected the request itself. `pending_job_id` names a Slurm
    job this attempt left running on Pegasus, so that a later run can pick it
    up instead of submitting another.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        status_code: int | None = None,
        pending_job_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.pending_job_id = pending_job_id


class SkyboxRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: str = Field(min_length=1)
    seed: int = Field(ge=0)
    scene_id: str = Field(pattern=SCENE_ID_PATTERN)


@dataclass(frozen=True)
class SkyboxResult:
    ok: bool
    attempts: int
    png: Path | None = None
    sidecar: Path | None = None
    error: str | None = None
    # On failure: a Slurm job that is still running on Pegasus, if any.
    pending_job_id: str | None = None


class ImageGenStatus(BaseModel):
    """What /health reports about the way to Pegasus."""

    login: str
    # Whether the shared SSH connection you opened by hand is up.
    connected: bool
    error: str | None = None
