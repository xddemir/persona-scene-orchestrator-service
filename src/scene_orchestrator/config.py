"""Configuration loading.

    SCENE_ORCHESTRATOR_CONFIG=configs/other.yaml   -> that file
    (unset)                                        -> configs/default.yaml

Relative paths resolve against the working directory, i.e. the project root.
Unknown keys are rejected: a typo should fail at startup, not be silently
ignored and discovered after a batch has run.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_ENV_VAR = "SCENE_ORCHESTRATOR_CONFIG"
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class ConfigError(Exception):
    """Raised for a missing, malformed or invalid configuration file."""


class ImageGenConfig(BaseModel):
    """How skyboxes are made: one Slurm job per skybox on Pegasus."""

    model_config = ConfigDict(extra="forbid")

    # The login node, reached through the shared SSH connection you open by
    # hand; control_path is that connection's socket.
    login: str
    control_path: str = "~/.ssh/pegasus.sock"

    # image-gen on Pegasus, as set up by its scripts/pegasus_setup.sh.
    repo: str
    venv: str
    hf_home: str
    remote_out: str

    # The GPU job.
    partition: str = "RTXA6000"
    account: str
    gpus: int = Field(default=1, gt=0)
    time_limit: str = "00:30:00"

    poll_interval_s: float = Field(default=15.0, gt=0)
    # Submit -> done, queue included. After that the job is cancelled.
    job_timeout_s: float = Field(default=3600.0, gt=0)
    ssh_timeout_s: float = Field(default=60.0, gt=0)
    # Extra attempts after a failed one. The brief: retry once, then fail.
    retries: int = Field(default=1, ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_dir: Path = Path("out")
    image_gen: ImageGenConfig


def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not path.is_file():
        raise ConfigError(
            f"config file not found: {path} (looked in {Path.cwd()}). "
            f"Run from the project root, or set {CONFIG_ENV_VAR}."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return AppConfig.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"invalid config {path}: {exc}") from exc
