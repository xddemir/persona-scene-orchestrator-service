from __future__ import annotations

from pathlib import Path

import pytest

from fakes import PEGASUS
from scene_orchestrator.config import AppConfig, ImageGenConfig


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


@pytest.fixture
def config(out_dir: Path) -> AppConfig:
    return AppConfig(out_dir=out_dir, image_gen=ImageGenConfig(**PEGASUS))
