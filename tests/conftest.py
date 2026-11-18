from __future__ import annotations

from pathlib import Path

import pytest

from fakes import PEGASUS
from scene_orchestrator.config import AppConfig, ImageGenConfig, PlacesConfig


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


@pytest.fixture
def config(out_dir: Path) -> AppConfig:
    # Every place in places_v1.yaml is still a candidate.
    return AppConfig(
        out_dir=out_dir,
        image_gen=ImageGenConfig(**PEGASUS),
        places=PlacesConfig(allow_candidates=True),
    )
