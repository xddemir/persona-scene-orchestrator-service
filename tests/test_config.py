"""Config files: the committed one loads, and mistakes fail loudly."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fakes import PEGASUS
from scene_orchestrator.config import CONFIG_ENV_VAR, ConfigError, load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _write(tmp_path: Path, image_gen: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"image_gen": image_gen}), encoding="utf-8")
    return path


def test_default_config_describes_pegasus():
    image_gen = load_config(CONFIGS / "default.yaml").image_gen
    assert image_gen.login == "demir@login1.pegasus.kl.dfki.de"
    assert image_gen.remote_out == "/netscratch/demir/image-gen/out"
    assert image_gen.partition == "RTXA6000"
    assert image_gen.account == "ei-external"
    assert image_gen.retries == 1


def test_env_var_selects_the_file(tmp_path, monkeypatch):
    path = _write(tmp_path, {**PEGASUS, "partition": "A100-80GB"})
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    assert load_config().image_gen.partition == "A100-80GB"


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="partiton"):
        load_config(_write(tmp_path, {**PEGASUS, "partiton": "RTXA6000"}))


def test_missing_settings_are_named(tmp_path):
    settings = {k: v for k, v in PEGASUS.items() if k != "login"}
    with pytest.raises(ConfigError, match="login"):
        load_config(_write(tmp_path, settings))


def test_missing_file_says_where_it_looked(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
