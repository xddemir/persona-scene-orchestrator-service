"""PersonaProfile and SceneSpec: valid input builds, bad input is rejected."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from scene_orchestrator.models import PersonaProfile, SceneSpec
from scene_orchestrator.models.schema import scene_spec_schema_text

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "schema" / "scene_spec.schema.json"

PERSONA = {
    "participant_id": "p07",
    "source": "bfi2s",
    "traits": {
        "openness": 0.8,
        "conscientiousness": 0.5,
        "extraversion": 0.2,
        "agreeableness": 0.6,
        "neuroticism": 0.7,
    },
}

SPEC = {
    "scene_id": "p07",
    "seed": 1234,
    "biome": "forest_clearing",
    "skybox": {"resolution": [2048, 1024]},
    "terrain": {
        "profile": "gentle_slope",
        "ground_material": "moss",
        "water": {"present": True, "motion": 0.2},
    },
    "props": [
        {"asset": "pine_tall", "count": 120, "distribution": "poisson",
         "min_spacing": 3.0, "exclude_radius": 8.0, "scale_jitter": 0.15},
        {"asset": "boulder", "count": 12, "distribution": "clustered",
         "exclude_radius": 6.0, "cluster_count": 3},
    ],
    "lighting": {"sun_elevation_deg": 12, "sun_azimuth_deg": 95,
                 "color_temperature_k": 3200, "fog_density": 0.04},
    "spatial": {"enclosure": 0.7, "sightline_distance_m": 40,
                "prospect_refuge_bias": "refuge"},
    "audio": [{"layer": "birdsong", "gain": 0.4, "event_rate_per_min": 6},
              {"layer": "stream", "gain": 0.3}],
    "motion": {"wind_strength": 0.2, "wind_direction_deg": 270},
    "activity": {"type": "seated_breathing"},
}


def _with(data: dict, dotted: str, value) -> dict:
    """A deep copy of `data` with one field replaced, e.g. "props.0.count"."""
    data = copy.deepcopy(data)
    *path, last = dotted.split(".")
    target = data
    for key in path:
        target = target[int(key)] if isinstance(target, list) else target[key]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value
    return data


# -- persona ----------------------------------------------------------------


def test_a_valid_persona_builds():
    persona = PersonaProfile.model_validate(PERSONA)
    assert persona.schema_version == "1.0"
    assert persona.traits.neuroticism == 0.7
    assert persona.confidence is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("traits.openness", 1.2),
        ("traits.neuroticism", -0.1),
        ("source", "survey"),
    ],
)
def test_an_invalid_persona_is_rejected(field, value):
    with pytest.raises(ValidationError):
        PersonaProfile.model_validate(_with(PERSONA, field, value))


# -- scene spec -------------------------------------------------------------


def test_a_valid_scene_spec_builds():
    spec = SceneSpec.model_validate(SPEC)
    assert spec.spec_version == "0.1"
    assert spec.skybox.projection == "equirect"
    assert spec.skybox.resolution == (2048, 1024)
    assert spec.props[1].cluster_count == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lighting.sun_elevation_deg", 91),
        ("lighting.sun_azimuth_deg", 361),
        ("lighting.color_temperature_k", 1500),
        ("lighting.fog_density", 0.2),
        ("spatial.enclosure", 1.5),
        ("spatial.sightline_distance_m", 5),
        ("terrain.water.motion", -0.1),
        ("props.0.count", 1001),
        ("props.0.scale_jitter", 2.0),
        ("audio.0.gain", 1.1),
        ("motion.wind_strength", 1.01),
        ("motion.wind_direction_deg", -1),
        ("skybox.resolution", [0, 1024]),
        ("seed", -1),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(_with(SPEC, field, value))


def test_an_unknown_biome_is_rejected():
    with pytest.raises(ValidationError, match="biome"):
        SceneSpec.model_validate(_with(SPEC, "biome", "volcano"))


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(_with(SPEC, "lighting.sun_brightness", 1.0))


def test_a_scene_id_cannot_escape_out():
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(_with(SPEC, "scene_id", "../p07"))


# -- JSON Schema for Unity --------------------------------------------------


def test_the_committed_schema_matches_the_models():
    assert SCHEMA_FILE.read_text(encoding="utf-8") == scene_spec_schema_text(), (
        "schema/scene_spec.schema.json is stale: "
        "run python -m scene_orchestrator.models.schema"
    )
