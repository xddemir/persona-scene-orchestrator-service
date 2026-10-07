"""PersonaProfile and SceneSpec: valid input builds, bad input is rejected."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from scene_orchestrator.models import (
    Lighting,
    Motion,
    PersonaProfile,
    ProceduralSky,
    SceneSpec,
    SkyClouds,
    Weather,
)
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
    "procedural_sky": {
        "atmosphere_thickness": 1.4, "sky_tint": [0.5, 0.6, 0.8],
        "horizon_color": [0.85, 0.88, 0.9], "twilight_color": [1.0, 0.6, 0.5],
        "ground_color": [0.3, 0.35, 0.2],
        "exposure": 1.1, "sun_size": 0.05, "sun_halo": 0.4,
        "stars": {"density": 0.8, "brightness": 0.6, "milky_way": 0.7},
        "clouds": {"coverage": 0.5, "softness": 0.6, "brightness": 0.8, "detail": 0.7,
                   "banding": 0.3, "offset": [12.5, 80.0]},
    },
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
    "lighting": {"sun_elevation_deg": 12, "sun_azimuth_deg": 95, "time_of_day": "morning",
                 "moon_elevation_deg": 40, "moon_azimuth_deg": 250, "moon_phase": 0.7,
                 "color_temperature_k": 3200, "intensity": 0.6, "fog_density": 0.04},
    "weather": {"precipitation": 0.0, "kind": "rain"},
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
    assert spec.spec_version == "0.2"
    assert spec.skybox.projection == "equirect"
    assert spec.skybox.resolution == (2048, 1024)
    assert spec.skybox.rotation_deg == 0
    assert spec.sky_mode == "panorama"
    assert spec.procedural_sky.sky_tint == (0.5, 0.6, 0.8)
    assert spec.props[1].cluster_count == 3


def test_a_spec_without_a_procedural_sky_is_rejected():
    """It is the fallback for a missing panorama, so it is never optional."""
    spec = {key: value for key, value in SPEC.items() if key != "procedural_sky"}
    with pytest.raises(ValidationError, match="procedural_sky"):
        SceneSpec.model_validate(spec)


def test_the_procedural_sky_repeats_nothing_from_lighting_or_weather():
    """Sun, moon, the light's colour, fog and rain are stated once, for both sky modes."""
    shared = set(Lighting.model_fields) | set(Weather.model_fields)
    assert not set(ProceduralSky.model_fields) & shared


def test_the_sun_may_stand_below_the_horizon():
    night = _with(_with(SPEC, "lighting.sun_elevation_deg", -25), "lighting.time_of_day", "night")
    assert SceneSpec.model_validate(night).lighting.sun_elevation_deg == -25


@pytest.mark.parametrize(
    ("section", "field"),
    [(None, "weather"), ("lighting", "time_of_day"), ("lighting", "moon_phase"),
     ("procedural_sky", "stars"), ("procedural_sky", "twilight_color")],
)
def test_a_spec_without_its_weather_or_night_sky_is_rejected(section, field):
    spec = copy.deepcopy(SPEC)
    del (spec[section] if section else spec)[field]
    with pytest.raises(ValidationError, match=field):
        SceneSpec.model_validate(spec)


def test_clouds_take_their_wind_from_motion():
    """They drift with the scene's wind, so they carry no speed or direction."""
    assert not set(SkyClouds.model_fields) & set(Motion.model_fields)
    assert not any("wind" in name or "speed" in name for name in SkyClouds.model_fields)


def test_a_procedural_sky_without_clouds_is_rejected():
    sky = {key: value for key, value in SPEC["procedural_sky"].items() if key != "clouds"}
    with pytest.raises(ValidationError, match="clouds"):
        SceneSpec.model_validate({**SPEC, "procedural_sky": sky})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lighting.sun_elevation_deg", 91),
        ("lighting.sun_elevation_deg", -91),
        ("lighting.sun_azimuth_deg", 361),
        ("lighting.time_of_day", "dusk"),
        ("lighting.moon_elevation_deg", -1),
        ("lighting.moon_azimuth_deg", 361),
        ("lighting.moon_phase", 1.1),
        ("lighting.color_temperature_k", 1500),
        ("lighting.color_temperature_k", 10001),
        ("lighting.intensity", 1.2),
        ("lighting.fog_density", 0.2),
        ("weather.precipitation", 1.1),
        ("weather.kind", "hail"),
        ("spatial.enclosure", 1.5),
        ("spatial.sightline_distance_m", 5),
        ("terrain.water.motion", -0.1),
        ("props.0.count", 1001),
        ("props.0.scale_jitter", 2.0),
        ("audio.0.gain", 1.1),
        ("motion.wind_strength", 1.01),
        ("motion.wind_direction_deg", -1),
        ("skybox.resolution", [0, 1024]),
        ("skybox.rotation_deg", 361),
        ("procedural_sky.atmosphere_thickness", 0.4),
        ("procedural_sky.atmosphere_thickness", 2.6),
        ("procedural_sky.sky_tint", [0.5, 0.6, 1.2]),
        ("procedural_sky.sky_tint", [0.5, 0.6]),
        ("procedural_sky.ground_color", [-0.1, 0.3, 0.2]),
        ("procedural_sky.exposure", 2.1),
        ("procedural_sky.sun_size", 0.6),
        ("procedural_sky.horizon_color", [0.5, 0.6, 1.2]),
        ("procedural_sky.twilight_color", [1.2, 0.6, 0.5]),
        ("procedural_sky.sun_halo", 1.1),
        ("procedural_sky.stars.density", 1.1),
        ("procedural_sky.stars.brightness", -0.1),
        ("procedural_sky.stars.milky_way", 2),
        ("procedural_sky.clouds.coverage", 1.1),
        ("procedural_sky.clouds.softness", -0.1),
        ("procedural_sky.clouds.brightness", 1.5),
        ("procedural_sky.clouds.detail", -0.2),
        ("procedural_sky.clouds.banding", 2),
        ("procedural_sky.clouds.offset", [12.5, 100.5]),
        ("procedural_sky.clouds.offset", [12.5]),
        ("seed", -1),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    SceneSpec.model_validate(SPEC)  # so the one changed field is what fails
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(_with(SPEC, field, value))


def test_an_unknown_sky_mode_is_rejected():
    with pytest.raises(ValidationError, match="sky_mode"):
        SceneSpec.model_validate(_with(SPEC, "sky_mode", "hdri"))


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
