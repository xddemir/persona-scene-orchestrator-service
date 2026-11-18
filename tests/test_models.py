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
    "archetype": "forest",
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
        "water": {"present": True, "motion": 0.2, "kind": "river"},
        "spawn": {"placement": "centre", "facing_deg": 185},
    },
    "props": [
        {"asset": "pine_large", "count": 120, "distribution": "poisson",
         "min_spacing": 3.0, "exclude_radius": 8.0, "scale_jitter": 0.15},
        {"asset": "boulder", "count": 12, "distribution": "clustered",
         "exclude_radius": 6.0, "cluster_count": 3},
    ],
    "focal": {"asset": "pine_large", "distance_m": 30, "azimuth_deg": 185},
    "framing": {"asset": "pine_large", "distance_m": 4, "azimuth_deg": 120},
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
    assert spec.spec_version == "0.4"
    assert spec.catalogue_version == "v1"
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
        ("seed", 2**32),  # Unity casts it to a 32-bit int
        ("terrain.spawn.facing_deg", 361),
        ("terrain.spawn.facing_deg", -1),
        ("focal.distance_m", -1),
        ("focal.azimuth_deg", 361),
        ("framing.azimuth_deg", -1),
        # 0 would read as "the archetype's" in Unity, so it can't be asked for.
        ("terrain.amplitude_m", 0),
        ("terrain.feature_scale_m", -5),
        ("terrain.roughness", 0),
        ("terrain.roughness", 1.1),
        ("terrain.water.distance_m", 0),
        ("spatial.silhouette_distance_m", 0),
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


# -- the land and the view (0.3) --------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("archetype", "volcano"),
        ("terrain.spawn.placement", "cave"),
        ("terrain.macro_shape", "volcano"),
        ("terrain.water.kind", "pond"),
        ("terrain.landform", "volcano"),
        ("terrain.position", "middle"),
        ("terrain.water_relation", "under"),
        ("terrain.vegetation", "lush"),
        ("terrain.season", "fall"),
    ],
)
def test_names_unity_does_not_build_are_rejected(field, value):
    with pytest.raises(ValidationError, match=field.rsplit(".", 1)[-1]):
        SceneSpec.model_validate(_with(SPEC, field, value))


@pytest.mark.parametrize(
    ("section", "field"),
    [(None, "archetype"), (None, "focal"), (None, "framing"), ("terrain", "spawn"),
     ("terrain.spawn", "placement"), ("terrain.water", "kind")],
)
def test_a_spec_without_its_land_or_view_is_rejected(section, field):
    """Unity reads the facing only with a position or a placement: SPEC has
    no position, so its placement can't go."""
    spec = copy.deepcopy(SPEC)
    target = spec
    for key in section.split(".") if section else ():
        target = target[key]
    del target[field]
    with pytest.raises(ValidationError, match=field):
        SceneSpec.model_validate(spec)


@pytest.mark.parametrize(("present", "kind"), [(True, "none"), (False, "river")])
def test_water_present_must_agree_with_its_kind(present, kind):
    spec = _with(_with(SPEC, "terrain.water.present", present), "terrain.water.kind", kind)
    with pytest.raises(ValidationError, match="present"):
        SceneSpec.model_validate(spec)


def test_a_water_azimuth_needs_its_distance():
    """Unity reads azimuth_deg only together with distance_m."""
    with pytest.raises(ValidationError, match="distance_m"):
        SceneSpec.model_validate(_with(SPEC, "terrain.water.azimuth_deg", 90))
    placed = _with(_with(SPEC, "terrain.water.azimuth_deg", 90), "terrain.water.distance_m", 20)
    assert SceneSpec.model_validate(placed).terrain.water.distance_m == 20


def test_overrides_default_to_null_so_unity_uses_the_archetype():
    """Unity reads null as 0 or "", which both mean the archetype's value."""
    spec = SceneSpec.model_validate(SPEC).model_dump(mode="json")
    terrain, water = spec["terrain"], spec["terrain"]["water"]
    for value in (terrain["ground_material"], terrain["macro_shape"], terrain["amplitude_m"],
                  terrain["feature_scale_m"], terrain["roughness"], water["distance_m"],
                  water["azimuth_deg"], spec["spatial"]["silhouette_distance_m"],
                  terrain["landform"], terrain["position"], terrain["water_relation"],
                  terrain["vegetation"], terrain["season"], spec["props"][0]["order"]):
        assert value is None


def test_terrain_profile_is_gone():
    """Unity stopped reading it; 0.4 dropped it."""
    with pytest.raises(ValidationError, match="profile"):
        SceneSpec.model_validate(_with(SPEC, "terrain.profile", "gentle_slope"))


# -- the place's axes (0.4): resolved against the setting, then checked -------


def _terrain(spec: dict, **fields) -> dict:
    """`spec` with these terrain fields; water is a kind, placement the spawn's."""
    spec = copy.deepcopy(spec)
    for name, value in fields.items():
        if name == "water":
            spec["terrain"]["water"] = {
                "present": value != "none", "motion": 0.2 if value != "none" else 0.0,
                "kind": value,
            }
        elif name == "placement":
            spec["terrain"]["spawn"]["placement"] = value
        else:
            spec["terrain"][name] = value
    return spec


@pytest.mark.parametrize(
    ("archetype", "fields", "why"),
    [
        # The setting's landform meets the spec's relation: a forest is a plain.
        ("forest", {"water_relation": "below"},
         r"plain \+ water below: a plain has nothing to fall from \(Unity would use distant\)"),
        ("hollow", {"water": "lake", "water_relation": "below"}, r"bowl \+ water below"),
        ("forest", {"water": "none", "water_relation": "beside"}, "with no water"),
        ("forest", {"water": "lake", "water_relation": "absent"}, "with water_relation absent"),
        # A hollow has no water of its own, so a water it is given needs its relation.
        ("hollow", {"water": "river"}, "with water_relation absent"),
        ("summit", {"water": "lake", "water_relation": "beside"}, "a summit never has a lake"),
        ("beach", {"water": "none", "water_relation": "absent"}, "a beach always has the sea"),
        ("beach", {"water": "lake"}, "a beach never has a lake"),
        ("meadow", {"water": "sea"}, "a meadow never has the sea"),
        ("forest", {"position": "top"}, r"plain \+ position top: a plain has no top or bottom"),
        ("forest", {"placement": "crest"}, r"plain \+ position top"),  # the older name
        ("hollow", {"landform": "valley", "position": "top", "placement": None},
         "the top of a valley is not a valley"),
        ("beach", {"water": "sea", "position": "bottom"}, "a shore's lowest ground is under the sea"),
        ("beach", {"water": "sea", "water_relation": "below"}, "the shore is the waterline"),
        ("overlook", {"water": "lake", "landform": "terraces", "water_relation": "beside"},
         "the terraced mountain stands on dry land"),
    ],
)
def test_axes_unity_would_change_are_rejected(archetype, fields, why):
    spec = _terrain({**SPEC, "archetype": archetype}, **fields)
    with pytest.raises(ValidationError, match=why):
        SceneSpec.model_validate(spec)


@pytest.mark.parametrize(
    ("archetype", "fields"),
    [
        ("forest", {"landform": "ridge", "water_relation": "below"}),  # a ridge has an edge
        ("overlook", {"water": "sea", "water_relation": "below", "position": "top",
                      "placement": None}),
        ("hollow", {"landform": "valley", "water": "river", "water_relation": "beside",
                    "position": "bottom", "placement": None}),
        ("hollow", {"water": "lake", "water_relation": "beside"}),
        ("meadow", {"landform": "valley", "water_relation": "distant", "water": "lake"}),
        ("beach", {"water": "sea", "position": "top", "placement": None, "vegetation": "sparse",
                   "season": "autumn"}),
        ("summit", {"water": "none", "landform": "terraces", "position": "bottom",
                    "placement": None, "vegetation": "dense"}),
        ("forest", {"macro_shape": "valley", "water_relation": "below"}),  # valley is a macro shape too
    ],
)
def test_axes_unity_builds_as_they_are_pass(archetype, fields):
    SceneSpec.model_validate(_terrain({**SPEC, "archetype": archetype}, **fields))


def test_position_wins_over_placement_and_still_carries_the_facing():
    spec = _terrain(SPEC, position="mid", placement=None)
    terrain = SceneSpec.model_validate(spec).terrain
    assert (terrain.position, terrain.spawn.placement, terrain.spawn.facing_deg) == ("mid", None, 185)
    # With both, the position is what is checked: crest (top) would fail on a plain.
    SceneSpec.model_validate(_terrain(SPEC, position="mid", placement="crest"))


def test_without_position_or_placement_the_facing_would_be_lost():
    with pytest.raises(ValidationError, match="ignores spawn.facing_deg"):
        SceneSpec.model_validate(_terrain(SPEC, placement=None))


def test_order_is_only_for_rings():
    ringed = _with(_with(SPEC, "props.0.distribution", "ringed"), "props.0.order", 0.8)
    assert SceneSpec.model_validate(ringed).props[0].order == 0.8
    with pytest.raises(ValidationError, match="order is only read for ringed"):
        SceneSpec.model_validate(_with(SPEC, "props.0.order", 0.8))  # poisson
    with pytest.raises(ValidationError):
        SceneSpec.model_validate(_with(ringed, "props.0.order", 1.2))


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
