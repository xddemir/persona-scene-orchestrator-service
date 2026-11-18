"""The backend's copy of Unity's catalogue, settings and axis tables.

Two kinds of test. The first pins today's numbers here, so a change to the
copy is a deliberate one. The second reads the Unity project itself
(UnityRelaxVR, its working tree) and fails where the copy has drifted from
it. Those skip when the project isn't on this machine: set UNITY_PROJECT to
its folder (in WSL: /mnt/c/Users/<you>/Unity/UnityRelaxVR).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml

from scene_orchestrator.axes import (
    LANDFORMS,
    POSITIONS,
    SEASONS,
    SPAWN_TABLE,
    VEGETATIONS,
    WATER_KINDS,
    WATER_RELATIONS,
    WATER_TABLE,
    setting_problems,
)
from scene_orchestrator.catalogue import CATALOGUE_V1, RESERVED_PROPS, EnvironmentRules, Need
from scene_orchestrator.mapping.rule_mapper import LINEAR_RULES, PLACES
from scene_orchestrator.models import (
    Landform,
    Position,
    Season,
    Vegetation,
    WaterKind,
    WaterRelation,
)

# -- today's numbers, pinned ------------------------------------------------

# From Assets/Catalogue/Settings/*.asset on 2026-11-18 (the working tree).
SETTINGS_V1: dict[str, dict[str, Any]] = {
    "hollow": dict(
        landform="bowl", position="bottom", water="none", water_relation="absent",
        vegetation="dense", season="summer", ground="moss_rock_mix", water_distance_m=15,
        water_bearing_deg=0, water_width_m=6, horizon_m=200, mountain_arc_deg=360,
        woods_arc_deg=360, woods_density=1.0, clearing_m=(0, 0), fog_floor=0,
        rules=EnvironmentRules(sea="never", mountains="must"),
    ),
    "overlook": dict(
        landform="ridge", position="top", water="none", water_relation="absent",
        vegetation="sparse", season="summer", ground="dry_meadow", water_distance_m=15,
        water_bearing_deg=0, water_width_m=6, horizon_m=340, mountain_arc_deg=200,
        woods_arc_deg=200, woods_density=2.0, clearing_m=(0, 0), fog_floor=0,
        rules=EnvironmentRules(),
    ),
    "summit": dict(
        landform="terraces", position="top", water="none", water_relation="absent",
        vegetation="sparse", season="summer", ground="alpine_rock", water_distance_m=15,
        water_bearing_deg=0, water_width_m=6, horizon_m=420, mountain_arc_deg=360,
        woods_arc_deg=360, woods_density=1.0, clearing_m=(0, 0), fog_floor=0,
        rules=EnvironmentRules(sea="never", river="never", lake="never", rocks="must",
                               mountains="must"),
    ),
    "forest": dict(
        landform="plain", position="mid", water="lake", water_relation="beside",
        vegetation="dense", season="summer", ground="moss_rock_mix", water_distance_m=20,
        water_bearing_deg=40, water_width_m=14, horizon_m=90, mountain_arc_deg=0,
        woods_arc_deg=360, woods_density=1.2, clearing_m=(24, 11), fog_floor=0.02,
        rules=EnvironmentRules(trees="must", mountains="never", vegetation="must"),
    ),
    "meadow": dict(
        landform="plain", position="mid", water="lake", water_relation="distant",
        vegetation="moderate", season="summer", ground="dry_meadow", water_distance_m=32,
        water_bearing_deg=30, water_width_m=36, horizon_m=280, mountain_arc_deg=150,
        woods_arc_deg=360, woods_density=0.6, clearing_m=(0, 0), fog_floor=0,
        rules=EnvironmentRules(sea="never", vegetation="must"),
    ),
    "beach": dict(
        landform="shore", position="mid", water="sea", water_relation="beside",
        vegetation="sparse", season="summer", ground="sand_grass_mix", water_distance_m=14,
        water_bearing_deg=0, water_width_m=10, horizon_m=260, mountain_arc_deg=360,
        woods_arc_deg=180, woods_density=1.2, clearing_m=(0, 0), fog_floor=0,
        rules=EnvironmentRules(sea="must", river="never", lake="never"),
    ),
}

# The props that place something in Unity today: entries with prefabs, and
# rock_cluster, which the Rocks section's "rocks" answers to.
PLACING_PROPS_V1 = {
    "pine_large", "pine_medium", "grass_tuft", "bush", "water_lily", "boulder",
    "rock_small", "rock_cluster",
}


def _fields(setting) -> dict[str, Any]:
    return {name: getattr(setting, name) for name in SETTINGS_V1["hollow"]}


@pytest.mark.parametrize("archetype", sorted(SETTINGS_V1))
def test_each_setting_is_pinned(archetype):
    assert _fields(CATALOGUE_V1.archetypes[archetype]) == SETTINGS_V1[archetype]


def test_the_catalogue_is_pinned():
    assert set(CATALOGUE_V1.archetypes) == set(SETTINGS_V1)
    assert CATALOGUE_V1.props == PLACING_PROPS_V1
    assert CATALOGUE_V1.grounds == {"moss_rock_mix", "dry_meadow", "sand_grass_mix", "alpine_rock"}
    # gone in 0.4: nothing in Unity places them
    assert not CATALOGUE_V1.props & {"fern", "driftwood", "birch_light"}


def test_mountains_stand_where_they_should():
    has = {name: s.has_mountains for name, s in CATALOGUE_V1.archetypes.items()}
    assert has == {"hollow": True, "overlook": True, "summit": True, "forest": False,
                   "meadow": True, "beach": True}


def test_every_setting_keeps_its_own_rules():
    for archetype in CATALOGUE_V1.archetypes:
        assert setting_problems(archetype) == []
        setting = CATALOGUE_V1.archetypes[archetype]
        assert setting.rules.allows(setting.water)


def test_the_settings_use_the_specs_names():
    for setting in CATALOGUE_V1.archetypes.values():
        assert setting.landform in get_args(Landform)
        assert setting.position in get_args(Position)
        assert setting.water in get_args(WaterKind)
        assert setting.water_relation in get_args(WaterRelation)
        assert setting.vegetation in get_args(Vegetation)
        assert setting.season in get_args(Season)
        assert setting.ground in CATALOGUE_V1.grounds
        assert (setting.water != "none") == (setting.water_relation != "absent")


def test_the_axes_are_the_specs_literals():
    assert set(LANDFORMS) == set(get_args(Landform))
    assert set(POSITIONS) == set(get_args(Position))
    assert set(WATER_KINDS) == set(get_args(WaterKind))
    assert set(WATER_RELATIONS) == set(get_args(WaterRelation))
    assert set(VEGETATIONS) == set(get_args(Vegetation))
    assert set(SEASONS) == set(get_args(Season))
    for table, columns in ((WATER_TABLE, WATER_RELATIONS), (SPAWN_TABLE, POSITIONS)):
        assert list(table) == list(LANDFORMS)
        for row in table.values():
            assert list(row) == list(columns)


def test_every_place_asks_only_for_props_that_place_something():
    for place in PLACES.places:
        assert place.names <= PLACING_PROPS_V1 - {"water_lily"}, place.id


# -- what Unity's GroundCover reads: don't break it -------------------------

# GroundCover.cs turns these spec fields into how much grass, ferns and
# flowers grow, between these ends (its EnclosureRange, ExposureRange,
# SunElevationRange, CloudDetailRange and ScaleJitterRange). Changing a rule's
# ends here means changing the matching constant in GroundCover.cs with it.
GROUND_COVER_RULES = {
    "spatial.enclosure": (0.15, 0.80),
    "procedural_sky.exposure": (0.7, 1.5),
    "lighting.sun_elevation_deg": (-30, 60),
    "procedural_sky.clouds.detail": (0.20, 0.90),
    "props[].scale_jitter": (0.05, 0.30),
}
GROUND_COVER_CONSTANTS = {
    "spatial.enclosure": "EnclosureRange",
    "procedural_sky.exposure": "ExposureRange",
    "lighting.sun_elevation_deg": "SunElevationRange",
    "procedural_sky.clouds.detail": "CloudDetailRange",
    "props[].scale_jitter": "ScaleJitterRange",
}


@pytest.mark.parametrize("parameter", sorted(GROUND_COVER_RULES))
def test_the_rules_ground_cover_reads_keep_their_ends(parameter):
    rules = {rule.parameter: rule for rule in LINEAR_RULES}
    assert (rules[parameter].at_0, rules[parameter].at_1) == GROUND_COVER_RULES[parameter]


def test_the_rules_unity_also_reads_for_the_land_are_unchanged():
    """Unity sizes the forest's clearing by enclosure and the grass's reach by the sightline."""
    rules = {rule.parameter: rule for rule in LINEAR_RULES}
    assert (rules["spatial.enclosure"].trait, rules["spatial.enclosure"].at_0,
            rules["spatial.enclosure"].at_1) == ("neuroticism", 0.15, 0.80)
    assert (rules["spatial.sightline_distance_m"].trait, rules["spatial.sightline_distance_m"].at_0,
            rules["spatial.sightline_distance_m"].at_1) == ("neuroticism", 200, 40)


# -- against the Unity project itself ---------------------------------------


def _unity_project() -> Path | None:
    candidates = [os.environ.get("UNITY_PROJECT"), Path.home() / "Unity" / "UnityRelaxVR"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "Assets" / "Catalogue").is_dir():
            return Path(candidate)
    return None


UNITY = _unity_project()
NEEDS = get_args(Need)  # may, must, never: Unity's Need enum, in order
needs_unity = pytest.mark.skipif(
    UNITY is None, reason="the Unity project isn't here: set UNITY_PROJECT to UnityRelaxVR"
)
SCRIPTS = Path("Assets/Scripts/SceneBuilder/Composition")


def _unity_asset(path: Path) -> dict[str, Any]:
    """A ScriptableObject asset's fields. Unity's YAML header and tags are
    left out; the rest is plain YAML."""
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text[text.index("MonoBehaviour:"):])["MonoBehaviour"]


def _cs(name: str) -> str:
    return (UNITY / SCRIPTS / name).read_text(encoding="utf-8")


def _enum(source: str, name: str) -> list[str]:
    body = re.search(rf"enum {name}\s*\{{([^}}]*)\}}", source).group(1)
    body = re.sub(r"//[^\n]*", "", body)
    return [part.strip().split("=")[0].strip() for part in body.split(",") if part.strip()]


def _names(values: list[str]) -> list[str]:
    return [value.lower() for value in values]


@needs_unity
def test_unitys_enums_are_in_the_order_the_assets_store():
    source = _cs("SettingProfile.cs")
    assert _names(_enum(source, "Landform")) == list(LANDFORMS)
    assert _names(_enum(source, "SpawnPosition")) == list(POSITIONS)
    assert _names(_enum(source, "WaterKind")) == list(WATER_KINDS)
    assert _names(_enum(source, "WaterRelation")) == list(WATER_RELATIONS)
    assert _names(_enum(source, "Vegetation")) == list(VEGETATIONS)
    assert _names(_enum(source, "Season")) == list(SEASONS)
    assert _names(_enum(_cs("EnvironmentRules.cs"), "Need")) == list(NEEDS)


def _unity_setting(data: dict[str, Any]) -> dict[str, Any]:
    needs = NEEDS
    clearing = data.get("clearingMetres") or {"x": 0, "y": 0}
    return dict(
        landform=LANDFORMS[data["landform"]],
        position=POSITIONS[data["position"]],
        water=WATER_KINDS[data["water"]],
        water_relation=WATER_RELATIONS[data["waterRelation"]],
        vegetation=VEGETATIONS[data["vegetation"]],
        season=SEASONS[data["season"]],
        ground=data["ground"],
        water_distance_m=data["waterDistanceMetres"],
        water_bearing_deg=data["waterBearingDeg"],
        water_width_m=data["waterWidthMetres"],
        horizon_m=data["horizonDistanceMetres"],
        mountain_arc_deg=data["mountainArcDeg"],
        woods_arc_deg=data["woodsArcDeg"],
        woods_density=data["woodsDensity"],
        clearing_m=(clearing["x"], clearing["y"]),
        fog_floor=data.get("minimumFogDensity", 0),
        rules=EnvironmentRules(**{need: needs[value] for need, value in data["rules"].items()}),
    )


@needs_unity
def test_every_setting_matches_its_unity_asset():
    folder = UNITY / "Assets" / "Catalogue" / "Settings"
    seen = {}
    for path in sorted(folder.glob("*.asset")):
        data = _unity_asset(path)
        seen[data["archetype"]] = _unity_setting(data)
    assert set(seen) == set(CATALOGUE_V1.archetypes)
    for archetype, unity in seen.items():
        assert _fields(CATALOGUE_V1.archetypes[archetype]) == unity, archetype


def _catalogue() -> dict[str, Any]:
    return _unity_asset(UNITY / "Assets" / "Catalogue" / "AssetCatalogue_v1.asset")


def _places_something(catalogue: dict[str, Any], name: str) -> bool:
    """AssetCatalogue.FindProp: an entry's own name first, then the others'
    other names; found only with at least one prefab."""
    entries = [*catalogue["props"], catalogue["smallRocks"], catalogue["rocks"],
               catalogue["roadRocks"]]
    entry = next((e for e in entries if e["name"] == name), None) or next(
        (e for e in entries if name in (e.get("alsoCalled") or [])), None
    )
    return entry is not None and any(
        prefab and prefab.get("fileID", 0) != 0 for prefab in entry.get("prefabs") or []
    )


@needs_unity
def test_every_places_props_focal_and_framing_place_something_in_unity():
    catalogue = _catalogue()
    assert catalogue["catalogueVersion"] == CATALOGUE_V1.version
    for place in PLACES.places:
        for name in sorted(place.names):
            assert _places_something(catalogue, name), f"{place.id}: {name} places nothing"
    for name in CATALOGUE_V1.props | RESERVED_PROPS:
        assert _places_something(catalogue, name), name
    for name in ("fern", "driftwood", "birch_light"):  # why no place lists them
        assert not _places_something(catalogue, name)


@needs_unity
def test_the_grounds_match_unitys():
    assert {m["name"] for m in _catalogue()["materials"]} == CATALOGUE_V1.grounds


def _cs_table(source: str, name: str) -> dict[str, list[tuple[str, str] | None]]:
    """A SceneAxes table, row by row: None for OK, else (what Unity uses, why)."""
    reasons = dict(re.findall(r'const string (\w+) = "([^"]*)";', source))
    body = re.search(rf"{name} =\s*\{{(.*?)\n\s*\}};", source, re.S).group(1)
    rows = {}
    for row, cells in re.findall(r"/\*\s*(\w+)\s*\*/\s*\{(.*?)\},", body):
        rows[row.lower()] = [
            None if cell == "OK" else (use.lower(), reasons[why])
            for cell, use, why in re.findall(r"(OK|Use\((\w+),\s*(\w+)\))", cells)
        ]
    return rows


@needs_unity
@pytest.mark.parametrize(("name", "table"), [("WaterTable", WATER_TABLE), ("SpawnTable", SPAWN_TABLE)])
def test_the_axis_tables_match_unitys(name, table):
    unity = _cs_table(_cs("SceneAxes.cs"), name)
    ours = {row: [None if fix is None else (fix.use, fix.why) for fix in cells.values()]
            for row, cells in table.items()}
    assert ours == unity


@needs_unity
def test_ground_cover_reads_the_ends_pinned_here():
    source = _cs("GroundCover.cs")
    for parameter, constant in GROUND_COVER_CONSTANTS.items():
        match = re.search(rf"{constant} = new Vector2\(([-\d.]+)f, ([-\d.]+)f\)", source)
        assert (float(match.group(1)), float(match.group(2))) == pytest.approx(
            GROUND_COVER_RULES[parameter]
        ), constant
