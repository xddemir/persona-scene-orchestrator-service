"""The place's axes and which combinations Unity builds as asked.

A copy of Unity's Composition/SceneAxes.cs (the two tables) and of what
Landscape.Resolve and EnvironmentRules.Apply do with a spec's terrain. Unity
quietly changes an axis combination that makes no sense (a plain with water
below, a terraced mountain with a lake) and logs a substitution. Here such a
combination is an error instead, so the spec on disk says what the headset
shows.

Every check runs on resolved axes: the spec's value, else its archetype's
setting's (CATALOGUE_V1). A terrain that leaves landform out can't be checked
on its own; it is the setting's landform that meets the spec's water.
SceneSpec and places_v1.yaml both use resolve_axes and axis_problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import NamedTuple

from .catalogue import CATALOGUE_V1, describe_water

# Each axis's values, in the order of Unity's enums (SettingProfile.cs).
LANDFORMS = ("plain", "bowl", "ridge", "peak", "shore", "terraces", "valley")
POSITIONS = ("bottom", "mid", "top")
WATER_KINDS = ("none", "river", "lake", "sea")
WATER_RELATIONS = ("absent", "beside", "below", "distant")
VEGETATIONS = ("sparse", "moderate", "dense")
SEASONS = ("summer", "spring", "autumn", "winter")

# Older specs' spawn.placement, as the position it means (Landscape.PositionOfPlacement).
PLACEMENT_POSITION = {
    "basin_floor": "bottom",
    "centre": "mid",
    "shore": "mid",
    "crest": "top",
    "summit": "top",
}


class Fix(NamedTuple):
    """A table cell that is not OK: what Unity uses instead, and why."""

    use: str
    why: str


_PLAIN_IS_LEVEL = "a plain has no top or bottom"
_NOTHING_TO_FALL_FROM = "a plain has nothing to fall from"
_BOWL_HAS_NO_EDGE = "a bowl has no edge to fall from"
_SHORE_NEEDS_WATER = "a shore without water is just a plain"
_SHORE_IS_THE_WATERLINE = "the shore is the waterline"
_SHORE_BOTTOM_IS_UNDERWATER = "a shore's lowest ground is under the sea"
_MOUNTAIN_IS_DRY = "the terraced mountain stands on dry land"
_VALLEY_HAS_NO_TOP = "the top of a valley is not a valley"
OK = None

# landform x water relation (SceneAxes.WaterTable). None: OK.
WATER_TABLE: dict[str, dict[str, Fix | None]] = {
    "plain": {"absent": OK, "beside": OK,
              "below": Fix("distant", _NOTHING_TO_FALL_FROM), "distant": OK},
    "bowl": {"absent": OK, "beside": OK,
             "below": Fix("distant", _BOWL_HAS_NO_EDGE), "distant": OK},
    "ridge": {"absent": OK, "beside": OK, "below": OK, "distant": OK},
    "peak": {"absent": OK, "beside": OK, "below": OK, "distant": OK},
    "shore": {"absent": Fix("plain", _SHORE_NEEDS_WATER), "beside": OK,
              "below": Fix("beside", _SHORE_IS_THE_WATERLINE), "distant": OK},
    "terraces": {"absent": OK, "beside": Fix("absent", _MOUNTAIN_IS_DRY),
                 "below": Fix("absent", _MOUNTAIN_IS_DRY),
                 "distant": Fix("absent", _MOUNTAIN_IS_DRY)},
    "valley": {"absent": OK, "beside": OK, "below": OK, "distant": OK},
}

# landform x position (SceneAxes.SpawnTable). None: OK.
SPAWN_TABLE: dict[str, dict[str, Fix | None]] = {
    "plain": {"bottom": Fix("mid", _PLAIN_IS_LEVEL), "mid": OK,
              "top": Fix("mid", _PLAIN_IS_LEVEL)},
    "bowl": {"bottom": OK, "mid": OK, "top": OK},
    "ridge": {"bottom": OK, "mid": OK, "top": OK},
    "peak": {"bottom": OK, "mid": OK, "top": OK},
    "shore": {"bottom": Fix("mid", _SHORE_BOTTOM_IS_UNDERWATER), "mid": OK, "top": OK},
    "terraces": {"bottom": OK, "mid": OK, "top": OK},
    "valley": {"bottom": OK, "mid": OK, "top": Fix("mid", _VALLEY_HAS_NO_TOP)},
}


@dataclass(frozen=True)
class Axes:
    """A place's axes, every one settled: the spec's, else the setting's."""

    landform: str
    position: str
    water: str
    water_relation: str
    vegetation: str
    season: str


def setting_relation(archetype: str) -> str:
    """The setting's water relation as Unity reads it: water from before
    relations (named, with relation absent) is beside."""
    setting = CATALOGUE_V1.archetypes[archetype]
    if setting.water != "none" and setting.water_relation == "absent":
        return "beside"
    return setting.water_relation


def resolve_axes(
    archetype: str,
    *,
    landform: str | None = None,
    macro_shape: str | None = None,
    position: str | None = None,
    placement: str | None = None,
    water: str | None = None,
    water_relation: str | None = None,
    vegetation: str | None = None,
    season: str | None = None,
) -> Axes:
    """The axes Unity settles on before it checks them (Landscape.Resolve):
    each is the spec's, else the setting's. The older names come first and
    the axes win: landform over macro_shape, position over spawn.placement."""
    setting = CATALOGUE_V1.archetypes[archetype]
    return Axes(
        landform=landform or macro_shape or setting.landform,
        position=position or (PLACEMENT_POSITION[placement] if placement else setting.position),
        water=water or setting.water,
        water_relation=water_relation or setting_relation(archetype),
        vegetation=vegetation or setting.vegetation,
        season=season or setting.season,
    )


def setting_problems(archetype: str) -> list[str]:
    """Where the setting's own values break its rules (EnvironmentRules.Problems):
    a Must it lacks, or a Never it has, which Unity would take away."""
    setting = CATALOGUE_V1.archetypes[archetype]
    rules, name = setting.rules, f"a {archetype}"
    problems = []
    woods = setting.woods_density > 0 and setting.woods_arc_deg > 0
    if rules.trees == "must" and not woods:
        problems.append(f"{name} must have woods, but its woods density or sector is 0")
    if rules.trees == "never" and woods:
        problems.append(f"{name} never has woods, but its setting has some (Unity leaves them out)")
    if rules.mountains == "must" and setting.mountain_arc_deg <= 0:
        problems.append(f"{name} must have mountains, but its mountain sector is 0°")
    if rules.mountains == "never" and setting.mountain_arc_deg > 0:
        problems.append(f"{name} never has mountains, but its setting has a sector "
                        "(Unity leaves it out)")
    return problems


def axis_problems(archetype: str, axes: Axes) -> list[str]:
    """Everything Unity would change about these axes, one line each; empty
    when it builds them as they are. In Unity's order: the environment's
    rules, then the water against its relation, then SceneAxes' two tables."""
    rules, name = CATALOGUE_V1.archetypes[archetype].rules, f"a {archetype}"
    problems = setting_problems(archetype)

    if not rules.allows(axes.water):
        if axes.water == "none":
            why = f"{name} always has {describe_water(rules.must_water)}"
        elif rules.water(axes.water) == "never":
            why = f"{name} never has {describe_water(axes.water)}"
        else:
            why = f"{name}'s water is always {describe_water(rules.must_water)}"
        problems.append(f"water {axes.water}: {why}")

    if axes.water != "none" and axes.water_relation == "absent":
        problems.append(f"water {axes.water} with water_relation absent: "
                        "name where the water is, or no water")
    elif axes.water == "none" and axes.water_relation != "absent":
        problems.append(f"water_relation {axes.water_relation} with no water: "
                        "there is no water to place")

    fix = WATER_TABLE[axes.landform][axes.water_relation]
    if fix is not None:
        problems.append(f"{axes.landform} + water {axes.water_relation}: {fix.why} "
                        f"(Unity would use {fix.use})")
    fix = SPAWN_TABLE[axes.landform][axes.position]
    if fix is not None:
        problems.append(f"{axes.landform} + position {axes.position}: {fix.why} "
                        f"(Unity would use {fix.use})")
    return problems


@dataclass(frozen=True)
class Combination:
    """One combination of the five axes that Unity builds as it is."""

    landform: str
    position: str
    water: str
    water_relation: str
    vegetation: str


def axis_space() -> list[Combination]:
    """Every combination of landform x position x water x water relation x
    vegetation that both tables accept, with the water and its relation
    agreeing. It knows no settings: an environment's rules narrow it further
    (no sea at a summit), the season widens it."""
    return [
        Combination(landform, position, water, relation, vegetation)
        for landform, position, water, relation, vegetation in product(
            LANDFORMS, POSITIONS, WATER_KINDS, WATER_RELATIONS, VEGETATIONS
        )
        if SPAWN_TABLE[landform][position] is None
        and WATER_TABLE[landform][relation] is None
        and (water != "none") == (relation != "absent")
    ]
