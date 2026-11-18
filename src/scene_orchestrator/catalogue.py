"""Unity's asset catalogue v1: the names a spec may carry, and the six settings.

A hand copy of what the Unity project (UnityRelaxVR) builds from:

    Assets/Catalogue/AssetCatalogue_v1.asset      props and grounds
    Assets/Catalogue/Settings/<Name>.asset         the settings, one per archetype
    Composition/SettingProfile.cs                  what a setting holds
    Composition/EnvironmentRules.cs                May / Must / Never per setting

tests/test_catalogue.py pins every number here, and compares them with the
Unity assets themselves when the Unity project is on this machine, so a change
on either side fails a test instead of drifting.

This lives outside mapping/ because SceneSpec checks its terrain axes against
the settings (axes.py), and the models can't import the mapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Need = Literal["may", "must", "never"]  # EnvironmentRules' Need: May, Must, Never

# The kinds of water, in the order EnvironmentRules looks for a Must.
WATER_KINDS_BY_NEED_ORDER = ("sea", "river", "lake")


@dataclass(frozen=True)
class EnvironmentRules:
    """What one environment must have and never has (EnvironmentRules.cs).

    may: as the setting and the spec say. must: always, a spec can't take it
    away. never: a spec can't ask for it. Where a spec breaks these, Unity
    puts it right and logs a substitution; SceneSpec rejects such a spec
    instead (axes.py).
    """

    sea: Need = "may"
    river: Need = "may"
    lake: Need = "may"
    trees: Need = "may"  # the woods round the participant
    rocks: Need = "may"  # the ground's stones and boulders
    mountains: Need = "may"  # the ridges and mountain prefabs on the horizon
    vegetation: Need = "may"  # the grass and flowers

    def water(self, kind: str) -> Need:
        """The need for a kind of water; may for none."""
        return getattr(self, kind) if kind in WATER_KINDS_BY_NEED_ORDER else "may"

    @property
    def must_have_water(self) -> bool:
        return any(self.water(kind) == "must" for kind in WATER_KINDS_BY_NEED_ORDER)

    @property
    def must_water(self) -> str:
        """The first kind of water that is a must (sea, river, lake); none without one."""
        return next((k for k in WATER_KINDS_BY_NEED_ORDER if self.water(k) == "must"), "none")

    def allows(self, kind: str) -> bool:
        """None only when no kind is a must; a kind when it isn't never, and is
        the must when there is one."""
        if kind == "none":
            return not self.must_have_water
        need = self.water(kind)
        return need == "must" if self.must_have_water else need != "never"


@dataclass(frozen=True)
class Setting:
    """One archetype's SettingProfile: its axes and the numbers that tune them.

    Unity fills in every axis the spec leaves out from these. The water's
    distance, bearing and width belong to the setting's own water and
    relation; another kind or relation gets that one's defaults.
    """

    # The axes (SettingProfile.cs's enums, as the spec spells them).
    landform: str
    position: str  # where on the land the participant sits
    water: str
    water_relation: str
    vegetation: str
    season: str
    ground: str
    # The setting's own water: from the participant, and from their facing.
    water_distance_m: float
    water_bearing_deg: float
    water_width_m: float
    # The horizon.
    horizon_m: float  # how far away the horizon mountains' crests stand
    mountain_arc_deg: float  # how wide a sector has mountains; 0: none
    # The woods round the participant.
    woods_arc_deg: float
    woods_density: float
    # The clearing they sit in, at spatial.enclosure 0 and 1; (0, 0): none of its own.
    clearing_m: tuple[float, float]
    # minimumFogDensity: Unity never renders less fog here, whatever the spec
    # says. 0 when the asset has none.
    fog_floor: float
    rules: EnvironmentRules = field(default_factory=EnvironmentRules)

    @property
    def has_mountains(self) -> bool:
        """Mountains stand on its horizon: a sector of them, and not ruled out."""
        return self.mountain_arc_deg > 0 and self.rules.mountains != "never"


@dataclass(frozen=True)
class Catalogue:
    version: str
    props: frozenset[str]
    grounds: frozenset[str]
    archetypes: dict[str, Setting]


# The single source for every catalogue name a spec may carry.
CATALOGUE_V1 = Catalogue(
    version="v1",
    # Only props that place something: a catalogue entry with prefabs, or a
    # name an entry answers to (rock_cluster: the Rocks section's "rocks").
    # fern and driftwood have entries without prefabs, and birch_light has
    # none, so they are left out; ferns come from the ground cover anyway.
    props=frozenset({
        "pine_large", "pine_medium", "grass_tuft", "bush", "water_lily",
        "boulder", "rock_small", "rock_cluster",
    }),
    # Normally not sent: each archetype has its own (below).
    grounds=frozenset({"moss_rock_mix", "dry_meadow", "sand_grass_mix", "alpine_rock"}),
    archetypes={
        "hollow": Setting(
            landform="bowl", position="bottom", water="none", water_relation="absent",
            vegetation="dense", season="summer", ground="moss_rock_mix",
            water_distance_m=15.0, water_bearing_deg=0.0, water_width_m=6.0,
            horizon_m=200.0, mountain_arc_deg=360.0,
            woods_arc_deg=360.0, woods_density=1.0, clearing_m=(0.0, 0.0), fog_floor=0.0,
            rules=EnvironmentRules(sea="never", mountains="must"),
        ),
        "overlook": Setting(
            landform="ridge", position="top", water="none", water_relation="absent",
            vegetation="sparse", season="summer", ground="dry_meadow",
            water_distance_m=15.0, water_bearing_deg=0.0, water_width_m=6.0,
            horizon_m=340.0, mountain_arc_deg=200.0,
            woods_arc_deg=200.0, woods_density=2.0, clearing_m=(0.0, 0.0), fog_floor=0.0,
            rules=EnvironmentRules(),
        ),
        # 3 steps, 30 m high, a path winding up
        "summit": Setting(
            landform="terraces", position="top", water="none", water_relation="absent",
            vegetation="sparse", season="summer", ground="alpine_rock",
            water_distance_m=15.0, water_bearing_deg=0.0, water_width_m=6.0,
            horizon_m=420.0, mountain_arc_deg=360.0,
            woods_arc_deg=360.0, woods_density=1.0, clearing_m=(0.0, 0.0), fog_floor=0.0,
            rules=EnvironmentRules(sea="never", river="never", lake="never",
                                   rocks="must", mountains="must"),
        ),
        # Woods closed all round and no mountains: the fog floor hides the
        # small terrain's edge.
        "forest": Setting(
            landform="plain", position="mid", water="lake", water_relation="beside",
            vegetation="dense", season="summer", ground="moss_rock_mix",
            water_distance_m=20.0, water_bearing_deg=40.0, water_width_m=14.0,
            horizon_m=90.0, mountain_arc_deg=0.0,
            woods_arc_deg=360.0, woods_density=1.2, clearing_m=(24.0, 11.0), fog_floor=0.02,
            rules=EnvironmentRules(trees="must", mountains="never", vegetation="must"),
        ),
        "meadow": Setting(
            landform="plain", position="mid", water="lake", water_relation="distant",
            vegetation="moderate", season="summer", ground="dry_meadow",
            water_distance_m=32.0, water_bearing_deg=30.0, water_width_m=36.0,
            horizon_m=280.0, mountain_arc_deg=150.0,
            woods_arc_deg=360.0, woods_density=0.6, clearing_m=(0.0, 0.0), fog_floor=0.0,
            rules=EnvironmentRules(sea="never", vegetation="must"),
        ),
        "beach": Setting(
            landform="shore", position="mid", water="sea", water_relation="beside",
            vegetation="sparse", season="summer", ground="sand_grass_mix",
            water_distance_m=14.0, water_bearing_deg=0.0, water_width_m=10.0,
            horizon_m=260.0, mountain_arc_deg=360.0,
            woods_arc_deg=180.0, woods_density=1.2, clearing_m=(0.0, 0.0), fog_floor=0.0,
            rules=EnvironmentRules(sea="must", river="never", lake="never"),
        ),
    },
)

# Never in `props`: Unity places these itself, on the horizon, along the
# summit's terraces and along the paths. The ground adds grass, small rocks,
# boulders, bushes, water lilies and reeds on its own too.
RESERVED_PROPS = frozenset({"mountain", "cliff", "road_rocks"})


def describe_water(kind: str) -> str:
    """A kind of water in words, as Unity's logs say it."""
    return {"sea": "the sea", "river": "a river", "lake": "a lake"}.get(kind, "no water")
