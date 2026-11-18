"""SceneSpec: the central contract between the orchestrator and Unity.

Built from a persona by rules first; the skybox prompt is derived from it
afterwards, so the sky and the procedural scene cannot disagree. Unity
validates the same structure against schema/scene_spec.schema.json.

Every numeric field is range-constrained, so an out-of-range value fails here
rather than producing a strange scene in the headset. Props name assets from
the curated Unity library; no geometry is ever generated.

The sky comes in two modes. "panorama" is the generated 360 image (`skybox`);
"procedural" is a sky drawn by a shader in Unity (`procedural_sky`).
`procedural_sky` is filled in on every spec, whichever mode is chosen, so Unity
always has a sky to fall back on. `lighting` and `weather` belong to both
modes: where the sun and the moon stand, the time of day, the light's colour
and strength, fog and rain are stated there once, never repeated in
`procedural_sky`.

The sun may stand below the horizon. The spec says where it is and what is in
the sky (clouds, stars, the moon's phase, rain); how a sky looks at that hour
is the renderer's business, the shader's or the image model's.

Unity builds the land from the spec too (0.3). `archetype` picks one of the
settings in Unity's asset catalogue (`catalogue_version`); the setting gives
every terrain value the spec leaves out. Unity reads a missing number as 0 and
a missing string as "", and both mean "the setting's". So the override fields
below are None by default, and a deliberate 0 can't be expressed.

0.4 adds the place's axes to `terrain` (landform, position, water_relation,
vegetation, season), each None for the setting's. Unity quietly changes a
combination of them that makes no sense; this model rejects it instead, after
resolving every axis against the archetype's setting (axes.py), so a spec on
disk is what the headset shows.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..axes import axis_problems, resolve_axes
from ..outputs import SCENE_ID_PATTERN

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Degrees = Annotated[float, Field(ge=0.0, le=360.0)]  # clockwise from north (+Z)
Pixels = Annotated[int, Field(gt=0)]
Rgb = tuple[Unit, Unit, Unit]
CloudOffset = Annotated[float, Field(ge=0.0, le=100.0)]

# Unity casts the seed to a 32-bit int, so it stays below 2**32.
MAX_SEED = 2**32 - 1

Biome = Literal[
    "coastal_pine", "open_meadow", "forest_clearing", "snowy_valley",
    "lakeside", "rocky_shore", "birch_grove", "alpine_basin",
]
SkyMode = Literal["panorama", "procedural"]
TimeOfDay = Literal["night", "sunrise", "morning", "midday", "afternoon", "sunset"]
PrecipitationKind = Literal["rain", "snow"]
# The settings in Unity's asset catalogue.
Archetype = Literal["hollow", "overlook", "summit", "forest", "meadow", "beach"]
# The place's axes (Unity's SettingProfile.cs enums); axes.py has their tables.
Landform = Literal["plain", "bowl", "ridge", "peak", "shore", "terraces", "valley"]
MacroShape = Landform  # the older name Unity still reads; landform wins over it
Position = Literal["bottom", "mid", "top"]
WaterRelation = Literal["absent", "beside", "below", "distant"]
Vegetation = Literal["sparse", "moderate", "dense"]
Season = Literal["spring", "summer", "autumn", "winter"]
# The older name for position: basin_floor is bottom; centre and shore are
# mid; crest and summit are top.
Placement = Literal["centre", "basin_floor", "crest", "summit", "shore"]
WaterKind = Literal["river", "lake", "sea", "none"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Skybox(_Strict):
    uri: str | None = None  # set once the image exists
    prompt: str | None = None  # derived from this spec
    projection: Literal["equirect"] = "equirect"
    resolution: tuple[Pixels, Pixels]  # (width, height)
    rotation_deg: Degrees = 0.0  # turns the panorama about the vertical axis


class SkyClouds(_Strict):
    """Clouds drawn by the sky shader from noise. They drift with `motion`'s
    wind, so there is no speed or direction of their own here."""

    coverage: Unit  # 0 clear sky .. 1 overcast
    softness: Unit  # 0 crisp edges .. 1 diffuse
    brightness: Unit  # 0 grey .. 1 white
    detail: Unit  # 0 smooth shapes .. 1 intricate
    banding: Unit  # 0 scattered puffs .. 1 regular rows along the wind
    # Where in the noise field the clouds are taken from: from the seed, so
    # two participants with like traits still get different clouds.
    offset: tuple[CloudOffset, CloudOffset]


class SkyStars(_Strict):
    """The stars of the night sky. They come out as the sun goes down, so
    nothing here says whether it is night: `lighting` does."""

    density: Unit  # 0 a few bright stars .. 1 a crowded field
    brightness: Unit  # 0 washed out .. 1 as on a dark, clear night
    milky_way: Unit  # 0 none .. 1 a bright band across the sky


class ProceduralSky(_Strict):
    """Parameters of the procedural sky shader in the Unity project: a colour
    gradient from horizon to zenith, the ground below, a sun with its glow,
    clouds, and the night sky.

    No sun or moon angle, colour temperature, fog or rain here: those are in
    `lighting` and `weather`, which both sky modes read. The colours are the
    daytime ones; the shader takes them down to dusk and night with the sun.
    """

    atmosphere_thickness: float = Field(ge=0.5, le=2.5)  # how far up the horizon's haze reaches
    sky_tint: Rgb  # the colour overhead, by day
    horizon_color: Rgb  # by day
    twilight_color: Rgb  # the glow where the sun rises or sets
    ground_color: Rgb
    exposure: float = Field(ge=0.5, le=2.0)
    sun_size: float = Field(ge=0.01, le=0.5)
    sun_halo: Unit  # strength of the glow around the sun
    stars: SkyStars
    clouds: SkyClouds


class Water(_Strict):
    """Unity goes by `kind`; `present` says the same thing, for older readers.
    Where it lies from the participant is terrain.water_relation."""

    present: bool
    motion: Unit  # not read by Unity yet
    kind: WaterKind
    # Both None: Unity places the water by the relation, from the facing (the
    # setting's own water at its own bearing). With distance_m, azimuth_deg is
    # absolute. For experiments only: the mapper never sends them.
    distance_m: float | None = Field(default=None, gt=0.0)
    azimuth_deg: Degrees | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Water:
        if self.present != (self.kind != "none"):
            raise ValueError(f"present must be {self.kind != 'none'} for kind {self.kind!r}")
        if self.azimuth_deg is not None and self.distance_m is None:
            raise ValueError("azimuth_deg is only read with distance_m")
        return self


class Spawn(_Strict):
    """Which way the participant looks, and in older specs where they sit.

    Unity reads facing_deg only when the spec says where the participant sits:
    terrain.position, or placement here. terrain.position wins over placement,
    so placement is None whenever position is set.
    """

    placement: Placement | None = None
    facing_deg: Degrees


class Terrain(_Strict):
    # The place's axes. Each None: the archetype's setting's. Which
    # combinations Unity builds as they are is checked on SceneSpec, once
    # every axis is resolved against the setting.
    landform: Landform | None = None
    position: Position | None = None  # where on the land the participant sits
    water_relation: WaterRelation | None = None  # where the water is from them
    vegetation: Vegetation | None = None  # every prop count x0.5, x1 or x1.6 in Unity
    season: Season | None = None  # the ground's style for that time of year
    # None: the archetype's ground. A name Unity doesn't know leaves the
    # terrain bare: no paint, no grass, no rocks.
    ground_material: str | None = None
    # Overrides of the archetype's tuned heightmap. None in normal output; for
    # experiments only: the mapper never sends them. 0 would mean "the
    # archetype's", so it is not allowed.
    macro_shape: MacroShape | None = None  # the older name for landform
    amplitude_m: float | None = Field(default=None, gt=0.0)  # lowest to highest point
    feature_scale_m: float | None = Field(default=None, gt=0.0)  # size of the bumps
    roughness: float | None = Field(default=None, gt=0.0, le=1.0)
    water: Water
    spawn: Spawn

    @model_validator(mode="after")
    def _facing_is_read(self) -> Terrain:
        if self.position is None and self.spawn.placement is None:
            raise ValueError(
                "give terrain.position (or the older spawn.placement): without one, "
                "Unity ignores spawn.facing_deg"
            )
        return self


class Anchor(_Strict):
    """One prop placed on purpose, seen from where the participant sits."""

    asset: str  # a prop name in the asset catalogue
    distance_m: float = Field(ge=0.0)
    azimuth_deg: Degrees


class Prop(_Strict):
    asset: str  # a name from the curated Unity asset library
    # Unity multiplies it by the vegetation's scale when terrain.vegetation is set.
    count: int = Field(ge=0, le=1000)
    distribution: Literal["poisson", "clustered", "ringed", "scattered"]
    min_spacing: float | None = Field(default=None, ge=0.0)
    exclude_radius: float = Field(ge=0.0)
    scale_jitter: Unit | None = None
    cluster_count: int | None = Field(default=None, ge=1)
    # ringed only: how exactly the props keep to their rings. 1: on them; 0:
    # each strays up to about a third of a ring step (Unity's PropPlacer).
    order: Unit | None = None

    @model_validator(mode="after")
    def _order_is_for_rings(self) -> Prop:
        if self.order is not None and self.distribution != "ringed":
            raise ValueError(f"order is only read for ringed, not {self.distribution}")
        return self


class Lighting(_Strict):
    sun_elevation_deg: float = Field(ge=-90.0, le=90.0)  # below 0: under the horizon
    sun_azimuth_deg: Degrees
    # Named from the sun's elevation, and from which half of the day it is.
    time_of_day: TimeOfDay
    moon_elevation_deg: float = Field(ge=0.0, le=90.0)
    moon_azimuth_deg: Degrees
    moon_phase: Unit  # 0 new moon .. 1 full moon
    # The next two are of the key light: the sun, and at night the moon.
    color_temperature_k: float = Field(ge=2000.0, le=10000.0)
    intensity: Unit  # 1 is full daylight
    fog_density: float = Field(ge=0.0, le=0.1)


class Weather(_Strict):
    precipitation: Unit  # 0 dry .. 1 steady
    kind: PrecipitationKind  # what falls when something does


class Spatial(_Strict):
    enclosure: Unit  # also how tall the horizon mountains are (the archetype's min -> max)
    sightline_distance_m: float = Field(ge=10.0, le=300.0)
    prospect_refuge_bias: Literal["prospect", "refuge", "balanced"]
    # How far the horizon mountains stand. None: the archetype's. Unity moves
    # them out to at least 290 m.
    silhouette_distance_m: float | None = Field(default=None, gt=0.0)


class AudioLayer(_Strict):
    layer: str
    gain: Unit
    event_rate_per_min: float | None = Field(default=None, ge=0.0)


class Motion(_Strict):
    wind_strength: Unit
    wind_direction_deg: Degrees


class Activity(_Strict):
    type: str


class SceneSpec(_Strict):
    spec_version: str = "0.4"
    scene_id: str = Field(pattern=SCENE_ID_PATTERN)  # also the folder under out/
    seed: int = Field(ge=0, le=MAX_SEED)  # drives all procedural placement
    catalogue_version: str = "v1"  # the Unity asset catalogue the names are from
    archetype: Archetype  # the Unity setting: landform, water, ground, horizon
    biome: Biome
    sky_mode: SkyMode = "panorama"  # which of the next two Unity renders
    skybox: Skybox
    procedural_sky: ProceduralSky
    terrain: Terrain
    props: list[Prop]
    # The prop the view is composed around: Unity keeps a 40 degree corridor
    # to it clear of scattered props.
    focal: Anchor
    # A prop close by at the edge of view, so distance reads. Unity keeps it
    # 3-5 m away.
    framing: Anchor
    lighting: Lighting
    weather: Weather
    spatial: Spatial
    audio: list[AudioLayer]
    motion: Motion
    activity: Activity

    @model_validator(mode="after")
    def _unity_builds_the_place_as_given(self) -> SceneSpec:
        """Resolved first, then checked: a terrain that leaves landform out
        meets the setting's landform, and that pair is what Unity builds."""
        terrain = self.terrain
        axes = resolve_axes(
            self.archetype,
            landform=terrain.landform,
            macro_shape=terrain.macro_shape,
            position=terrain.position,
            placement=terrain.spawn.placement,
            water=terrain.water.kind,
            water_relation=terrain.water_relation,
            vegetation=terrain.vegetation,
            season=terrain.season,
        )
        problems = axis_problems(self.archetype, axes)
        if problems:
            raise ValueError(
                f"Unity would not build this {self.archetype} as given: " + "; ".join(problems)
            )
        return self
