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
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..outputs import SCENE_ID_PATTERN

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Degrees = Annotated[float, Field(ge=0.0, le=360.0)]
Pixels = Annotated[int, Field(gt=0)]
Rgb = tuple[Unit, Unit, Unit]
CloudOffset = Annotated[float, Field(ge=0.0, le=100.0)]

Biome = Literal[
    "coastal_pine", "open_meadow", "forest_clearing", "snowy_valley",
    "lakeside", "rocky_shore", "birch_grove", "alpine_basin",
]
SkyMode = Literal["panorama", "procedural"]
TimeOfDay = Literal["night", "sunrise", "morning", "midday", "afternoon", "sunset"]
PrecipitationKind = Literal["rain", "snow"]


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
    present: bool
    motion: Unit


class Terrain(_Strict):
    profile: Literal["flat", "gentle_slope", "rolling", "basin"]
    ground_material: str
    water: Water


class Prop(_Strict):
    asset: str  # a name from the curated Unity asset library
    count: int = Field(ge=0, le=1000)
    distribution: Literal["poisson", "clustered", "ringed", "scattered"]
    min_spacing: float | None = Field(default=None, ge=0.0)
    exclude_radius: float = Field(ge=0.0)
    scale_jitter: Unit | None = None
    cluster_count: int | None = Field(default=None, ge=1)


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
    enclosure: Unit
    sightline_distance_m: float = Field(ge=10.0, le=300.0)
    prospect_refuge_bias: Literal["prospect", "refuge", "balanced"]


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
    spec_version: str = "0.2"
    scene_id: str = Field(pattern=SCENE_ID_PATTERN)  # also the folder under out/
    seed: int = Field(ge=0)  # drives all procedural placement
    biome: Biome
    sky_mode: SkyMode = "panorama"  # which of the next two Unity renders
    skybox: Skybox
    procedural_sky: ProceduralSky
    terrain: Terrain
    props: list[Prop]
    lighting: Lighting
    weather: Weather
    spatial: Spatial
    audio: list[AudioLayer]
    motion: Motion
    activity: Activity
