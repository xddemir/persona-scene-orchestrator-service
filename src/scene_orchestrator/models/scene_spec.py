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
always has a sky to fall back on. `lighting` belongs to both modes: sun angle,
colour temperature and fog are stated there once, never repeated in
`procedural_sky`.
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


class ProceduralSky(_Strict):
    """Parameters of the procedural sky shader in the Unity project: a colour
    gradient from horizon to zenith, the ground below, a sun with its glow,
    and clouds.

    No sun angle, colour temperature or fog here: those are in `lighting`,
    which both sky modes read.
    """

    atmosphere_thickness: float = Field(ge=0.5, le=2.5)  # how far up the horizon's haze reaches
    sky_tint: Rgb  # the colour overhead
    horizon_color: Rgb
    ground_color: Rgb
    exposure: float = Field(ge=0.5, le=2.0)
    sun_size: float = Field(ge=0.01, le=0.5)
    sun_halo: Unit  # strength of the glow around the sun
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
    sun_elevation_deg: float = Field(ge=0.0, le=90.0)
    sun_azimuth_deg: Degrees
    color_temperature_k: float = Field(ge=2000.0, le=7000.0)
    fog_density: float = Field(ge=0.0, le=0.1)


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
    spec_version: str = "0.1"
    scene_id: str = Field(pattern=SCENE_ID_PATTERN)  # also the folder under out/
    seed: int = Field(ge=0)  # drives all procedural placement
    biome: Biome
    sky_mode: SkyMode = "panorama"  # which of the next two Unity renders
    skybox: Skybox
    procedural_sky: ProceduralSky
    terrain: Terrain
    props: list[Prop]
    lighting: Lighting
    spatial: Spatial
    audio: list[AudioLayer]
    motion: Motion
    activity: Activity
