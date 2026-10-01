"""SceneSpec: the central contract between the orchestrator and Unity.

Built from a persona by rules first; the skybox prompt is derived from it
afterwards, so the sky and the procedural scene cannot disagree. Unity
validates the same structure against schema/scene_spec.schema.json.

Every numeric field is range-constrained, so an out-of-range value fails here
rather than producing a strange scene in the headset. Props name assets from
the curated Unity library; no geometry is ever generated.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..outputs import SCENE_ID_PATTERN

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Degrees = Annotated[float, Field(ge=0.0, le=360.0)]
Pixels = Annotated[int, Field(gt=0)]

Biome = Literal[
    "coastal_pine", "open_meadow", "forest_clearing", "snowy_valley",
    "lakeside", "rocky_shore", "birch_grove", "alpine_basin",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Skybox(_Strict):
    uri: str | None = None  # set once the image exists
    prompt: str | None = None  # derived from this spec
    projection: Literal["equirect"] = "equirect"
    resolution: tuple[Pixels, Pixels]  # (width, height)


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
    skybox: Skybox
    terrain: Terrain
    props: list[Prop]
    lighting: Lighting
    spatial: Spatial
    audio: list[AudioLayer]
    motion: Motion
    activity: Activity
