"""RuleSpecMapper: PersonaProfile + seed + scene_id -> SceneSpec.

A pure, deterministic function. No LLM, and no randomness except one
random.Random(seed), drawn in a fixed order. The same persona and seed always
give a byte-identical spec, which is what makes a scene reproducible and the
mapping inspectable.

Every rule is a module-level constant below, so the whole mapping can be
printed in the thesis (`rules_markdown()`). Each rule names the trait it
derives from. Values interpolate linearly between the trait's two poles:
`at_0` when the trait is 0.0, `at_1` when it is 1.0.

Asset, ground-material and audio-layer names stand in for the curated Unity
library and must match its names once that library exists.
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass

from ..models import (
    Activity,
    AudioLayer,
    Lighting,
    Motion,
    PersonaProfile,
    ProceduralSky,
    Prop,
    SceneSpec,
    Skybox,
    SkyClouds,
    Spatial,
    Terrain,
    Traits,
    Water,
)

# -- trait -> parameter: linear rules ---------------------------------------


@dataclass(frozen=True)
class LinearRule:
    parameter: str
    trait: str
    at_0: float
    at_1: float
    rationale: str


LINEAR_RULES: tuple[LinearRule, ...] = (
    # neuroticism -> shelter and calm
    LinearRule("spatial.enclosure", "neuroticism", 0.15, 0.80,
               "more enclosure, more sense of shelter"),
    LinearRule("spatial.sightline_distance_m", "neuroticism", 200, 40,
               "shorter views, fewer open exposures"),
    LinearRule("lighting.fog_density", "neuroticism", 0.005, 0.045,
               "soft haze closes the space in"),
    LinearRule("motion.wind_strength", "neuroticism", 0.45, 0.12,
               "stiller air, less agitation"),
    LinearRule("terrain.water.motion", "neuroticism", 0.40, 0.10,
               "calmer water"),
    LinearRule("procedural_sky.atmosphere_thickness", "neuroticism", 0.7, 2.2,
               "denser, softer sky"),
    # A factor on SKY_TINT_SATURATION, not a saturation itself.
    LinearRule("procedural_sky.sky_tint.saturation_scale", "neuroticism", 1.0, 0.45,
               "more muted sky colour"),
    LinearRule("procedural_sky.clouds.coverage", "neuroticism", 0.20, 0.70,
               "more cloud overhead, a sheltering cover"),
    LinearRule("procedural_sky.clouds.softness", "neuroticism", 0.25, 0.80,
               "softer, more diffuse clouds"),
    # extraversion -> brightness and liveliness
    LinearRule("lighting.sun_elevation_deg", "extraversion", 6, 40,
               "higher sun, brighter scene"),
    LinearRule("lighting.color_temperature_k", "extraversion", 2900, 5400,
               "warm low light -> neutral daylight"),
    LinearRule("audio.birds.gain", "extraversion", 0.05, 0.25,
               "livelier soundscape"),
    LinearRule("audio.birds.event_rate_per_min", "extraversion", 1, 12,
               "more frequent bird calls"),
    LinearRule("procedural_sky.exposure", "extraversion", 0.7, 1.5,
               "brighter sky"),
    LinearRule("procedural_sky.sun_size", "extraversion", 0.03, 0.12,
               "larger, more present sun disc"),
    LinearRule("procedural_sky.sun_halo", "extraversion", 0.15, 0.70,
               "stronger glow around the sun"),
    LinearRule("procedural_sky.clouds.brightness", "extraversion", 0.70, 1.00,
               "brighter, whiter clouds"),
    # openness -> variety
    LinearRule("props[].scale_jitter", "openness", 0.05, 0.30,
               "more size variation"),
    LinearRule("procedural_sky.clouds.detail", "openness", 0.20, 0.90,
               "more intricate, varied cloud shapes"),
    # How much of HORIZON_ACCENT is mixed into the horizon, not a colour itself.
    LinearRule("procedural_sky.horizon_color.accent", "openness", 0.0, 0.5,
               "a second colour at the horizon"),
    # conscientiousness -> order
    LinearRule("procedural_sky.clouds.banding", "conscientiousness", 0.0, 1.0,
               "clouds in regular rows rather than scattered"),
)

# neuroticism -> prospect/refuge (Appleton). Between the two: "balanced".
REFUGE_ABOVE = 0.6
PROSPECT_BELOW = 0.4

# conscientiousness -> order in prop placement.
RINGED_ABOVE = 0.65  # neat, regular arrangement
CLUSTERED_ABOVE = 0.35  # grouped; below this, "scattered"

# openness -> number of prop types: 2 + round(3 * openness), so 2..5.
PROP_TYPES_MIN, PROP_TYPES_SPAN = 2, 3

# openness -> biome: familiar landscapes for low openness, less common ones for
# high. (upper bound of openness, biomes); the seed picks within the tier.
BIOME_TIERS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0.4, ("open_meadow", "forest_clearing")),
    (0.7, ("lakeside", "birch_grove", "coastal_pine")),
    (1.0, ("alpine_basin", "rocky_shore", "snowy_valley")),
)

# agreeableness: UNMAPPED. No parameter depends on it yet. Which (if any)
# environmental quality agreeableness should drive is an open question in the
# thesis, so it is left out on purpose rather than mapped arbitrarily.

# -- derived from spatial.enclosure (itself from neuroticism) ---------------

# More enclosure -> more and denser props, standing closer to the viewer.
PROP_COUNT = (30, 150)  # per prop type at weight 1.0, enclosure 0 -> 1
EXCLUDE_RADIUS_M = (25.0, 6.0)  # clear area around the viewer
MIN_SPACING_M = (6.0, 2.5)
PROPS_PER_CLUSTER = 20  # for "clustered": cluster_count = count / this

# -- fixed (not persona-driven yet) -----------------------------------------

SKYBOX_RESOLUTION = (2048, 1024)  # equirectangular 2:1, image-gen's default
ACTIVITY = "seated_relaxation"

# -- what each biome is made of ---------------------------------------------


@dataclass(frozen=True)
class BiomeKit:
    terrain: str
    ground_material: str
    water: bool
    # (asset, weight): weight scales the count. The first is the biome's
    # signature asset and is always placed. At least 5, for openness = 1.
    props: tuple[tuple[str, float], ...]
    ambience: tuple[tuple[str, float], ...]  # (layer, gain), fixed
    birds: str  # gain and rate come from extraversion


BIOMES: dict[str, BiomeKit] = {
    "open_meadow": BiomeKit(
        "gentle_slope", "meadow_grass", False,
        (("tall_grass_clump", 1.0), ("wildflower_patch", 0.8), ("oak_solitary", 0.1),
         ("fence_post", 0.15), ("hay_bale", 0.05)),
        (("wind_grass", 0.3), ("insects", 0.15)), "birdsong_meadow",
    ),
    "forest_clearing": BiomeKit(
        "flat", "forest_floor_moss", False,
        (("pine_tall", 1.0), ("fern", 0.8), ("mushroom_cluster", 0.3),
         ("mossy_log", 0.2), ("boulder", 0.15)),
        (("wind_canopy", 0.25), ("leaves_rustle", 0.2)), "birdsong_forest",
    ),
    "lakeside": BiomeKit(
        "gentle_slope", "lake_shore_pebbles", True,
        (("reed_bed", 1.0), ("birch_tall", 0.4), ("willow", 0.3),
         ("smooth_stone", 0.3), ("wooden_jetty", 0.02)),
        (("lake_lapping", 0.35), ("wind_reeds", 0.2)), "birdsong_wetland",
    ),
    "birch_grove": BiomeKit(
        "rolling", "birch_leaf_litter", False,
        (("birch_tall", 1.0), ("fern", 0.6), ("birch_sapling", 0.5),
         ("wildflower_patch", 0.4), ("mossy_log", 0.2)),
        (("wind_canopy", 0.25), ("leaves_rustle", 0.2)), "birdsong_forest",
    ),
    "coastal_pine": BiomeKit(
        "rolling", "sandy_needles", True,
        (("pine_coastal", 1.0), ("dune_grass", 0.8), ("beach_shrub", 0.4),
         ("driftwood", 0.2), ("boulder", 0.15)),
        (("waves_gentle", 0.4), ("wind_pines", 0.2)), "seabirds",
    ),
    "alpine_basin": BiomeKit(
        "basin", "alpine_turf", True,
        (("spruce_alpine", 1.0), ("alpine_flowers", 0.8), ("dwarf_pine", 0.5),
         ("boulder", 0.4), ("scree_patch", 0.3)),
        (("wind_high", 0.3), ("stream", 0.25)), "birdsong_alpine",
    ),
    "rocky_shore": BiomeKit(
        "rolling", "wet_rock", True,
        (("sea_rock", 1.0), ("coastal_grass", 0.6), ("tide_pool", 0.3),
         ("kelp_pile", 0.2), ("driftwood", 0.2)),
        (("waves_rocky", 0.45), ("wind_coast", 0.25)), "seabirds",
    ),
    "snowy_valley": BiomeKit(
        "basin", "snow", False,
        (("spruce_snowy", 1.0), ("snow_drift", 0.6), ("frozen_shrub", 0.4),
         ("boulder_snowy", 0.3), ("deadwood", 0.15)),
        (("wind_snow", 0.3), ("snow_settle", 0.1)), "birdsong_winter",
    ),
}

# -- the procedural sky, per biome ------------------------------------------

# The biome gives the sky its hue (degrees on the colour wheel); neuroticism
# then mutes it, through the saturation_scale rule above. Saturation and
# brightness at neuroticism 0 are the same for every biome, so how muted a sky
# looks depends on the persona alone.
BIOME_SKY_HUE: dict[str, float] = {
    "open_meadow": 210.0,  # clear summer blue
    "forest_clearing": 200.0,
    "lakeside": 205.0,
    "birch_grove": 196.0,
    "coastal_pine": 198.0,  # towards turquoise, sea air
    "alpine_basin": 220.0,  # deeper blue at altitude
    "rocky_shore": 215.0,
    "snowy_valley": 225.0,  # cold, towards violet
}
SKY_TINT_SATURATION = 0.45
SKY_TINT_VALUE = 0.85

# The horizon is a paler, brighter version of the same colour, so it is muted
# by neuroticism along with it. Openness then mixes in HORIZON_ACCENT (the
# horizon_color.accent rule): a second, warmer colour low in the sky.
HORIZON_SATURATION_SCALE = 0.4
HORIZON_VALUE = 0.95
HORIZON_ACCENT = (1.0, 0.82, 0.66)  # soft peach

# Clouds are cut out of a noise field; the seed says where. Unity units.
CLOUD_OFFSET_MAX = 100.0

# What the sky shader paints below the horizon: the biome's ground, as RGB 0..1.
BIOME_GROUND_COLOR: dict[str, tuple[float, float, float]] = {
    "open_meadow": (0.36, 0.45, 0.24),
    "forest_clearing": (0.22, 0.28, 0.18),
    "lakeside": (0.42, 0.42, 0.38),
    "birch_grove": (0.45, 0.38, 0.24),
    "coastal_pine": (0.62, 0.55, 0.42),
    "alpine_basin": (0.38, 0.42, 0.30),
    "rocky_shore": (0.33, 0.34, 0.36),
    "snowy_valley": (0.85, 0.87, 0.90),
}


# -- the mapper -------------------------------------------------------------


class RuleSpecMapper:
    def map(self, persona: PersonaProfile, seed: int, scene_id: str) -> SceneSpec:
        return map_persona(persona, seed, scene_id)


def map_persona(persona: PersonaProfile, seed: int, scene_id: str) -> SceneSpec:
    traits = persona.traits
    rng = random.Random(seed)  # the only source of variation besides traits

    # Draws happen in this fixed order; reordering them changes every scene.
    biome = _pick_biome(traits.openness, rng)
    kit = BIOMES[biome]
    prop_assets = _pick_prop_types(kit, traits.openness, rng)
    sun_azimuth = _r(rng.uniform(0.0, 360.0))
    wind_direction = _r(rng.uniform(0.0, 360.0))
    cloud_offset = (
        _r(rng.uniform(0.0, CLOUD_OFFSET_MAX)),
        _r(rng.uniform(0.0, CLOUD_OFFSET_MAX)),
    )

    enclosure = _linear("spatial.enclosure", traits)
    distribution = _distribution(traits.conscientiousness)

    props = []
    for asset, weight in prop_assets:
        count = min(1000, max(1, round(weight * _lerp(*PROP_COUNT, enclosure))))
        props.append(Prop(
            asset=asset,
            count=count,
            distribution=distribution,
            min_spacing=_r(_lerp(*MIN_SPACING_M, enclosure)),
            exclude_radius=_r(_lerp(*EXCLUDE_RADIUS_M, enclosure)),
            scale_jitter=_linear("props[].scale_jitter", traits),
            cluster_count=(
                max(1, count // PROPS_PER_CLUSTER) if distribution == "clustered" else None
            ),
        ))

    audio = [AudioLayer(layer=layer, gain=gain) for layer, gain in kit.ambience]
    audio.append(AudioLayer(
        layer=kit.birds,
        gain=_linear("audio.birds.gain", traits),
        event_rate_per_min=_linear("audio.birds.event_rate_per_min", traits),
    ))

    return SceneSpec(
        scene_id=scene_id,
        seed=seed,
        biome=biome,
        skybox=Skybox(resolution=SKYBOX_RESOLUTION),
        # Filled in whatever the sky mode: it costs nothing, and it is what
        # Unity renders when there is no panorama.
        procedural_sky=_procedural_sky(biome, traits, cloud_offset),
        terrain=Terrain(
            profile=kit.terrain,
            ground_material=kit.ground_material,
            water=Water(
                present=kit.water,
                motion=_linear("terrain.water.motion", traits) if kit.water else 0.0,
            ),
        ),
        props=props,
        lighting=Lighting(
            sun_elevation_deg=_linear("lighting.sun_elevation_deg", traits),
            sun_azimuth_deg=sun_azimuth,
            color_temperature_k=_linear("lighting.color_temperature_k", traits),
            fog_density=_linear("lighting.fog_density", traits),
        ),
        spatial=Spatial(
            enclosure=enclosure,
            sightline_distance_m=_linear("spatial.sightline_distance_m", traits),
            prospect_refuge_bias=_prospect_refuge(traits.neuroticism),
        ),
        audio=audio,
        motion=Motion(
            wind_strength=_linear("motion.wind_strength", traits),
            wind_direction_deg=wind_direction,
        ),
        activity=Activity(type=ACTIVITY),
    )


# -- the rules, applied -----------------------------------------------------

_RULES = {rule.parameter: rule for rule in LINEAR_RULES}


def _linear(parameter: str, traits: Traits) -> float:
    rule = _RULES[parameter]
    return _r(_lerp(rule.at_0, rule.at_1, getattr(traits, rule.trait)))


def _prospect_refuge(neuroticism: float) -> str:
    if neuroticism > REFUGE_ABOVE:
        return "refuge"
    if neuroticism < PROSPECT_BELOW:
        return "prospect"
    return "balanced"


def _distribution(conscientiousness: float) -> str:
    if conscientiousness > RINGED_ABOVE:
        return "ringed"
    if conscientiousness > CLUSTERED_ABOVE:
        return "clustered"
    return "scattered"


def _procedural_sky(
    biome: str, traits: Traits, cloud_offset: tuple[float, float]
) -> ProceduralSky:
    hue = BIOME_SKY_HUE[biome] / 360.0
    saturation = SKY_TINT_SATURATION * _linear(
        "procedural_sky.sky_tint.saturation_scale", traits
    )
    pale = colorsys.hsv_to_rgb(hue, saturation * HORIZON_SATURATION_SCALE, HORIZON_VALUE)
    accent = _linear("procedural_sky.horizon_color.accent", traits)
    return ProceduralSky(
        atmosphere_thickness=_linear("procedural_sky.atmosphere_thickness", traits),
        sky_tint=_rgb(colorsys.hsv_to_rgb(hue, saturation, SKY_TINT_VALUE)),
        horizon_color=_rgb(_lerp(p, a, accent) for p, a in zip(pale, HORIZON_ACCENT)),
        ground_color=BIOME_GROUND_COLOR[biome],
        exposure=_linear("procedural_sky.exposure", traits),
        sun_size=_linear("procedural_sky.sun_size", traits),
        sun_halo=_linear("procedural_sky.sun_halo", traits),
        clouds=SkyClouds(
            coverage=_linear("procedural_sky.clouds.coverage", traits),
            softness=_linear("procedural_sky.clouds.softness", traits),
            brightness=_linear("procedural_sky.clouds.brightness", traits),
            detail=_linear("procedural_sky.clouds.detail", traits),
            banding=_linear("procedural_sky.clouds.banding", traits),
            offset=cloud_offset,
        ),
    )


def _rgb(channels) -> tuple[float, float, float]:
    r, g, b = (_r(channel) for channel in channels)
    return (r, g, b)


def _pick_biome(openness: float, rng: random.Random) -> str:
    for upper, biomes in BIOME_TIERS:
        if openness < upper or upper == BIOME_TIERS[-1][0]:
            return rng.choice(biomes)
    raise AssertionError("unreachable: the last tier covers openness = 1.0")


def _pick_prop_types(
    kit: BiomeKit, openness: float, rng: random.Random
) -> list[tuple[str, float]]:
    n = PROP_TYPES_MIN + _round_half_up(PROP_TYPES_SPAN * openness)
    signature, *others = kit.props
    chosen = rng.sample(others, n - 1)
    # Keep the kit's order, so a spec reads the same way every time.
    return [signature] + [p for p in others if p in chosen]


def _lerp(at_0: float, at_1: float, t: float) -> float:
    return at_0 + (at_1 - at_0) * t


def _round_half_up(x: float) -> int:
    # Not round(): Python rounds halves to even, so 2.5 -> 2 but 3.5 -> 4.
    return int(x + 0.5)


def _r(x: float) -> float:
    # Three decimals: keeps the JSON readable and the thesis tables short.
    return round(x, 3)


# -- for the thesis ---------------------------------------------------------


def rules_markdown() -> str:
    """The linear rules as a Markdown table, ready to paste into the thesis."""
    lines = [
        "| Parameter | Trait | at 0.0 | at 1.0 | Rationale |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rule in LINEAR_RULES:
        lines.append(
            f"| `{rule.parameter}` | {rule.trait} | {rule.at_0:g} | {rule.at_1:g} "
            f"| {rule.rationale} |"
        )
    return "\n".join(lines)
