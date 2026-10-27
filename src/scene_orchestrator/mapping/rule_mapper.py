"""RuleSpecMapper: PersonaProfile + seed + scene_id -> SceneSpec.

A pure, deterministic function. No LLM, and no randomness except one
random.Random(seed), drawn in a fixed order. The same persona and seed always
give a byte-identical spec, which is what makes a scene reproducible and the
mapping inspectable.

Every rule is a module-level constant below, so the whole mapping can be
printed in the thesis (`rules_markdown()`). Each rule names the trait it
derives from. Values interpolate linearly between the trait's two poles:
`at_0` when the trait is 0.0, `at_1` when it is 1.0.

Prop, ground and archetype names are those of Unity's asset catalogue
(CATALOGUE_V1). Audio-layer names stand in for a sound library that doesn't
exist yet.
"""

from __future__ import annotations

import colorsys
import random
from dataclasses import dataclass

from ..models import (
    Activity,
    Anchor,
    AudioLayer,
    Lighting,
    Motion,
    PersonaProfile,
    ProceduralSky,
    Prop,
    SceneSpec,
    Skybox,
    SkyClouds,
    SkyStars,
    Spatial,
    Spawn,
    Terrain,
    TimeOfDay,
    Traits,
    Water,
    Weather,
)

# -- Unity's asset catalogue ------------------------------------------------


@dataclass(frozen=True)
class Setting:
    """An archetype's defaults in Unity (its SettingProfile). Unity fills in
    whatever the spec leaves out from these; placement and water are always
    sent, since Unity reads the facing only with a placement."""

    landform: str
    placement: str
    water: str
    ground: str
    horizon_m: float  # how far away the horizon mountains stand


@dataclass(frozen=True)
class Catalogue:
    version: str
    props: frozenset[str]
    grounds: frozenset[str]
    archetypes: dict[str, Setting]


# The single source for every catalogue name a spec may carry.
CATALOGUE_V1 = Catalogue(
    version="v1",
    # Only these place anything. pine_large, pine_medium, birch_light, fern and
    # driftwood have no model yet; Unity lays them out all the same and shows
    # them once the art is there.
    props=frozenset({
        "pine_large", "pine_medium", "birch_light", "rock_cluster", "fern",
        "grass_tuft", "driftwood", "rock_small", "boulder",
    }),
    # Normally not sent: each archetype has its own (below).
    grounds=frozenset({"moss_rock_mix", "dry_meadow", "sand_grass_mix", "alpine_rock"}),
    archetypes={
        "hollow": Setting("bowl", "basin_floor", "none", "moss_rock_mix", 300.0),
        "overlook": Setting("ridge", "crest", "none", "dry_meadow", 340.0),
        # 3 steps, 30 m high, a path winding up
        "summit": Setting("terraces", "summit", "none", "alpine_rock", 420.0),
        "forest": Setting("plain", "centre", "river", "moss_rock_mix", 300.0),
        "meadow": Setting("plain", "centre", "lake", "dry_meadow", 330.0),
        "beach": Setting("shore", "shore", "sea", "sand_grass_mix", 300.0),
    },
)

# In the catalogue, but never in `props`: Unity places these itself, on the
# horizon and along the summit's terraces. The ground adds grass, small
# rocks, boulders, and reeds at the water, on its own too.
RESERVED_PROPS = frozenset({"mountain", "cliff"})

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
    LinearRule("lighting.moon_phase", "neuroticism", 0.0, 1.0,
               "a fuller moon, so the night is never quite dark"),
    LinearRule("motion.wind_strength", "neuroticism", 0.45, 0.12,
               "stiller air, less agitation"),
    LinearRule("terrain.water.motion", "neuroticism", 0.40, 0.10,
               "calmer water"),
    LinearRule("procedural_sky.atmosphere_thickness", "neuroticism", 0.7, 2.2,
               "denser, softer sky"),
    # A factor on SKY_TINT_SATURATION, not a saturation itself.
    LinearRule("procedural_sky.sky_tint.saturation_scale", "neuroticism", 1.0, 0.45,
               "more muted sky colour"),
    LinearRule("procedural_sky.clouds.coverage", "neuroticism", 0.05, 0.95,
               "from a clear sky to a closed, sheltering cover"),
    LinearRule("procedural_sky.clouds.softness", "neuroticism", 0.25, 0.80,
               "softer, more diffuse clouds"),
    LinearRule("procedural_sky.stars.brightness", "neuroticism", 1.0, 0.35,
               "haze and moonlight leave fewer stars to see"),
    LinearRule("focal.distance_m", "neuroticism", 40, 25,
               "the view's focal point nearer, the space held closer"),
    # extraversion -> brightness and liveliness
    LinearRule("lighting.sun_elevation_deg", "extraversion", -30, 60,
               "from a starlit night, past dawn or dusk, to high noon"),
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
    LinearRule("procedural_sky.stars.density", "openness", 0.15, 1.0,
               "a richer, more crowded star field"),
    LinearRule("procedural_sky.stars.milky_way", "openness", 0.0, 1.0,
               "the band of the Milky Way comes out"),
    # conscientiousness -> order
    LinearRule("procedural_sky.clouds.banding", "conscientiousness", 0.0, 1.0,
               "clouds in regular rows rather than scattered"),
)

# neuroticism -> prospect/refuge (Appleton). Between the two: "balanced".
REFUGE_ABOVE = 0.6
PROSPECT_BELOW = 0.4

# neuroticism -> rain (snow in a snowy biome): none up to here, then rising to
# steady at 1.0. The far end of the sheltering cloud cover: a veil that closes
# the view, and a sound that masks every other.
RAIN_ABOVE = 0.7

# extraversion -> time of day, by where its rule above puts the sun.
NIGHT_BELOW_DEG = -6.0  # the end of civil twilight
LOW_SUN_BELOW_DEG = 10.0  # from there up to here the sun is rising or setting
MIDDAY_FROM_DEG = 40.0

# conscientiousness -> which half of the day (rising early goes with it).
# Above this the sun is rising or climbing; else it is sinking or setting.
MORNING_ABOVE = 0.5

# conscientiousness -> order in prop placement.
RINGED_ABOVE = 0.65  # neat, regular arrangement
CLUSTERED_ABOVE = 0.35  # grouped; below this, "scattered"
# Unity builds only "poisson" and "clustered" so far, and treats the other two
# as poisson, with a warning. Until it builds them, each is sent as the
# nearest it has: ringed (orderly) as clustered, scattered (loose) as poisson.
DISTRIBUTION_STAND_IN = {"ringed": "clustered", "scattered": "poisson"}

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

# -- the land: archetype, from biome and prospect/refuge ---------------------

# biome (openness) x prospect_refuge_bias (neuroticism) -> the Unity setting.
# The biome keeps the sky; neuroticism picks the land within it: a hollow to
# shelter in, a view to look out over, or between the two. No seed draw, so
# the sky (biome) and the land (archetype) can't disagree.
ARCHETYPE_BY_BIOME: dict[str, dict[str, str]] = {
    "open_meadow": {"refuge": "hollow", "balanced": "meadow", "prospect": "overlook"},
    "forest_clearing": {"refuge": "hollow", "balanced": "forest", "prospect": "overlook"},
    "birch_grove": {"refuge": "hollow", "balanced": "forest", "prospect": "overlook"},
    "lakeside": {"refuge": "forest", "balanced": "meadow", "prospect": "meadow"},
    "coastal_pine": {"refuge": "forest", "balanced": "beach", "prospect": "beach"},
    "rocky_shore": {"refuge": "beach", "balanced": "beach", "prospect": "beach"},
    "alpine_basin": {"refuge": "hollow", "balanced": "overlook", "prospect": "summit"},
    "snowy_valley": {"refuge": "hollow", "balanced": "hollow", "prospect": "summit"},
}

# Where the participant sits and what water there is: the archetype's
# (CATALOGUE_V1), always sent. The water's sound follows its kind: no lake
# lapping without a lake, waves only at the sea. It comes before the biome's
# own ambience.
WATER_SOUND: dict[str, tuple[str, float]] = {
    "river": ("stream", 0.25),
    "lake": ("lake_lapping", 0.35),
    "sea": ("waves_gentle", 0.40),
}
SEA_SOUND_BY_BIOME: dict[str, tuple[str, float]] = {
    "rocky_shore": ("waves_rocky", 0.45),  # the sea breaks on rock
}

# Unity's fog is exponential squared: exp(-(density * d)**2) of a thing d away
# still shows. Capping density at FOG_HORIZON_DEPTH / horizon leaves
# exp(-1.21), about 30%, of the horizon mountains, so a view keeps its depth.
# A hollow is not capped: there the fog is the shelter.
FOG_HORIZON_DEPTH = 1.1
FOG_UNCAPPED = frozenset({"hollow"})
FOG_CAP: dict[str, float] = {
    name: round(FOG_HORIZON_DEPTH / setting.horizon_m, 4)
    for name, setting in CATALOGUE_V1.archetypes.items()
    if name not in FOG_UNCAPPED
}

# -- the view: facing, focal and framing ------------------------------------

# Which way the participant faces, from the light. A low sun (sunrise or
# sunset) is faced: its glow is ahead, and so is the sea or river, which Unity
# lays out in front. At night the moon is faced. By day the sun stands to one
# side, so relief reads and nothing glares: this many degrees from it, to the
# left or right (side and angle from the seed).
LOW_SUN_TIMES = frozenset({"sunrise", "sunset"})
DAY_SUN_OFFSET_DEG = (60.0, 120.0)

# The focal prop stands where the participant faces, as far as its linear
# rule says. In a forest or a hollow it is the biome's tree; elsewhere, and
# where the biome has none, a rock cluster.
FOCAL_TREE_ARCHETYPES = frozenset({"forest", "hollow"})
FOCAL_ROCK = "rock_cluster"
# The framing prop: the biome's tree (a rock cluster where it has none),
# close by at the edge of view, this far to the left or right of the facing
# (side and angle from the seed). Unity keeps it within 3-5 m.
FRAMING_DISTANCE_M = 4.0
FRAMING_OFFSET_DEG = (55.0, 80.0)

# -- derived from spatial.enclosure (itself from neuroticism) ---------------

# More enclosure -> more and denser props, standing closer to the viewer.
PROP_COUNT = (30, 150)  # per prop type at weight 1.0, enclosure 0 -> 1
EXCLUDE_RADIUS_M = (25.0, 6.0)  # clear area around the viewer
MIN_SPACING_M = (6.0, 2.5)
PROPS_PER_CLUSTER = 20  # for "clustered": cluster_count = count / this

# -- derived from lighting.sun_elevation_deg (itself from extraversion) ------

# The key light is the sun, and at night the moon. For the sun: (elevation,
# value) points, linear in between and flat beyond the ends.
SUN_COLOR_TEMPERATURE_K = (
    (-6.0, 2100.0), (0.0, 2400.0), (10.0, 3600.0), (25.0, 5000.0), (45.0, 5800.0),
)
SUN_INTENSITY = ((-6.0, 0.08), (0.0, 0.25), (10.0, 0.60), (25.0, 1.0))
MOON_COLOR_TEMPERATURE_K = 8000.0  # cool, as moonlight is seen
MOON_INTENSITY = (0.03, 0.18)  # new moon -> full moon (lighting.moon_phase)

# The glow of a low sun, as (hue in degrees, saturation): rose in the morning,
# amber in the evening. Neuroticism mutes it along with the rest of the sky.
SUNRISE_GLOW = (8.0, 0.45)
SUNSET_GLOW = (22.0, 0.80)

# The moon stands as far round from the sun as its phase says (a thin crescent
# beside it, a full moon opposite), give or take what the seed adds. How high
# it stands is up to the seed alone.
MOON_AZIMUTH_SPREAD_DEG = 25.0
MOON_ELEVATION_DEG = (25.0, 60.0)

# -- fixed (not persona-driven yet) -----------------------------------------

SKYBOX_RESOLUTION = (2048, 1024)  # equirectangular 2:1, image-gen's default
ACTIVITY = "seated_relaxation"

SNOW_BIOMES = frozenset({"snowy_valley"})  # where what falls is snow, not rain

# -- what each biome is made of ---------------------------------------------


@dataclass(frozen=True)
class BiomeKit:
    terrain: str  # terrain.profile; Unity no longer reads it
    tree: str | None  # the biome's tree, for the focal and framing props; None: it has none
    # (catalogue prop, weight): weight scales the count. The first is the
    # biome's signature prop and is always placed. Exactly 5, all different:
    # openness = 1 places them all, and five entries keep the seed's draws
    # (rng.sample) the same as before the kits were re-mapped to the catalogue.
    props: tuple[tuple[str, float], ...]
    ambience: tuple[tuple[str, float], ...]  # (layer, gain), fixed; the water's sound is added by its kind
    birds: str  # gain and rate come from extraversion


# Each entry keeps its place in the kit, so the seed picks the same entries
# as before. Comments name the old asset where an entry stands in for one the
# catalogue doesn't have.
BIOMES: dict[str, BiomeKit] = {
    "open_meadow": BiomeKit(
        "gentle_slope", "birch_light",
        (("grass_tuft", 1.0),
         ("rock_cluster", 0.05),  # for wildflower_patch
         ("birch_light", 0.1),  # for oak_solitary
         ("pine_medium", 0.05),  # for fence_post
         ("driftwood", 0.05)),  # for hay_bale
        (("wind_grass", 0.3), ("insects", 0.15)), "birdsong_meadow",
    ),
    "forest_clearing": BiomeKit(
        "flat", "pine_large",
        (("pine_large", 1.0), ("fern", 0.8),
         ("pine_medium", 0.3),  # for mushroom_cluster
         ("driftwood", 0.2),  # mossy_log
         ("boulder", 0.15)),
        (("wind_canopy", 0.25), ("leaves_rustle", 0.2)), "birdsong_forest",
    ),
    "lakeside": BiomeKit(
        "gentle_slope", "birch_light",
        (("grass_tuft", 1.0),  # for reed_bed: the ground grows the reeds at the water
         ("birch_light", 0.4),
         ("fern", 0.3),  # for willow
         ("rock_cluster", 0.3),  # smooth_stone
         ("driftwood", 0.05)),  # for wooden_jetty
        (("wind_reeds", 0.2),), "birdsong_wetland",
    ),
    "birch_grove": BiomeKit(
        "rolling", "birch_light",
        (("birch_light", 1.0), ("fern", 0.6),
         ("pine_medium", 0.2),  # for birch_sapling
         ("grass_tuft", 0.4),  # wildflower_patch
         ("driftwood", 0.2)),  # mossy_log
        (("wind_canopy", 0.25), ("leaves_rustle", 0.2)), "birdsong_forest",
    ),
    "coastal_pine": BiomeKit(
        "rolling", "pine_large",
        (("pine_large", 1.0), ("grass_tuft", 0.8),
         ("pine_medium", 0.4),  # for beach_shrub
         ("driftwood", 0.2), ("boulder", 0.15)),
        (("wind_pines", 0.2),), "seabirds",
    ),
    "alpine_basin": BiomeKit(
        "basin", "pine_large",
        (("pine_large", 1.0), ("grass_tuft", 0.8), ("pine_medium", 0.5),
         ("boulder", 0.4), ("rock_cluster", 0.3)),
        (("wind_high", 0.3),), "birdsong_alpine",
    ),
    "rocky_shore": BiomeKit(
        "rolling", None,
        (("rock_cluster", 1.0), ("grass_tuft", 0.6),
         ("boulder", 0.3),  # for tide_pool
         ("rock_small", 0.2),  # for kelp_pile
         ("driftwood", 0.2)),
        (("wind_coast", 0.25),), "seabirds",
    ),
    "snowy_valley": BiomeKit(
        "basin", "pine_large",
        (("pine_large", 1.0),
         ("pine_medium", 0.4),  # for snow_drift
         ("rock_cluster", 0.2),  # for frozen_shrub
         ("boulder", 0.3),
         ("driftwood", 0.15)),  # deadwood
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
    moon_spread = rng.uniform(-MOON_AZIMUTH_SPREAD_DEG, MOON_AZIMUTH_SPREAD_DEG)
    moon_elevation = _r(rng.uniform(*MOON_ELEVATION_DEG))
    # Added in 0.3, after all of the above, so those keep their values. Each
    # is drawn whether or not it is used.
    sun_side = _side(rng)
    sun_offset = rng.uniform(*DAY_SUN_OFFSET_DEG)
    framing_side = _side(rng)
    framing_offset = rng.uniform(*FRAMING_OFFSET_DEG)

    sun_elevation = _linear("lighting.sun_elevation_deg", traits)
    time_of_day = _time_of_day(sun_elevation, traits.conscientiousness)
    moon_phase = _linear("lighting.moon_phase", traits)
    moon_azimuth = _r((sun_azimuth + 180.0 * moon_phase + moon_spread) % 360.0)
    night = time_of_day == "night"

    bias = _prospect_refuge(traits.neuroticism)
    archetype = ARCHETYPE_BY_BIOME[biome][bias]
    setting = CATALOGUE_V1.archetypes[archetype]
    facing = _facing(time_of_day, sun_azimuth, moon_azimuth, sun_side * sun_offset)
    water = setting.water

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

    sound = _water_sound(biome, water)
    audio = [AudioLayer(layer=layer, gain=gain) for layer, gain in (*sound, *kit.ambience)]
    audio.append(AudioLayer(
        layer=kit.birds,
        gain=_linear("audio.birds.gain", traits),
        event_rate_per_min=_linear("audio.birds.event_rate_per_min", traits),
    ))

    fog_density = _linear("lighting.fog_density", traits)
    if archetype in FOG_CAP:
        fog_density = min(fog_density, FOG_CAP[archetype])

    return SceneSpec(
        scene_id=scene_id,
        seed=seed,
        catalogue_version=CATALOGUE_V1.version,
        archetype=archetype,
        biome=biome,
        skybox=Skybox(resolution=SKYBOX_RESOLUTION),
        # Filled in whatever the sky mode: it costs nothing, and it is what
        # Unity renders when there is no panorama.
        procedural_sky=_procedural_sky(biome, traits, cloud_offset),
        # The landform, ground and horizon are left to the archetype.
        terrain=Terrain(
            profile=kit.terrain,
            water=Water(
                present=water != "none",
                motion=_linear("terrain.water.motion", traits) if water != "none" else 0.0,
                kind=water,
            ),
            spawn=Spawn(placement=setting.placement, facing_deg=facing),
        ),
        props=props,
        focal=Anchor(
            asset=_focal_asset(kit, archetype),
            distance_m=_linear("focal.distance_m", traits),
            azimuth_deg=facing,
        ),
        framing=Anchor(
            asset=kit.tree or FOCAL_ROCK,
            distance_m=FRAMING_DISTANCE_M,
            azimuth_deg=_r((facing + framing_side * framing_offset) % 360.0),
        ),
        lighting=Lighting(
            sun_elevation_deg=sun_elevation,
            sun_azimuth_deg=sun_azimuth,
            time_of_day=time_of_day,
            moon_elevation_deg=moon_elevation,
            moon_azimuth_deg=moon_azimuth,
            moon_phase=moon_phase,
            color_temperature_k=_r(
                MOON_COLOR_TEMPERATURE_K
                if night
                else _interpolate(SUN_COLOR_TEMPERATURE_K, sun_elevation)
            ),
            intensity=_r(
                _lerp(*MOON_INTENSITY, moon_phase)
                if night
                else _interpolate(SUN_INTENSITY, sun_elevation)
            ),
            fog_density=fog_density,
        ),
        weather=Weather(
            precipitation=_precipitation(traits.neuroticism),
            kind="snow" if biome in SNOW_BIOMES else "rain",
        ),
        spatial=Spatial(
            enclosure=enclosure,
            sightline_distance_m=_linear("spatial.sightline_distance_m", traits),
            prospect_refuge_bias=bias,
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


def _precipitation(neuroticism: float) -> float:
    return _r(max(0.0, (neuroticism - RAIN_ABOVE) / (1.0 - RAIN_ABOVE)))


def _is_morning(conscientiousness: float) -> bool:
    return conscientiousness > MORNING_ABOVE


def _time_of_day(sun_elevation_deg: float, conscientiousness: float) -> TimeOfDay:
    if sun_elevation_deg < NIGHT_BELOW_DEG:
        return "night"
    morning = _is_morning(conscientiousness)
    if sun_elevation_deg < LOW_SUN_BELOW_DEG:
        return "sunrise" if morning else "sunset"
    if sun_elevation_deg >= MIDDAY_FROM_DEG:
        return "midday"
    return "morning" if morning else "afternoon"


def _distribution(conscientiousness: float) -> str:
    """The rule's distribution, as sent: see DISTRIBUTION_STAND_IN."""
    intended = _intended_distribution(conscientiousness)
    return DISTRIBUTION_STAND_IN.get(intended, intended)


def _intended_distribution(conscientiousness: float) -> str:
    if conscientiousness > RINGED_ABOVE:
        return "ringed"
    if conscientiousness > CLUSTERED_ABOVE:
        return "clustered"
    return "scattered"


def _side(rng: random.Random) -> float:
    """-1 (left, anticlockwise) or +1 (right, clockwise), from the seed."""
    return 1.0 if rng.random() < 0.5 else -1.0


def _facing(
    time_of_day: str, sun_azimuth: float, moon_azimuth: float, day_offset: float
) -> float:
    if time_of_day == "night":
        return moon_azimuth
    if time_of_day in LOW_SUN_TIMES:
        return sun_azimuth
    return _r((sun_azimuth + day_offset) % 360.0)


def _focal_asset(kit: BiomeKit, archetype: str) -> str:
    if archetype in FOCAL_TREE_ARCHETYPES and kit.tree:
        return kit.tree
    return FOCAL_ROCK


def _water_sound(biome: str, kind: str) -> tuple[tuple[str, float], ...]:
    if kind == "none":
        return ()
    if kind == "sea" and biome in SEA_SOUND_BY_BIOME:
        return (SEA_SOUND_BY_BIOME[biome],)
    return (WATER_SOUND[kind],)


def _procedural_sky(
    biome: str, traits: Traits, cloud_offset: tuple[float, float]
) -> ProceduralSky:
    hue = BIOME_SKY_HUE[biome] / 360.0
    mute = _linear("procedural_sky.sky_tint.saturation_scale", traits)
    saturation = SKY_TINT_SATURATION * mute
    pale = colorsys.hsv_to_rgb(hue, saturation * HORIZON_SATURATION_SCALE, HORIZON_VALUE)
    accent = _linear("procedural_sky.horizon_color.accent", traits)
    glow_hue, glow_saturation = (
        SUNRISE_GLOW if _is_morning(traits.conscientiousness) else SUNSET_GLOW
    )
    return ProceduralSky(
        atmosphere_thickness=_linear("procedural_sky.atmosphere_thickness", traits),
        sky_tint=_rgb(colorsys.hsv_to_rgb(hue, saturation, SKY_TINT_VALUE)),
        horizon_color=_rgb(_lerp(p, a, accent) for p, a in zip(pale, HORIZON_ACCENT)),
        twilight_color=_rgb(
            colorsys.hsv_to_rgb(glow_hue / 360.0, glow_saturation * mute, 1.0)
        ),
        ground_color=BIOME_GROUND_COLOR[biome],
        exposure=_linear("procedural_sky.exposure", traits),
        sun_size=_linear("procedural_sky.sun_size", traits),
        sun_halo=_linear("procedural_sky.sun_halo", traits),
        stars=SkyStars(
            density=_linear("procedural_sky.stars.density", traits),
            brightness=_linear("procedural_sky.stars.brightness", traits),
            milky_way=_linear("procedural_sky.stars.milky_way", traits),
        ),
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


def _interpolate(points: tuple[tuple[float, float], ...], x: float) -> float:
    """The value at x on the line through `points`, flat beyond its two ends."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return _lerp(y0, y1, (x - x0) / (x1 - x0))
    return points[-1][1]


def _round_half_up(x: float) -> int:
    # Not round(): Python rounds halves to even, so 2.5 -> 2 but 3.5 -> 4.
    return int(x + 0.5)


def _r(x: float) -> float:
    # Three decimals: keeps the JSON readable and the thesis tables short.
    return round(x, 3)


# -- for the thesis ---------------------------------------------------------


def rules_markdown() -> str:
    """The rules as Markdown tables, ready to paste into the thesis."""
    sections = (
        ("Linear rules", _linear_table()),
        ("Archetype: biome (openness) × prospect/refuge (neuroticism)", _archetype_table()),
        (f"Per archetype (Unity catalogue {CATALOGUE_V1.version})", _setting_table()),
        ("Facing, focal and framing", _view_table()),
        ("Prop distribution (conscientiousness)", _distribution_table()),
        ("Biome kits", _kit_table()),
    )
    return "\n\n".join(f"### {title}\n\n{table}" for title, table in sections)


def _table(header: tuple[str, ...], rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _linear_table() -> str:
    return _table(
        ("Parameter", "Trait", "at 0.0", "at 1.0", "Rationale"),
        ((f"`{r.parameter}`", r.trait, f"{r.at_0:g}", f"{r.at_1:g}", r.rationale)
         for r in LINEAR_RULES),
    )


def _archetype_table() -> str:
    biases = ("refuge", "balanced", "prospect")
    return _table(
        ("Biome", *biases),
        ((biome, *(row[b] for b in biases)) for biome, row in ARCHETYPE_BY_BIOME.items()),
    )


def _setting_table() -> str:
    def sound(name: str, kind: str) -> str:
        if kind == "none":
            return "none"
        layers = {_water_sound(b, kind)[0][0] for b, row in ARCHETYPE_BY_BIOME.items()
                  if name in row.values()}
        return f"{kind} ({', '.join(sorted(layers))})"

    return _table(
        ("Archetype", "Landform", "Placement", "Water (sound)", "Ground", "Horizon (m)",
         "Fog cap"),
        ((name, s.landform, s.placement, sound(name, s.water), s.ground, f"{s.horizon_m:g}",
          f"{FOG_CAP[name]:g}" if name in FOG_CAP else "none")
         for name, s in CATALOGUE_V1.archetypes.items()),
    )


def _view_table() -> str:
    day = "–".join(f"{d:g}" for d in DAY_SUN_OFFSET_DEG)
    side = "–".join(f"{d:g}" for d in FRAMING_OFFSET_DEG)
    focal = _RULES["focal.distance_m"]
    return _table(
        ("Field", "Rule"),
        (
            ("`terrain.spawn.facing_deg`", "sunrise, sunset: the sun's azimuth"),
            ("", "night: the moon's azimuth"),
            ("", f"day: the sun's azimuth ± {day}° (side and angle from the seed)"),
            ("`focal.azimuth_deg`", "the facing"),
            ("`focal.distance_m`", f"{focal.at_0:g} m → {focal.at_1:g} m by {focal.trait}"),
            ("`focal.asset`", f"the biome's tree in {' and '.join(sorted(FOCAL_TREE_ARCHETYPES))}; "
                              f"else {FOCAL_ROCK}"),
            ("`framing.asset`", f"the biome's tree; {FOCAL_ROCK} where it has none"),
            ("`framing.distance_m`", f"{FRAMING_DISTANCE_M:g} m"),
            ("`framing.azimuth_deg`", f"the facing ± {side}° (side and angle from the seed)"),
        ),
    )


def _distribution_table() -> str:
    rows = (
        (f"> {RINGED_ABOVE:g}", "ringed"),
        (f"> {CLUSTERED_ABOVE:g}", "clustered"),
        (f"≤ {CLUSTERED_ABOVE:g}", "scattered"),
    )
    return _table(
        ("Conscientiousness", "Rule", "Sent until Unity builds it"),
        ((when, rule, DISTRIBUTION_STAND_IN.get(rule, rule)) for when, rule in rows),
    )


def _kit_table() -> str:
    return _table(
        ("Biome", "Tree", "Props (weight; the first always placed)"),
        ((biome, kit.tree or "none", ", ".join(f"{a} {w:g}" for a, w in kit.props))
         for biome, kit in BIOMES.items()),
    )
