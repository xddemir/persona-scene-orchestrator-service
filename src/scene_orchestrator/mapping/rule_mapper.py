"""RuleSpecMapper: PersonaProfile + seed + scene_id -> SceneSpec.

A pure, deterministic function. No LLM, and no randomness except one
random.Random(seed), drawn in a fixed order. The same persona and seed always
give a byte-identical spec, which is what makes a scene reproducible and the
mapping inspectable.

Every rule is a module-level constant below, so the whole mapping can be
printed in the thesis (`rules_markdown()`). Each rule names the trait it
derives from. Values interpolate linearly between the trait's two poles:
`at_0` when the trait is 0.0, `at_1` when it is 1.0.

The place
---------
The persona chooses which place the scene is at, but only from a fixed list
of places, each tuned by hand and checked in the VR headset
(places_v1.yaml). It never sets a raw terrain number. A place is one Unity
setting (archetype) with its landform, where the participant sits on it, the
kind of water and where that lies from them, and how much grows; its own
props (focal, framing, and a list, most typical first); a biome for the sky's
colour and the sounds; and, for some, a season. The mapper picks the place
mainly by shelter (pick_place). Light, fog, wind, prop density and colour
still vary freely with the persona, by the linear rules, within each place's
safe range (the fog's floor and cap).

Which props a scene has is the place's, chosen by hand; the persona sets only
amounts: how many of the list's kinds (openness takes the first 2..5), how
many of each, how varied in size, and whether they stand in rings. Asset
choice is not part of the research claim and has no theory behind it, and
every free choice would add combinations to check in the headset.

Why: shelter mostly comes from the terrain, from how enclosed the land is,
where one sits on it and how far one sees. With one fixed landscape, a
participant's own scene and the yoked scene would differ only in light, fog
and props, and RQ3 (are the scenes distinct?) would likely fail. Driving the
heightmap's numbers from the answers instead produces odd in-between
landscapes, and that oddness would be a confound against the hand-made
baseline scene. A short list of whole places, each checked in the headset,
keeps the terrain a real choice without making it strange.

The seed only moves things around: which way the sun, the moon and the wind
come from, where the clouds are, which side the sun and the framing prop
stand on, and (in Unity) where each prop stands. It never decides anything the
persona should: not the place, not the biome, not which prop kinds. The draws
are listed in map_persona.

Prop, ground and archetype names are those of Unity's asset catalogue
(CATALOGUE_V1, in catalogue.py). Audio-layer names stand in for a sound
library that doesn't exist yet.
"""

from __future__ import annotations

import colorsys
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..axes import LANDFORMS, POSITIONS, SPAWN_TABLE, WATER_TABLE, axis_space
from ..catalogue import CATALOGUE_V1, RESERVED_PROPS, Catalogue, Setting  # noqa: F401  (re-exported)
from ..models import (
    Activity,
    Anchor,
    AudioLayer,
    Grass,
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
from .places import PLACES_FILE, Place, Places, load_places

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
RINGED_ABOVE = 0.65  # neat, regular arrangement: rings round the participant
CLUSTERED_ABOVE = 0.35  # grouped; below this, "scattered"
# Unity builds poisson, clustered and ringed (PropPlacer), and treats anything
# else as poisson, with a warning. Until it builds scattered (loose), that is
# sent as the nearest it has.
DISTRIBUTION_STAND_IN = {"scattered": "poisson"}

# conscientiousness -> props[].order, sent only with ringed: how exactly the
# props keep to their rings. Stretched over the ringed band, so it runs the
# whole 0..1 that Unity reads: loose rings just above RINGED_ABOVE (each
# strays up to about a third of a ring step), exact rings at 1.

# openness -> number of prop types: 2 + round(3 * openness), so 2..5, taken
# from the front of the place's list.
PROP_TYPES_MIN, PROP_TYPES_SPAN = 2, 3

# agreeableness: UNMAPPED. No parameter depends on it yet. Which (if any)
# environmental quality agreeableness should drive is an open question in the
# thesis, so it is left out on purpose rather than mapped arbitrarily.

# -- the place: picked from places_v1.yaml ----------------------------------

# The pick rule (pick_place): shelter picks the band, novelty the place in it.
SHELTER_LEVELS = (1, 5)  # shelter_level runs 1 (open) .. 5 (sheltered)
# Novelties closer than this count as equally near: a tie, so list order decides.
NOVELTY_TIE = 1e-9

# The place's shelter_level -> spatial.prospect_refuge_bias (Appleton), so
# the bias can't disagree with the place. Between the two: "balanced".
REFUGE_FROM_LEVEL = 4
PROSPECT_TO_LEVEL = 2


def _shelter(persona: PersonaProfile) -> float:
    """How much shelter the persona seeks, 0..1. Today neuroticism, the same
    trait as spatial.enclosure. The preference-profile adapter will replace it
    with the profile's shelter dimension."""
    return persona.traits.neuroticism


def _novelty(persona: PersonaProfile) -> float:
    """How unusual a place the persona seeks, 0..1. Today openness. The
    preference-profile adapter will replace it with the profile's variety
    dimension."""
    return persona.traits.openness


class NoUsablePlace(LookupError):
    """No place may be used: none is verified, and candidates are not allowed."""


def usable_places(places: Sequence[Place], allow_candidates: bool) -> list[Place]:
    """Verified places, and candidates too while they are allowed. Never rejected ones."""
    return [
        p for p in places
        if p.status == "verified" or (allow_candidates and p.status == "candidate")
    ]


def pick_place(
    places: Sequence[Place], shelter: float, novelty: float, *, allow_candidates: bool
) -> Place:
    """The place for a persona. Pure: the same inputs always give the same place.

    The target band is 1 + 4 * shelter, rounded half up (2.5 -> 3, 3.5 -> 4).
    The band used is the nearest one with a usable place, the more sheltered
    on a tie. Within it, the place with the nearest novelty; on a tie, the
    first in the list.
    """
    pool = usable_places(places, allow_candidates)
    if not pool:
        statuses = sorted({p.status for p in places})
        raise NoUsablePlace(
            "no usable place: none is verified"
            + ("" if allow_candidates else ", and places.allow_candidates is false")
            + f" (statuses in the list: {', '.join(statuses) or 'none'}). Check places "
            "in the headset and mark them verified, or allow candidates in the config."
        )
    low, high = SHELTER_LEVELS
    target = _round_half_up(low + (high - low) * shelter)
    band = min({p.shelter_level for p in pool}, key=lambda level: (abs(level - target), -level))
    in_band = [(i, p) for i, p in enumerate(pool) if p.shelter_level == band]
    return min(
        in_band,
        key=lambda ip: (round(abs(ip[1].novelty - novelty) / NOVELTY_TIE), ip[0]),
    )[1]


def _prospect_refuge(shelter_level: int) -> str:
    if shelter_level >= REFUGE_FROM_LEVEL:
        return "refuge"
    if shelter_level <= PROSPECT_TO_LEVEL:
        return "prospect"
    return "balanced"


# The water's sound follows its kind: no lake lapping without a lake, waves
# only at the sea. It comes before the biome's own ambience.
WATER_SOUND: dict[str, tuple[str, float]] = {
    "river": ("stream", 0.25),
    "lake": ("lake_lapping", 0.35),
    "sea": ("waves_gentle", 0.40),
}
SEA_SOUND_BY_BIOME: dict[str, tuple[str, float]] = {
    "rocky_shore": ("waves_rocky", 0.45),  # the sea breaks on rock
}

# Unity's fog is exponential squared: exp(-(density * d)**2) of a thing d away
# still shows. Where mountains stand on the horizon, capping density at
# FOG_HORIZON_DEPTH / horizon leaves exp(-1.21), about 30%, of them, so a view
# keeps its depth. A setting without mountains has nothing to keep in view,
# and a hollow is not capped either: there the fog is the shelter.
FOG_HORIZON_DEPTH = 1.1
FOG_UNCAPPED = frozenset({"hollow"})
FOG_CAP: dict[str, float] = {
    name: round(FOG_HORIZON_DEPTH / setting.horizon_m, 4)
    for name, setting in CATALOGUE_V1.archetypes.items()
    if setting.has_mountains and name not in FOG_UNCAPPED
}
# Never below a setting's fog floor (its minimumFogDensity): Unity raises it
# to that anyway, so the spec says what is rendered.

# -- the view: facing, focal and framing ------------------------------------

# Which way the participant faces, from the light. A low sun (sunrise or
# sunset) is faced: its glow is ahead, and so is water Unity lays out ahead.
# At night the moon is faced. By day the sun stands to one side, so relief
# reads and nothing glares: this many degrees from it, to the left or right
# (side and angle from the seed).
LOW_SUN_TIMES = frozenset({"sunrise", "sunset"})
DAY_SUN_OFFSET_DEG = (60.0, 120.0)

# The focal prop (the place's) stands where the participant faces, as far as
# its linear rule says. The framing prop (the place's) stands close by at the
# edge of view, this far to the left or right of the facing (side and angle
# from the seed). Unity keeps it within 3-5 m.
FRAMING_DISTANCE_M = 4.0
FRAMING_OFFSET_DEG = (55.0, 80.0)

# -- derived from spatial.enclosure (itself from neuroticism) ---------------

# More enclosure -> more and denser props, standing closer to the viewer.
# Unity then scales every count by the place's vegetation (x0.5, x1, x1.6).
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

# -- the grass, by season ---------------------------------------------------

# terrain.grass for a place that names a season: a copy of Unity's own table
# (GrassLook.OfSeason), so the spec on disk says what the headset shows.
# Unity's grass, fern and reed billboards take these; flowers keep their
# colours, only how many there are changes. A place without a season sends
# nothing, and Unity uses its table for the setting's season.
SEASON_GRASS: dict[str, Grass | None] = {
    # FlipGameDev's billboards as they are: the look the grounds were tuned in.
    "summer": None,
    # Fresh, light green growth, not yet at full height, and more flowers.
    "spring": Grass(
        healthy_color=(0.55, 0.95, 0.40), dry_color=(0.45, 0.80, 0.35),
        height_scale=0.85, grass_density=0.9, flower_density=1.8,
    ),
    # Golden to brown, a little thinner, with a few late flowers left.
    "autumn": Grass(
        healthy_color=(0.85, 0.68, 0.32), dry_color=(0.62, 0.42, 0.22),
        grass_density=0.8, flower_density=0.15,
    ),
    # Snow white to frosted grey, short and thin, and no flowers at all
    # (no_flowers, since Unity reads a flower_density of 0 as "the season's").
    "winter": Grass(
        healthy_color=(0.93, 0.96, 1.00), dry_color=(0.78, 0.82, 0.86),
        width_scale=0.9, height_scale=0.6, grass_density=0.45, no_flowers=True,
    ),
}

# -- what each biome sounds like --------------------------------------------

# A biome no longer brings props: each place lists its own (places_v1.yaml).
# It gives the sky its colour (below), whether rain or snow falls, and this.


@dataclass(frozen=True)
class BiomeSound:
    ambience: tuple[tuple[str, float], ...]  # (layer, gain), fixed; the water's sound is added by its kind
    birds: str  # gain and rate come from extraversion


BIOME_SOUNDS: dict[str, BiomeSound] = {
    "open_meadow": BiomeSound((("wind_grass", 0.3), ("insects", 0.15)), "birdsong_meadow"),
    "forest_clearing": BiomeSound(
        (("wind_canopy", 0.25), ("leaves_rustle", 0.2)), "birdsong_forest"
    ),
    "lakeside": BiomeSound((("wind_reeds", 0.2),), "birdsong_wetland"),
    # A pine_medium grove until a birch model exists; the id stays.
    "birch_grove": BiomeSound((("wind_canopy", 0.25), ("leaves_rustle", 0.2)), "birdsong_forest"),
    "coastal_pine": BiomeSound((("wind_pines", 0.2),), "seabirds"),
    "alpine_basin": BiomeSound((("wind_high", 0.3),), "birdsong_alpine"),
    "rocky_shore": BiomeSound((("wind_coast", 0.25),), "seabirds"),
    "snowy_valley": BiomeSound((("wind_snow", 0.3), ("snow_settle", 0.1)), "birdsong_winter"),
}

# The places, checked against Unity's catalogue and axis tables at import: a
# list Unity would not build as written stops the service from starting.
PLACES: Places = load_places(PLACES_FILE)

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
    """map_persona with the place rule's settings.

    allow_candidates: candidates count as usable (while places are being
    checked; false for the study). place_id: always this place, whatever the
    persona (headset review).
    """

    def __init__(
        self,
        *,
        allow_candidates: bool = False,
        place_id: str | None = None,
        places: Places | None = None,
    ) -> None:
        self.allow_candidates = allow_candidates
        self.place_id = place_id
        self.places = places or PLACES

    def map(self, persona: PersonaProfile, seed: int, scene_id: str) -> SceneSpec:
        return map_persona(persona, seed, scene_id, place=self.place_for(persona))

    def place_for(self, persona: PersonaProfile) -> Place:
        if self.place_id is not None:
            return self.places.get(self.place_id)
        return pick_place(
            self.places.places, _shelter(persona), _novelty(persona),
            allow_candidates=self.allow_candidates,
        )

    def provenance(self, persona: PersonaProfile) -> dict[str, Any]:
        """For the manifest: the place, its status as loaded, the candidate
        flag and the hash of the list, so a scene can be traced to both."""
        place = self.place_for(persona)
        entry: dict[str, Any] = {
            "id": place.id,
            "status": place.status,
            "allow_candidates": self.allow_candidates,
            "places_sha256": self.places.sha256,
        }
        if self.place_id is not None:
            entry["forced"] = True  # picked by hand, not by the rule
        return {"place": entry}


def map_persona(
    persona: PersonaProfile,
    seed: int,
    scene_id: str,
    *,
    place: Place | None = None,
    allow_candidates: bool = False,
) -> SceneSpec:
    """The spec for a persona at a place: `place`, else the one pick_place
    gives (from PLACES, with or without candidates).

    The seed's draws, in this order (reordering them changes every scene).
    Each decides only where or which way something is, and each is drawn
    whether or not it is used:

         1. sun azimuth
         2. wind direction
       3-4. cloud offset (x, then y)
         5. moon azimuth spread: give or take from where its phase puts it
         6. moon elevation
         7. sun side: left or right of the facing, by day
         8. sun offset: how far to that side, by day
         9. framing side: left or right of the facing
        10. framing offset: how far to that side
    """
    traits = persona.traits
    if place is None:
        place = pick_place(
            PLACES.places, _shelter(persona), _novelty(persona),
            allow_candidates=allow_candidates,
        )
    biome, archetype = place.biome, place.setting
    sounds = BIOME_SOUNDS[biome]
    setting = CATALOGUE_V1.archetypes[archetype]
    prop_assets = _pick_prop_types(place, traits.openness)

    rng = random.Random(seed)  # the only source of variation besides traits
    sun_azimuth = _r(rng.uniform(0.0, 360.0))
    wind_direction = _r(rng.uniform(0.0, 360.0))
    cloud_offset = (
        _r(rng.uniform(0.0, CLOUD_OFFSET_MAX)),
        _r(rng.uniform(0.0, CLOUD_OFFSET_MAX)),
    )
    moon_spread = rng.uniform(-MOON_AZIMUTH_SPREAD_DEG, MOON_AZIMUTH_SPREAD_DEG)
    moon_elevation = _r(rng.uniform(*MOON_ELEVATION_DEG))
    sun_side = _side(rng)
    sun_offset = rng.uniform(*DAY_SUN_OFFSET_DEG)
    framing_side = _side(rng)
    framing_offset = rng.uniform(*FRAMING_OFFSET_DEG)

    sun_elevation = _linear("lighting.sun_elevation_deg", traits)
    time_of_day = _time_of_day(sun_elevation, traits.conscientiousness)
    moon_phase = _linear("lighting.moon_phase", traits)
    moon_azimuth = _r((sun_azimuth + 180.0 * moon_phase + moon_spread) % 360.0)
    night = time_of_day == "night"
    facing = _facing(time_of_day, sun_azimuth, moon_azimuth, sun_side * sun_offset)
    water = place.water

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
            order=_order(traits.conscientiousness) if distribution == "ringed" else None,
        ))

    sound = _water_sound(biome, water)
    audio = [AudioLayer(layer=layer, gain=gain) for layer, gain in (*sound, *sounds.ambience)]
    audio.append(AudioLayer(
        layer=sounds.birds,
        gain=_linear("audio.birds.gain", traits),
        event_rate_per_min=_linear("audio.birds.event_rate_per_min", traits),
    ))

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
        # The place's axes, all of them; the heightmap's numbers, the ground
        # and where the water lies are left to the setting.
        terrain=Terrain(
            landform=place.landform,
            position=place.position,
            water_relation=place.water_relation,
            vegetation=place.vegetation,
            season=place.season,
            grass=SEASON_GRASS[place.season] if place.season else None,
            water=Water(
                present=water != "none",
                motion=_linear("terrain.water.motion", traits) if water != "none" else 0.0,
                kind=water,
            ),
            # position says where the participant sits; Unity reads the facing from here.
            spawn=Spawn(facing_deg=facing),
        ),
        props=props,
        focal=Anchor(
            asset=place.focal,
            distance_m=_linear("focal.distance_m", traits),
            azimuth_deg=facing,
        ),
        framing=Anchor(
            asset=place.framing,
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
            fog_density=_fog_density(traits, archetype, setting),
        ),
        weather=Weather(
            precipitation=_precipitation(traits.neuroticism),
            kind="snow" if biome in SNOW_BIOMES else "rain",
        ),
        spatial=Spatial(
            enclosure=enclosure,
            sightline_distance_m=_linear("spatial.sightline_distance_m", traits),
            prospect_refuge_bias=_prospect_refuge(place.shelter_level),
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


def _fog_density(traits: Traits, archetype: str, setting: Setting) -> float:
    """The rule's fog, capped where the horizon mountains should show, and
    never below the setting's floor, which Unity would raise it to."""
    fog = _linear("lighting.fog_density", traits)
    if archetype in FOG_CAP:
        fog = min(fog, FOG_CAP[archetype])
    return max(fog, setting.fog_floor)


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


def _order(conscientiousness: float) -> float:
    """props[].order for ringed props: (C - 0.65) / 0.35, the ringed band
    stretched over 0..1."""
    return _r(min(1.0, max(0.0, (conscientiousness - RINGED_ABOVE) / (1.0 - RINGED_ABOVE))))


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


def _pick_prop_types(place: Place, openness: float) -> list[tuple[str, float]]:
    """The first n of the place's list, n = 2 + round(3 * openness): no seed
    draw, so the persona sets how many kinds and the place which ones."""
    n = PROP_TYPES_MIN + _round_half_up(PROP_TYPES_SPAN * openness)
    return list(place.props[:n])


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
        (f"Places ({PLACES.path.name})", _places_table()),
        ("The rule that picks a place", _pick_table()),
        ("Every place Unity can build", _space_table()),
        (f"Per archetype (Unity catalogue {CATALOGUE_V1.version})", _setting_table()),
        ("Facing, focal and framing", _view_table()),
        ("Prop distribution (conscientiousness)", _distribution_table()),
        ("Props per place", _place_props_table()),
        ("Grass by season (`terrain.grass`)", _grass_table()),
        ("Biome sounds", _sound_table()),
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


def _grass_table() -> str:
    def rgb(color: tuple[float, float, float] | None) -> str:
        return "–" if color is None else "(" + ", ".join(f"{c:.2f}" for c in color) + ")"

    def times(value: float | None) -> str:
        return "×1" if value is None else f"×{value:g}"

    def flowers(grass: Grass) -> str:
        return "none" if grass.no_flowers else times(grass.flower_density)

    rows = []
    for season, grass in SEASON_GRASS.items():
        if grass is None:
            rows.append((season, "not sent: FlipGameDev's billboards as they are",
                         "", "", "", "", ""))
            continue
        rows.append((season, rgb(grass.healthy_color), rgb(grass.dry_color),
                     f"{times(grass.width_scale)} / {times(grass.height_scale)}",
                     times(grass.grass_density), flowers(grass),
                     "–" if grass.noise_spread is None else f"{grass.noise_spread:g}"))
    note = ("Sent only for a place that names a season; without one, `terrain.grass` is "
            "null and Unity uses the same table for the setting's season.")
    return note + "\n\n" + _table(
        ("Season", "Healthy colour", "Dry colour", "Width / height", "Grass", "Flowers",
         "Noise spread"),
        rows,
    )


def _places_table() -> str:
    def water(p: Place) -> str:
        return "none" if p.water == "none" else f"{p.water} {p.water_relation}"

    return _table(
        ("Place", "Setting", "Landform", "Position", "Water", "Vegetation", "Season", "Biome",
         "Shelter", "Novelty", "Status"),
        ((p.id, p.setting, p.landform, p.position, water(p), p.vegetation,
          p.season or "(setting's)", p.biome, str(p.shelter_level), f"{p.novelty:g}", p.status)
         for p in PLACES.places),
    )


def _pick_table() -> str:
    low, high = SHELTER_LEVELS
    return _table(
        ("Step", "Rule"),
        (
            ("shelter", "neuroticism, 0..1 (the preference profile's shelter will replace it)"),
            ("novelty", "openness, 0..1 (the preference profile's variety will replace it)"),
            ("target band", f"{low} + {high - low} × shelter, rounded half up "
                            "(2.5 → 3, 3.5 → 4)"),
            ("usable", "verified places; candidates too while `places.allow_candidates` is "
                       "true; rejected never"),
            ("band", "the nearest `shelter_level` with a usable place; on a tie, the more "
                     "sheltered"),
            ("place", "in that band, the nearest novelty; on a tie, the first in the list"),
            ("`spatial.prospect_refuge_bias`", f"from the place: shelter_level ≥ "
                                              f"{REFUGE_FROM_LEVEL} refuge, ≤ "
                                              f"{PROSPECT_TO_LEVEL} prospect, else balanced"),
            ("seed", "no part in it"),
        ),
    )


def _space_table() -> str:
    """Every combination SceneAxes builds as it is, by landform and position,
    with the places that use it."""
    space = axis_space()
    rows = []
    for landform in LANDFORMS:
        for position in POSITIONS:
            if SPAWN_TABLE[landform][position] is not None:
                continue
            here = [c for c in space if (c.landform, c.position) == (landform, position)]
            used = [p for p in PLACES.places if (p.landform, p.position) == (landform, position)]
            rows.append((
                landform, position, _water_options(landform), str(len(here)),
                ", ".join(f"{p.id} ({_place_water(p)}, {p.vegetation})" for p in used) or "–",
            ))
    used = {(p.landform, p.position, p.water, p.water_relation, p.vegetation)
            for p in PLACES.places}
    summary = (
        f"Unity builds {len(space)} combinations of landform × position × water × water "
        f"relation × vegetation as they are (SceneAxes' two tables, the water agreeing with "
        f"its relation; before any setting's rules, and with the season on top). "
        f"{PLACES.path.name} uses {len(used)} of them, in {len(PLACES.places)} places."
    )
    return summary + "\n\n" + _table(
        ("Landform", "Position", "Water", "Combinations", "In the places"), rows
    )


def _water_options(landform: str) -> str:
    relations = [r for r, fix in WATER_TABLE[landform].items() if fix is None and r != "absent"]
    options = ["none"] if WATER_TABLE[landform]["absent"] is None else []
    if relations:
        options.append(f"river, lake or sea {', '.join(relations)}")
    return "; ".join(options)


def _place_water(place: Place) -> str:
    return "no water" if place.water == "none" else f"{place.water} {place.water_relation}"


def _setting_table() -> str:
    def water(s: Setting) -> str:
        return "none" if s.water == "none" else f"{s.water} {s.water_relation}"

    return _table(
        ("Archetype", "Landform", "Position", "Water", "Vegetation", "Ground", "Horizon (m)",
         "Mountains", "Fog floor", "Fog cap"),
        ((name, s.landform, s.position, water(s), s.vegetation, s.ground, f"{s.horizon_m:g}",
          f"{s.mountain_arc_deg:g}°" if s.has_mountains else "none", f"{s.fog_floor:g}",
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
            ("`focal.asset`", "the place's focal"),
            ("`framing.asset`", "the place's framing prop"),
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
    span = 1.0 - RINGED_ABOVE
    order = {"ringed": f"(C − {RINGED_ABOVE:g}) / {span:g}: loose rings → exact rings"}
    return _table(
        ("Conscientiousness", "Rule", "Sent until Unity builds it", "`props[].order`"),
        ((when, rule, DISTRIBUTION_STAND_IN.get(rule, rule), order.get(rule, "not sent"))
         for when, rule in rows),
    )


def _place_props_table() -> str:
    return _table(
        ("Place", "Focal", "Framing", "Props, most typical first (weight); openness takes 2–5"),
        ((p.id, p.focal, p.framing, ", ".join(f"{a} {w:g}" for a, w in p.props))
         for p in PLACES.places),
    )


def _sound_table() -> str:
    return _table(
        ("Biome", "Ambience (gain)", "Birds"),
        ((biome, ", ".join(f"{layer} {gain:g}" for layer, gain in s.ambience), s.birds)
         for biome, s in BIOME_SOUNDS.items()),
    )
