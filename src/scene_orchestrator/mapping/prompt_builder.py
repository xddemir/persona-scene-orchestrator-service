"""SceneSpec -> skybox prompt text. Deterministic; no LLM.

The prompt is derived from the spec, never the other way round, so the sky
can't contradict the procedural scene (e.g. "misty dusk" over noon lighting).
Each numeric field falls into a band, and each band is a fixed phrase. The
bands are module constants, so they can be printed in the thesis next to the
mapping rules.

The image and the procedural sky are two renderings of one description: the
hour, the cloud cover, the moon, the stars and the rain named here are the
ones the sky shader draws from the same spec. The view named (the place's
landform, where the participant sits on it, and its water) is the land Unity
builds under it.
"""

from __future__ import annotations

from typing import Protocol

from ..axes import resolve_axes
from ..models import SceneSpec

# The 360 model's trigger phrase. Required, and always first.
TRIGGER = "equirectangular 360 view"

DEFAULT_NEGATIVE_PROMPT = "people, text, watermark, buildings, distorted horizon"

STYLE = ("photorealistic", "serene")

# Bands: (comparison, threshold, phrase). The first match wins, and the last
# entry (no comparison) is the fallback. An empty phrase leaves the slot out.
# Sheltered (> 0.55) was widened from 0.6 so a clearly anxious persona like
# the P01 fixture reads as such.
Band = tuple[str | None, float | None, str]

ENCLOSURE_BANDS: tuple[Band, ...] = (  # spatial.enclosure
    (">", 0.55, "a sheltered"),
    ("<", 0.35, "an open"),
    (None, None, "a quiet"),
)
# lighting.time_of_day. The two tables after it replace single entries: at
# sunrise and sunset when the sun is still, or already, under the horizon
# (lighting.sun_elevation_deg < 0), and by day under a closed sky.
TIME_PHRASES: dict[str, str] = {
    "night": "at night",
    "sunrise": "at sunrise",
    "morning": "in the morning sun",
    "midday": "at midday sun",
    "afternoon": "at soft afternoon sun",
    "sunset": "at sunset",
}
SUN_DOWN_PHRASES: dict[str, str] = {
    "sunrise": "at dawn before sunrise",
    "sunset": "at dusk after sunset",
}
SUNLESS_PHRASES: dict[str, str] = {
    "morning": "in the morning",
    "midday": "at midday",
    "afternoon": "in the afternoon",
}
# More cloud than this closes the sky: no sun, moon or stars are named under
# it. Below the point where rain starts (rule_mapper.RAIN_ABOVE gives 0.68
# cover), so rain never falls from a sky described as open.
CLOSED_ABOVE = 0.65
CLOUD_BANDS: tuple[Band, ...] = (  # procedural_sky.clouds.coverage
    (">", 0.85, "overcast sky"),
    (">", CLOSED_ABOVE, "mostly cloudy sky"),
    (">", 0.40, "scattered clouds"),
    (">", 0.15, "a few clouds"),
    (None, None, "cloudless sky"),
)
MOON_BANDS: tuple[Band, ...] = (  # lighting.moon_phase, at night under a sky that shows it
    (">", 0.75, "a full moon"),
    (">", 0.35, "a half moon"),
    (">", 0.10, "a crescent moon"),
    (None, None, ""),  # a new moon is not to be seen
)
# The next two are scaled by stars.brightness first: what is seen of them.
STAR_BANDS: tuple[Band, ...] = (  # procedural_sky.stars.density
    (">", 0.55, "a sky crowded with stars"),
    (">", 0.20, "many stars"),
    (None, None, "a few faint stars"),
)
MILKY_WAY_BANDS: tuple[Band, ...] = (  # procedural_sky.stars.milky_way
    (">", 0.50, "the Milky Way"),
    (None, None, ""),
)
RAIN_BANDS: tuple[Band, ...] = (  # weather.precipitation, kind "rain"
    (">", 0.50, "steady rain"),
    (">", 0.05, "light rain"),
    (None, None, ""),
)
SNOW_BANDS: tuple[Band, ...] = (  # weather.precipitation, kind "snow"
    (">", 0.50, "steady snowfall"),
    (">", 0.05, "light snowfall"),
    (None, None, ""),
)
FOG_BANDS: tuple[Band, ...] = (  # lighting.fog_density
    (">", 0.03, "heavy drifting mist"),
    (">", 0.015, "light haze"),
    (None, None, "clear air"),
)
LIGHT_BANDS: tuple[Band, ...] = (  # lighting.color_temperature_k, by day under an open sky
    ("<", 3500, "warm dim light"),
    (">", 4800, "bright cool light"),
    (None, None, "neutral daylight"),
)
CLOSED_SKY_LIGHT = "soft grey light"  # by day, instead of the band above
NIGHT_LIGHT_BANDS: tuple[Band, ...] = (  # lighting.moon_phase, at night
    (">", 0.35, "soft moonlight"),
    (None, None, "dim starlight"),
)
WATER_BANDS: tuple[Band, ...] = (  # terrain.water.motion, only if water is present
    ("<", 0.2, "calm water"),  # not "in the distance": the water may be beside
    (None, None, "gently moving water"),
)
WIND_BANDS: tuple[Band, ...] = (  # motion.wind_strength
    (">", 0.3, "a steady breeze through the foliage"),
    (None, None, ""),
)

BIOME_PHRASES: dict[str, str] = {
    "coastal_pine": "coastal pine clearing",
    "open_meadow": "wildflower meadow",  # not "open meadow": "an open open meadow"
    "forest_clearing": "forest clearing",
    "snowy_valley": "snowy valley",
    "lakeside": "lakeside shore",  # only ever at a lake (places_v1.yaml)
    "rocky_shore": "rocky shoreline",
    "birch_grove": "birch grove",
    "alpine_basin": "alpine basin",
}

# The view from where the participant sits: the place's landform and their
# position on it (terrain.landform, terrain.position, else the archetype's),
# so the painted horizon agrees with the land Unity builds. Every pair Unity
# builds as it is (axes.SPAWN_TABLE) has one.
VIEW_PHRASES: dict[tuple[str, str], str] = {
    ("plain", "mid"): "",  # level ground: what grows on it says more (PLAIN_PHRASES)
    ("bowl", "bottom"): "seen from a hollow among low hills",
    ("bowl", "mid"): "seen from the slope of a hollow",
    ("bowl", "top"): "seen from the rim of a hollow",
    ("ridge", "bottom"): "seen from the foot of a ridge",
    ("ridge", "mid"): "seen from a hillside below a ridge",
    ("ridge", "top"): "seen from a ridge overlooking the land below",
    ("peak", "bottom"): "seen from the foot of a hill",
    ("peak", "mid"): "seen from a hillside",
    ("peak", "top"): "seen from a hilltop",
    ("shore", "mid"): "seen from a beach",
    ("shore", "top"): "seen from high above a rocky shore",
    ("terraces", "bottom"): "seen from the foot of a terraced mountain",
    ("terraces", "mid"): "seen from a terrace halfway up a mountain",
    ("terraces", "top"): "seen from a high mountain top",
    ("valley", "bottom"): "seen from the floor of a valley",
    ("valley", "mid"): "seen from the side of a valley",
}
# On a plain, the archetype's woods or grass are the view.
PLAIN_PHRASES: dict[str, str] = {
    "forest": "in a clearing among tall trees",
    "meadow": "in open grassland",
}
PLAIN_PHRASE = "on open level ground"  # a plain in any other archetype

# The water and where it lies from the participant (terrain.water.kind and
# terrain.water_relation, else the archetype's). Named only where it is: no
# sea in a forest panorama, no lake where Unity lays a river.
WATER_PLACE_PHRASES: dict[tuple[str, str], str] = {
    ("river", "beside"): "a river running close by",
    ("river", "below"): "a river far below",
    ("river", "distant"): "a river in the distance",
    ("lake", "beside"): "the shore of a lake close by",
    ("lake", "below"): "a lake far below",
    ("lake", "distant"): "a lake in the distance",
    ("sea", "beside"): "the open sea ahead",
    ("sea", "below"): "the sea far below",
    ("sea", "distant"): "the sea in the distance",
}
# At sunrise and sunset the participant faces the sun, and Unity lays the sea
# ahead. Under a closed sky no sun is named, so these don't apply there.
SEA_TOWARD_SUN: dict[str, str] = {
    "sunrise": "toward the rising sun",
    "sunset": "toward the setting sun",
}


class PromptBuilder(Protocol):
    def build(self, spec: SceneSpec) -> str: ...


class TemplatePromptBuilder:
    def build(self, spec: SceneSpec) -> str:
        lighting, sky = spec.lighting, spec.procedural_sky
        night = lighting.time_of_day == "night"
        closed = sky.clouds.coverage > CLOSED_ABOVE
        # The moon and the stars: only at night, and only where they show.
        heavens = night and not closed

        time = TIME_PHRASES[lighting.time_of_day]
        if lighting.sun_elevation_deg < 0:
            time = SUN_DOWN_PHRASES.get(lighting.time_of_day, time)
        elif closed:
            time = SUNLESS_PHRASES.get(lighting.time_of_day, time)
        if night:
            light = band(NIGHT_LIGHT_BANDS, lighting.moon_phase)
        elif closed:
            light = CLOSED_SKY_LIGHT
        else:
            light = band(LIGHT_BANDS, lighting.color_temperature_k)
        water = (
            band(WATER_BANDS, spec.terrain.water.motion)
            if spec.terrain.water.present
            else ""
        )
        view, where_water = view_phrases(spec, sun_named=not closed)
        slots = [
            TRIGGER,
            f"{band(ENCLOSURE_BANDS, spec.spatial.enclosure)} {BIOME_PHRASES[spec.biome]} {time}",
            view,
            where_water,
            band(CLOUD_BANDS, sky.clouds.coverage),
            band(MOON_BANDS, lighting.moon_phase) if heavens else "",
            band(STAR_BANDS, sky.stars.density * sky.stars.brightness) if heavens else "",
            band(MILKY_WAY_BANDS, sky.stars.milky_way * sky.stars.brightness) if heavens else "",
            band(
                SNOW_BANDS if spec.weather.kind == "snow" else RAIN_BANDS,
                spec.weather.precipitation,
            ),
            band(FOG_BANDS, lighting.fog_density),
            light,
            water,
            band(WIND_BANDS, spec.motion.wind_strength),
            *STYLE,
        ]
        # Empty slots are dropped, so there are never double or trailing commas.
        return ", ".join(slot for slot in slots if slot)


def view_phrases(spec: SceneSpec, *, sun_named: bool) -> tuple[str, str]:
    """(where the participant sits, where the water is), each "" when there
    is nothing to say: from the place's axes, resolved as Unity resolves them."""
    terrain = spec.terrain
    axes = resolve_axes(
        spec.archetype,
        landform=terrain.landform,
        macro_shape=terrain.macro_shape,
        position=terrain.position,
        placement=terrain.spawn.placement,
        water=terrain.water.kind,
        water_relation=terrain.water_relation,
    )
    view = VIEW_PHRASES[(axes.landform, axes.position)]
    if axes.landform == "plain":
        view = PLAIN_PHRASES.get(spec.archetype, PLAIN_PHRASE)
    water = WATER_PLACE_PHRASES.get((axes.water, axes.water_relation), "")
    toward = SEA_TOWARD_SUN.get(spec.lighting.time_of_day) if sun_named else None
    if axes.water == "sea" and toward:
        water = f"{water}, {toward}"
    return view, water


def band(bands: tuple[Band, ...], value: float) -> str:
    for comparison, threshold, phrase in bands:
        if (
            comparison is None
            or (comparison == ">" and value > threshold)
            or (comparison == "<" and value < threshold)
        ):
            return phrase
    raise ValueError("a band table must end with a fallback entry")


def attach_prompt(spec: SceneSpec, builder: PromptBuilder | None = None) -> SceneSpec:
    """A copy of `spec` with skybox.prompt filled in, so the spec is self-contained."""
    prompt = (builder or TemplatePromptBuilder()).build(spec)
    return spec.model_copy(
        update={"skybox": spec.skybox.model_copy(update={"prompt": prompt})}
    )
