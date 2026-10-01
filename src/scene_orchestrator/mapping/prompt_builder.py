"""SceneSpec -> skybox prompt text. Deterministic; no LLM.

The prompt is derived from the spec, never the other way round, so the sky
can't contradict the procedural scene (e.g. "misty dusk" over noon lighting).
Each numeric field falls into a band, and each band is a fixed phrase. The
bands are module constants, so they can be printed in the thesis next to the
mapping rules.
"""

from __future__ import annotations

from typing import Protocol

from ..models import SceneSpec

# The 360 model's trigger phrase. Required, and always first.
TRIGGER = "equirectangular 360 view"

DEFAULT_NEGATIVE_PROMPT = "people, text, watermark, buildings, distorted horizon"

STYLE = ("photorealistic", "serene")

# Bands: (comparison, threshold, phrase). The first match wins, and the last
# entry (no comparison) is the fallback. An empty phrase leaves the slot out.
# Sheltered (> 0.55) and low sun (< 18 deg) were widened from 0.6 and 12 so a
# clearly anxious, introverted persona like the P01 fixture reads as such.
Band = tuple[str | None, float | None, str]

ENCLOSURE_BANDS: tuple[Band, ...] = (  # spatial.enclosure
    (">", 0.55, "a sheltered"),
    ("<", 0.35, "an open"),
    (None, None, "a quiet"),
)
SUN_BANDS: tuple[Band, ...] = (  # lighting.sun_elevation_deg
    ("<", 18, "low sun"),
    (">", 30, "midday sun"),
    (None, None, "soft afternoon sun"),
)
FOG_BANDS: tuple[Band, ...] = (  # lighting.fog_density
    (">", 0.03, "heavy drifting mist"),
    (">", 0.015, "light haze"),
    (None, None, "clear air"),
)
LIGHT_BANDS: tuple[Band, ...] = (  # lighting.color_temperature_k
    ("<", 3500, "warm dim light"),
    (">", 4800, "bright cool light"),
    (None, None, "neutral daylight"),
)
WATER_BANDS: tuple[Band, ...] = (  # terrain.water.motion, only if water is present
    ("<", 0.2, "calm water in the distance"),
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
    "lakeside": "lakeside shore",
    "rocky_shore": "rocky shoreline",
    "birch_grove": "birch grove",
    "alpine_basin": "alpine basin",
}


class PromptBuilder(Protocol):
    def build(self, spec: SceneSpec) -> str: ...


class TemplatePromptBuilder:
    def build(self, spec: SceneSpec) -> str:
        water = (
            band(WATER_BANDS, spec.terrain.water.motion)
            if spec.terrain.water.present
            else ""
        )
        slots = [
            TRIGGER,
            f"{band(ENCLOSURE_BANDS, spec.spatial.enclosure)} "
            f"{BIOME_PHRASES[spec.biome]} at {band(SUN_BANDS, spec.lighting.sun_elevation_deg)}",
            band(FOG_BANDS, spec.lighting.fog_density),
            band(LIGHT_BANDS, spec.lighting.color_temperature_k),
            water,
            band(WIND_BANDS, spec.motion.wind_strength),
            *STYLE,
        ]
        # Empty slots are dropped, so there are never double or trailing commas.
        return ", ".join(slot for slot in slots if slot)


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
