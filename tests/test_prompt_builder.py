"""TemplatePromptBuilder: SceneSpec -> prompt text."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from scene_orchestrator.adapters import ChatbotV2Adapter
from scene_orchestrator.mapping import (
    DEFAULT_NEGATIVE_PROMPT,
    PromptBuilder,
    RuleSpecMapper,
    TemplatePromptBuilder,
    attach_prompt,
)
from scene_orchestrator.mapping.prompt_builder import (
    CLOSED_ABOVE,
    MILKY_WAY_BANDS,
    MOON_BANDS,
    RAIN_BANDS,
    SNOW_BANDS,
    STAR_BANDS,
)
from scene_orchestrator.models import PersonaProfile, SceneSpec, Traits, Weather

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "personas" / "chatbot"
BUILDER: PromptBuilder = TemplatePromptBuilder()  # also checks it fits the protocol
TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")


def _spec(alias: str) -> SceneSpec:
    raw = json.loads((FIXTURES / f"{alias}.json").read_text(encoding="utf-8"))
    return RuleSpecMapper().map(ChatbotV2Adapter().adapt(raw), 1234, alias.lower())


def _all_specs():
    """Specs across the whole trait space, every biome tier, several seeds."""
    for values in itertools.product((0.0, 0.5, 1.0), repeat=5):
        persona = PersonaProfile(
            participant_id="t", source="stub", traits=Traits(**dict(zip(TRAITS, values)))
        )
        for seed in (0, 1, 2):
            yield RuleSpecMapper().map(persona, seed, "t")


# -- the fixtures -----------------------------------------------------------


def test_p01_prompt_is_sheltered_before_sunrise_and_misty():
    prompt = BUILDER.build(_spec("P01"))
    assert "sheltered" in prompt
    assert "at dawn before sunrise" in prompt
    assert "mostly cloudy sky" in prompt
    assert "heavy drifting mist" in prompt


def test_p02_prompt_is_open_bright_and_clear():
    prompt = BUILDER.build(_spec("P02"))
    assert "open" in prompt
    assert "midday sun" in prompt
    assert "a few clouds" in prompt
    assert "clear air" in prompt
    assert "bright cool light" in prompt


@pytest.mark.parametrize(
    ("alias", "phrases"),
    [
        ("P04", ["at night", "cloudless sky", "a sky crowded with stars", "the Milky Way",
                 "dim starlight"]),
        ("P06", ["at night", "overcast sky", "soft moonlight"]),
        ("P08", ["at sunrise", "scattered clouds", "warm dim light"]),
        ("P09", ["at dusk after sunset", "a few clouds"]),
        ("P10", ["at night", "scattered clouds", "a half moon", "a few faint stars",
                 "soft moonlight"]),
        ("P11", ["at sunset", "warm dim light"]),
        ("P12", ["in the morning,", "mostly cloudy sky", "soft grey light"]),
        ("P13", ["in the afternoon,", "mostly cloudy sky"]),
    ],
)
def test_the_fixtures_read_as_their_hour_and_weather(alias, phrases):
    prompt = BUILDER.build(_spec(alias))
    for phrase in phrases:
        assert phrase in prompt


@pytest.mark.parametrize(
    ("kind", "amount", "phrase"),
    [("rain", 0.83, "steady rain"), ("rain", 0.3, "light rain"),
     ("snow", 0.83, "steady snowfall"), ("snow", 0.3, "light snowfall")],
)
def test_what_falls_is_named_by_kind_and_amount(kind, amount, phrase):
    spec = _spec("P06")  # overcast, at night
    spec = spec.model_copy(update={"weather": Weather(precipitation=amount, kind=kind)})
    assert phrase in BUILDER.build(spec)


def test_a_moonless_night_names_no_moon():
    prompt = BUILDER.build(_spec("P04"))
    assert not any(phrase in prompt for _, _, phrase in MOON_BANDS if phrase)


@pytest.mark.parametrize("spec", list(_all_specs()))
def test_the_sky_named_is_one_that_can_be_seen(spec):
    prompt = BUILDER.build(spec)
    heavens = [phrase for bands in (MOON_BANDS, STAR_BANDS, MILKY_WAY_BANDS)
               for _, _, phrase in bands if phrase and phrase in prompt]
    falling = [phrase for bands in (RAIN_BANDS, SNOW_BANDS)
               for _, _, phrase in bands if phrase and phrase in prompt]
    closed = spec.procedural_sky.clouds.coverage > CLOSED_ABOVE

    if spec.lighting.time_of_day != "night" or closed:
        assert not heavens  # no moon or stars by day, or behind the cloud
    if closed:
        assert " sun" not in prompt.replace("sunrise", "").replace("sunset", "")
    else:
        assert not falling  # nothing falls from an open sky
    assert ("at night" in prompt) == (spec.lighting.time_of_day == "night")


# -- shape of every prompt --------------------------------------------------


@pytest.mark.parametrize("spec", list(_all_specs()))
def test_every_prompt_is_well_formed(spec):
    prompt = BUILDER.build(spec)
    assert prompt.startswith("equirectangular 360 view, ")
    assert ",," not in prompt.replace(" ", "")
    assert not prompt.rstrip().endswith(",")
    assert "  " not in prompt
    assert prompt.endswith("photorealistic, serene")


def test_the_same_spec_gives_the_same_prompt():
    assert BUILDER.build(_spec("P01")) == BUILDER.build(_spec("P01"))


def test_slots_without_a_phrase_are_left_out():
    spec = _spec("P02")
    still = spec.model_copy(update={"motion": spec.motion.model_copy(update={"wind_strength": 0.1})})
    dry = still.model_copy(update={"terrain": still.terrain.model_copy(
        update={"water": still.terrain.water.model_copy(update={"present": False})})})
    prompt = BUILDER.build(dry)
    assert "breeze" not in prompt
    assert "water" not in prompt


# -- storing it on the spec -------------------------------------------------


def test_attach_prompt_fills_the_spec_and_leaves_the_original_alone():
    spec = _spec("P02")
    with_prompt = attach_prompt(spec)
    assert with_prompt.skybox.prompt == BUILDER.build(spec)
    assert spec.skybox.prompt is None
    SceneSpec.model_validate(json.loads(with_prompt.model_dump_json()))


def test_default_negative_prompt():
    assert DEFAULT_NEGATIVE_PROMPT == "people, text, watermark, buildings, distorted horizon"
