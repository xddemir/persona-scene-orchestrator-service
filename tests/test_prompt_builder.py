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
from scene_orchestrator.models import PersonaProfile, SceneSpec, Traits

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


def test_p01_prompt_is_sheltered_low_and_misty():
    prompt = BUILDER.build(_spec("P01"))
    assert "sheltered" in prompt
    assert "low sun" in prompt
    assert "heavy drifting mist" in prompt


def test_p02_prompt_is_open_bright_and_clear():
    prompt = BUILDER.build(_spec("P02"))
    assert "open" in prompt
    assert "midday sun" in prompt
    assert "clear air" in prompt


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
