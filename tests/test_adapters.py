"""ChatbotV2Adapter against the fixtures in fixtures/personas/chatbot/."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scene_orchestrator.adapters import ChatbotV2Adapter, PersonaAdapter, PersonaAdapterError
from scene_orchestrator.models import PersonaProfile

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "personas" / "chatbot"


def _load(alias: str) -> dict:
    return json.loads((FIXTURES / f"{alias}.json").read_text(encoding="utf-8"))


def _adapt(raw: dict) -> PersonaProfile:
    adapter: PersonaAdapter = ChatbotV2Adapter()  # also checks it fits the protocol
    return adapter.adapt(raw)


# -- the inversion ----------------------------------------------------------


def test_emotional_stability_becomes_inverted_neuroticism():
    """P01 reports Emotional Stability 32/100: low stability, so HIGH neuroticism."""
    persona = _adapt(_load("P01"))
    assert persona.traits.neuroticism == 0.68
    assert persona.traits.neuroticism != 0.32


def test_the_inversion_separates_opposite_participants():
    p01, p02 = _adapt(_load("P01")), _adapt(_load("P02"))
    assert p01.traits.neuroticism > 0.5 > p02.traits.neuroticism  # anxious vs stable
    assert p02.traits.neuroticism == pytest.approx(0.2)


# -- round trip -------------------------------------------------------------


@pytest.mark.parametrize("alias", ["P01", "P02"])
def test_complete_fixtures_become_valid_profiles(alias):
    persona = _adapt(_load(alias))
    assert isinstance(persona, PersonaProfile)
    assert persona.participant_id == alias  # from user.alias
    assert persona.source == "chatbot"
    # Survives serialisation unchanged.
    assert PersonaProfile.model_validate_json(persona.model_dump_json()) == persona


def test_scores_are_scaled_to_0_1():
    traits = _adapt(_load("P01")).traits
    assert traits.openness == 0.78
    assert traits.conscientiousness == 0.55
    assert traits.extraversion == 0.31
    assert traits.agreeableness == 0.64


def test_certainty_becomes_confidence_per_trait():
    assert _adapt(_load("P01")).confidence == {
        "openness": 0.9,
        "conscientiousness": 0.6,
        "extraversion": 0.9,
        "agreeableness": 0.6,
        "neuroticism": 0.9,
    }


def test_a_missing_dimension_raises_and_names_it():
    with pytest.raises(PersonaAdapterError, match=r"'P03'.*missing.*Agreeableness"):
        _adapt(_load("P03"))


# -- other malformed input --------------------------------------------------


def _with_first_dimension(**changes) -> dict:
    raw = copy.deepcopy(_load("P01"))
    raw["assessment"]["dimensions"][0].update(changes)
    return raw


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (_with_first_dimension(name="Curiosity"), "unknown dimension 'Curiosity'"),
        (_with_first_dimension(name="Conscientiousness"), "appears twice"),
        (_with_first_dimension(certainty="very high"), "certainty 'very high'"),
        (_with_first_dimension(score=120), "outside 0..100"),
        (_with_first_dimension(score="78"), "numeric"),
        ({"assessment": {"dimensions": []}}, "no 'user'"),
    ],
)
def test_malformed_profiles_raise_with_a_reason(raw, message):
    with pytest.raises(PersonaAdapterError, match=message):
        _adapt(raw)


def test_dimension_names_ignore_case_and_spacing():
    raw = _with_first_dimension(name="  openness to experience ")
    assert _adapt(raw).traits.openness == 0.78
