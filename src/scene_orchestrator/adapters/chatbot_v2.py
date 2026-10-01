"""Adapter for the chatbot thesis' `conversational_bigfive_v2` output.

    {"user": {"alias": "P01", ...},
     "assessment": {"dimensions": [
         {"name": "Emotional Stability", "score": 32, "scale_max": 100, "certainty": "high"},
         ...]}}

Their format differs from ours in three ways that matter:

- Scores run 0..scale_max; ours run 0..1.
- They report Emotional Stability, the opposite pole of neuroticism, so
  neuroticism = 1 - score/scale_max. Getting this backwards would quietly
  give anxious participants their opposite's scene.
- Dimension names are prose, so they go through a lookup table.

Anything missing or unrecognised raises. A defaulted trait would be a
fabricated personality score, and it would end up in the thesis data.
"""

from __future__ import annotations

from typing import Any

from ..models import PersonaProfile, Traits
from .base import PersonaAdapterError

# Their dimension name -> (our trait, whether their scale runs the other way).
DIMENSIONS: dict[str, tuple[str, bool]] = {
    "openness to experience": ("openness", False),
    "conscientiousness": ("conscientiousness", False),
    "extraversion": ("extraversion", False),
    "agreeableness": ("agreeableness", False),
    "emotional stability": ("neuroticism", True),
}

CERTAINTY: dict[str, float] = {"high": 0.9, "medium": 0.6, "low": 0.3}


class ChatbotV2Adapter:
    def participant_id(self, raw: dict[str, Any]) -> str:
        alias = _get(raw, "user", "alias")
        if not isinstance(alias, str) or not alias.strip():
            raise PersonaAdapterError(f"chatbot profile has an empty user.alias: {alias!r}")
        return alias.strip()

    def adapt(self, raw: dict[str, Any]) -> PersonaProfile:
        alias = self.participant_id(raw)
        where = f"chatbot profile {alias!r}"

        traits: dict[str, float] = {}
        confidence: dict[str, float] = {}
        for dimension in _get(raw, "assessment", "dimensions"):
            name = str(dimension.get("name", "")).strip()
            if name.lower() not in DIMENSIONS:
                raise PersonaAdapterError(
                    f"{where}: unknown dimension {name!r}; "
                    f"expected one of {sorted(n.title() for n in DIMENSIONS)}"
                )
            trait, inverted = DIMENSIONS[name.lower()]
            if trait in traits:
                raise PersonaAdapterError(f"{where}: dimension {name!r} appears twice")

            traits[trait] = _normalise(dimension, where, name, inverted)
            certainty = dimension.get("certainty")
            if certainty not in CERTAINTY:
                raise PersonaAdapterError(
                    f"{where}: {name!r} has certainty {certainty!r}; "
                    f"expected one of {list(CERTAINTY)}"
                )
            confidence[trait] = CERTAINTY[certainty]

        missing = [
            their_name.title()
            for their_name, (trait, _) in DIMENSIONS.items()
            if trait not in traits
        ]
        if missing:
            raise PersonaAdapterError(
                f"{where}: missing dimension(s) {missing}. Not defaulting them: "
                "a guessed trait would be a fabricated personality score."
            )

        return PersonaProfile(
            participant_id=alias,
            source="chatbot",
            traits=Traits(**traits),
            confidence=confidence,
        )


def _normalise(dimension: dict[str, Any], where: str, name: str, inverted: bool) -> float:
    score, scale_max = dimension.get("score"), dimension.get("scale_max")
    if not isinstance(score, (int, float)) or not isinstance(scale_max, (int, float)):
        raise PersonaAdapterError(f"{where}: {name!r} needs numeric score and scale_max")
    if scale_max <= 0 or not 0 <= score <= scale_max:
        raise PersonaAdapterError(
            f"{where}: {name!r} score {score} is outside 0..{scale_max}"
        )
    # (max - score) / max rather than 1 - score/max: same value, but exact
    # for whole-number scores (32 -> 0.68, not 0.6799999999999999).
    return (scale_max - score) / scale_max if inverted else score / scale_max


def _get(raw: dict[str, Any], *keys: str) -> Any:
    value: Any = raw
    for depth, key in enumerate(keys):
        if not isinstance(value, dict) or key not in value:
            raise PersonaAdapterError(
                f"chatbot profile has no {'.'.join(keys[: depth + 1])!r}"
            )
        value = value[key]
    return value
