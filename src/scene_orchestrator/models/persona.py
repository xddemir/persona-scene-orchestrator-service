"""PersonaProfile: the input contract.

Profiles come from a parallel thesis (a chatbot) or a BFI-2-S questionnaire.
This is our own type, so nothing here depends on theirs: their output is
converted at the boundary, and `source="stub"` covers synthetic profiles.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Traits(BaseModel):
    """Big Five scores, normalised to 0..1."""

    model_config = ConfigDict(extra="forbid")

    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)


class PersonaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    participant_id: str  # pseudonymous
    source: Literal["chatbot", "bfi2s", "stub"]
    traits: Traits
    # Per-trait confidence, when the source provides one.
    confidence: dict[str, float] | None = None
