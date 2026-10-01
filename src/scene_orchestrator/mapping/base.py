"""The SpecMapper interface: persona in, SceneSpec out."""

from __future__ import annotations

from typing import Protocol

from ..models import PersonaProfile, SceneSpec


class SpecMapper(Protocol):
    def map(self, persona: PersonaProfile, seed: int, scene_id: str) -> SceneSpec: ...
