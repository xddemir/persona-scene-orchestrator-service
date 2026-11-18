"""The SpecMapper interface: persona in, SceneSpec out."""

from __future__ import annotations

from typing import Any, Protocol

from ..models import PersonaProfile, SceneSpec


class SpecMapper(Protocol):
    def map(self, persona: PersonaProfile, seed: int, scene_id: str) -> SceneSpec: ...

    def provenance(self, persona: PersonaProfile) -> dict[str, Any]:
        """What the manifest records about how this persona's spec was made
        (for the rule mapper: the place, its status, the list's hash)."""
        ...
