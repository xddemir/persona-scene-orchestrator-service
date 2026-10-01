"""The boundary between other people's persona formats and ours.

Profiles arrive in formats we don't control. Each format gets one adapter
that converts it into a PersonaProfile, so nothing past this point ever sees
a foreign shape, and a change on their side is a change in one file here.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..models import PersonaProfile


class PersonaAdapterError(ValueError):
    """The raw profile cannot be converted. The message says what is wrong."""


class PersonaAdapter(Protocol):
    def adapt(self, raw: dict[str, Any]) -> PersonaProfile: ...

    def participant_id(self, raw: dict[str, Any]) -> str:
        """Just the id, without converting the rest: enough to name a scene
        whose persona may still turn out to be invalid."""
        ...
