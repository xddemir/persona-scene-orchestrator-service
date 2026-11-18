"""Checking the places of places_v1.yaml in the headset.

    python -m scene_orchestrator.places review                      # every place
    python -m scene_orchestrator.places review --place forest_lake
    python -m scene_orchestrator.places mark forest_lake verified --note "reads as a lake in the woods"
    python -m scene_orchestrator.places mark summit_top rejected --note "the top's edge floats"

review builds each place's scene from a neutral persona (every trait 0.5)
and seed 0, with the place forced and the procedural sky (no image, no
Pegasus). Each goes through the same pipeline as a participant's scene, so it
ends the same way: out/place-<id>/scene_spec.json and an "ok" entry in
out/manifest.json. In Unity, type place-<id> as the participant number and
press Fetch; GET /scenes/place-<id> answers with it. The server can be
running meanwhile: it reads the manifest on every request.

mark sets one place's status, checked (today) and note in places_v1.yaml,
and leaves every other line of the file, comments included, as it was. The
file is checked before it is written. A running server keeps the places it
started with; restart it (or run it with --reload-include "*.yaml").
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config
from .mapping.places import PLACES_FILE, Place, PlacesError, parse_places, set_place_fields
from .mapping.rule_mapper import BIOMES, PLACES, RuleSpecMapper
from .models import PersonaProfile, Traits
from .outputs import write_bytes_atomic
from .pipeline import SceneOutcome, ScenePipeline

REVIEW_SEED = 0
NEUTRAL = 0.5  # every trait: the middle of every linear rule


def neutral_persona(participant_id: str) -> PersonaProfile:
    value = dict.fromkeys(Traits.model_fields, NEUTRAL)
    return PersonaProfile(participant_id=participant_id, source="stub", traits=Traits(**value))


class _NeutralAdapter:
    """Stands in for the chatbot adapter: every review scene's persona is the
    neutral one, under the scene's id."""

    def adapt(self, raw: dict[str, Any]) -> PersonaProfile:
        return neutral_persona(self.participant_id(raw))

    def participant_id(self, raw: dict[str, Any]) -> str:
        return raw["participant_id"]


def review(
    places: list[Place], out_dir: Path, *, allow_candidates: bool
) -> dict[str, SceneOutcome]:
    """Build and register each place's review scene; scene id -> outcome."""
    outcomes = {}
    for place in places:
        mapper = RuleSpecMapper(allow_candidates=allow_candidates, place_id=place.id)
        # No image client: a procedural scene never asks for one.
        pipeline = ScenePipeline(None, out_dir, adapter=_NeutralAdapter(), mapper=mapper)
        outcomes[place.scene_id] = pipeline.run(
            {"participant_id": place.scene_id}, place.scene_id, REVIEW_SEED, "procedural"
        )
    return outcomes


def mark(
    place_id: str, status: str, note: str | None = None, *,
    path: Path = PLACES_FILE, today: date | None = None,
) -> Place:
    """Set a place's status, checked and (if given) note in the file at `path`."""
    raw = path.read_bytes()
    before = parse_places(raw, kits=_kits(), path=path)
    before.get(place_id)  # an unknown id fails here, naming the file
    checked = today or date.today()
    fields = {"status": status, "checked": checked.isoformat()}
    if note is not None:
        # A JSON string is a valid YAML double-quoted one.
        fields["note"] = json.dumps(note, ensure_ascii=False)
    text = set_place_fields(raw.decode("utf-8"), place_id, fields)
    after = parse_places(text.encode("utf-8"), kits=_kits(), path=path)

    # The edit changed this place's three fields and nothing else.
    place = after.get(place_id)
    expected = before.get(place_id).model_copy(update={
        "status": status, "checked": checked,
        "note": note if note is not None else before.get(place_id).note,
    })
    others_before = [p for p in before.places if p.id != place_id]
    others_after = [p for p in after.places if p.id != place_id]
    if place != expected or others_before != others_after:
        raise PlacesError(f"{path}: could not set {place_id}'s fields cleanly; edit it by hand")
    write_bytes_atomic(path, text.encode("utf-8"))
    return place


def _kits() -> dict[str, frozenset[str]]:
    return {biome: kit.names for biome, kit in BIOMES.items()}


def _describe(place: Place) -> str:
    water = "no water" if place.water == "none" else f"{place.water} {place.water_relation}"
    season = f", {place.season}" if place.season else ""
    return (f"{place.setting}, {place.landform} {place.position}, {water}, {place.vegetation}"
            f"{season}, {place.biome} ({place.status})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scene_orchestrator.places",
        description="Review the places of places_v1.yaml in the headset, and mark them.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    review_cmd = commands.add_parser(
        "review", help="build each place's scene as place-<id>, for Unity to fetch"
    )
    review_cmd.add_argument("--place", help="only this place")
    review_cmd.add_argument("--config", type=Path, default=None, help="instead of the default config")

    mark_cmd = commands.add_parser("mark", help="set a place's status after a headset check")
    mark_cmd.add_argument("place")
    mark_cmd.add_argument("status", choices=("verified", "rejected"))
    mark_cmd.add_argument("--note", default=None, help="what was seen; left as it was if omitted")

    args = parser.parse_args(argv)
    try:
        if args.command == "mark":
            # The module's PLACES_FILE, looked up now, not mark()'s default.
            place = mark(args.place, args.status, args.note, path=PLACES_FILE)
            print(f"{place.id}: {place.status}, checked {place.checked}. "
                  "Restart the server to use it.")
            return 0

        config = load_config(args.config)
        places = [PLACES.get(args.place)] if args.place else list(PLACES.places)
        outcomes = review(places, config.out_dir, allow_candidates=config.places.allow_candidates)
    except (ConfigError, PlacesError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for place in places:
        outcome = outcomes[place.scene_id]
        result = outcome.status if outcome.error is None else f"{outcome.status}: {outcome.error}"
        print(f"{place.scene_id}: {result}. {_describe(place)}")
    print(f"In Unity, fetch place-<id> as the participant number (written to {config.out_dir}).")
    return 0 if all(o.status == "ok" for o in outcomes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
