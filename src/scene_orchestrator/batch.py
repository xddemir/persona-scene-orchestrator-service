"""Scenes for a whole queue of participants, one after another.

    python -m scene_orchestrator.batch queue.json --personas path/to/personas
    python -m scene_orchestrator.batch queue.json --personas ... --sky-mode procedural

The queue is a JSON list with one entry per participant:

    [{"participant_id": "P01"},
     {"participant_id": "P02", "sky_mode": "procedural"},
     {"participant_id": "P07", "persona_file": "elsewhere/p07.json", "seed": 1234}]

Without a persona_file, the persona is <personas>/<participant_id>.json. Each
entry goes through the same pipeline as POST /scenes, so it ends in the same
files and the same manifest entry.

It can be stopped and started again at any point. A scene the manifest already
has as "ok" in the wanted sky mode is skipped. A scene whose Slurm job was
submitted but never seen to finish (the runner was killed, the SSH connection
dropped) has that job's id in the manifest, and the next run waits for that
job instead of submitting another. Anything else that is not "ok" is redone.

Exit status: 0 when every scene is ok, 1 when any is failed or fell back to
the procedural sky, 2 when nothing was run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .clients.image_gen import SlurmImageGenClient
from .config import ConfigError, load_config
from .manifest import read_manifest, update_manifest
from .models import SkyMode
from .outputs import SCENE_ID_PATTERN
from .pipeline import ScenePipeline

SKIPPED = "skipped"


class QueueError(Exception):
    """The queue cannot be run as given. The message says what is wrong."""


class QueueEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(pattern=SCENE_ID_PATTERN)  # also the scene_id
    persona_file: Path | None = None
    # Omitted: derived from the participant id, as for POST /scenes.
    seed: int | None = Field(default=None, ge=0)
    # Omitted: the run's --sky-mode.
    sky_mode: SkyMode | None = None


def read_queue(path: Path) -> list[QueueEntry]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        entries = TypeAdapter(list[QueueEntry]).validate_python(raw)
    except (OSError, ValueError) as exc:
        raise QueueError(f"cannot read the queue {path}: {exc}") from exc
    repeated = [p for p, n in Counter(e.participant_id for e in entries).items() if n > 1]
    if repeated:
        raise QueueError(
            f"{path}: {repeated} listed more than once; a participant has one scene"
        )
    return entries


def run_batch(
    entries: list[QueueEntry],
    pipeline: ScenePipeline,
    out_dir: Path,
    *,
    personas_dir: Path | None = None,
    sky_mode: SkyMode = "panorama",
    log: Callable[[str], None] = print,
) -> dict[str, str]:
    """Run each entry in turn; return scene_id -> ok | fallback | failed | skipped."""
    results: dict[str, str] = {}
    for entry in entries:
        scene_id = entry.participant_id
        mode = entry.sky_mode or sky_mode

        done = read_manifest(out_dir).get(scene_id, {})
        # Entries written before sky modes existed are panoramas.
        if done.get("status") == "ok" and done.get("sky_mode", "panorama") == mode:
            results[scene_id] = SKIPPED
            log(f"{scene_id}: already ok, skipped")
            continue

        try:
            raw = _read_persona(entry, personas_dir)
        except QueueError as exc:
            # The pipeline never starts, so its manifest entry is written here.
            update_manifest(out_dir, scene_id, "failed", mode, [], str(exc))
            results[scene_id] = "failed"
            log(f"{scene_id}: failed: {exc}")
            continue

        outcome = pipeline.run(
            raw, scene_id, entry.seed, mode,
            on_progress=lambda message, scene_id=scene_id: log(f"{scene_id}: {message}"),
        )
        results[scene_id] = outcome.status
        log(f"{scene_id}: {outcome.status}" + (f": {outcome.error}" if outcome.error else ""))
    return results


def _read_persona(entry: QueueEntry, personas_dir: Path | None) -> dict[str, Any]:
    if entry.persona_file is not None:
        path = entry.persona_file
    elif personas_dir is not None:
        path = personas_dir / f"{entry.participant_id}.json"
    else:
        raise QueueError("no persona_file in the queue entry and no personas folder given")
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise QueueError(f"persona file not found: {path}") from None
    except (OSError, ValueError) as exc:
        raise QueueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise QueueError(f"{path} is not a JSON object")
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scene_orchestrator.batch",
        description="Generate the scenes for a queue of participants. Resumable.",
    )
    parser.add_argument("queue", type=Path, help="JSON list of {participant_id, ...}")
    parser.add_argument(
        "--personas", type=Path, default=None,
        help="folder holding <participant_id>.json for entries without a persona_file",
    )
    parser.add_argument(
        "--sky-mode", choices=get_args(SkyMode), default="panorama",
        help="for entries that don't set their own (default: panorama)",
    )
    parser.add_argument("--config", type=Path, default=None, help="instead of the default config")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        entries = read_queue(args.queue)
        # Checked up front, never defaulted: a persona taken from the wrong
        # folder would give a participant somebody else's scene, marked "ok".
        if args.personas is None and any(e.persona_file is None for e in entries):
            raise QueueError(
                "give --personas, or a persona_file for every entry in the queue"
            )
    except (ConfigError, QueueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    client = SlurmImageGenClient(config.image_gen)
    if any((e.sky_mode or args.sky_mode) == "panorama" for e in entries):
        # Without the connection no image can be made at all, and every scene
        # would be written as a fallback. Better to run nothing.
        status = client.status()
        if not status.connected:
            print(f"error: {status.error}", file=sys.stderr)
            return 2

    results = run_batch(
        entries,
        ScenePipeline(client, config.out_dir),
        config.out_dir,
        personas_dir=args.personas,
        sky_mode=args.sky_mode,
    )
    counts = Counter(results.values())
    print(", ".join(f"{n} {status}" for status, n in sorted(counts.items())) or "empty queue")
    return 0 if set(results.values()) <= {"ok", SKIPPED} else 1


if __name__ == "__main__":
    raise SystemExit(main())
