"""The on-disk output contract Unity reads.

    out/<scene_id>/skybox_<seed>.png
    out/<scene_id>/skybox_<seed>.json    # image-gen's sidecar, `file` rewritten

Every write is atomic: a job killed mid-write (walltime, OOM, node failure)
must never leave a half-written file where Unity expects a whole one. Within a
scene, the manifest is written last, so once it says "ok" every file it lists
is already complete.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The same rule image-gen enforces, since the id is passed on to it. It becomes
# a directory name, so it must not be able to escape out/; a leading
# alphanumeric also rules out "." and "..".
SCENE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_SCENE_ID_RE = re.compile(SCENE_ID_PATTERN)


def validate_scene_id(scene_id: str) -> str:
    if not _SCENE_ID_RE.match(scene_id or ""):
        raise ValueError(
            f"invalid scene_id {scene_id!r}: use 1-64 characters of "
            "[A-Za-z0-9._-], starting with a letter or digit"
        )
    return scene_id


def scene_dir(out_dir: Path, scene_id: str) -> Path:
    return Path(out_dir) / validate_scene_id(scene_id)


def skybox_png_path(out_dir: Path, scene_id: str, seed: int) -> Path:
    return scene_dir(out_dir, scene_id) / f"skybox_{seed}.png"


def skybox_sidecar_path(out_dir: Path, scene_id: str, seed: int) -> Path:
    return scene_dir(out_dir, scene_id) / f"skybox_{seed}.json"


def scene_spec_path(out_dir: Path, scene_id: str) -> Path:
    return scene_dir(out_dir, scene_id) / "scene_spec.json"


def relative_to_out(out_dir: Path, path: Path) -> str:
    """Forward slashes: written on Windows or Linux, read by Unity on either."""
    return Path(path).resolve().relative_to(Path(out_dir).resolve()).as_posix()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Leading dot: out_dir may sit inside a Unity project's Assets/, and Unity
    # skips hidden files, so it never imports the half-written temp file.
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_bytes_atomic(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
