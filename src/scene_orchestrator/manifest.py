"""out/manifest.json: one entry per scene, the first file Unity reads.

    {"P01_A": {"status": "ok",
               "files": ["scene_spec.json", "skybox_48213.png", "skybox_48213.json"],
               "updated_at": "2026-10-01T12:00:00Z"}}

status is ok | failed | fallback, and failed entries carry an "error". File
names are relative to out/<scene_id>/. Unity never has to know whether
generation worked: it reads the status and uses the scene or a fallback.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

from .outputs import utc_now_iso, write_json

ManifestStatus = Literal["ok", "failed", "fallback"]

# Read-modify-write of one shared file: one writer at a time within a process.
_LOCK = threading.Lock()


def manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / "manifest.json"


def read_manifest(out_dir: Path) -> dict[str, Any]:
    path = manifest_path(out_dir)
    if not path.is_file():
        return {}
    # utf-8-sig tolerates a BOM from Windows tooling; files we write have none.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def update_manifest(
    out_dir: Path,
    scene_id: str,
    status: ManifestStatus,
    files: list[str],
    error: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"status": status, "files": list(files), "updated_at": utc_now_iso()}
    if error is not None:
        entry["error"] = error
    with _LOCK:
        manifest = read_manifest(out_dir)
        manifest[scene_id] = entry
        write_json(manifest_path(out_dir), dict(sorted(manifest.items())))
    return entry
