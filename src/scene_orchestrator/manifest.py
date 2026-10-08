"""out/manifest.json: one entry per scene, the first file Unity reads.

    {"P01": {"status": "ok",
             "sky_mode": "panorama",
             "files": ["scene_spec.json", "skybox_48213.png", "skybox_48213.json"],
             "updated_at": "2026-10-01T12:00:00Z"}}

status is ok | failed | fallback:

    ok        the scene as requested
    fallback  the panorama could not be made, so the spec was written with the
              procedural sky instead; Unity can still render the scene
    failed    there is no usable scene

sky_mode is the mode of the spec on disk, so "procedural" for a fallback.
Entries that are not ok carry an "error". File names are relative to
out/<scene_id>/. Unity never has to know whether generation worked: it shows
the image when the spec names one, and the spec's procedural sky when not.

An entry may also carry "slurm_job_id" and "seed": a skybox job that was
submitted and not seen to finish (the scene is still being generated, the
process was killed, or the SSH connection dropped). The next run of that scene
waits for that job instead of submitting another.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

from .models import SkyMode
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
    sky_mode: SkyMode,
    files: list[str],
    error: str | None = None,
    *,
    slurm_job_id: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "status": status,
        "sky_mode": sky_mode,
        "files": list(files),
        "updated_at": utc_now_iso(),
    }
    if error is not None:
        entry["error"] = error
    if slurm_job_id is not None:
        # The seed says which request the job belongs to.
        entry["slurm_job_id"] = slurm_job_id
        entry["seed"] = seed
    with _LOCK:
        manifest = read_manifest(out_dir)
        manifest[scene_id] = entry
        write_json(manifest_path(out_dir), dict(sorted(manifest.items())))
    return entry
