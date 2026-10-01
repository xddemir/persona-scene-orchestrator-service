"""Atomic writes, safe inside a Unity project's Assets/ folder."""

from __future__ import annotations

from scene_orchestrator.outputs import write_bytes_atomic


def test_writes_land_whole_and_leave_no_temp_file(tmp_path):
    target = tmp_path / "P01" / "skybox_1.png"
    write_bytes_atomic(target, b"pixels")
    assert target.read_bytes() == b"pixels"
    assert [p.name for p in target.parent.iterdir()] == ["skybox_1.png"]


def test_the_temp_file_is_hidden_from_unity(tmp_path, monkeypatch):
    """Unity ignores dot-files, so it never imports a half-written temp file."""
    seen = []
    import scene_orchestrator.outputs as outputs

    real_replace = outputs.os.replace
    monkeypatch.setattr(outputs.os, "replace", lambda src, dst: (seen.append(src), real_replace(src, dst)))
    write_bytes_atomic(tmp_path / "scene_spec.json", b"{}")
    (tmp,) = seen
    assert tmp.name.startswith(".")
