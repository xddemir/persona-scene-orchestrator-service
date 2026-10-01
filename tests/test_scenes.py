"""/scenes end to end: persona -> spec -> prompt -> (simulated) Slurm -> files.

One scene per participant: scene_id is the participant id. Only the Pegasus
login node is simulated (tests/fakes.py); the adapter, mapper, prompt builder,
Slurm client, file writing and manifest are all real.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import PNG, FakePegasus, make_client
from scene_orchestrator.api.app import create_app
from scene_orchestrator.manifest import read_manifest
from scene_orchestrator.mapping import RuleSpecMapper
from scene_orchestrator.models import SceneSpec
from scene_orchestrator.pipeline import UNFINISHED, ScenePipeline, derive_seed

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "personas" / "chatbot"


def _persona_file(alias: str) -> str:
    return str(FIXTURES / f"{alias}.json")


def _raw(alias: str) -> dict:
    return json.loads((FIXTURES / f"{alias}.json").read_text(encoding="utf-8"))


@contextmanager
def _serve(config, pegasus=None):
    app = create_app(config, image_gen=make_client(pegasus or FakePegasus()))
    with TestClient(app) as client:
        yield client


def _run(client, **body) -> dict:
    response = client.post("/scenes", json=body)
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["status"] == "queued"
    assert client.app.state.scenes.wait_idle()
    return client.get(f"/scenes/{accepted['scene_id']}").json()


# -- the happy path, P01 and P02 --------------------------------------------


def test_p01_scene_is_built_end_to_end(config, out_dir):
    with _serve(config) as client:
        scene = _run(client, persona_file=_persona_file("P01"))

    seed = derive_seed("P01")
    assert scene["scene_id"] == "P01"
    assert scene["status"] == "ready"
    assert scene["error"] is None
    assert scene["files"] == ["scene_spec.json", f"skybox_{seed}.png", f"skybox_{seed}.json"]

    # scene_spec.json: valid, self-contained, and what the API returned
    on_disk = SceneSpec.model_validate_json((out_dir / "P01" / "scene_spec.json").read_text())
    assert on_disk == SceneSpec.model_validate(scene["spec"])
    assert on_disk.scene_id == "P01"
    assert on_disk.seed == seed
    assert on_disk.skybox.prompt.startswith("equirectangular 360 view, a sheltered ")
    assert on_disk.skybox.uri == f"skybox_{seed}.png"
    assert (out_dir / "P01" / on_disk.skybox.uri).read_bytes() == PNG

    # manifest.json
    entry = read_manifest(out_dir)["P01"]
    assert entry["status"] == "ok"
    assert entry["files"] == scene["files"]
    assert entry["updated_at"].endswith("Z")
    assert "error" not in entry


def test_p02_inline_persona_and_the_manifest_holds_both(config, out_dir):
    with _serve(config) as client:
        _run(client, persona_file=_persona_file("P01"))
        scene = _run(client, persona=_raw("P02"))

    assert scene["scene_id"] == "P02"
    assert scene["status"] == "ready"
    spec = SceneSpec.model_validate_json((out_dir / "P02" / "scene_spec.json").read_text())
    assert spec.skybox.prompt.startswith("equirectangular 360 view, an open ")

    manifest = read_manifest(out_dir)
    assert list(manifest) == ["P01", "P02"]
    assert {e["status"] for e in manifest.values()} == {"ok"}


def test_one_call_returns_everything_for_the_participant(config):
    with _serve(config) as client:
        scene = _run(client, persona_file=_persona_file("P01"))

    assert scene["persona"]["participant_id"] == "P01"
    assert scene["persona"]["traits"]["neuroticism"] == 0.68
    assert scene["persona"]["confidence"]["neuroticism"] == 0.9
    assert scene["detail"].startswith("Slurm job ")  # Slurm progress
    assert scene["detail"].endswith(": COMPLETED")
    assert scene["spec"]["skybox"]["prompt"].startswith("equirectangular 360 view")


def test_the_prompt_sent_to_pegasus_is_the_one_on_the_spec(config):
    pegasus = FakePegasus()
    with _serve(config, pegasus) as client:
        scene = _run(client, persona_file=_persona_file("P01"))
    assert scene["spec"]["skybox"]["prompt"] in pegasus.submitted[0].replace("'", "")


def test_the_spec_is_available_before_the_image(out_dir):
    built = []
    ScenePipeline(make_client(FakePegasus()), out_dir).run(
        _raw("P01"), "P01", on_built=lambda persona, spec: built.append(spec)
    )
    before_image, after_image = built
    assert before_image.skybox.prompt is not None
    assert before_image.skybox.uri is None
    assert after_image.skybox.uri.startswith("skybox_")


# -- seeds ------------------------------------------------------------------


def test_a_given_seed_is_used(config):
    with _serve(config) as client:
        scene = _run(client, persona_file=_persona_file("P01"), seed=7)
    assert scene["spec"]["seed"] == 7
    assert scene["spec"]["skybox"]["uri"] == "skybox_7.png"


def test_derived_seeds_are_stable_and_differ_by_participant():
    assert derive_seed("P01") == derive_seed("P01")
    assert derive_seed("P01") != derive_seed("P02")
    assert 0 <= derive_seed("P01") < 2**32


# -- every failure ends in the manifest -------------------------------------


def test_a_failed_slurm_job_is_failed_in_the_manifest(config, out_dir):
    with _serve(config, FakePegasus(jobs=[("FAILED",)])) as client:
        scene = _run(client, persona_file=_persona_file("P01"))

    assert scene["status"] == "failed"
    entry = read_manifest(out_dir)["P01"]
    assert entry["status"] == "failed"
    assert "CUDA out of memory" in entry["error"]
    assert entry["files"] == ["scene_spec.json"]  # what was attempted, no image
    spec = SceneSpec.model_validate_json((out_dir / "P01" / "scene_spec.json").read_text())
    assert spec.skybox.uri is None


def test_an_invalid_persona_is_failed_in_the_manifest(config, out_dir):
    with _serve(config) as client:
        scene = _run(client, persona_file=_persona_file("P03"))

    assert scene["scene_id"] == "P03"
    assert scene["status"] == "failed"
    assert scene["persona"] is None  # it could not be converted
    entry = read_manifest(out_dir)["P03"]
    assert entry["status"] == "failed"
    assert "Agreeableness" in entry["error"]
    assert entry["files"] == []


def test_no_connection_to_pegasus_is_failed_in_the_manifest(config, out_dir):
    with _serve(config, FakePegasus(connected=False)) as client:
        _run(client, persona_file=_persona_file("P02"))
    entry = read_manifest(out_dir)["P02"]
    assert entry["status"] == "failed"
    assert "no shared SSH connection" in entry["error"]


def test_an_unexpected_bug_is_failed_in_the_manifest(config, out_dir, monkeypatch):
    def broken(self, persona, seed, scene_id):
        raise RuntimeError("mapper bug")

    monkeypatch.setattr(RuleSpecMapper, "map", broken)
    with _serve(config) as client:
        scene = _run(client, persona_file=_persona_file("P01"))
    assert scene["status"] == "failed"
    assert read_manifest(out_dir)["P01"]["error"] == "RuntimeError: mapper bug"


def test_a_scene_is_in_the_manifest_from_the_moment_it_starts(out_dir):
    """Even a process killed mid-job leaves Unity an entry, not a gap."""
    seen = {}

    def on_stage(stage):
        seen[stage] = read_manifest(out_dir).get("P01")

    ScenePipeline(make_client(FakePegasus()), out_dir).run(_raw("P01"), "P01", on_stage=on_stage)

    assert seen["generating_image"]["status"] == "failed"
    assert seen["generating_image"]["error"] == UNFINISHED
    assert read_manifest(out_dir)["P01"]["status"] == "ok"


def test_stages_are_reported_in_order(out_dir):
    stages = []
    ScenePipeline(make_client(FakePegasus()), out_dir).run(
        _raw("P01"), "P01", on_stage=stages.append
    )
    assert stages == ["building_spec", "generating_image"]


# -- requests that can't name a scene ---------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},  # no persona at all
        {"persona_file": "x.json", "persona": {}},  # both
        {"persona_file": "does/not/exist.json"},
        {"persona": {"assessment": {}}},  # no user.alias
        {"persona": {"user": {"alias": "../P01"}}},  # alias that is no safe folder name
        {"persona_file": "x.json", "condition": "A"},  # no conditions any more
    ],
)
def test_requests_without_a_usable_scene_id_are_422(config, body):
    with _serve(config) as client:
        assert client.post("/scenes", json=body).status_code == 422


def test_unknown_scene_is_404(config):
    with _serve(config) as client:
        assert client.get("/scenes/P99").status_code == 404


# -- downloading a scene's files (for Unity) --------------------------------


def test_every_listed_file_can_be_downloaded(config, out_dir):
    with _serve(config) as client:
        scene = _run(client, persona_file=_persona_file("P01"))
        png, spec_json = scene["spec"]["skybox"]["uri"], "scene_spec.json"

        image = client.get(f"/scenes/P01/files/{png}")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"
        assert image.content == PNG

        spec = client.get(f"/scenes/P01/files/{spec_json}")
        assert spec.headers["content-type"] == "application/json"
        assert SceneSpec.model_validate_json(spec.content) == SceneSpec.model_validate(scene["spec"])

        for name in scene["files"]:
            assert client.get(f"/scenes/P01/files/{name}").status_code == 200


@pytest.mark.parametrize(
    ("scene_id", "name"),
    [
        ("P01", "skybox_1.png"),  # not this scene's file
        ("P01", "manifest.json"),  # exists in out/, but isn't listed
        ("P01", "..%2Fmanifest.json"),  # path traversal
        ("P99", "scene_spec.json"),  # unknown scene
        ("..", "manifest.json"),
    ],
)
def test_only_files_listed_in_the_manifest_are_served(config, scene_id, name):
    with _serve(config) as client:
        _run(client, persona_file=_persona_file("P01"))
        assert client.get(f"/scenes/{scene_id}/files/{name}").status_code == 404


def test_a_failed_scene_has_no_image_to_download(config):
    with _serve(config, FakePegasus(jobs=[("FAILED",)])) as client:
        scene = _run(client, persona_file=_persona_file("P01"))
        assert scene["files"] == ["scene_spec.json"]
        png = f"skybox_{derive_seed('P01')}.png"
        assert client.get(f"/scenes/P01/files/{png}").status_code == 404


def test_after_a_restart_the_scene_is_read_back_from_disk(config):
    with _serve(config) as client:
        _run(client, persona_file=_persona_file("P01"))
    with _serve(config) as fresh:  # a new app: nothing in memory
        scene = fresh.get("/scenes/P01").json()
    assert scene["status"] == "ready"
    assert scene["spec"]["scene_id"] == "P01"
    assert scene["files"][0] == "scene_spec.json"
    assert scene["persona"] is None  # only known while the server is up
