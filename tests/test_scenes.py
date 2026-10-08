"""/scenes end to end: persona -> spec -> prompt -> (simulated) Slurm -> files.

One scene per participant: scene_id is the participant id. Only the Pegasus
login node is simulated (tests/fakes.py); the adapter, mapper, prompt builder,
Slurm client, file writing and manifest are all real.

Both sky modes: "panorama" goes through the Slurm job, "procedural" stops
before it. A panorama that cannot be made falls back to the procedural sky.
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import PNG, FakePegasus, make_client
from scene_orchestrator.api.app import create_app
from scene_orchestrator.config import AutoCreateConfig
from scene_orchestrator.manifest import manifest_path, read_manifest
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
    assert on_disk.sky_mode == "panorama"
    assert on_disk.procedural_sky.atmosphere_thickness == 1.72  # there all the same

    # manifest.json
    entry = read_manifest(out_dir)["P01"]
    assert list(entry) == ["status", "sky_mode", "files", "updated_at"]
    assert entry["status"] == "ok"
    assert entry["sky_mode"] == "panorama"
    assert entry["files"] == scene["files"]
    assert entry["updated_at"].endswith("Z")


def test_p02_inline_persona_and_the_manifest_holds_both(config, out_dir):
    with _serve(config) as client:
        _run(client, persona_file=_persona_file("P01"))
        scene = _run(client, persona=_raw("P02"))

    assert scene["scene_id"] == "P02"
    assert scene["status"] == "ready"
    spec = SceneSpec.model_validate_json((out_dir / "P02" / "scene_spec.json").read_text())
    assert spec.skybox.prompt.startswith("equirectangular 360 view, an open ")

    assert spec.sky_mode == "panorama"
    assert spec.skybox.uri == f"skybox_{derive_seed('P02')}.png"

    manifest = read_manifest(out_dir)
    assert list(manifest) == ["P01", "P02"]
    assert {e["status"] for e in manifest.values()} == {"ok"}
    assert {e["sky_mode"] for e in manifest.values()} == {"panorama"}


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


# -- sky_mode "procedural": no image at all ---------------------------------


@pytest.mark.parametrize("alias", ["P01", "P02"])
def test_a_procedural_scene_is_complete_without_an_image(config, out_dir, alias):
    pegasus = FakePegasus(connected=False)  # never needed
    with _serve(config, pegasus) as client:
        scene = _run(client, persona_file=_persona_file(alias), sky_mode="procedural")

    assert scene["status"] == "ready"
    assert scene["error"] is None
    assert scene["files"] == ["scene_spec.json"]
    assert pegasus.submitted == []

    on_disk = SceneSpec.model_validate_json((out_dir / alias / "scene_spec.json").read_text())
    assert on_disk == SceneSpec.model_validate(scene["spec"])
    assert on_disk.sky_mode == "procedural"
    assert on_disk.skybox.uri is None
    assert on_disk.skybox.prompt.startswith("equirectangular 360 view, ")  # still derived
    assert [f.name for f in (out_dir / alias).iterdir()] == ["scene_spec.json"]

    entry = read_manifest(out_dir)[alias]
    assert entry["status"] == "ok"
    assert entry["sky_mode"] == "procedural"
    assert entry["files"] == ["scene_spec.json"]
    assert "error" not in entry


def test_the_sky_mode_changes_nothing_else_in_the_spec(out_dir):
    specs = {}
    for mode in ("panorama", "procedural"):
        outcome = ScenePipeline(make_client(FakePegasus()), out_dir / mode).run(
            _raw("P01"), "P01", sky_mode=mode
        )
        specs[mode] = outcome.spec.model_dump()
    for spec in specs.values():
        del spec["sky_mode"], spec["skybox"]["uri"]
    assert specs["panorama"] == specs["procedural"]


def test_a_procedural_scene_never_reaches_the_image_stage(out_dir):
    stages = []
    ScenePipeline(make_client(FakePegasus()), out_dir).run(
        _raw("P01"), "P01", sky_mode="procedural", on_stage=stages.append
    )
    assert stages == ["building_spec"]


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


# -- no panorama: fall back to the procedural sky ---------------------------


@pytest.mark.parametrize("alias", ["P01", "P02"])
def test_a_failed_slurm_job_falls_back_to_the_procedural_sky(config, out_dir, alias):
    pegasus = FakePegasus(jobs=[("FAILED",)])
    with _serve(config, pegasus) as client:
        scene = _run(client, persona_file=_persona_file(alias))

    assert len(pegasus.submitted) == 2  # the existing retry came first
    assert scene["status"] == "ready"  # Unity can render it
    assert "CUDA out of memory" in scene["error"]

    entry = read_manifest(out_dir)[alias]
    assert entry["status"] == "fallback"
    assert entry["sky_mode"] == "procedural"
    assert "CUDA out of memory" in entry["error"]
    assert entry["files"] == ["scene_spec.json"]
    assert "slurm_job_id" not in entry  # that job is over

    spec = SceneSpec.model_validate_json((out_dir / alias / "scene_spec.json").read_text())
    assert spec == SceneSpec.model_validate(scene["spec"])
    assert spec.sky_mode == "procedural"  # flipped from the "panorama" asked for
    assert spec.skybox.uri is None
    assert spec.skybox.prompt is not None  # what was attempted


def test_the_fallback_sky_is_the_one_a_procedural_request_gets(out_dir):
    fallen_back = ScenePipeline(make_client(FakePegasus(jobs=[("FAILED",)])), out_dir / "a").run(
        _raw("P01"), "P01"
    )
    asked_for = ScenePipeline(make_client(FakePegasus()), out_dir / "b").run(
        _raw("P01"), "P01", sky_mode="procedural"
    )
    assert fallen_back.status == "fallback" and asked_for.status == "ok"
    assert fallen_back.usable and asked_for.usable
    assert fallen_back.spec == asked_for.spec


# -- every failure ends in the manifest -------------------------------------


def test_an_invalid_persona_is_failed_in_the_manifest(config, out_dir):
    with _serve(config) as client:
        scene = _run(client, persona_file=_persona_file("P03"))

    assert scene["scene_id"] == "P03"
    assert scene["status"] == "failed"
    assert scene["persona"] is None  # it could not be converted
    entry = read_manifest(out_dir)["P03"]
    assert entry["status"] == "failed"
    assert entry["sky_mode"] == "panorama"  # as asked for; there is no spec
    assert "Agreeableness" in entry["error"]
    assert entry["files"] == []
    assert not (out_dir / "P03").exists()


def test_no_connection_to_pegasus_is_a_fallback_in_the_manifest(config, out_dir):
    with _serve(config, FakePegasus(connected=False)) as client:
        _run(client, persona_file=_persona_file("P02"))
    entry = read_manifest(out_dir)["P02"]
    assert entry["status"] == "fallback"
    assert "no shared SSH connection" in entry["error"]


def test_a_persona_is_only_used_for_its_own_participant(out_dir):
    outcome = ScenePipeline(make_client(FakePegasus()), out_dir).run(_raw("P02"), "P01")
    assert outcome.status == "failed" and not outcome.usable
    assert "'P02'" in outcome.error and "'P01'" in outcome.error
    assert read_manifest(out_dir)["P01"]["status"] == "failed"
    assert not (out_dir / "P01").exists()


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
    assert "slurm_job_id" not in seen["generating_image"]  # nothing submitted yet
    assert read_manifest(out_dir)["P01"]["status"] == "ok"


def test_the_slurm_job_is_in_the_manifest_from_the_moment_it_is_submitted(out_dir):
    """What a later run needs to pick the job up instead of resubmitting."""
    seen = []

    def on_progress(message):
        seen.append((message, read_manifest(out_dir)["P01"]))

    ScenePipeline(make_client(FakePegasus()), out_dir).run(
        _raw("P01"), "P01", on_progress=on_progress
    )

    message, entry = seen[1]
    assert message == "Slurm job 123456: PENDING"
    assert entry["status"] == "failed" and entry["error"] == UNFINISHED
    assert entry["slurm_job_id"] == "123456"
    assert entry["seed"] == derive_seed("P01")
    assert "slurm_job_id" not in read_manifest(out_dir)["P01"]  # done: nothing pending


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
        {"persona_file": _persona_file("P01"), "sky_mode": "hdri"},
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


def test_a_fallback_scene_has_no_image_to_download(config):
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


def test_after_a_restart_a_fallback_scene_is_still_ready(config):
    with _serve(config, FakePegasus(jobs=[("FAILED",)])) as client:
        _run(client, persona_file=_persona_file("P01"))
    with _serve(config) as fresh:
        scene = fresh.get("/scenes/P01").json()
    assert scene["status"] == "ready"
    assert scene["spec"]["sky_mode"] == "procedural"
    assert "CUDA out of memory" in scene["error"]


# -- a scene Unity asks for before anyone created it ------------------------


def _auto(config, sky_mode="procedural"):
    return config.model_copy(
        update={"auto_create": AutoCreateConfig(personas_dir=FIXTURES, sky_mode=sky_mode)}
    )


def test_a_missing_scene_is_built_when_it_is_asked_for(config, out_dir):
    with _serve(_auto(config)) as client:
        scene = client.get("/scenes/P04").json()  # no POST before it

        assert scene["status"] == "ready"  # in that same answer: nothing to poll for
        assert scene["spec"]["scene_id"] == "P04"
        assert scene["spec"]["sky_mode"] == "procedural"
        assert scene["spec"]["seed"] == derive_seed("P04")
        assert scene["files"] == ["scene_spec.json"]
        assert client.get("/scenes/P04/files/scene_spec.json").status_code == 200
    assert read_manifest(out_dir)["P04"]["status"] == "ok"


def test_a_missing_panorama_scene_is_started_only_once(config):
    pegasus = FakePegasus()
    with _serve(_auto(config, "panorama"), pegasus) as client:
        client.get("/scenes/P04")
        client.get("/scenes/P04")  # Unity asking again
        assert client.app.state.scenes.wait_idle()
        scene = client.get("/scenes/P04").json()

    assert scene["status"] == "ready"
    assert scene["spec"]["skybox"]["uri"] == f"skybox_{derive_seed('P04')}.png"
    assert len(pegasus.submitted) == 1


def test_a_scene_that_exists_is_not_built_again(config):
    with _serve(_auto(config)) as client:
        _run(client, persona_file=_persona_file("P04"), seed=7)
    with _serve(_auto(config)) as fresh:  # nothing in memory, only the files on disk
        assert fresh.get("/scenes/P04").json()["spec"]["seed"] == 7


def test_a_participant_without_a_persona_file_is_still_404(config, out_dir):
    with _serve(_auto(config)) as client:
        assert client.get("/scenes/P99").status_code == 404
    assert "P99" not in read_manifest(out_dir)


def test_a_missing_scene_whose_persona_wont_convert_is_failed(config):
    with _serve(_auto(config)) as client:
        scene = client.get("/scenes/P03").json()
    assert scene["status"] == "failed"
    assert "Agreeableness" in scene["error"]


# -- a scene Unity asks for that can't be answered with what is on disk -----


def _fetch(client, scene_id) -> dict:
    """Ask the way Unity does: again, for as long as the scene is generating."""
    while True:
        scene = client.get(f"/scenes/{scene_id}").json()
        if scene["status"] in ("ready", "failed"):
            return scene
        assert client.app.state.scenes.wait_idle()


def _much_later(out_dir, scene_id):
    """Date the scene's manifest entry back, past the wait before a fallback's
    image is tried again."""
    manifest = read_manifest(out_dir)
    manifest[scene_id]["updated_at"] = "2026-01-01T00:00:00Z"
    manifest_path(out_dir).write_text(json.dumps(manifest))


@pytest.mark.parametrize("deleted", ["P04", "."])  # its folder, or all of out/
def test_a_scene_whose_files_are_gone_is_built_again(config, out_dir, deleted):
    with _serve(_auto(config)) as client:
        client.get("/scenes/P04")
        shutil.rmtree(out_dir / deleted)

        scene = _fetch(client, "P04")  # not answered from memory
        assert scene["status"] == "ready"
        assert client.get("/scenes/P04/files/scene_spec.json").status_code == 200
    assert read_manifest(out_dir)["P04"]["status"] == "ok"


def test_a_scene_missing_only_its_image_is_built_again(config, out_dir):
    pegasus = FakePegasus()
    with _serve(_auto(config, "panorama"), pegasus) as client:
        png = _fetch(client, "P04")["spec"]["skybox"]["uri"]
        (out_dir / "P04" / png).unlink()

        scene = _fetch(client, "P04")
        assert client.get(f"/scenes/P04/files/{png}").content == PNG
    assert scene["spec"]["skybox"]["uri"] == png
    assert len(pegasus.submitted) == 2


def test_a_deleted_fallback_scene_comes_back_with_its_image(config, out_dir):
    """Made without the connection, its folder deleted from out/ (the manifest
    still lists it), asked for again with the connection open."""
    pegasus = FakePegasus(connected=False)
    with _serve(_auto(config, "panorama"), pegasus) as client:
        assert _fetch(client, "P02")["spec"]["sky_mode"] == "procedural"
    shutil.rmtree(out_dir / "P02")

    pegasus.connected = True
    with _serve(_auto(config, "panorama"), pegasus) as fresh:
        scene = _fetch(fresh, "P02")
        for name in scene["files"]:
            assert fresh.get(f"/scenes/P02/files/{name}").status_code == 200

    seed = derive_seed("P02")
    assert scene["status"] == "ready" and scene["error"] is None
    assert scene["spec"]["sky_mode"] == "panorama"
    assert scene["files"] == ["scene_spec.json", f"skybox_{seed}.png", f"skybox_{seed}.json"]


def test_a_fallback_scene_gets_its_image_once_pegasus_can_be_reached(config, out_dir):
    pegasus = FakePegasus(connected=False)
    with _serve(_auto(config, "panorama"), pegasus) as client:
        assert _fetch(client, "P04")["spec"]["sky_mode"] == "procedural"
        _much_later(out_dir, "P04")
        assert _fetch(client, "P04")["spec"]["sky_mode"] == "procedural"  # still no connection
        assert pegasus.submitted == []

        pegasus.connected = True
        scene = _fetch(client, "P04")

    assert scene["status"] == "ready" and scene["error"] is None
    assert scene["spec"]["sky_mode"] == "panorama"
    assert scene["spec"]["skybox"]["uri"] == f"skybox_{derive_seed('P04')}.png"
    assert scene["spec"]["procedural_sky"] is not None  # both skies on the one spec
    entry = read_manifest(out_dir)["P04"]
    assert entry["status"] == "ok" and entry["sky_mode"] == "panorama"
    assert len(pegasus.submitted) == 1


def test_the_image_is_tried_again_after_a_restart_too(config, out_dir):
    with _serve(_auto(config, "panorama"), FakePegasus(connected=False)) as client:
        _fetch(client, "P04")
    _much_later(out_dir, "P04")
    with _serve(_auto(config, "panorama"), FakePegasus()) as fresh:
        assert _fetch(fresh, "P04")["spec"]["sky_mode"] == "panorama"


def test_a_failed_attempt_is_not_repeated_by_the_next_question(config, out_dir):
    """Unity asks again every few seconds: after an attempt that failed, it has
    to get the fallback, not set off another attempt."""
    pegasus = FakePegasus(jobs=[("FAILED",)])  # reachable, but no job succeeds
    with _serve(_auto(config, "panorama"), pegasus) as client:
        first = _fetch(client, "P04")
        again = _fetch(client, "P04")
        assert client.app.state.scenes.wait_idle()

        assert len(pegasus.submitted) == 2  # the one attempt, with its retry
        assert first["status"] == again["status"] == "ready"
        assert again["spec"]["sky_mode"] == "procedural"
        assert "CUDA out of memory" in again["error"]

        _much_later(out_dir, "P04")
        _fetch(client, "P04")
        assert len(pegasus.submitted) == 4  # asked for again later: one more attempt


def test_the_image_is_tried_again_for_the_same_seed(config, out_dir):
    pegasus = FakePegasus(connected=False)
    with _serve(_auto(config, "panorama"), pegasus) as client:
        _run(client, persona_file=_persona_file("P04"), seed=7)
        _much_later(out_dir, "P04")
        pegasus.connected = True
        scene = _fetch(client, "P04")
    assert scene["spec"]["seed"] == 7
    assert scene["spec"]["skybox"]["uri"] == "skybox_7.png"


def test_a_scene_asked_for_as_procedural_stays_procedural(config, out_dir):
    pegasus = FakePegasus()  # reachable all along
    with _serve(_auto(config, "panorama"), pegasus) as client:
        _run(client, persona_file=_persona_file("P04"), sky_mode="procedural")
        _much_later(out_dir, "P04")
        scene = _fetch(client, "P04")
    assert scene["spec"]["sky_mode"] == "procedural"
    assert pegasus.submitted == []


def test_a_scene_cut_short_by_a_restart_is_picked_up(config, out_dir):
    pegasus = FakePegasus(jobs=[("RUNNING", "COMPLETED")])
    answer, stopped = pegasus._sacct, []

    def sacct(job_id):
        if not stopped:
            stopped.append(job_id)
            raise KeyboardInterrupt  # the server is stopped while the job runs
        return answer(job_id)

    pegasus._sacct = sacct
    with pytest.raises(KeyboardInterrupt):
        ScenePipeline(make_client(pegasus), out_dir).run(_raw("P04"), "P04")
    entry = read_manifest(out_dir)["P04"]
    assert entry["error"] == UNFINISHED and entry["slurm_job_id"] == "123456"

    with _serve(_auto(config, "panorama"), pegasus) as restarted:
        scene = _fetch(restarted, "P04")

    assert scene["status"] == "ready"
    assert scene["spec"]["skybox"]["uri"] == f"skybox_{derive_seed('P04')}.png"
    assert len(pegasus.submitted) == 1  # the job from before, not a second one


def test_without_auto_create_nothing_is_built_again(config, out_dir):
    with _serve(config, FakePegasus(connected=False)) as client:
        _run(client, persona_file=_persona_file("P04"))
    _much_later(out_dir, "P04")
    pegasus = FakePegasus()
    with _serve(config, pegasus) as fresh:
        scene = fresh.get("/scenes/P04").json()
    assert scene["spec"]["sky_mode"] == "procedural"
    assert pegasus.submitted == []
