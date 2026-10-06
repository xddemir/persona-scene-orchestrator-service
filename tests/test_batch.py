"""The batch runner: a queue of participants through the same pipeline.

Only the Pegasus login node is simulated (tests/fakes.py). Resuming is tested
the way it happens: a first run that is cut short, then a second one over the
same out/ folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from fakes import PEGASUS, PNG, FakePegasus, make_client
from scene_orchestrator.batch import QueueEntry, QueueError, main, read_queue, run_batch
from scene_orchestrator.manifest import read_manifest
from scene_orchestrator.models import SceneSpec
from scene_orchestrator.pipeline import UNFINISHED, ScenePipeline, derive_seed

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "personas" / "chatbot"


def _queue(*participants: str, **fields) -> list[QueueEntry]:
    return [QueueEntry(participant_id=p, **fields) for p in participants]


def _batch(pegasus, out_dir, entries, **options) -> dict[str, str]:
    options.setdefault("personas_dir", FIXTURES)
    options.setdefault("log", lambda _: None)
    return run_batch(entries, ScenePipeline(make_client(pegasus), out_dir), out_dir, **options)


def _spec(out_dir: Path, scene_id: str) -> SceneSpec:
    return SceneSpec.model_validate_json((out_dir / scene_id / "scene_spec.json").read_text())


# -- the whole pipeline, both sky modes -------------------------------------


def test_a_panorama_batch_writes_every_scene_and_the_manifest(out_dir):
    pegasus = FakePegasus()
    results = _batch(pegasus, out_dir, _queue("P01", "P02"))

    assert results == {"P01": "ok", "P02": "ok"}
    assert len(pegasus.submitted) == 2
    manifest = read_manifest(out_dir)
    for alias in ("P01", "P02"):
        seed = derive_seed(alias)
        spec = _spec(out_dir, alias)
        assert spec.scene_id == alias and spec.seed == seed
        assert spec.sky_mode == "panorama"
        assert spec.skybox.uri == f"skybox_{seed}.png"
        assert (out_dir / alias / spec.skybox.uri).read_bytes() == PNG
        assert manifest[alias]["status"] == "ok"
        assert manifest[alias]["sky_mode"] == "panorama"
        assert manifest[alias]["files"] == [
            "scene_spec.json", f"skybox_{seed}.png", f"skybox_{seed}.json"
        ]


def test_a_procedural_batch_needs_no_pegasus(out_dir):
    pegasus = FakePegasus(connected=False)
    results = _batch(pegasus, out_dir, _queue("P01", "P02"), sky_mode="procedural")

    assert results == {"P01": "ok", "P02": "ok"}
    assert pegasus.submitted == []
    manifest = read_manifest(out_dir)
    for alias in ("P01", "P02"):
        spec = _spec(out_dir, alias)
        assert spec.sky_mode == "procedural"
        assert spec.skybox.uri is None
        assert manifest[alias] == {
            "status": "ok", "sky_mode": "procedural", "files": ["scene_spec.json"],
            "updated_at": manifest[alias]["updated_at"],
        }


def test_an_entry_can_set_its_own_sky_mode_seed_and_persona_file(out_dir):
    entries = [
        QueueEntry(participant_id="P01", sky_mode="procedural", seed=7),
        QueueEntry(participant_id="P02", persona_file=FIXTURES / "P02.json"),
    ]
    assert _batch(FakePegasus(), out_dir, entries, personas_dir=None) == {
        "P01": "failed",  # no personas folder, and no persona_file of its own
        "P02": "ok",
    }
    assert _batch(FakePegasus(), out_dir, entries) == {"P01": "ok", "P02": "skipped"}
    assert _spec(out_dir, "P01").seed == 7
    assert _spec(out_dir, "P01").sky_mode == "procedural"
    assert _spec(out_dir, "P02").sky_mode == "panorama"


# -- one bad scene doesn't stop the rest ------------------------------------


def test_a_failed_scene_is_recorded_and_the_batch_goes_on(out_dir):
    results = _batch(FakePegasus(), out_dir, _queue("P03", "P99", "P01"))

    assert results == {"P03": "failed", "P99": "failed", "P01": "ok"}
    manifest = read_manifest(out_dir)
    assert "Agreeableness" in manifest["P03"]["error"]  # the missing dimension
    assert "persona file not found" in manifest["P99"]["error"]
    assert manifest["P99"]["files"] == []


def test_a_panorama_that_cannot_be_made_is_a_fallback(out_dir):
    results = _batch(FakePegasus(jobs=[("FAILED",)]), out_dir, _queue("P01", "P02"))

    assert results == {"P01": "fallback", "P02": "fallback"}
    for alias in ("P01", "P02"):
        assert _spec(out_dir, alias).sky_mode == "procedural"
        assert read_manifest(out_dir)[alias]["status"] == "fallback"


# -- resuming ---------------------------------------------------------------


def test_scenes_already_ok_are_skipped(out_dir):
    pegasus = FakePegasus()
    _batch(pegasus, out_dir, _queue("P01"))
    before = read_manifest(out_dir)["P01"]

    log = []
    results = _batch(pegasus, out_dir, _queue("P01", "P02"), log=log.append)

    assert results == {"P01": "skipped", "P02": "ok"}
    assert len(pegasus.submitted) == 2  # one each, never a second for P01
    assert read_manifest(out_dir)["P01"] == before
    assert "P01: already ok, skipped" in log


def test_ok_in_the_other_sky_mode_is_not_what_was_asked_for(out_dir):
    pegasus = FakePegasus()
    _batch(pegasus, out_dir, _queue("P01"), sky_mode="procedural")
    assert _batch(pegasus, out_dir, _queue("P01")) == {"P01": "ok"}
    assert _spec(out_dir, "P01").sky_mode == "panorama"
    assert len(pegasus.submitted) == 1


def test_an_entry_from_before_sky_modes_counts_as_a_panorama(out_dir):
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(
        json.dumps({"P01": {"status": "ok", "files": ["scene_spec.json"], "updated_at": "x"}})
    )
    assert _batch(FakePegasus(), out_dir, _queue("P01")) == {"P01": "skipped"}


def test_failed_and_fallback_scenes_are_redone(out_dir):
    assert _batch(FakePegasus(jobs=[("FAILED",)]), out_dir, _queue("P01")) == {"P01": "fallback"}
    assert _batch(FakePegasus(), out_dir, _queue("P01")) == {"P01": "ok"}
    assert _spec(out_dir, "P01").sky_mode == "panorama"
    assert "error" not in read_manifest(out_dir)["P01"]


def test_a_runner_killed_mid_job_picks_the_job_up_instead_of_resubmitting(out_dir):
    pegasus = FakePegasus(jobs=[("RUNNING", "RUNNING", "COMPLETED")])
    answer, killed = pegasus._sacct, []

    def sacct(job_id):
        if not killed:
            killed.append(job_id)
            raise KeyboardInterrupt  # Ctrl-C while waiting for the job
        return answer(job_id)

    pegasus._sacct = sacct
    with pytest.raises(KeyboardInterrupt):
        _batch(pegasus, out_dir, _queue("P01", "P02"))

    # What the dead runner left behind: the job, by id.
    entry = read_manifest(out_dir)["P01"]
    assert entry["status"] == "failed" and entry["error"] == UNFINISHED
    assert entry["slurm_job_id"] == "123456"
    assert entry["seed"] == derive_seed("P01")
    assert "P02" not in read_manifest(out_dir)

    log = []
    results = _batch(pegasus, out_dir, _queue("P01", "P02"), log=log.append)

    assert results == {"P01": "ok", "P02": "ok"}
    assert len(pegasus.submitted) == 2  # P01's one job, then P02's
    assert "P01: Slurm job 123456: picked up from an earlier run" in log
    sidecar = json.loads((out_dir / "P01" / f"skybox_{derive_seed('P01')}.json").read_text())
    assert sidecar["slurm_job_id"] == "123456"
    assert "slurm_job_id" not in read_manifest(out_dir)["P01"]
    assert _spec(out_dir, "P01").sky_mode == "panorama"


def test_a_job_outliving_a_lost_connection_is_picked_up_next_time(out_dir):
    pegasus = FakePegasus(jobs=[("RUNNING", "COMPLETED")], lose_connection=True)
    assert _batch(pegasus, out_dir, _queue("P01")) == {"P01": "fallback"}

    entry = read_manifest(out_dir)["P01"]
    assert entry["status"] == "fallback"
    assert entry["slurm_job_id"] == "123456"  # it keeps running on Pegasus
    assert _spec(out_dir, "P01").sky_mode == "procedural"  # usable meanwhile

    # Still no connection: the job must not be forgotten.
    pegasus.lose_connection = False
    assert _batch(pegasus, out_dir, _queue("P01")) == {"P01": "fallback"}
    assert read_manifest(out_dir)["P01"]["slurm_job_id"] == "123456"

    pegasus.connected = True
    assert _batch(pegasus, out_dir, _queue("P01")) == {"P01": "ok"}
    assert len(pegasus.submitted) == 1
    assert _spec(out_dir, "P01").sky_mode == "panorama"
    assert _spec(out_dir, "P01").skybox.uri == f"skybox_{derive_seed('P01')}.png"


def test_a_job_for_another_seed_is_not_picked_up(out_dir):
    pegasus = FakePegasus(lose_connection=True)
    _batch(pegasus, out_dir, _queue("P01"))
    pegasus.lose_connection, pegasus.connected = False, True

    assert _batch(pegasus, out_dir, _queue("P01", seed=7)) == {"P01": "ok"}
    assert len(pegasus.submitted) == 2  # the old job would be another image
    assert _spec(out_dir, "P01").skybox.uri == "skybox_7.png"


# -- the queue file ---------------------------------------------------------


def _write_queue(tmp_path: Path, content) -> Path:
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_the_queue_is_a_json_list_of_participants(tmp_path):
    path = _write_queue(tmp_path, [
        {"participant_id": "P01"},
        {"participant_id": "P02", "sky_mode": "procedural", "seed": 3, "persona_file": "x.json"},
    ])
    first, second = read_queue(path)
    assert (first.participant_id, first.sky_mode, first.seed) == ("P01", None, None)
    assert (second.sky_mode, second.seed, second.persona_file) == ("procedural", 3, Path("x.json"))


@pytest.mark.parametrize(
    "content",
    [
        {"participant_id": "P01"},  # not a list
        [{"participant_id": "../P01"}],  # no safe folder name
        [{"participant_id": "P01", "condition": "A"}],  # no conditions
        [{"participant_id": "P01", "sky_mode": "hdri"}],
        [{"participant_id": "P01"}, {"participant_id": "P01"}],  # one scene each
    ],
)
def test_a_queue_that_cannot_be_run_is_rejected_whole(tmp_path, content):
    with pytest.raises(QueueError):
        read_queue(_write_queue(tmp_path, content))


def test_a_missing_queue_file_is_a_queue_error(tmp_path):
    with pytest.raises(QueueError, match="cannot read the queue"):
        read_queue(tmp_path / "nope.json")


# -- the command line -------------------------------------------------------


@pytest.fixture
def config_file(tmp_path: Path, out_dir: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"out_dir": str(out_dir), "image_gen": PEGASUS}))
    return path


def test_the_cli_runs_a_procedural_queue(tmp_path, out_dir, config_file, capsys):
    queue = _write_queue(tmp_path, [{"participant_id": "P01"}, {"participant_id": "P03"}])
    args = [str(queue), "--personas", str(FIXTURES), "--sky-mode", "procedural",
            "--config", str(config_file)]

    assert main(args) == 1  # P03's persona is incomplete
    assert read_manifest(out_dir)["P01"]["status"] == "ok"
    assert read_manifest(out_dir)["P03"]["status"] == "failed"
    assert capsys.readouterr().out.splitlines()[-1] == "1 failed, 1 ok"

    assert main(args) == 1  # again: P01 is done, P03 is retried and fails again
    assert "P01: already ok, skipped" in capsys.readouterr().out


def test_the_cli_does_not_guess_where_personas_are(tmp_path, out_dir, config_file, capsys):
    queue = _write_queue(tmp_path, [{"participant_id": "P01"}])
    assert main([str(queue), "--sky-mode", "procedural", "--config", str(config_file)]) == 2
    assert "--personas" in capsys.readouterr().err
    assert not out_dir.exists()


def test_the_cli_runs_nothing_when_panoramas_are_wanted_but_pegasus_is_unreachable(
    tmp_path, out_dir, config_file, capsys, monkeypatch
):
    from scene_orchestrator.clients.image_gen import PegasusShell

    monkeypatch.setattr(PegasusShell, "check", lambda self: "no shared SSH connection")
    queue = _write_queue(tmp_path, [{"participant_id": "P01"}])

    assert main([str(queue), "--personas", str(FIXTURES), "--config", str(config_file)]) == 2
    assert "no shared SSH connection" in capsys.readouterr().err
    assert not out_dir.exists()  # nobody written off as a fallback
