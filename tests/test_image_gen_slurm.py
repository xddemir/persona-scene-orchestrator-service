"""SlurmImageGenClient against a simulated Pegasus.

Covers the job it submits, the happy path, the retry policy, and every
failure ending in a failed result rather than an exception. No cluster, and
no real waiting.
"""

from __future__ import annotations

import json
import shlex

from fakes import NOT_CONNECTED, PNG, FakePegasus, make_client
from scene_orchestrator.clients.image_gen import SkyboxRequest

REQUEST = SkyboxRequest(prompt="a calm forest", seed=1234, scene_id="p07")


def _generate_line(script: str) -> list[str]:
    (line,) = [ln for ln in script.splitlines() if ln.startswith("image-gen gen")]
    return shlex.split(line)


# -- the job ----------------------------------------------------------------


def test_the_job_asks_for_a_gpu_and_runs_image_gen_on_it(tmp_path):
    pegasus = FakePegasus()
    make_client(pegasus).generate_skybox(REQUEST, tmp_path)
    (script,) = pegasus.submitted

    assert script.startswith("#!/bin/bash\n")
    for line in (
        "#SBATCH --partition=RTXA6000",
        "#SBATCH --account=ei-external",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --time=00:30:00",
        "#SBATCH --output=/netscratch/demir/image-gen/out/.slurm/%j.out",
        "source /netscratch/demir/venvs/image-gen/bin/activate",
        "cd /netscratch/demir/image-gen-service",
    ):
        assert line in script.splitlines()
    assert "HF_HUB_OFFLINE=1" in script  # compute nodes have no internet
    assert _generate_line(script) == [
        "image-gen", "gen", "--backend", "local_diffusers", "--kind", "skybox",
        "--prompt", "a calm forest", "--seed", "1234", "--scene-id", "p07",
        "--out", "/netscratch/demir/image-gen/out",
    ]


def test_a_prompt_cannot_break_out_of_the_job_script(tmp_path):
    prompt = "it's a $HOME `test`; rm -rf /"
    pegasus = FakePegasus()
    request = SkyboxRequest(prompt=prompt, seed=1, scene_id="p07")
    make_client(pegasus).generate_skybox(request, tmp_path)

    args = _generate_line(pegasus.submitted[0])
    assert args[args.index("--prompt") + 1] == prompt


# -- happy path -------------------------------------------------------------


def test_fetches_png_and_sidecar_under_the_briefs_names(tmp_path):
    result = make_client(FakePegasus()).generate_skybox(REQUEST, tmp_path)

    assert result.ok
    assert result.attempts == 1
    assert result.png == tmp_path / "p07" / "skybox_1234.png"
    assert result.png.read_bytes() == PNG

    sidecar = json.loads(result.sidecar.read_text(encoding="utf-8"))
    assert result.sidecar.name == "skybox_1234.json"
    assert sidecar["file"] == "skybox_1234.png"
    assert sidecar["upstream_file"] == "/netscratch/demir/image-gen/out/p07/p07_skybox_1234.png"
    assert sidecar["slurm_job_id"] == "123456"
    assert sidecar["steps"] == 30  # the rest of image-gen's metadata is kept


def test_progress_follows_the_slurm_job(tmp_path):
    seen = []
    make_client(FakePegasus()).generate_skybox(REQUEST, tmp_path, seen.append)
    assert seen == [
        "Slurm job 123456: submitted",
        "Slurm job 123456: PENDING",
        "Slurm job 123456: RUNNING",
        "Slurm job 123456: COMPLETED",
    ]


def test_a_job_not_yet_in_accounting_counts_as_pending(tmp_path):
    pegasus = FakePegasus(jobs=[("", "RUNNING", "COMPLETED")])
    assert make_client(pegasus).generate_skybox(REQUEST, tmp_path).ok


# -- retry policy -----------------------------------------------------------


def test_a_failed_job_is_retried_once(tmp_path):
    pegasus = FakePegasus(jobs=[("RUNNING", "FAILED"), ("RUNNING", "COMPLETED")])
    result = make_client(pegasus).generate_skybox(REQUEST, tmp_path)

    assert result.ok
    assert result.attempts == 2
    assert len(pegasus.submitted) == 2


def test_failing_twice_is_a_failed_result_with_the_jobs_log(tmp_path):
    result = make_client(FakePegasus(jobs=[("FAILED",)])).generate_skybox(REQUEST, tmp_path)

    assert not result.ok
    assert result.attempts == 2
    assert "attempt 1" in result.error and "attempt 2" in result.error
    assert "CUDA out of memory" in result.error
    assert "/netscratch/demir/image-gen/out/.slurm/123456.out" in result.error
    assert not (tmp_path / "p07").exists()


def test_retries_are_configurable(tmp_path):
    result = make_client(FakePegasus(jobs=[("FAILED",)]), retries=0).generate_skybox(
        REQUEST, tmp_path
    )
    assert not result.ok
    assert result.attempts == 1


def test_a_cancelled_job_counts_as_failed(tmp_path):
    pegasus = FakePegasus(jobs=[("CANCELLED by 4242",)])
    result = make_client(pegasus, retries=0).generate_skybox(REQUEST, tmp_path)
    assert not result.ok
    assert "CANCELLED" in result.error


def test_a_job_stuck_in_the_queue_times_out_and_is_cancelled(tmp_path):
    pegasus = FakePegasus(jobs=[("PENDING",)])
    result = make_client(pegasus).generate_skybox(REQUEST, tmp_path)

    assert not result.ok
    assert result.attempts == 2
    assert "still PENDING after 60s" in result.error
    assert pegasus.cancelled == ["123456", "123457"]


# -- failing fast, with a reason --------------------------------------------


def test_no_shared_connection_fails_at_once_without_submitting(tmp_path):
    pegasus = FakePegasus(connected=False)
    result = make_client(pegasus).generate_skybox(REQUEST, tmp_path)

    assert not result.ok
    assert result.attempts == 1
    assert NOT_CONNECTED in result.error
    assert pegasus.submitted == []


def test_a_job_slurm_refuses_is_not_retried(tmp_path):
    pegasus = FakePegasus(sbatch_error="sbatch: error: Invalid account or account/partition")
    result = make_client(pegasus).generate_skybox(REQUEST, tmp_path)

    assert not result.ok
    assert result.attempts == 1
    assert "Invalid account" in result.error


def test_losing_the_connection_mid_job_says_the_job_keeps_running(tmp_path):
    pegasus = FakePegasus(lose_connection=True)
    result = make_client(pegasus).generate_skybox(REQUEST, tmp_path)

    assert not result.ok
    assert result.attempts == 1
    assert "keeps running on Pegasus" in result.error


# -- health -----------------------------------------------------------------


def test_status_reports_the_shared_connection():
    assert make_client(FakePegasus()).status().connected
    status = make_client(FakePegasus(connected=False)).status()
    assert not status.connected
    assert status.error == NOT_CONNECTED
