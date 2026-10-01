"""HTTP surface: health, the skybox job lifecycle, and request validation."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from fakes import NOT_CONNECTED, FakePegasus, make_client
from scene_orchestrator import __version__
from scene_orchestrator.api.app import create_app
from scene_orchestrator.clients.image_gen import SlurmImageGenClient


@contextmanager
def _serve(config, pegasus):
    with TestClient(create_app(config, image_gen=make_client(pegasus))) as client:
        yield client


@pytest.fixture
def client(config):
    with _serve(config, FakePegasus()) as client:
        yield client


def _post(client, **body):
    payload = {"prompt": "a calm forest", "seed": 1234, "scene_id": "p07"}
    payload.update(body)
    return client.post("/skyboxes", json=payload)


def _await_job(client, job_id):
    assert client.app.state.runner.wait_idle()
    return client.get(f"/skyboxes/{job_id}").json()


# -- health -----------------------------------------------------------------


def test_health_reports_version_and_the_connection(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["image_gen"] == {
        "login": "demir@login1.pegasus.kl.dfki.de",
        "connected": True,
        "error": None,
    }


def test_health_says_how_to_open_a_missing_connection(config):
    with _serve(config, FakePegasus(connected=False)) as client:  # starts anyway
        image_gen = client.get("/health").json()["image_gen"]
    assert image_gen["connected"] is False
    assert image_gen["error"] == NOT_CONNECTED


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


# -- skybox jobs ------------------------------------------------------------


def test_skybox_job_is_queued_then_ready_with_files(client, out_dir):
    response = _post(client)
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert job["files"] is None

    done = _await_job(client, job["job_id"])
    assert done["status"] == "ready"
    assert done["attempts"] == 1
    assert done["detail"] == "Slurm job 123456: COMPLETED"
    assert done["files"] == {
        "png": "p07/skybox_1234.png",
        "sidecar": "p07/skybox_1234.json",
    }
    assert (out_dir / "p07" / "skybox_1234.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_failed_generation_surfaces_on_the_job(config):
    with _serve(config, FakePegasus(jobs=[("FAILED",)])) as client:
        done = _await_job(client, _post(client).json()["job_id"])

    assert done["status"] == "failed"
    assert done["attempts"] == 2
    assert "CUDA out of memory" in done["error"]
    assert done["files"] is None


def test_a_job_without_the_connection_fails_and_says_so(config):
    with _serve(config, FakePegasus(connected=False)) as client:
        done = _await_job(client, _post(client).json()["job_id"])

    assert done["status"] == "failed"
    assert done["attempts"] == 1
    assert NOT_CONNECTED in done["error"]


def test_a_client_bug_fails_the_job_not_the_worker(client, monkeypatch):
    def crash(self, request, out_dir, on_progress=None):
        raise RuntimeError("bug")

    monkeypatch.setattr(SlurmImageGenClient, "generate_skybox", crash)
    done = _await_job(client, _post(client).json()["job_id"])
    assert done["status"] == "failed"
    assert done["error"] == "RuntimeError: bug"

    monkeypatch.undo()
    assert _await_job(client, _post(client).json()["job_id"])["status"] == "ready"


@pytest.mark.parametrize(
    "body",
    [
        {"scene_id": "../escape"},
        {"scene_id": ""},
        {"seed": -1},
        {"prompt": ""},
        {"promt": "typo"},  # unknown field: rejected, not silently ignored
    ],
)
def test_invalid_requests_are_rejected_before_queueing(client, body):
    assert _post(client, **body).status_code == 422


def test_unknown_job_is_404(client):
    assert client.get("/skyboxes/nope").status_code == 404
