"""HTTP surface: the server starts and answers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scene_orchestrator import __version__
from scene_orchestrator.api.app import create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as client:
        yield client


def test_health_reports_ok_and_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
