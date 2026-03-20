from __future__ import annotations

from fastapi.testclient import TestClient

from finance_tracker.api.main import create_app


def test_health_endpoint():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_version_endpoint():
    client = TestClient(create_app())
    r = client.get("/version")
    assert r.status_code == 200
    assert "version" in r.json()
