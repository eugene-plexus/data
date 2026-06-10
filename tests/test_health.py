"""Tests for /healthz.

With the data-preparation engine wired in, a successful build makes
`app.state.engine` non-None and health reports `ok`. Safe mode and engine
build failures still surface as `degraded` (covered in test_safe_mode.py).
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_healthz_is_reachable_and_well_formed(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "data"
    assert "version" in body


def test_healthz_reports_ok_when_engine_built(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    # The engine builds against the tmp_path-seeded config, so health is ok.
    assert body["status"] == "ok"
    assert body["safeMode"] is False
