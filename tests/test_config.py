"""Tests for the config protocol endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_config_schema_lists_data_fields(client: TestClient) -> None:
    response = client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "data"
    keys = {f["key"] for f in body["fields"]}
    # `port` is not here — owned by the watchdog topology via
    # EUGENE_PLEXUS_DAT_BIND_PORT.
    assert keys == {
        "dataRoot",
        "inboxDir",
        "archiveDir",
        "defaultBlockSize",
        "logLevel",
    }


def test_get_config_returns_defaults(client: TestClient) -> None:
    response = client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert "port" not in body
    assert body["logLevel"] == "INFO"
    assert body["defaultBlockSize"] == 1024


def test_patch_config_validates_per_field(client: TestClient) -> None:
    response = client.patch(
        "/v1/config",
        json={
            "defaultBlockSize": 2048,  # valid integer
            "logLevel": "DEBUG",  # valid enum, requiresRestart
            "blockSize_typo": 512,  # unknown field
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body["applied"]) == {"defaultBlockSize", "logLevel"}
    rejected = {r["key"] for r in body["rejected"]}
    assert rejected == {"blockSize_typo"}
    # logLevel is requiresRestart
    assert body["requiresRestart"] is True
    assert "logLevel" in body["pendingRestart"]


def test_patch_config_rejects_bad_enum(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"logLevel": "VERBOSE"})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "logLevel"


def test_patch_config_rejects_below_minimum(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"defaultBlockSize": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "defaultBlockSize"


def test_patch_config_rejects_unknown_field(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"madeUpKey": "anything"})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "madeUpKey"
    assert "unknown field" in body["rejected"][0]["message"]


def test_config_test_succeeds_when_dirs_writable(client: TestClient) -> None:
    response = client.post("/v1/config/test", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "data"
    assert body["ok"] is True
    assert "latencyMs" in body
