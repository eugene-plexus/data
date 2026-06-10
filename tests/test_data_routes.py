"""Tests for the data domain routes (v0.3 skeleton).

Mutating + per-resource endpoints return 501 (engine not implemented);
the list endpoints (datasets, tokenizers) return real empty responses.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_list_datasets_returns_empty(client: TestClient) -> None:
    response = client.get("/v1/data/datasets")
    assert response.status_code == 200
    assert response.json() == {"datasets": []}


def test_list_tokenizers_returns_empty(client: TestClient) -> None:
    response = client.get("/v1/data/tokenizers")
    assert response.status_code == 200
    assert response.json() == {"tokenizers": []}


def test_create_dataset_returns_501(client: TestClient) -> None:
    response = client.post(
        "/v1/data/datasets",
        json={"datasetId": str(uuid4()), "name": "my-corpus"},
    )
    assert response.status_code == 501
    body = response.json()
    assert body["component"] == "data"
    assert "not implemented" in body["detail"].lower()


def test_get_dataset_returns_501(client: TestClient) -> None:
    response = client.get(f"/v1/data/datasets/{uuid4()}")
    assert response.status_code == 501


def test_delete_dataset_returns_501(client: TestClient) -> None:
    response = client.delete(f"/v1/data/datasets/{uuid4()}")
    assert response.status_code == 501


def test_import_dataset_returns_501(client: TestClient) -> None:
    response = client.post(
        f"/v1/data/datasets/{uuid4()}/import",
        json={"sourceKind": "local_path", "location": "/data/raw/corpus.txt"},
    )
    assert response.status_code == 501


def test_pretokenize_dataset_returns_501(client: TestClient) -> None:
    response = client.post(
        f"/v1/data/datasets/{uuid4()}/pretokenize",
        json={"tokenizerId": str(uuid4()), "blockSize": 1024},
    )
    assert response.status_code == 501


def test_train_tokenizer_returns_501(client: TestClient) -> None:
    response = client.post(
        "/v1/data/tokenizers/train",
        json={"name": "bpe-32k", "vocabSize": 32000, "sourceDatasetIds": [str(uuid4())]},
    )
    assert response.status_code == 501
    body = response.json()
    assert body["component"] == "data"
    assert "not implemented" in body["detail"].lower()
