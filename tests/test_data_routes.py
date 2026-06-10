"""Tests for the data domain routes against the real preparation engine.

The engine is driven in `inline` mode so import / train / pretokenize jobs
run synchronously and the 202 response already reflects the terminal state —
no polling races. The full happy path (create -> import -> train tokenizer ->
pretokenize) plus error mappings are exercised end to end.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

_CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a journey of a thousand miles begins with a single step. "
    "to be or not to be that is the question. "
    "all that glitters is not gold and every cloud has a silver lining. "
) * 12


@pytest.fixture
def inline_client(client: TestClient) -> TestClient:
    """A client whose engine runs jobs synchronously for deterministic asserts."""
    client.app.state.engine.inline = True
    return client


@pytest.fixture
def corpus_file(tmp_path: Path) -> str:
    path = tmp_path / "corpus.txt"
    path.write_text(_CORPUS, encoding="utf-8")
    return str(path)


def _create_dataset(client: TestClient, name: str = "my-corpus") -> str:
    dataset_id = str(uuid4())
    resp = client.post("/v1/data/datasets", json={"datasetId": dataset_id, "name": name})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["datasetId"] == dataset_id
    assert body["status"] == "empty"
    return dataset_id


def test_list_datasets_starts_empty(client: TestClient) -> None:
    resp = client.get("/v1/data/datasets")
    assert resp.status_code == 200
    assert resp.json() == {"datasets": []}


def test_list_tokenizers_starts_empty(client: TestClient) -> None:
    resp = client.get("/v1/data/tokenizers")
    assert resp.status_code == 200
    assert resp.json() == {"tokenizers": []}


def test_full_pipeline(inline_client: TestClient, corpus_file: str) -> None:
    client = inline_client

    # 1. create
    dataset_id = _create_dataset(client)

    # 2. import a local text file
    resp = client.post(
        f"/v1/data/datasets/{dataset_id}/import",
        json={"sourceKind": "local_path", "location": corpus_file},
    )
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "ready"  # inline -> terminal state already

    # 3. train a tokenizer over the dataset
    resp = client.post(
        "/v1/data/tokenizers/train",
        json={"name": "bpe-tiny", "vocabSize": 300, "sourceDatasetIds": [dataset_id]},
    )
    assert resp.status_code == 202, resp.text
    tok = resp.json()
    assert tok["status"] == "ready"
    assert tok["vocabFingerprint"]
    tokenizer_id = tok["tokenizerId"]

    # 4. pretokenize into blocks
    resp = client.post(
        f"/v1/data/datasets/{dataset_id}/pretokenize",
        json={"tokenizerId": tokenizer_id, "blockSize": 8},
    )
    assert resp.status_code == 202, resp.text
    manifest = resp.json()
    assert manifest["status"] == "ready"
    assert manifest["shardCount"] >= 1
    assert manifest["tokenCount"] > 0
    assert manifest["blockSize"] == 8
    # the dataset's fingerprint must match the tokenizer it was built with
    assert manifest["vocabFingerprint"] == tok["vocabFingerprint"]

    # 5. it shows up in the listings
    listed = client.get("/v1/data/datasets").json()["datasets"]
    assert any(d["datasetId"] == dataset_id and d["status"] == "ready" for d in listed)
    assert client.get("/v1/data/tokenizers").json()["tokenizers"][0]["tokenizerId"] == tokenizer_id


def test_delete_dataset(inline_client: TestClient) -> None:
    dataset_id = _create_dataset(inline_client, name="to-delete")
    assert inline_client.delete(f"/v1/data/datasets/{dataset_id}").status_code == 204
    assert inline_client.get(f"/v1/data/datasets/{dataset_id}").status_code == 404


def test_delete_unknown_dataset_404(client: TestClient) -> None:
    assert client.delete(f"/v1/data/datasets/{uuid4()}").status_code == 404


def test_pretokenize_unknown_dataset_404(client: TestClient) -> None:
    resp = client.post(
        f"/v1/data/datasets/{uuid4()}/pretokenize",
        json={"tokenizerId": str(uuid4()), "blockSize": 8},
    )
    assert resp.status_code == 404


def test_train_tokenizer_without_sources_400(client: TestClient) -> None:
    resp = client.post("/v1/data/tokenizers/train", json={"name": "x", "vocabSize": 300})
    assert resp.status_code == 400
    assert resp.json()["component"] == "data"


def test_url_import_is_not_implemented(inline_client: TestClient) -> None:
    dataset_id = _create_dataset(inline_client, name="from-url")
    resp = inline_client.post(
        f"/v1/data/datasets/{dataset_id}/import",
        json={"sourceKind": "url", "location": "https://example.com/corpus.txt"},
    )
    assert resp.status_code == 501
    assert "url" in resp.json()["detail"].lower()
