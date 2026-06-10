"""Data domain routes: datasets + tokenizers (v0.3 skeleton).

v0.3 SKELETON. The real data-preparation engine (import / clean / split /
tokenizer-train / pretokenize) is not implemented yet, so the mutating
and per-resource read endpoints return `501 Not Implemented` with a
standard `Problem` body. The two list endpoints are real:

  * `GET /v1/data/datasets`   returns an empty list (no datasets exist yet).
  * `GET /v1/data/tokenizers` returns an empty list (no tokenizers yet).

When the engine lands it replaces the 501s; the wire shapes here are the
long-term contract from specs/openapi/data.yaml.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response, status

from .._generated.common_models import Problem
from .._generated.models import (
    DatasetManifest,
    ImportRequest,
    V1DataDatasetsDatasetIdPretokenizePostRequest,
    V1DataDatasetsGetResponse,
    V1DataTokenizersGetResponse,
    V1DataTokenizersTrainPostRequest,
)

router = APIRouter(tags=["data"])

_ENGINE_NOT_IMPLEMENTED = (
    "data preparation engine not implemented in the v0.3 skeleton; "
    "this repo ships the control-plane wire shape only"
)


def _not_implemented(operation: str) -> Response:
    problem = Problem(
        type="https://github.com/eugene-plexus/data#engine-not-implemented",
        title="Data engine not implemented",
        status=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"{operation}: {_ENGINE_NOT_IMPLEMENTED}.",
        component="data",
    )
    return Response(
        content=problem.model_dump_json(exclude_none=True),
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        media_type="application/problem+json",
    )


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


@router.get("/v1/data/datasets", response_model=V1DataDatasetsGetResponse)
async def list_datasets(request: Request) -> V1DataDatasetsGetResponse:
    """List dataset manifests. The skeleton has no engine and therefore no
    datasets — returns an empty list rather than 501 so callers polling for
    dataset inventory get a valid empty result."""
    return V1DataDatasetsGetResponse(datasets=[])


@router.post("/v1/data/datasets", status_code=status.HTTP_201_CREATED)
async def create_dataset(request: Request, body: DatasetManifest) -> Response:
    return _not_implemented("createDataset")


@router.get("/v1/data/datasets/{dataset_id}")
async def get_dataset(request: Request, dataset_id: UUID) -> Response:
    return _not_implemented("getDataset")


@router.delete("/v1/data/datasets/{dataset_id}", status_code=204)
async def delete_dataset(request: Request, dataset_id: UUID) -> Response:
    return _not_implemented("deleteDataset")


@router.post("/v1/data/datasets/{dataset_id}/import", status_code=202)
async def import_dataset(request: Request, dataset_id: UUID, body: ImportRequest) -> Response:
    return _not_implemented("importDataset")


@router.post("/v1/data/datasets/{dataset_id}/pretokenize", status_code=202)
async def pretokenize_dataset(
    request: Request,
    dataset_id: UUID,
    body: V1DataDatasetsDatasetIdPretokenizePostRequest,
) -> Response:
    return _not_implemented("pretokenizeDataset")


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #


@router.get("/v1/data/tokenizers", response_model=V1DataTokenizersGetResponse)
async def list_tokenizers(request: Request) -> V1DataTokenizersGetResponse:
    """List tokenizers. Empty in the skeleton (no engine to have trained
    any), returned as a valid empty list rather than 501."""
    return V1DataTokenizersGetResponse(tokenizers=[])


@router.post("/v1/data/tokenizers/train", status_code=202)
async def train_tokenizer(
    request: Request,
    body: V1DataTokenizersTrainPostRequest,
) -> Response:
    return _not_implemented("trainTokenizer")
