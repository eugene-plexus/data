"""Data domain routes: datasets + tokenizers.

Drives the `DataEngine` (app.state.engine). When the engine is unavailable —
safe mode, or a build failure that left the component degraded — every
endpoint returns `503` with a standard `Problem` body and config stays
reachable so an operator can repair it. Engine-level errors map to HTTP
codes: not-found -> 404, precondition/validation -> 400, deferred feature
(e.g. url/scrape import) -> 501.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse

from .._generated.common_models import Problem
from .._generated.models import (
    DatasetManifest,
    ImportRequest,
    V1DataDatasetsDatasetIdPretokenizePostRequest,
    V1DataTokenizersTrainPostRequest,
)
from ..engine.engine import (
    BadRequestError,
    DataEngine,
    EngineError,
    NotFoundError,
    UnsupportedError,
)

router = APIRouter(tags=["data"])


def _problem(status_code: int, title: str, detail: str) -> JSONResponse:
    slug = title.replace(" ", "-").lower()
    body = Problem(
        type=f"https://github.com/eugene-plexus/data#{slug}",
        title=title,
        status=status_code,
        detail=detail,
        component="data",
    )
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        content=body.model_dump(exclude_none=True),
    )


def _engine_or_problem(request: Request) -> DataEngine | JSONResponse:
    engine: DataEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        detail = getattr(request.app.state, "engine_error", None) or "data engine unavailable"
        return _problem(status.HTTP_503_SERVICE_UNAVAILABLE, "Data engine unavailable", detail)
    return engine


def _map_engine_error(e: EngineError) -> JSONResponse:
    if isinstance(e, NotFoundError):
        return _problem(status.HTTP_404_NOT_FOUND, "Not found", str(e))
    if isinstance(e, UnsupportedError):
        return _problem(status.HTTP_501_NOT_IMPLEMENTED, "Not implemented", str(e))
    if isinstance(e, BadRequestError):
        return _problem(status.HTTP_400_BAD_REQUEST, "Bad request", str(e))
    return _problem(status.HTTP_500_INTERNAL_SERVER_ERROR, "Engine error", str(e))


def _ok(model: object, status_code: int = status.HTTP_200_OK) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=model.model_dump(mode="json"))  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
@router.get("/v1/data/datasets")
async def list_datasets(request: Request) -> JSONResponse:
    engine = _engine_or_problem(request)
    if isinstance(engine, JSONResponse):
        return engine
    return JSONResponse(
        content={"datasets": [m.model_dump(mode="json") for m in engine.list_datasets()]}
    )


@router.post("/v1/data/datasets")
async def create_dataset(request: Request, body: DatasetManifest) -> JSONResponse:
    engine = _engine_or_problem(request)
    if isinstance(engine, JSONResponse):
        return engine
    try:
        return _ok(engine.create_dataset(body), status.HTTP_201_CREATED)
    except EngineError as e:
        return _map_engine_error(e)


@router.get("/v1/data/datasets/{dataset_id}")
async def get_dataset(request: Request, dataset_id: UUID) -> JSONResponse:
    engine = _engine_or_problem(request)
    if isinstance(engine, JSONResponse):
        return engine
    manifest = engine.get_dataset(str(dataset_id))
    if manifest is None:
        return _problem(status.HTTP_404_NOT_FOUND, "Not found", f"dataset {dataset_id} not found")
    return _ok(manifest)


@router.delete("/v1/data/datasets/{dataset_id}")
async def delete_dataset(request: Request, dataset_id: UUID) -> Response:
    engine = _engine_or_problem(request)
    if isinstance(engine, JSONResponse):
        return engine
    if not engine.delete_dataset(str(dataset_id)):
        return _problem(status.HTTP_404_NOT_FOUND, "Not found", f"dataset {dataset_id} not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/v1/data/datasets/{dataset_id}/import")
async def import_dataset(request: Request, dataset_id: UUID, body: ImportRequest) -> JSONResponse:
    engine = _engine_or_problem(request)
    if isinstance(engine, JSONResponse):
        return engine
    try:
        manifest = engine.start_import(
            str(dataset_id),
            source_kind=body.sourceKind.value,
            location=body.location,
            data_column=body.dataColumn,
        )
        return _ok(manifest, status.HTTP_202_ACCEPTED)
    except EngineError as e:
        return _map_engine_error(e)


@router.post("/v1/data/datasets/{dataset_id}/pretokenize")
async def pretokenize_dataset(
    request: Request,
    dataset_id: UUID,
    body: V1DataDatasetsDatasetIdPretokenizePostRequest,
) -> JSONResponse:
    engine = _engine_or_problem(request)
    if isinstance(engine, JSONResponse):
        return engine
    try:
        manifest = engine.start_pretokenize(
            str(dataset_id),
            tokenizer_id=str(body.tokenizerId),
            block_size=body.blockSize,
        )
        return _ok(manifest, status.HTTP_202_ACCEPTED)
    except EngineError as e:
        return _map_engine_error(e)


# --------------------------------------------------------------------------- #
# Tokenizers
# --------------------------------------------------------------------------- #
@router.get("/v1/data/tokenizers")
async def list_tokenizers(request: Request) -> JSONResponse:
    engine = _engine_or_problem(request)
    if isinstance(engine, JSONResponse):
        return engine
    return JSONResponse(
        content={"tokenizers": [t.model_dump(mode="json") for t in engine.list_tokenizers()]}
    )


@router.post("/v1/data/tokenizers/train")
async def train_tokenizer(
    request: Request,
    body: V1DataTokenizersTrainPostRequest,
) -> JSONResponse:
    engine = _engine_or_problem(request)
    if isinstance(engine, JSONResponse):
        return engine
    try:
        spec = engine.start_train_tokenizer(
            name=body.name,
            vocab_size=body.vocabSize,
            min_frequency=body.minFrequency,
            source_dataset_ids=[str(s) for s in (body.sourceDatasetIds or [])],
        )
        return _ok(spec, status.HTTP_202_ACCEPTED)
    except EngineError as e:
        return _map_engine_error(e)
