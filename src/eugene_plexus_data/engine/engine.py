"""DataEngine — orchestrates import, tokenizer training, and pretokenization.

Owns the on-disk layout under the configured data root, an in-memory mirror of
dataset manifests + tokenizer specs (rebuilt from disk on startup), and a small
thread-pool job runner. Long operations (import / train / pretokenize) run off
the request thread and report progress through the resource's ``status`` field,
which the UI/peers poll. `tokenizers` releases the GIL during training and the
Arrow writes are I/O-bound, so threads — not subprocesses — are the right
isolation here; the subprocess-per-run model is reserved for `trainer` GPU runs.
"""

from __future__ import annotations

import itertools
import json
import logging
import shutil
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from uuid import UUID, uuid4

from .._generated.models import DatasetManifest, SourceKind, Status, Status1, TokenizerSpec
from . import importer
from .pretokenizer import pretokenize
from .tokenizer_trainer import train_bpe

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Engine-level errors (routes map these to HTTP status codes)
# --------------------------------------------------------------------------- #
class EngineError(Exception):
    """Base for engine errors with an HTTP-friendly mapping."""


class NotFoundError(EngineError):
    """A referenced dataset or tokenizer does not exist (-> 404)."""


class BadRequestError(EngineError):
    """The request is malformed or violates a precondition (-> 400)."""


class UnsupportedError(EngineError):
    """A valid-but-not-yet-implemented operation (-> 501)."""


# --------------------------------------------------------------------------- #
# Internal records (persisted as JSON; projected to the generated wire models)
# --------------------------------------------------------------------------- #
def _from_dict(cls: type, data: dict) -> object:
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class DatasetRecord:
    dataset_id: str
    name: str
    source_kind: str | None = None
    location: str | None = None
    data_column: str | None = None
    block_size: int | None = None
    tokenizer_id: str | None = None
    vocab_fingerprint: str | None = None
    shard_count: int | None = None
    token_count: int | None = None
    block_count: int | None = None
    status: str = "empty"
    error: str | None = None

    def to_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            datasetId=UUID(self.dataset_id),
            name=self.name,
            sourceKind=SourceKind(self.source_kind) if self.source_kind else None,
            blockSize=self.block_size,
            vocabFingerprint=self.vocab_fingerprint,
            shardCount=self.shard_count,
            tokenCount=self.token_count,
            status=Status(self.status),
        )


@dataclass
class TokenizerRecord:
    tokenizer_id: str
    name: str
    vocab_size: int | None = None
    vocab_fingerprint: str | None = None
    min_frequency: int | None = None
    source_dataset_ids: list[str] = field(default_factory=list)
    status: str = "training"
    error: str | None = None

    def to_spec(self) -> TokenizerSpec:
        return TokenizerSpec(
            tokenizerId=UUID(self.tokenizer_id),
            name=self.name,
            vocabSize=self.vocab_size,
            vocabFingerprint=self.vocab_fingerprint,
            status=Status1(self.status),
        )


class DataEngine:
    def __init__(
        self,
        *,
        data_root: Path,
        inbox_dir: Path,
        archive_dir: Path,
        default_block_size: int,
    ) -> None:
        self.data_root = Path(data_root)
        self.inbox_dir = Path(inbox_dir)
        self.archive_dir = Path(archive_dir)
        self.default_block_size = int(default_block_size)
        self.datasets_dir = self.data_root / "datasets"
        self.tokenizers_dir = self.data_root / "tokenizers"
        for d in (self.datasets_dir, self.tokenizers_dir, self.inbox_dir, self.archive_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._datasets: dict[str, DatasetRecord] = {}
        self._tokenizers: dict[str, TokenizerRecord] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="data-job")
        # When True, jobs run synchronously on the calling thread. Used by
        # tests for deterministic assertions without polling.
        self.inline = False

        self._load_existing()

    # ----------------------------------------------------------------- #
    # Path helpers
    # ----------------------------------------------------------------- #
    def _dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / dataset_id

    def _raw_dir(self, dataset_id: str) -> Path:
        return self._dataset_dir(dataset_id) / "raw"

    def _arrow_dir(self, dataset_id: str) -> Path:
        return self._dataset_dir(dataset_id) / "arrow"

    def _tokenizer_dir(self, tokenizer_id: str) -> Path:
        return self.tokenizers_dir / tokenizer_id

    def _tokenizer_path(self, tokenizer_id: str) -> Path:
        return self._tokenizer_dir(tokenizer_id) / "tokenizer.json"

    # ----------------------------------------------------------------- #
    # Persistence
    # ----------------------------------------------------------------- #
    def _load_existing(self) -> None:
        for manifest_path in self.datasets_dir.glob("*/manifest.json"):
            try:
                rec = _from_dict(DatasetRecord, json.loads(manifest_path.read_text("utf-8")))
            except (OSError, ValueError, TypeError):
                log.warning("skipping unreadable dataset manifest %s", manifest_path)
                continue
            assert isinstance(rec, DatasetRecord)
            # A job that was running when the process died can't resume; mark it.
            if rec.status in ("importing", "pretokenizing"):
                rec.status = "error"
                rec.error = "job interrupted by a restart; re-run it"
            self._datasets[rec.dataset_id] = rec

        for spec_path in self.tokenizers_dir.glob("*/spec.json"):
            try:
                rec_t = _from_dict(TokenizerRecord, json.loads(spec_path.read_text("utf-8")))
            except (OSError, ValueError, TypeError):
                log.warning("skipping unreadable tokenizer spec %s", spec_path)
                continue
            assert isinstance(rec_t, TokenizerRecord)
            if rec_t.status == "training":
                rec_t.status = "error"
                rec_t.error = "training interrupted by a restart; re-run it"
            self._tokenizers[rec_t.tokenizer_id] = rec_t

    def _save_dataset(self, rec: DatasetRecord) -> None:
        path = self._dataset_dir(rec.dataset_id) / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(rec), indent=2), encoding="utf-8")

    def _save_tokenizer(self, rec: TokenizerRecord) -> None:
        path = self._tokenizer_dir(rec.tokenizer_id) / "spec.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(rec), indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- #
    # Job submission
    # ----------------------------------------------------------------- #
    def _submit(self, fn: Callable[..., None], *args: object) -> None:
        if self.inline:
            fn(*args)
        else:
            self._executor.submit(fn, *args)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ----------------------------------------------------------------- #
    # Datasets
    # ----------------------------------------------------------------- #
    def list_datasets(self) -> list[DatasetManifest]:
        with self._lock:
            return [r.to_manifest() for r in self._datasets.values()]

    def get_dataset(self, dataset_id: str) -> DatasetManifest | None:
        with self._lock:
            rec = self._datasets.get(dataset_id)
            return rec.to_manifest() if rec else None

    def create_dataset(self, manifest: DatasetManifest) -> DatasetManifest:
        dataset_id = str(manifest.datasetId)
        with self._lock:
            if dataset_id in self._datasets:
                raise BadRequestError(f"dataset {dataset_id} already exists")
            rec = DatasetRecord(
                dataset_id=dataset_id,
                name=manifest.name,
                source_kind=manifest.sourceKind.value if manifest.sourceKind else None,
                status="empty",
            )
            self._datasets[dataset_id] = rec
            self._save_dataset(rec)
            return rec.to_manifest()

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._lock:
            rec = self._datasets.pop(dataset_id, None)
            if rec is None:
                return False
            src = self._dataset_dir(dataset_id)
            if src.exists():
                dest = self.archive_dir / f"dataset-{dataset_id}"
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                shutil.move(str(src), str(dest))
            return True

    def start_import(
        self, dataset_id: str, *, source_kind: str, location: str, data_column: str | None
    ) -> DatasetManifest:
        with self._lock:
            rec = self._datasets.get(dataset_id)
            if rec is None:
                raise NotFoundError(f"dataset {dataset_id} not found")
            if source_kind == "url":
                raise UnsupportedError("url / scrape import is deferred to a later release")
            if source_kind not in ("local_path", "huggingface"):
                raise BadRequestError(f"unsupported sourceKind: {source_kind}")
            rec.source_kind = source_kind
            rec.location = location
            rec.data_column = data_column
            rec.status = "importing"
            rec.error = None
            self._save_dataset(rec)
        self._submit(self._do_import, dataset_id)
        return self.get_dataset(dataset_id)  # type: ignore[return-value]

    def _do_import(self, dataset_id: str) -> None:
        with self._lock:
            rec = self._datasets.get(dataset_id)
        if rec is None:
            return  # deleted before the job ran
        raw_dir = self._raw_dir(dataset_id)
        try:
            if rec.source_kind == "huggingface":
                importer.import_huggingface(
                    rec.location or "", data_column=rec.data_column, dest_dir=raw_dir
                )
            else:
                importer.import_local_path(
                    rec.location or "", data_column=rec.data_column, dest_dir=raw_dir
                )
            status, error = "ready", None
        except Exception as e:  # surface any failure as the dataset's error status
            log.exception("import failed for dataset %s", dataset_id)
            status, error = "error", str(e)
        with self._lock:
            if dataset_id not in self._datasets:
                return  # deleted mid-job; don't resurrect its manifest
            rec.status = status
            rec.error = error
            self._save_dataset(rec)

    def start_pretokenize(
        self, dataset_id: str, *, tokenizer_id: str, block_size: int
    ) -> DatasetManifest:
        with self._lock:
            rec = self._datasets.get(dataset_id)
            if rec is None:
                raise NotFoundError(f"dataset {dataset_id} not found")
            if not importer.has_documents(self._raw_dir(dataset_id)):
                raise BadRequestError("dataset has no imported data; import before pretokenizing")
            tok = self._tokenizers.get(tokenizer_id)
            if tok is None:
                raise NotFoundError(f"tokenizer {tokenizer_id} not found")
            if tok.status != "ready":
                raise BadRequestError(
                    f"tokenizer {tokenizer_id} is not ready (status={tok.status})"
                )
            rec.tokenizer_id = tokenizer_id
            rec.block_size = block_size
            rec.status = "pretokenizing"
            rec.error = None
            self._save_dataset(rec)
        self._submit(self._do_pretokenize, dataset_id, tokenizer_id, block_size)
        return self.get_dataset(dataset_id)  # type: ignore[return-value]

    def _do_pretokenize(self, dataset_id: str, tokenizer_id: str, block_size: int) -> None:
        with self._lock:
            rec = self._datasets.get(dataset_id)
            tok = self._tokenizers.get(tokenizer_id)
        if rec is None or tok is None:
            return  # dataset or tokenizer deleted before the job ran
        fingerprint = tok.vocab_fingerprint
        try:
            meta = pretokenize(
                importer.iter_documents(self._raw_dir(dataset_id)),
                tokenizer_path=self._tokenizer_path(tokenizer_id),
                block_size=block_size,
                out_dir=self._arrow_dir(dataset_id),
            )
            with self._lock:
                if dataset_id not in self._datasets:
                    return  # deleted mid-job; don't resurrect its manifest
                rec.shard_count = meta["shardCount"]
                rec.block_count = meta["blockCount"]
                rec.token_count = meta["tokenCount"]
                rec.vocab_fingerprint = fingerprint
                rec.status = "ready"
                rec.error = None
                self._save_dataset(rec)
        except Exception as e:  # surface any failure as the dataset's error status
            log.exception("pretokenize failed for dataset %s", dataset_id)
            with self._lock:
                if dataset_id not in self._datasets:
                    return
                rec.status = "error"
                rec.error = str(e)
                self._save_dataset(rec)

    # ----------------------------------------------------------------- #
    # Tokenizers
    # ----------------------------------------------------------------- #
    def list_tokenizers(self) -> list[TokenizerSpec]:
        with self._lock:
            return [r.to_spec() for r in self._tokenizers.values()]

    def start_train_tokenizer(
        self,
        *,
        name: str,
        vocab_size: int,
        min_frequency: int | None,
        source_dataset_ids: list[str] | None,
    ) -> TokenizerSpec:
        source_ids = [str(s) for s in (source_dataset_ids or [])]
        if not source_ids:
            raise BadRequestError("trainTokenizer requires at least one sourceDatasetId")
        with self._lock:
            for ds_id in source_ids:
                if ds_id not in self._datasets:
                    raise NotFoundError(f"source dataset {ds_id} not found")
                if not importer.has_documents(self._raw_dir(ds_id)):
                    raise BadRequestError(f"source dataset {ds_id} has no imported data")
            tokenizer_id = str(uuid4())
            rec = TokenizerRecord(
                tokenizer_id=tokenizer_id,
                name=name,
                vocab_size=vocab_size,
                min_frequency=min_frequency or 2,
                source_dataset_ids=source_ids,
                status="training",
            )
            self._tokenizers[tokenizer_id] = rec
            self._save_tokenizer(rec)
        self._submit(self._do_train, tokenizer_id)
        return rec.to_spec()

    def _do_train(self, tokenizer_id: str) -> None:
        with self._lock:
            rec = self._tokenizers.get(tokenizer_id)
        if rec is None:
            return  # deleted before the job ran
        source_ids = list(rec.source_dataset_ids)

        def documents() -> Iterator[str]:
            return itertools.chain.from_iterable(
                importer.iter_documents(self._raw_dir(ds_id)) for ds_id in source_ids
            )

        try:
            fingerprint, actual_vocab = train_bpe(
                documents(),
                vocab_size=rec.vocab_size or 0,
                min_frequency=rec.min_frequency or 2,
                out_path=self._tokenizer_path(tokenizer_id),
            )
            with self._lock:
                if tokenizer_id not in self._tokenizers:
                    return  # deleted mid-job; don't resurrect its spec
                rec.vocab_fingerprint = fingerprint
                rec.vocab_size = actual_vocab
                rec.status = "ready"
                rec.error = None
                self._save_tokenizer(rec)
        except Exception as e:  # surface any failure as the tokenizer's error status
            log.exception("tokenizer training failed for %s", tokenizer_id)
            with self._lock:
                if tokenizer_id not in self._tokenizers:
                    return
                rec.status = "error"
                rec.error = str(e)
                self._save_tokenizer(rec)
