"""Raw-data import and the shared document iterator.

Imports text from a local path or the HuggingFace Hub and stores it as plain
UTF-8 JSONL shards (one ``{"text": ...}`` record per line); no text
preprocessing or annotation is applied. JSONL-shard storage (rather than one
file per document) scales to a HuggingFace split's millions of rows without
exhausting inodes, while keeping clean per-document boundaries for the
EOS-separated pretokenizer.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

# Plain-text file extensions imported whole (one document per file).
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}

# Documents per raw JSONL shard before rolling to the next file.
_RAW_DOCS_PER_SHARD = 50_000


def iter_documents(raw_dir: Path) -> Iterator[str]:
    """Yield every imported document from a dataset's raw JSONL shards."""
    for shard in sorted(raw_dir.glob("raw-*.jsonl")):
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") if isinstance(obj, dict) else None
                if isinstance(text, str) and text:
                    yield text


def has_documents(raw_dir: Path) -> bool:
    """True if any raw JSONL shard exists (cheap precondition check)."""
    return any(raw_dir.glob("raw-*.jsonl"))


def _write_jsonl_shards(documents: Iterator[str], dest_dir: Path) -> int:
    """Write `documents` to rolling JSONL shards; return the document count."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for stale in dest_dir.glob("raw-*.jsonl"):
        stale.unlink()

    count = 0
    shard_idx = 0
    in_shard = 0
    handle = None
    try:
        for doc in documents:
            doc = doc.strip()
            if not doc:
                continue
            if handle is None or in_shard >= _RAW_DOCS_PER_SHARD:
                if handle is not None:
                    handle.close()
                handle = (dest_dir / f"raw-{shard_idx:05d}.jsonl").open("w", encoding="utf-8")
                shard_idx += 1
                in_shard = 0
            handle.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
            in_shard += 1
            count += 1
    finally:
        if handle is not None:
            handle.close()
    return count


def _parquet_documents(path: Path, column: str) -> Iterator[str]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[column])  # raises if the column is absent
    for val in table.column(column).to_pylist():
        if isinstance(val, str) and val.strip():
            yield val


def _local_documents(location: str, data_column: str | None) -> Iterator[str]:
    src = Path(location)
    if not src.exists():
        raise FileNotFoundError(f"local import source not found: {location}")
    column = data_column or "text"
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    for path in files:
        suffix = path.suffix.lower()
        if suffix in _TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if text.strip():
                yield text
        elif suffix == ".jsonl":
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for raw_line in handle:  # stream — never load the whole file into RAM
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    val = obj.get(column) if isinstance(obj, dict) else None
                    if isinstance(val, str) and val.strip():
                        yield val
        elif suffix == ".parquet":
            yield from _parquet_documents(path, column)
        # Other file types are ignored rather than failing the whole import.


def import_local_path(location: str, *, data_column: str | None, dest_dir: Path) -> int:
    """Import a local file or directory tree into `dest_dir` as JSONL shards."""
    count = _write_jsonl_shards(_local_documents(location, data_column), dest_dir)
    if count == 0:
        raise ValueError(f"no importable text found at {location}")
    return count


def import_huggingface(
    location: str,
    *,
    data_column: str | None,
    dest_dir: Path,
    split: str = "train",
) -> int:
    """Import a HuggingFace Hub dataset split into `dest_dir` as JSONL shards."""
    from datasets import load_dataset  # heavy import — deferred to call time

    column = data_column or "text"
    dataset = load_dataset(location, split=split)
    if column not in dataset.column_names:
        raise ValueError(
            f"column {column!r} not found in HuggingFace dataset {location!r}; "
            f"available columns: {dataset.column_names}"
        )

    def docs() -> Iterator[str]:
        for val in dataset[column]:
            if isinstance(val, str) and val.strip():
                yield val

    count = _write_jsonl_shards(docs(), dest_dir)
    if count == 0:
        raise ValueError(f"no non-empty text in column {column!r} of {location!r}")
    return count
