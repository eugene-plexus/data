"""Unit tests for the engine's algorithm modules (no HTTP / auth).

Covers the importer's JSONL round-trip, byte-level BPE training, and the
pretokenizer's block math + Arrow shard shape — the load-bearing pieces
distilled from the CLLM data stack.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from eugene_plexus_data.engine import importer
from eugene_plexus_data.engine.pretokenizer import pretokenize
from eugene_plexus_data.engine.tokenizer_trainer import fingerprint_file, train_bpe

_DOCS = [
    "the quick brown fox jumps over the lazy dog " * 20,
    "a journey of a thousand miles begins with a single step " * 20,
    "to be or not to be that is the question " * 20,
]


def test_importer_jsonl_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "doc.txt"
    src.write_text("  hello world  \n", encoding="utf-8")
    raw = tmp_path / "raw"

    assert importer.import_local_path(str(src), data_column=None, dest_dir=raw) == 1
    assert importer.has_documents(raw)
    assert list(importer.iter_documents(raw)) == ["hello world"]


def test_train_bpe_produces_tokenizer_and_fingerprint(tmp_path: Path) -> None:
    out = tmp_path / "tokenizer.json"
    fingerprint, vocab_size = train_bpe(iter(_DOCS), vocab_size=300, min_frequency=1, out_path=out)
    assert out.is_file()
    assert fingerprint == fingerprint_file(out)  # fingerprint is SHA256 of the saved file
    assert vocab_size >= 256  # byte-level alphabet floor


def test_pretokenize_block_math_and_arrow_shape(tmp_path: Path) -> None:
    tok = tmp_path / "tokenizer.json"
    train_bpe(iter(_DOCS), vocab_size=300, min_frequency=1, out_path=tok)

    out = tmp_path / "arrow"
    meta = pretokenize(
        iter(_DOCS), tokenizer_path=tok, block_size=8, out_dir=out, blocks_per_shard=10
    )

    assert meta["blockCount"] >= 1
    assert meta["tokenCount"] == meta["blockCount"] * 8
    shards = sorted(out.glob("shard-*.arrow"))
    assert len(shards) == meta["shardCount"] >= 1

    with pa.memory_map(str(shards[0]), "r") as mm:
        table = pa.ipc.open_file(mm).read_all()
    assert table.num_rows >= 1
    first_block = table.column("input_ids")[0].as_py()
    assert len(first_block) == 8  # each row is exactly one block_size window


def test_pretokenize_discards_trailing_partial_block(tmp_path: Path) -> None:
    tok = tmp_path / "tokenizer.json"
    train_bpe(iter(_DOCS), vocab_size=300, min_frequency=1, out_path=tok)
    out = tmp_path / "arrow"
    # A block size larger than the whole corpus -> zero complete blocks.
    meta = pretokenize(["short doc"], tokenizer_path=tok, block_size=100_000, out_dir=out)
    assert meta["blockCount"] == 0
    assert meta["shardCount"] == 0
    assert meta["tokenCount"] == 0
