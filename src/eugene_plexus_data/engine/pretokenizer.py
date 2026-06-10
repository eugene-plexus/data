"""Pretokenization into fixed-size token blocks.

Each document is encoded, documents are separated by the EOS token and
concatenated into a running buffer, and the buffer is sliced into fixed
``block_size`` windows saved as memory-mapped Arrow IPC shards. A trailing
partial block (< block_size) is discarded — the standard fixed-context
pretraining convention.

``pyarrow`` and ``tokenizers`` are imported at call time so the component can
boot and serve config + health even if a dependency is missing; a
pretokenize request then fails with a clear error instead of the whole
service refusing to start.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .tokenizer_trainer import EOS_TOKEN

# Blocks held in memory before a shard is flushed to disk. Keeps peak RAM
# bounded for large corpora while keeping small datasets to a single shard.
_DEFAULT_BLOCKS_PER_SHARD = 50_000


def pretokenize(
    documents: Iterable[str],
    *,
    tokenizer_path: Path,
    block_size: int,
    out_dir: Path,
    blocks_per_shard: int = _DEFAULT_BLOCKS_PER_SHARD,
) -> dict[str, int]:
    """Tokenize `documents` into fixed blocks under `out_dir`.

    Returns ``{"blockSize", "shardCount", "blockCount", "tokenCount"}`` and
    writes ``shard-NNNNN.arrow`` files plus a ``pretokenize.json`` sidecar.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    import pyarrow as pa
    from tokenizers import Tokenizer

    # One Arrow column: each row is a block of `block_size` int32 token ids.
    arrow_schema = pa.schema([("input_ids", pa.list_(pa.int32()))])

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        eos_id = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("shard-*.arrow"):
        stale.unlink()

    buffer: list[int] = []
    pending: list[list[int]] = []
    shard_count = 0
    block_count = 0

    def flush() -> None:
        nonlocal shard_count, pending
        if not pending:
            return
        table = pa.table({"input_ids": pending}, schema=arrow_schema)
        shard_path = out_dir / f"shard-{shard_count:05d}.arrow"
        with (
            pa.OSFile(str(shard_path), "wb") as sink,
            pa.ipc.new_file(sink, arrow_schema) as writer,
        ):
            writer.write_table(table)
        shard_count += 1
        pending = []

    for doc in documents:
        if not doc:
            continue
        buffer.extend(tokenizer.encode(doc, add_special_tokens=False).ids)
        buffer.append(eos_id)
        while len(buffer) >= block_size:
            pending.append(buffer[:block_size])
            del buffer[:block_size]
            block_count += 1
            if len(pending) >= blocks_per_shard:
                flush()
    flush()

    meta = {
        "blockSize": block_size,
        "shardCount": shard_count,
        "blockCount": block_count,
        "tokenCount": block_count * block_size,
    }
    (out_dir / "pretokenize.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
