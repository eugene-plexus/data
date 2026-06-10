"""Data-preparation engine for the Eugene Plexus `data` component.

A generic, torch-free pipeline: import raw text, train a byte-level BPE
tokenizer, and pretokenize into fixed-size token blocks written as
memory-mapped Arrow shards. Imported text is stored as plain UTF-8 JSONL
shards; no special-token injection or text-annotation layers are applied.
"""

from __future__ import annotations

from .engine import DataEngine

__all__ = ["DataEngine"]
