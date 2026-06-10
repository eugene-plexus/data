"""Byte-level BPE tokenizer training.

GPT-2-style byte-level BPE: robust across languages and code, with no
out-of-vocabulary tokens. Only the standard ``pad`` / ``unk`` / ``bos`` /
``eos`` special tokens are added, plus any extras the caller supplies.

The ``tokenizers`` import is deferred to call time so the component can boot
(and serve config + health) even if the dependency is missing — a training
request then fails with a clear error rather than the whole service refusing
to start.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
DEFAULT_SPECIAL_TOKENS: list[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def fingerprint_file(path: Path) -> str:
    """SHA256 of a file's bytes — the tokenizer's vocab fingerprint.

    Consumers (trainer, eval) must pretokenize/evaluate with a tokenizer whose
    fingerprint matches the dataset's, or token ids won't line up.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def train_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int,
    min_frequency: int,
    out_path: Path,
    extra_special_tokens: Iterable[str] | None = None,
) -> tuple[str, int]:
    """Train a byte-level BPE tokenizer over an iterable of documents.

    Returns ``(vocab_fingerprint, actual_vocab_size)`` and writes
    ``out_path`` (a ``tokenizer.json``).
    """
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import TemplateProcessing
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN, byte_fallback=True))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = list(DEFAULT_SPECIAL_TOKENS)
    for tok in extra_special_tokens or []:
        if tok and tok not in special_tokens:
            special_tokens.append(tok)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add BOS/EOS framing for single-sequence encodes. Pretokenization passes
    # add_special_tokens=False and inserts EOS between documents itself, so
    # this only affects callers that encode a standalone string.
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is not None and eos_id is not None:
        tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS_TOKEN} $0 {EOS_TOKEN}",
            special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return fingerprint_file(out_path), tokenizer.get_vocab_size()
