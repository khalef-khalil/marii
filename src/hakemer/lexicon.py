from __future__ import annotations

import re
from collections import Counter
from itertools import chain

from nrclex.core import EMOTION_ORDER, _load_bundled_lexicon

NRC_DIM = len(EMOTION_ORDER)
SENTICNET_DIM = 5

_LEXICON = None


def _nrc_lexicon() -> dict[str, list[str]]:
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = _load_bundled_lexicon()
    return _LEXICON


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def nrc_document_vector(text: str) -> list[float]:
    """8-D NRC affect frequencies (NRCLex-compatible, surface tokens only)."""
    words = word_tokens(text)
    lexicon = _nrc_lexicon()
    matched = [w for w in words if w in lexicon]
    affect_list = list(chain.from_iterable(lexicon[w] for w in matched))
    counts = Counter(affect_list)
    total = sum(counts.values())
    if total == 0:
        return [0.0] * NRC_DIM
    return [float(counts.get(emotion, 0)) / float(total) for emotion in EMOTION_ORDER]


def senticnet_document_vector(text: str) -> list[float]:
    raise NotImplementedError(
        "SenticNet M3 is not bundled yet; run the NRC campaign first or add SenticNet lookup."
    )


def document_lexicon_vector(text: str, source: str) -> list[float]:
    if source == "nrc":
        return nrc_document_vector(text)
    if source == "senticnet":
        return senticnet_document_vector(text)
    raise ValueError(f"Unknown lexicon source: {source}")


def lexicon_dim(source: str) -> int:
    if source == "nrc":
        return NRC_DIM
    if source == "senticnet":
        return SENTICNET_DIM
    raise ValueError(source)
