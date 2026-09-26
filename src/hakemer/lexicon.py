from __future__ import annotations

import re
from collections import Counter
from itertools import chain
from pathlib import Path

from nrclex.core import _load_bundled_lexicon

# Plutchik 8-D (ch. 3), not NRCLex's extended EMOTION_ORDER.
NRC_EMOTIONS = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
)
NRC_DIM = len(NRC_EMOTIONS)
SENTICNET_DIM = 5

ROOT = Path(__file__).resolve().parents[2]
SENTICNET_TSV = ROOT / "reference" / "lexicons" / "senticnet5_polarity.tsv"

_NRC_LEXICON: dict[str, list[str]] | None = None
_SENTIC_POLARITY: dict[str, float] | None = None


def _nrc_lexicon() -> dict[str, list[str]]:
    global _NRC_LEXICON
    if _NRC_LEXICON is None:
        _NRC_LEXICON = _load_bundled_lexicon()
    return _NRC_LEXICON


def _load_sentic_polarity() -> dict[str, float]:
    global _SENTIC_POLARITY
    if _SENTIC_POLARITY is not None:
        return _SENTIC_POLARITY
    if not SENTICNET_TSV.is_file():
        raise FileNotFoundError(
            f"Missing SenticNet table at {SENTICNET_TSV}. "
            "See reference/lexicons/README.md."
        )
    table: dict[str, float] = {}
    with SENTICNET_TSV.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("CONCEPT"):
                continue
            parts = re.split(r"\s+", line, maxsplit=2)
            if len(parts) != 3:
                continue
            concept, _label, intensity = parts[0].lower(), parts[1], parts[2]
            try:
                table[concept] = float(intensity)
            except ValueError:
                continue
    _SENTIC_POLARITY = table
    return table


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", (text or "").lower())


def _lookup_sentic_concept(token: str, table: dict[str, float]) -> float | None:
    for key in (token, token.replace("-", "_"), token.replace("'", "")):
        if key in table:
            return table[key]
    return None


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
    return [float(counts.get(emotion, 0)) / float(total) for emotion in NRC_EMOTIONS]


def senticnet_document_vector(text: str) -> list[float]:
    """5-D SenticNet descriptors (protocol ch. 3 / ch. 4)."""
    words = word_tokens(text)
    table = _load_sentic_polarity()
    polarities: list[float] = []
    covered = 0
    concepts: set[str] = set()
    for word in words:
        value = _lookup_sentic_concept(word, table)
        if value is None:
            continue
        covered += 1
        polarities.append(value)
        concepts.add(word)
    total = len(words)
    if not polarities or total == 0:
        return [0.0] * SENTICNET_DIM
    return [
        float(sum(polarities) / len(polarities)),
        float(max(polarities)),
        float(min(polarities)),
        float(covered) / float(total),
        float(len(concepts)),
    ]


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
