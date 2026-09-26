from __future__ import annotations

import nltk
from nltk.tokenize import sent_tokenize

_punkt_ready = False


def ensure_punkt() -> None:
    global _punkt_ready
    if _punkt_ready:
        return
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            name = resource.split("/")[-1]
            nltk.download(name, quiet=True)
    _punkt_ready = True


def split_phrases(text: str, max_phrases: int) -> list[str]:
    ensure_punkt()
    sents = sent_tokenize(text.strip() or "")
    if not sents:
        return [""]
    return sents[:max_phrases]
