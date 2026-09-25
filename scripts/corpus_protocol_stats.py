#!/usr/bin/env python3
"""GoEmotions simplified: truncation (M=4, 32 tokens) and NRC lexicon coverage."""

from __future__ import annotations

import json
import re
from pathlib import Path

import nltk
from datasets import load_dataset
from nrclex.core import _load_bundled_lexicon
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference" / "artifacts" / "corpus_protocol_stats.json"

M_MAX = 4
TOKENS_PER_PHRASE = 32
SPLITS = ("train", "validation", "test")
TOKENIZER = "roberta-base"


def ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize

    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    return sents if sents else [text.strip()]


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def analyze_split(rows, tokenizer) -> dict:
    lexicon = _load_bundled_lexicon()
    n = 0
    multi_sent = 0
    dropped_extra_sents = 0
    any_phrase_trunc = 0
    nrc_tokens = 0
    total_word_tokens = 0
    docs_any_nrc = 0

    for row in rows:
        text = row["text"]
        sents = split_sentences(text)
        n += 1
        if len(sents) > 1:
            multi_sent += 1
        if len(sents) > M_MAX:
            dropped_extra_sents += 1
        kept = sents[:M_MAX]
        doc_trunc = False
        doc_nrc = False
        for sent in kept:
            ids = tokenizer.encode(sent, add_special_tokens=False)
            if len(ids) > TOKENS_PER_PHRASE:
                doc_trunc = True
            for w in word_tokens(sent):
                total_word_tokens += 1
                if w in lexicon:
                    nrc_tokens += 1
                    doc_nrc = True
        if doc_trunc:
            any_phrase_trunc += 1
        if doc_nrc:
            docs_any_nrc += 1

    return {
        "examples": n,
        "pct_multi_sentence": round(100.0 * multi_sent / n, 2),
        "pct_drop_extra_sentences": round(100.0 * dropped_extra_sents / n, 2),
        "pct_any_phrase_token_truncation": round(100.0 * any_phrase_trunc / n, 2),
        "pct_tokens_in_nrc_lexicon": round(100.0 * nrc_tokens / max(total_word_tokens, 1), 2),
        "pct_docs_with_any_nrc_token": round(100.0 * docs_any_nrc / n, 2),
    }


def main() -> None:
    ensure_punkt()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    report = {
        "config": {
            "m_max_sentences": M_MAX,
            "tokens_per_phrase": TOKENS_PER_PHRASE,
            "tokenizer": TOKENIZER,
            "sentence_splitter": "nltk punkt",
            "nrc_lexicon": "nrclex bundled nrc_en.json",
        },
        "splits": {},
    }
    for split in SPLITS:
        report["splits"][split] = analyze_split(ds[split], tokenizer)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
