#!/usr/bin/env python3
"""Quick forward-pass smoke test for emotion-specific M3 (no GPU training)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from transformers import AutoTokenizer

from hakemer.config import TrainConfig
from hakemer.data import GoEmotionsTorchDataset, load_go_emotions_splits
from hakemer.metrics import go_emotions_label_names
from hakemer.goemotions_lexicon_map import SIMPLIFIED_GOEMOTIONS_LABELS
from hakemer.model import HAKEMER


def main() -> None:
    names = go_emotions_label_names()
    assert names == list(SIMPLIFIED_GOEMOTIONS_LABELS), "Label order drift vs HF dataset"
    train, _, _ = load_go_emotions_splits()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = GoEmotionsTorchDataset(
        train.select(range(4)),
        tok,
        128,
        28,
        use_m1=True,
        use_m3=True,
        lexicon_source="nrc",
        lexicon_fusion="emotion_specific",
    )
    batch = {k: torch.stack([ds[i][k] for i in range(4)]) for k in ds[0].keys()}
    cfg = TrainConfig(use_m1=True, use_m2=True, use_m3=True, lexicon_source="nrc", lexicon_fusion="emotion_specific")
    model = HAKEMER(cfg)
    logits = model(
        batch["input_ids"],
        batch["attention_mask"],
        phrase_mask=batch["phrase_mask"],
        lexicon_features=batch["lexicon_features"],
    )
    assert logits.shape == (4, 28)
    print("OK: emotion-specific M3 forward", logits.shape)


if __name__ == "__main__":
    main()
