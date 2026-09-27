"""Full GoEmotions test split for report supplements (ignore smoke-test caps)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from transformers import AutoTokenizer

from hakemer.config import TrainConfig
from hakemer.data import GoEmotionsTorchDataset, load_go_emotions_splits, make_dataloader
from hakemer.eval_checkpoint import config_from_run_dir
from hakemer.metrics import f1_by_gold_cardinality, logits_to_preds
from hakemer.model import HAKEMER
from hakemer.train import evaluate, forward_batch, resolve_device, set_seed
import numpy as np
import torch


def full_test_config(config: TrainConfig) -> TrainConfig:
    return replace(config, max_eval_samples=None, max_train_samples=None)


def cardinality_is_reportable(card: dict) -> bool:
    one = card.get("1") or {}
    return int(one.get("n", 0)) >= 4000


def eval_cardinality_full_test(run_dir: Path, device: str = "auto") -> dict:
    run_dir = run_dir.resolve()
    config = full_test_config(config_from_run_dir(run_dir))
    set_seed(config.seed)
    dev = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    _, _, test_hf = load_go_emotions_splits()
    ds_kw = dict(
        use_m1=config.use_m1,
        use_m3=config.use_m3,
        lexicon_source=config.lexicon_source,
        lexicon_fusion=config.lexicon_fusion,
        max_phrases=config.max_phrases,
        phrase_max_length=config.phrase_max_length,
    )
    test_ds = GoEmotionsTorchDataset(
        test_hf,
        tokenizer,
        config.max_length,
        config.num_labels,
        None,
        **ds_kw,
    )
    loader = make_dataloader(test_ds, 16, shuffle=False)
    model = HAKEMER(config).to(dev)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=dev))
    model.eval()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = forward_batch(model, batch, dev)
            y_true_all.append(batch["labels"].numpy())
            y_pred_all.append(logits_to_preds(logits.cpu().numpy(), threshold=config.decision_threshold))
    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)
    card = f1_by_gold_cardinality(y_true, y_pred)
    if not cardinality_is_reportable(card):
        raise RuntimeError(
            f"Cardinality n={card.get('1', {}).get('n')} looks like a subsampled eval; "
            "use a full-protocol checkpoint (Colab baseline campaign)."
        )
    return card


def make_full_test_loader(run_dir: Path, batch_size: int = 16):
    config = full_test_config(config_from_run_dir(run_dir))
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    _, _, test_hf = load_go_emotions_splits()
    ds_kw = dict(
        use_m1=config.use_m1,
        use_m3=config.use_m3,
        lexicon_source=config.lexicon_source,
        lexicon_fusion=config.lexicon_fusion,
        max_phrases=config.max_phrases,
        phrase_max_length=config.phrase_max_length,
    )
    test_ds = GoEmotionsTorchDataset(
        test_hf,
        tokenizer,
        config.max_length,
        config.num_labels,
        None,
        **ds_kw,
    )
    return config, make_dataloader(test_ds, batch_size, shuffle=False)
