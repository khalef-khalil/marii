from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from hakemer.data import GoEmotionsTorchDataset, load_go_emotions_splits, make_dataloader
from hakemer.eval_checkpoint import config_from_run_dir
from hakemer.metrics import go_emotions_label_names, logits_to_preds
from hakemer.model import HAKEMER
from hakemer.train import forward_batch, resolve_device, set_seed


def collect_test_predictions(
    run_dir: Path,
    *,
    device: str = "auto",
    batch_size: int = 16,
    threshold: float = 0.5,
    max_examples: int = 8,
) -> dict:
    run_dir = run_dir.resolve()
    config = config_from_run_dir(run_dir)
    set_seed(config.seed)
    dev = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    _, _, test_hf = load_go_emotions_splits()
    ds_kw = dict(
        use_m1=config.use_m1,
        use_m3=config.use_m3,
        lexicon_source=config.lexicon_source,
        max_phrases=config.max_phrases,
        phrase_max_length=config.phrase_max_length,
    )
    test_ds = GoEmotionsTorchDataset(
        test_hf,
        tokenizer,
        config.max_length,
        config.num_labels,
        config.max_eval_samples,
        **ds_kw,
    )
    loader = make_dataloader(test_ds, batch_size, shuffle=False)
    model = HAKEMER(config).to(dev)
    ckpt = run_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    names = go_emotions_label_names()
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    texts_all: list[str] = []

    with torch.no_grad():
        for batch in loader:
            logits = forward_batch(model, batch, dev)
            preds = logits_to_preds(logits.cpu().numpy(), threshold=threshold)
            labels = batch["labels"].numpy()
            y_pred_all.append(preds)
            y_true_all.append(labels)
            batch_size_actual = labels.shape[0]
            start = len(texts_all)
            for i in range(batch_size_actual):
                idx = start + i
                if idx < len(test_hf):
                    texts_all.append(test_hf[idx]["text"])
                else:
                    texts_all.append("")

    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    pair_counts: dict[str, int] = defaultdict(int)
    for j, name in enumerate(names):
        gold_pos = y_true[:, j] == 1
        pred_pos = y_pred[:, j] == 1
        fn_mask = gold_pos & ~pred_pos
        fp_mask = ~gold_pos & pred_pos
        for k, other in enumerate(names):
            if k == j:
                continue
            if fn_mask.any() and (y_pred[fn_mask, k] == 1).any():
                pair_counts[f"FN {name} pred {other}"] += int((y_pred[fn_mask, k] == 1).sum())
            if fp_mask.any() and (y_true[fp_mask, k] == 1).any():
                pair_counts[f"FP {name} gold {other}"] += int((y_true[fp_mask, k] == 1).sum())

    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:15]

    rare_examples: dict[str, list[dict]] = {}
    for label in ("grief", "relief", "neutral"):
        if label not in names:
            continue
        j = names.index(label)
        cases: list[dict] = []
        scored: list[tuple[int, dict]] = []
        for i in range(len(y_true)):
            if y_true[i, j] == 0 and y_pred[i, j] == 0:
                continue
            gold = [names[k] for k in range(len(names)) if y_true[i, k] == 1]
            pred = [names[k] for k in range(len(names)) if y_pred[i, k] == 1]
            if y_true[i, j] == 1 and y_pred[i, j] == 1:
                kind, rank = "TP", 0
            elif y_true[i, j] == 1:
                kind, rank = "FN", 1
            else:
                kind, rank = "FP", 2
            scored.append(
                (
                    rank,
                    {
                        "kind": kind,
                        "text": texts_all[i][:280],
                        "gold": gold,
                        "pred": pred,
                    },
                )
            )
        scored.sort(key=lambda x: x[0])
        rare_examples[label] = [item for _, item in scored[:max_examples]]

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "seed": config.seed,
        "backbone": config.backbone,
        "threshold": threshold,
        "n_test": int(y_true.shape[0]),
        "top_confusion_pairs": [{"pattern": p, "count": c} for p, c in top_pairs],
        "rare_label_examples": rare_examples,
    }


def write_error_analysis(run_dir: Path, output: Path | None = None, **kwargs) -> Path:
    report = collect_test_predictions(run_dir, **kwargs)
    out = output or Path("reference/artifacts") / f"error_analysis_{run_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out
