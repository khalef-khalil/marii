#!/usr/bin/env python3
"""Add by_gold_cardinality to campaign JSON runs when local checkpoints exist."""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from hakemer.eval_checkpoint import config_from_run_dir
from hakemer.data import GoEmotionsTorchDataset, load_go_emotions_splits, make_dataloader
from hakemer.metrics import f1_by_gold_cardinality, logits_to_preds, logits_to_probs, multilabel_scores
from hakemer.model import HAKEMER
from hakemer.train import forward_batch, resolve_device, set_seed
import numpy as np
import torch
from transformers import AutoTokenizer


def eval_cardinality(run_dir: Path, device: str = "auto") -> dict:
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
        lexicon_fusion=config.lexicon_fusion,
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
    return f1_by_gold_cardinality(y_true, y_pred)


def patch_campaign(path: Path, root: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for run in data.get("runs", []):
        if run.get("test", {}).get("by_gold_cardinality"):
            continue
        out_dir = run.get("output_dir")
        run_dir: Path | None = None
        if out_dir:
            run_dir = Path(out_dir)
            if not run_dir.is_absolute():
                run_dir = root / run_dir
        else:
            seed = run.get("seed")
            backbone = data.get("backbone", "distilbert-base-uncased").replace("-", "_")
            step = data.get("step", "m1_m2")
            if seed is not None:
                candidate = root / "runs" / f"{backbone}_seed{seed}_{step}"
                if (candidate / "best_model.pt").is_file():
                    run_dir = candidate
        if run_dir is None or not (run_dir / "best_model.pt").is_file():
            continue
        card = eval_cardinality(run_dir)
        run.setdefault("test", {})["by_gold_cardinality"] = card
        changed = True
        print(f"Updated cardinality: {run_dir.name}")
    if changed:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return changed


def main() -> None:
    import os

    if os.environ.get("ALLOW_EVAL") != "1":
        print("Skipping checkpoint eval: set ALLOW_EVAL=1 (Colab/CUDA).")
        print("Run scripts/merge_archived_metrics_into_campaigns.py after Colab eval.")
        return
    art = root / "reference" / "artifacts"
    for path in sorted(art.glob("*_campaign.json")):
        patch_campaign(path, root)
    merge = root / "scripts" / "merge_archived_metrics_into_campaigns.py"
    import subprocess

    subprocess.run([sys.executable, str(merge)], check=True, cwd=root)
    subprocess.run([sys.executable, str(root / "scripts" / "render_cardinality_tex.py")], check=True, cwd=root)
    print("Done.")


if __name__ == "__main__":
    main()
