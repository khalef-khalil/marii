#!/usr/bin/env python3
"""Eval-only: cardinality + M3 lexicon zero/shuffle for saved checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hakemer.eval_checkpoint import config_from_run_dir
from hakemer.data import GoEmotionsTorchDataset, load_go_emotions_splits, make_dataloader
from hakemer.metrics import f1_by_gold_cardinality, logits_to_preds
from hakemer.model import HAKEMER
from hakemer.train import evaluate, forward_batch, resolve_device, set_seed
import numpy as np
import torch
from transformers import AutoTokenizer


def eval_cardinality(run_dir: Path, device: str) -> dict:
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


def supplement_run(run_dir: Path, device: str, dry_run: bool = False) -> bool:
    run_dir = run_dir.resolve()
    ckpt = run_dir / "best_model.pt"
    if not ckpt.is_file():
        return False
    config = config_from_run_dir(run_dir)
    metrics_path = run_dir / "metrics.json"
    summary: dict = {}
    if metrics_path.is_file():
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    changed = False
    test = summary.setdefault("test", {})
    if not test.get("by_gold_cardinality"):
        if dry_run:
            print(f"[dry-run] cardinality: {run_dir.name}")
        else:
            test["by_gold_cardinality"] = eval_cardinality(run_dir, device)
            changed = True
            print(f"Cardinality: {run_dir.name}")
    if config.use_m3:
        dev = resolve_device(device)
        set_seed(config.seed)
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
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()
        for mode in ("zero", "shuffle"):
            key = f"test_lexicon_{mode}"
            if key in summary:
                continue
            if dry_run:
                print(f"[dry-run] lexicon {mode}: {run_dir.name}")
                continue
            ablated = evaluate(
                model,
                loader,
                dev,
                threshold=config.decision_threshold,
                lexicon_ablation=mode,
            )
            summary[key] = {k: v for k, v in ablated.items() if k != "per_label"}
            changed = True
            print(f"Lexicon {mode}: {run_dir.name}")
    if changed and not dry_run:
        summary.setdefault("output_dir", str(run_dir))
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return changed


def main() -> None:
    p = argparse.ArgumentParser(description="Eval-only supplements for archived runs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--glob", default="distilbert_base_uncased_seed*")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    runs_root = root / "runs"
    if not runs_root.is_dir():
        print("No runs/ directory.", file=sys.stderr)
        sys.exit(1)
    any_changed = False
    for run_dir in sorted(runs_root.glob(f"{args.glob}")):
        if not run_dir.is_dir():
            continue
        if supplement_run(run_dir, args.device, dry_run=args.dry_run):
            any_changed = True
    merge = root / "scripts" / "merge_archived_metrics_into_campaigns.py"
    render_card = root / "scripts" / "render_cardinality_tex.py"
    render_lex = root / "scripts" / "render_lexicon_ablation_tex.py"
    if any_changed and not args.dry_run:
        subprocess.run([sys.executable, str(merge)], check=True, cwd=root)
        subprocess.run([sys.executable, str(render_card)], check=True, cwd=root)
        subprocess.run([sys.executable, str(render_lex)], check=True, cwd=root)
        print("Merged campaigns and regenerated LaTeX snippets.")
    elif not any_changed:
        print("No supplements applied (missing checkpoints or already complete).")


if __name__ == "__main__":
    main()
