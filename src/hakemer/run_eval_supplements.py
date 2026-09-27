#!/usr/bin/env python3
"""Eval-only: cardinality + M3 lexicon zero/shuffle for saved checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hakemer.eval_full_test import (
    cardinality_is_reportable,
    eval_cardinality_full_test,
    full_test_config,
    make_full_test_loader,
)
from hakemer.eval_checkpoint import config_from_run_dir
from hakemer.model import HAKEMER
from hakemer.train import evaluate, resolve_device, set_seed
import torch


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
    existing = test.get("by_gold_cardinality")
    if not existing or not cardinality_is_reportable(existing):
        if dry_run:
            print(f"[dry-run] cardinality: {run_dir.name}")
        else:
            test["by_gold_cardinality"] = eval_cardinality_full_test(run_dir, device)
            changed = True
            print(f"Cardinality: {run_dir.name} (n={test['by_gold_cardinality']['1']['n']})")
    if config.use_m3:
        dev = resolve_device(device)
        set_seed(config.seed)
        cfg_full, loader = make_full_test_loader(run_dir)
        model = HAKEMER(full_test_config(config_from_run_dir(run_dir))).to(dev)
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
                threshold=cfg_full.decision_threshold,
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
    p.add_argument("--no-merge", action="store_true", help="Skip merge_archived_metrics (Step~0 script)")
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
        try:
            if supplement_run(run_dir, args.device, dry_run=args.dry_run):
                any_changed = True
        except RuntimeError as exc:
            print(f"skip {run_dir.name}: {exc}", file=sys.stderr)
    merge = root / "scripts" / "merge_archived_metrics_into_campaigns.py"
    render_card = root / "scripts" / "render_cardinality_tex.py"
    render_lex = root / "scripts" / "render_lexicon_ablation_tex.py"
    if any_changed and not args.dry_run:
        if not args.no_merge:
            subprocess.run([sys.executable, str(merge)], check=True, cwd=root)
        subprocess.run([sys.executable, str(render_card)], check=True, cwd=root)
        if "baseline_plm" not in args.glob:
            subprocess.run([sys.executable, str(render_lex)], check=True, cwd=root)
        print("Regenerated LaTeX snippets.")
    elif not any_changed:
        print("No supplements applied (missing checkpoints or already complete).")


if __name__ == "__main__":
    main()
