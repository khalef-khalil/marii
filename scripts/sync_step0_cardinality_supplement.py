#!/usr/bin/env python3
"""Patch Step~0 supplement campaign with by_gold_cardinality from run metrics."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from hakemer.eval_full_test import cardinality_is_reportable

SEEDS = (42, 123, 456)
SLUG = "distilbert_base_uncased"
STEP = "baseline_plm"


def metrics_for_seed(root: Path, seed: int) -> Path | None:
    run_name = f"{SLUG}_seed{seed}_{STEP}"
    for candidate in (
        root / "runs" / run_name / "metrics.json",
        root / "reference" / "artifacts" / "step0_distilbert" / run_name / "metrics.json",
        root / "reference" / "artifacts" / "step_eval_supplements_distilbert" / run_name / "metrics.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    art = root / "reference" / "artifacts"
    supplement_campaign = (
        art / "step_eval_supplements_distilbert" / "campaigns" / f"{STEP}_{SLUG}_campaign.json"
    )
    if not supplement_campaign.is_file():
        raise SystemExit(f"Missing {supplement_campaign}")

    data = json.loads(supplement_campaign.read_text(encoding="utf-8"))
    changed = False
    dest_root = art / "step_eval_supplements_distilbert"
    for run in data.get("runs", []):
        seed = run.get("seed")
        if seed is None:
            continue
        mpath = metrics_for_seed(root, int(seed))
        if mpath is None:
            print(f"skip seed {seed}: no metrics.json")
            continue
        metrics = json.loads(mpath.read_text(encoding="utf-8"))
        card = metrics.get("test", {}).get("by_gold_cardinality")
        if not card or not cardinality_is_reportable(card):
            print(f"skip seed {seed}: missing or non-reportable cardinality in {mpath}")
            continue
        run.setdefault("test", {})["by_gold_cardinality"] = card
        changed = True
        out_dir = dest_root / mpath.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mpath, out_dir / "metrics.json")
        print(f"seed {seed}: cardinality from {mpath.parent.name}")

    if changed:
        supplement_campaign.write_text(json.dumps(data, indent=2), encoding="utf-8")
    import subprocess
    import sys

    subprocess.run([sys.executable, str(root / "scripts" / "render_cardinality_tex.py")], check=True)
    print("Updated supplement baseline campaign and cardinality_snippet.tex")


if __name__ == "__main__":
    main()
