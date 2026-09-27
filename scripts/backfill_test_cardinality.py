#!/usr/bin/env python3
"""Add by_gold_cardinality to campaign JSON runs when local checkpoints exist."""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from hakemer.eval_full_test import eval_cardinality_full_test


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
        card = eval_cardinality_full_test(run_dir)
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
