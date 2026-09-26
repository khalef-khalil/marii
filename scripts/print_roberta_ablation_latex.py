#!/usr/bin/env python3
"""Print LaTeX table rows for RoBERTa +M1 / +M1+M2 from campaign JSON files."""

from __future__ import annotations

import json
from pathlib import Path


def fmt_macro(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data["test_aggregate"]["f1_macro"]
    return f"${m['mean']:.3f} \\pm {m['std']:.3f}$".replace(".", "{,}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference/artifacts"
    dist_m1 = art / "m1_distilbert_base_uncased_campaign.json"
    dist_m2 = art / "m1_m2_distilbert_base_uncased_campaign.json"
    rob_m1 = art / "m1_roberta_base_campaign.json"
    rob_m2 = art / "m1_m2_roberta_base_campaign.json"
    for p in (dist_m1, dist_m2, rob_m1, rob_m2):
        if not p.is_file():
            print(f"Missing {p.name}", flush=True)
            return
    print("Paste into tab:expe-roberta-ablation:")
    print(
        f"+M1 & {fmt_macro(dist_m1)} & {fmt_macro(rob_m1)} \\\\"
        f"\n\\hline\n"
        f"+M1+M2 & {fmt_macro(dist_m2)} & {fmt_macro(rob_m2)} \\\\"
    )


if __name__ == "__main__":
    main()
