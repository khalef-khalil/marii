#!/usr/bin/env python3
"""Print LaTeX table rows for DistilBERT vs RoBERTa ablation from campaign JSON files."""

from __future__ import annotations

import json
from pathlib import Path


def fmt_macro(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data["test_aggregate"]["f1_macro"]
    return f"${m['mean']:.3f} \\pm {m['std']:.3f}$".replace(".", "{,}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference" / "artifacts"
    dist_slug, rob_slug = "distilbert_base_uncased", "roberta_base"
    rows = [
        ("+M1", f"m1_{dist_slug}_campaign.json", f"m1_{rob_slug}_campaign.json"),
        ("+M1+M2", f"m1_m2_{dist_slug}_campaign.json", f"m1_m2_{rob_slug}_campaign.json"),
        (
            "+M1+M2+M3 (NRC)",
            f"m1_m2_m3_nrc_{dist_slug}_campaign.json",
            f"m1_m2_m3_nrc_{rob_slug}_campaign.json",
        ),
        (
            "+M1+M2+M3 (SenticNet)",
            f"m1_m2_m3_senticnet_{dist_slug}_campaign.json",
            f"m1_m2_m3_senticnet_{rob_slug}_campaign.json",
        ),
        (
            "+M1+M2+M3+M4 (SenticNet)",
            f"m1_m2_m3_senticnet_m4_{dist_slug}_campaign.json",
            f"m1_m2_m3_senticnet_m4_{rob_slug}_campaign.json",
        ),
    ]
    lines: list[str] = []
    for label, dist_name, rob_name in rows:
        dist_p, rob_p = art / dist_name, art / rob_name
        if not dist_p.is_file() or not rob_p.is_file():
            print(f"Skip (missing pair): {label}", flush=True)
            continue
        lines.append(f"{label} & {fmt_macro(dist_p)} & {fmt_macro(rob_p)} \\\\")
        lines.append("\\hline")
    if not lines:
        print("No complete DistilBERT/RoBERTa pairs found.", flush=True)
        return
    print("Paste into tab:expe-roberta-ablation (or use render_roberta_ablation_tex.py):")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
