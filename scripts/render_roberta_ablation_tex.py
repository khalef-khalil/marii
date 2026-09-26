#!/usr/bin/env python3
"""Write roberta_ablation_extra_rows.tex from RoBERTa M3/M4 campaign JSON (if present)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt_macro(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data["test_aggregate"]["f1_macro"]
    return f"${m['mean']:.3f} \\pm {m['std']:.3f}$".replace(".", "{,}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference" / "artifacts"
    slug = "roberta_base"
    specs = [
        ("+M1+M2+M3 (NRC)", f"m1_m2_m3_nrc_{slug}_campaign.json", "distilbert"),
        ("+M1+M2+M3 (SenticNet)", f"m1_m2_m3_senticnet_{slug}_campaign.json", "distilbert"),
        ("+M1+M2+M3+M4 (SenticNet)", f"m1_m2_m3_senticnet_m4_{slug}_campaign.json", "distilbert"),
    ]
    dist_slug = "distilbert_base_uncased"
    dist_map = {
        "+M1+M2+M3 (NRC)": art / f"m1_m2_m3_nrc_{dist_slug}_campaign.json",
        "+M1+M2+M3 (SenticNet)": art / f"m1_m2_m3_senticnet_{dist_slug}_campaign.json",
        "+M1+M2+M3+M4 (SenticNet)": art / f"m1_m2_m3_senticnet_m4_{dist_slug}_campaign.json",
    }
    rows: list[str] = []
    for label, rob_name, _ in specs:
        rob_path = art / rob_name
        if not rob_path.is_file():
            continue
        dist_path = dist_map[label]
        if not dist_path.is_file():
            continue
        rows.append(
            f"{label} & {fmt_macro(dist_path)} & {fmt_macro(rob_path)} \\\\\n\\hline"
        )
    out = root / "roberta_ablation_extra_rows.tex"
    if not rows:
        if out.is_file():
            out.unlink()
        print("No RoBERTa M3/M4 JSON pairs found; removed snippet if any.", file=sys.stderr)
        return 0
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
