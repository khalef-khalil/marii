#!/usr/bin/env python3
"""Merge error_analysis_*.json under reference/artifacts into one aggregate file."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference/artifacts"
    paths = sorted(art.glob("error_analysis_*_m1_m2.json"))
    if not paths:
        paths = sorted(art.glob("error_analysis_*.json"))
    if not paths:
        print("No error_analysis_*.json found", flush=True)
        return

    pair_totals: dict[str, int] = defaultdict(int)
    reports = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        reports.append(data)
        for row in data.get("top_confusion_pairs", []):
            pair_totals[row["pattern"]] += int(row["count"])

    top = sorted(pair_totals.items(), key=lambda x: x[1], reverse=True)[:12]
    out = {
        "source_files": [p.name for p in paths],
        "n_reports": len(reports),
        "top_confusion_pairs_aggregate": [{"pattern": p, "count": c} for p, c in top],
        "reports": reports,
    }
    dest = art / "error_analysis_m1_m2_aggregate.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(dest)
    for p, c in top[:8]:
        print(f"  {c:4d}  {p}")


if __name__ == "__main__":
    main()
