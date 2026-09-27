#!/usr/bin/env python3
"""Compute Δ = mean F1(frequent group) − mean F1(rare group) from per_label metrics.json."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def delta_from_per_label(rows: list[dict], *, freq_min_support: int = 500, rare_max_support: int = 39) -> float:
    f_vals = [float(r["f1"]) for r in rows if int(r.get("support", 0)) >= freq_min_support]
    r_vals = [float(r["f1"]) for r in rows if int(r.get("support", 0)) < rare_max_support]
    if not f_vals or not r_vals:
        raise ValueError("Empty frequent or rare group; check support fields in per_label rows")
    return float(statistics.mean(f_vals) - statistics.mean(r_vals))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("metrics_json", type=Path, help="metrics.json with test.per_label")
    args = p.parse_args()
    data = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    per_label = data.get("test", {}).get("per_label")
    if not per_label:
        raise SystemExit("No test.per_label in metrics file")
    delta = delta_from_per_label(per_label)
    print(f"Δ frequent−rare = {delta:.4f}")


if __name__ == "__main__":
    main()
