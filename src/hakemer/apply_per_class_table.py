"""Print LaTeX rows for tab:expe-per-class-f1 from per_class_ladder JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FOCUS = [
    "grief",
    "relief",
    "pride",
    "nervousness",
    "embarrassment",
    "remorse",
    "fear",
    "desire",
    "admiration",
    "neutral",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "ladder_json",
        type=Path,
        default=Path("reference/artifacts/per_class_ladder_distilbert_seeds_42_123_456.json"),
        nargs="?",
    )
    args = p.parse_args()
    data = json.loads(args.ladder_json.read_text(encoding="utf-8"))
    by_name: dict[str, dict[str, dict]] = {}
    for cfg in data["configurations"]:
        key = cfg["configuration"]
        by_name[key] = {row["label"]: row for row in cfg["per_label"]}

    def f1(cfg_key: str, label: str) -> str:
        row = by_name.get(cfg_key, {}).get(label)
        if not row or row.get("f1_mean") is None:
            return "---"
        m, s = float(row["f1_mean"]), float(row["f1_std"])
        return f"${m:.3f} \\pm {s:.3f}$".replace(".", "{,}")

    step0 = "Step~0 PLM (flat)"
    m1 = "+M1"
    m2 = "+M1+M2"
    support = by_name.get(m2) or by_name.get(step0) or {}
    for label in FOCUS:
        sup = support.get(label, {}).get("support", 0)
        print(
            f"{label} & {sup} & {f1(step0, label)} & {f1(m1, label)} & {f1(m2, label)} \\\\"
        )
        print("\\hline")


if __name__ == "__main__":
    main()
