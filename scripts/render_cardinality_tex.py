#!/usr/bin/env python3
"""Build LaTeX rows for F1-macro by gold label count from campaign JSON runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def aggregate_cardinality(runs: list[dict]) -> dict[str, dict[str, float]]:
    keys = ("1", "2", "3", "4+")
    out: dict[str, dict[str, list[float]]] = {
        k: {"f1_macro": [], "exact_match": []} for k in keys
    }
    for run in runs:
        card = run.get("test", {}).get("by_gold_cardinality")
        if not card:
            continue
        for k in keys:
            if k not in card or card[k].get("n", 0) == 0:
                continue
            out[k]["f1_macro"].append(float(card[k]["f1_macro"]))
            out[k]["exact_match"].append(float(card[k]["exact_match"]))
    result: dict[str, dict[str, float]] = {}
    for k in keys:
        f1s = out[k]["f1_macro"]
        if not f1s:
            continue
        result[k] = {
            "f1_macro_mean": float(statistics.mean(f1s)),
            "f1_macro_std": float(statistics.pstdev(f1s)) if len(f1s) > 1 else 0.0,
        }
    return result


def fmt(m: float, s: float) -> str:
    return f"${m:.3f} \\pm {s:.3f}$".replace(".", "{,}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference" / "artifacts"
    slug = "distilbert_base_uncased"
    configs = [
        ("Step~0", f"baseline_plm_{slug}_campaign.json"),
        ("+M1", f"m1_{slug}_campaign.json"),
        ("+M1+M2", f"m1_m2_{slug}_campaign.json"),
    ]
    lines: list[str] = []
    for bucket in ("1", "2", "3", "4+"):
        row = [bucket]
        for _label, fname in configs:
            path = art / fname
            if not path.is_file():
                row.append("{---}")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            agg = aggregate_cardinality(data["runs"])
            if bucket not in agg:
                row.append("{---}")
            else:
                m = agg[bucket]["f1_macro_mean"]
                s = agg[bucket]["f1_macro_std"]
                row.append(fmt(m, s))
        lines.append(" & ".join(row) + " \\\\")
        lines.append("\\hline")
    out = art / "cardinality_snippet.tex"
    body = "\n".join(lines)
    out.write_text(body + "\n", encoding="utf-8")
    print(f"Wrote {out}\n{body}")


if __name__ == "__main__":
    main()
