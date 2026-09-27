#!/usr/bin/env python3
"""Build LaTeX rows for F1-macro by gold label count from campaign JSON runs."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tex_tabular_snippet import join_tabular_rows


def aggregate_cardinality(runs: list[dict]) -> dict[str, dict[str, float]]:
    keys = ("1", "2", "3", "4+")
    out: dict[str, dict[str, list[float]]] = {
        k: {"f1_macro": [], "exact_match": [], "n": []} for k in keys
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
            out[k]["n"].append(int(card[k]["n"]))
    result: dict[str, dict[str, float]] = {}
    for k in keys:
        f1s = out[k]["f1_macro"]
        if not f1s:
            continue
        ns = out[k]["n"]
        result[k] = {
            "f1_macro_mean": float(statistics.mean(f1s)),
            "f1_macro_std": float(statistics.pstdev(f1s)) if len(f1s) > 1 else 0.0,
            "n_mean": int(statistics.mean(ns)) if ns else 0,
        }
    return result


def fmt(m: float, s: float) -> str:
    return f"${m:.3f} \\pm {s:.3f}$".replace(".", "{,}")


def campaign_path(art: Path, fname: str) -> Path | None:
    primary = art / fname
    supplement = art / "step_eval_supplements_distilbert" / "campaigns" / fname
    for path in (supplement, primary):
        if not path.is_file():
            continue
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
        if any(r.get("test", {}).get("by_gold_cardinality") for r in runs):
            return path
    return primary if primary.is_file() else (supplement if supplement.is_file() else None)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference" / "artifacts"
    slug = "distilbert_base_uncased"
    configs = [
        ("Step~0", f"baseline_plm_{slug}_campaign.json"),
        ("+M1", f"m1_{slug}_campaign.json"),
        ("+M1+M2", f"m1_m2_{slug}_campaign.json"),
    ]
    support_n: dict[str, int] = {}
    row_lines: list[str] = []
    for bucket in ("1", "2", "3", "4+"):
        row = [bucket]
        for _label, fname in configs:
            path = campaign_path(art, fname)
            if path is None:
                if len(row) == 1:
                    row.append("{---}")
                row.append("{---}")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            agg = aggregate_cardinality(data["runs"])
            if bucket not in agg:
                if len(row) == 1:
                    row.append("{---}")
                row.append("{---}")
                continue
            if bucket not in support_n:
                support_n[bucket] = agg[bucket]["n_mean"]
            m = agg[bucket]["f1_macro_mean"]
            s = agg[bucket]["f1_macro_std"]
            if len(row) == 1:
                row.append(str(support_n.get(bucket, agg[bucket]["n_mean"])))
            row.append(fmt(m, s))
        row_lines.append(" & ".join(row) + " \\\\")
    out = art / "cardinality_snippet.tex"
    body = join_tabular_rows(row_lines)
    out.write_text(body + "\n", encoding="utf-8")
    print(f"Wrote {out}\n{body}")


if __name__ == "__main__":
    main()
