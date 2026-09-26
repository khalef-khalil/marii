from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from hakemer.metrics import go_emotions_label_names


def load_per_label_from_metrics(path: Path) -> list[dict] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test") or {}
    pl = test.get("per_label")
    if pl is None:
        return None
    return pl


def aggregate_per_label(runs: list[list[dict]]) -> list[dict]:
    names = go_emotions_label_names()
    by_label: dict[str, list[float]] = {n: [] for n in names}
    support: dict[str, int] = {}
    for pl in runs:
        for row in pl:
            label = str(row["label"])
            by_label[label].append(float(row["f1"]))
            support[label] = int(row["support"])
    out = []
    for name in names:
        vals = by_label[name]
        out.append(
            {
                "label": name,
                "support": support.get(name, 0),
                "f1_mean": float(statistics.mean(vals)) if vals else None,
                "f1_std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
            }
        )
    return out


LADDER_PATTERNS: list[tuple[str, str]] = [
    ("Step~0 PLM (flat)", "distilbert_base_uncased_seed{seed}_baseline_plm"),
    ("+M1", "distilbert_base_uncased_seed{seed}_m1"),
    ("+M1+M2", "distilbert_base_uncased_seed{seed}_m1_m2"),
    ("+M1+M2+M3 (NRC)", "distilbert_base_uncased_seed{seed}_m1_m2_m3_nrc"),
    ("+M1+M2+M3 (SenticNet)", "distilbert_base_uncased_seed{seed}_m1_m2_m3_senticnet"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-class ladder JSON from run metrics.json files")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--artifact-dir", type=Path, default=Path("reference/artifacts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    configs: list[dict] = []

    for display, pattern in LADDER_PATTERNS:
        per_seed_pl: list[list[dict]] = []
        seeds_ok: list[int] = []
        macros: list[float] = []
        for seed in seeds:
            run_name = pattern.format(seed=seed)
            metrics_path = args.runs_dir / run_name / "metrics.json"
            if not metrics_path.is_file():
                print(f"skip (no metrics): {metrics_path}")
                continue
            pl = load_per_label_from_metrics(metrics_path)
            if pl is None:
                print(f"skip (no per_label in test metrics): {metrics_path}")
                continue
            per_seed_pl.append(pl)
            seeds_ok.append(seed)
            test = json.loads(metrics_path.read_text())["test"]
            macros.append(float(test["f1_macro"]))
        if not per_seed_pl:
            continue
        configs.append(
            {
                "configuration": display,
                "seeds_evaluated": seeds_ok,
                "test_f1_macro_mean": float(statistics.mean(macros)),
                "per_label": aggregate_per_label(per_seed_pl),
            }
        )

    slug = "_".join(map(str, seeds))
    report = {
        "backbone": "distilbert-base-uncased",
        "split": "test",
        "source": "metrics.json test.per_label",
        "seeds_requested": seeds,
        "configurations": configs,
    }
    out_path = args.artifact_dir / f"per_class_ladder_distilbert_seeds_{slug}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(configs)} configurations)")


if __name__ == "__main__":
    main()
