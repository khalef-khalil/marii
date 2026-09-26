from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from hakemer.eval_checkpoint import eval_run_dir
from hakemer.metrics import go_emotions_label_names


LADDER: list[tuple[str, str]] = [
    ("Step~0 PLM (flat)", "distilbert_base_uncased_seed{seed}_baseline_plm"),
    ("+M1", "distilbert_base_uncased_seed{seed}_m1"),
    ("+M1+M2", "distilbert_base_uncased_seed{seed}_m1_m2"),
    ("+M1+M2+M3 (NRC)", "distilbert_base_uncased_seed{seed}_m1_m2_m3_nrc"),
    ("+M1+M2+M3 (SenticNet)", "distilbert_base_uncased_seed{seed}_m1_m2_m3_senticnet"),
]


def aggregate_per_label(runs: list[dict]) -> list[dict]:
    names = go_emotions_label_names()
    by_label: dict[str, list[float]] = {n: [] for n in names}
    support: dict[str, int] = {}
    for run in runs:
        for row in run["per_label"]:
            label = str(row["label"])
            by_label[label].append(float(row["f1"]))
            support[label] = int(row["support"])
    out = []
    for name in names:
        vals = by_label[name]
        out.append(
            {
                "label": name,
                "support": support[name],
                "f1_mean": float(statistics.mean(vals)) if vals else None,
                "f1_std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class F1 ladder from local run checkpoints")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--seeds", default="42", help="Comma-separated seeds (checkpoint must exist)")
    p.add_argument("--device", default="auto")
    p.add_argument("--artifact-dir", type=Path, default=Path("reference/artifacts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ladder_out: list[dict] = []

    for display, pattern in LADDER:
        seed_runs: list[dict] = []
        for seed in seeds:
            run_name = pattern.format(seed=seed)
            run_dir = args.runs_dir / run_name
            if not (run_dir / "best_model.pt").is_file():
                print(f"skip (no checkpoint): {run_dir}")
                continue
            print(f"eval: {run_dir}")
            seed_runs.append(eval_run_dir(run_dir, device=args.device))
        if not seed_runs:
            continue
        ladder_out.append(
            {
                "configuration": display,
                "seeds_evaluated": [r["seed"] for r in seed_runs],
                "test_f1_macro_mean": float(
                    statistics.mean(float(r["test"]["f1_macro"]) for r in seed_runs)
                ),
                "per_label": aggregate_per_label(seed_runs),
            }
        )

    report = {
        "backbone": "distilbert-base-uncased",
        "split": "test",
        "seeds_requested": seeds,
        "configurations": ladder_out,
    }
    slug = "_".join(map(str, seeds))
    out_path = args.artifact_dir / f"per_class_ladder_distilbert_seeds_{slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
