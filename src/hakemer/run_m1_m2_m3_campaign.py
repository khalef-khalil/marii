from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from hakemer.config import TrainConfig
from hakemer.train import train_loop

DEFAULT_SEEDS = (42, 123, 456)


def aggregate_runs(per_seed: list[dict]) -> dict:
    metric_keys = ("f1_micro", "f1_macro", "exact_match", "map")
    test_block: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = [float(row["test"][key]) for row in per_seed]
        test_block[key] = {
            "mean": float(statistics.mean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "values": values,
        }
    gates = [row.get("lexicon_gate_abs") for row in per_seed if row.get("lexicon_gate_abs") is not None]
    extra: dict = {}
    if gates:
        extra["lexicon_gate_abs"] = {
            "mean": float(statistics.mean(gates)),
            "std": float(statistics.pstdev(gates)) if len(gates) > 1 else 0.0,
            "values": gates,
        }
    return {
        "seeds": [row["seed"] for row in per_seed],
        "runs": per_seed,
        "test_aggregate": test_block,
        **extra,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1+M2+M3 lexicon campaign (H3, DistilBERT)")
    p.add_argument(
        "--lexicon",
        default="nrc",
        choices=["nrc", "senticnet"],
        help="Lexicon resource for M3 (SenticNet requires lookup implementation).",
    )
    p.add_argument("--backbone", default="distilbert-base-uncased", choices=[
        "distilbert-base-uncased", "roberta-base",
    ])
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--max-phrases", type=int, default=4)
    p.add_argument("--phrase-max-length", type=int, default=32)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--artifact-dir", default="reference/artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    per_seed: list[dict] = []

    for seed in seeds:
        config = TrainConfig(
            backbone=args.backbone,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            early_stopping_patience=args.early_stopping_patience,
            use_m1=True,
            use_m2=True,
            use_m3=True,
            lexicon_source=args.lexicon,
            max_phrases=args.max_phrases,
            phrase_max_length=args.phrase_max_length,
            output_dir=args.output_dir,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            device=args.device,
        )
        summary = train_loop(config)
        per_seed.append({"seed": seed, **summary})

    runs_clean = []
    for row in per_seed:
        item = dict(row)
        item.pop("output_dir", None)
        runs_clean.append(item)

    report = {
        "backbone": args.backbone,
        "step": f"m1_m2_m3_{args.lexicon}",
        "modules": {
            "use_m1": True,
            "use_m2": True,
            "use_m3": True,
            "use_m4": False,
            "lexicon_source": args.lexicon,
        },
        "protocol": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_phrases": args.max_phrases,
            "phrase_max_length": args.phrase_max_length,
            "early_stopping_patience": args.early_stopping_patience,
        },
        **aggregate_runs(runs_clean),
    }
    report["runs"] = runs_clean
    slug = args.backbone.replace("-", "_")
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / f"m1_m2_m3_{args.lexicon}_{slug}_campaign.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    agg = report["test_aggregate"]
    gate_msg = ""
    if "lexicon_gate_abs" in report:
        g = report["lexicon_gate_abs"]
        gate_msg = f" |g|={g['mean']:.4f}±{g['std']:.4f} "
    print(
        f"M1+M2+M3 ({args.lexicon}) done ({len(seeds)} seeds). "
        f"Test F1-macro={agg['f1_macro']['mean']:.4f}±{agg['f1_macro']['std']:.4f} "
        f"{gate_msg}-> {out_path}"
    )


if __name__ == "__main__":
    main()
