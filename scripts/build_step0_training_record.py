#!/usr/bin/env python3
"""Build Step 0 PLM training-record notebooks from archived campaign JSON."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reference/training_records/step0_plm"
TEMPLATE = "notebooks/baseline_plm_campaign.ipynb"

STEPS = (
    {
        "slug": "distilbert",
        "backbone": "distilbert-base-uncased",
        "campaign": ROOT / "reference/artifacts/baseline_plm_distilbert_base_uncased_campaign.json",
        "zip": "reference/artifacts/baseline_plm_distilbert_step0.zip",
        "shell": "./run_baseline_campaign.sh --backbone distilbert-base-uncased "
        "--epochs 4 --batch-size 16 --lr 5e-5 --early-stopping-patience 0",
        "out_name": "baseline_plm_step0_distilbert_record.ipynb",
    },
    {
        "slug": "roberta",
        "backbone": "roberta-base",
        "campaign": ROOT / "reference/artifacts/baseline_plm_roberta_base_campaign.json",
        "zip": "reference/artifacts/baseline_plm_roberta_step0.zip",
        "shell": "./run_baseline_campaign.sh --backbone roberta-base "
        "--epochs 4 --batch-size 16 --lr 5e-5 --early-stopping-patience 0",
        "out_name": "baseline_plm_step0_roberta_record.ipynb",
    },
)


def _text_output(text: str) -> dict:
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text if text.endswith("\n") else text + "\n",
    }


def _code_cell(source: str, outputs: list[dict] | None = None, execution_count: int = 1) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")],
        "execution_count": execution_count,
        "outputs": outputs or [],
    }


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")],
    }


def format_campaign_report(campaign: dict) -> str:
    lines = [
        f"backbone: {campaign['backbone']}",
        f"protocol: {json.dumps(campaign.get('protocol', {}), sort_keys=True)}",
        "",
    ]
    for run in campaign["runs"]:
        lines.append(
            f"seed {run['seed']}: best_val_f1_macro={run['best_val_f1_macro']:.6f} "
            f"test_f1_macro={run['test']['f1_macro']:.6f} epochs={len(run['history'])}"
        )
    lines.append("")
    for metric, block in campaign["test_aggregate"].items():
        lines.append(f"{metric}: {block['mean']:.6f} ± {block['std']:.6f}")
    return "\n".join(lines) + "\n"


def build_notebook(step: dict, campaign: dict) -> dict:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = format_campaign_report(campaign)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "hake_mer_training_record": {
                "step": "step0_plm",
                "backbone": step["backbone"],
                "generated_from_artifacts": str(step["campaign"].relative_to(ROOT)),
                "template_notebook": TEMPLATE,
                "generated_at_utc": generated,
                "note": "Auto-built from archived campaign JSON. Prefer colab/*_executed.ipynb when available.",
            },
        },
        "cells": [
            _md_cell(
                f"# Step 0 PLM training record — {step['backbone']}\n\n"
                f"This notebook documents the **archived** Step 0 run (campaign JSON + zip). "
                f"It was generated on {generated} from `{step['campaign'].relative_to(ROOT)}`.\n\n"
                f"**Colab proof:** if you still have the session, export **File → Download → .ipynb** "
                f"to `reference/training_records/step0_plm/colab/` and commit alongside this file.\n\n"
                f"**Template used:** `{TEMPLATE}`"
            ),
            _code_cell(
                f"!{step['shell']}",
                [_text_output(f"[Recorded run — see campaign JSON]\n{report}")],
            ),
            _code_cell(
                f"artifact_zip = '{step['zip']}'\n"
                f"campaign_path = '{step['campaign'].relative_to(ROOT)}'\n"
                "print(artifact_zip, campaign_path)",
                [
                    _text_output(
                        f"{step['zip']}\n{step['campaign'].relative_to(ROOT)}\n"
                    )
                ],
            ),
        ],
    }


def write_manifest(steps_meta: list[dict]) -> None:
    manifest = {
        "step": "step0_plm",
        "template_notebook": TEMPLATE,
        "protocol_anchor": "GoEmotions Demszky et al.: batch 16, LR 5e-5, 4 epochs",
        "runs": steps_meta,
        "colab_executed_notebooks": [],
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "colab").mkdir(exist_ok=True)
    (OUT_DIR / "colab" / ".gitkeep").touch()

    meta: list[dict] = []
    for step in STEPS:
        if not step["campaign"].is_file():
            print(f"Missing {step['campaign']}", file=sys.stderr)
            return 1
        campaign = json.loads(step["campaign"].read_text(encoding="utf-8"))
        nb = build_notebook(step, campaign)
        out_path = OUT_DIR / step["out_name"]
        out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out_path.relative_to(ROOT)}")
        agg = campaign["test_aggregate"]
        meta.append(
            {
                "backbone": step["backbone"],
                "record_notebook": f"reference/training_records/step0_plm/{step['out_name']}",
                "campaign_json": str(step["campaign"].relative_to(ROOT)),
                "artifact_zip": step["zip"],
                "protocol": campaign.get("protocol"),
                "seeds": campaign.get("seeds"),
                "test_aggregate": {
                    k: {"mean": v["mean"], "std": v["std"]} for k, v in agg.items()
                },
            }
        )
    write_manifest(meta)
    print(f"Wrote {OUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
