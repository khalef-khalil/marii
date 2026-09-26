#!/usr/bin/env python3
"""Generate report figures under img/expe/ from archived campaign JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "reference/artifacts"
OUT = ROOT / "img/expe"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def load_macro(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data["test_aggregate"]["f1_macro"]
    return float(m["mean"]), float(m["std"])


def fig_ablation_ladder() -> None:
    specs = [
        ("Step~0", "baseline_plm_distilbert_base_uncased_campaign.json"),
        ("+M1", "m1_distilbert_base_uncased_campaign.json"),
        ("+M1+M2", "m1_m2_distilbert_base_uncased_campaign.json"),
        ("+M3 NRC", "m1_m2_m3_nrc_distilbert_base_uncased_campaign.json"),
        ("+M3 SenticNet", "m1_m2_m3_senticnet_distilbert_base_uncased_campaign.json"),
        ("+M4", "m1_m2_m3_senticnet_m4_distilbert_base_uncased_campaign.json"),
    ]
    labels, means, stds = [], [], []
    for lab, fn in specs:
        p = ART / fn
        if not p.is_file():
            continue
        mu, sd = load_macro(p)
        labels.append(lab.replace("~", " "))
        means.append(mu)
        stds.append(sd)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    ax.bar(x, means, yerr=stds, capsize=4, color=colors[: len(labels)], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("F1-macro (test)")
    ax.set_ylim(0.42, 0.54)
    ax.set_title("DistilBERT: ablation ladder (mean ± std, 3 seeds)")
    fig.savefig(OUT / "expe_ablation_macro.pdf")
    fig.savefig(OUT / "expe_ablation_macro.png")
    plt.close(fig)


def fig_backbone_comparison() -> None:
    rows = [
        ("Step~0", "baseline_plm_distilbert_base_uncased_campaign.json", "baseline_plm_roberta_base_campaign.json"),
        ("+M1", "m1_distilbert_base_uncased_campaign.json", "m1_roberta_base_campaign.json"),
        ("+M1+M2", "m1_m2_distilbert_base_uncased_campaign.json", "m1_m2_roberta_base_campaign.json"),
    ]
    labels = [r[0].replace("~", " ") for r in rows]
    d_mean, d_std, r_mean, r_std = [], [], [], []
    for _, df, rf in rows:
        dm, ds = load_macro(ART / df)
        rm, rs = load_macro(ART / rf)
        d_mean.append(dm)
        d_std.append(ds)
        r_mean.append(rm)
        r_std.append(rs)
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w / 2, d_mean, w, yerr=d_std, label="DistilBERT", capsize=3, color="#4C72B0")
    ax.bar(x + w / 2, r_mean, w, yerr=r_std, label="RoBERTa", capsize=3, color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1-macro (test)")
    ax.set_ylim(0.44, 0.54)
    ax.legend()
    ax.set_title("Multi-backbone control (+M1, +M1+M2)")
    fig.savefig(OUT / "expe_backbone_macro.pdf")
    fig.savefig(OUT / "expe_backbone_macro.png")
    plt.close(fig)


def fig_label_support() -> None:
    data = json.loads(
        (ART / "per_class_ladder_distilbert_seeds_42_123_456.json").read_text(encoding="utf-8")
    )
    step0 = next(c for c in data["configurations"] if "Step" in c["configuration"])
    rows = sorted(step0["per_label"], key=lambda r: r["support"], reverse=True)
    labels = [r["label"] for r in rows]
    support = [int(r["support"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))
    ax.barh(y, support, color="#55A868", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Positive examples (GoEmotions test)")
    ax.set_title("Long-tail label support (test split)")
    fig.savefig(OUT / "expe_label_support.pdf")
    fig.savefig(OUT / "expe_label_support.png")
    plt.close(fig)


def fig_error_pairs() -> None:
    agg = json.loads((ART / "error_analysis_m1_m2_aggregate.json").read_text(encoding="utf-8"))
    pairs = agg["top_confusion_pairs_aggregate"][:10]
    labels = [p["pattern"] for p in pairs][::-1]
    counts = [p["count"] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(range(len(labels)), counts, color="#DD8452", edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Aggregated co-occurrence count (+M1+M2, 3 seeds)")
    ax.set_title("Top error motifs (test)")
    fig.savefig(OUT / "expe_error_pairs.pdf")
    fig.savefig(OUT / "expe_error_pairs.png")
    plt.close(fig)


def fig_per_class_delta() -> None:
    data = json.loads(
        (ART / "per_class_ladder_distilbert_seeds_42_123_456.json").read_text(encoding="utf-8")
    )
    by = {c["configuration"]: c for c in data["configurations"]}
    s0 = {r["label"]: r["f1_mean"] for r in by["Step~0 PLM (flat)"]["per_label"]}
    m2 = {r["label"]: r["f1_mean"] for r in by["+M1+M2"]["per_label"]}
    labels = sorted(s0.keys(), key=lambda l: s0[l] - m2.get(l, 0))
    pick = labels[:8] + labels[-4:]
    pick = list(dict.fromkeys(pick))
    x = np.arange(len(pick))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w / 2, [s0[l] for l in pick], w, label="Step~0", color="#4C72B0")
    ax.bar(x + w / 2, [m2[l] for l in pick], w, label="+M1+M2", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(pick, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1 (test, mean over seeds)")
    ax.set_ylim(0, 0.9)
    ax.legend()
    ax.set_title("Per-class F1: flat vs best stack (selected labels)")
    fig.savefig(OUT / "expe_per_class_selected.pdf")
    fig.savefig(OUT / "expe_per_class_selected.png")
    plt.close(fig)


def fig_training_curves() -> None:
    data = json.loads((ART / "m1_m2_distilbert_base_uncased_campaign.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in data["runs"]:
        hist = run["history"]
        epochs = [h["epoch"] for h in hist]
        val = [h["val_f1_macro"] for h in hist]
        ax.plot(epochs, val, marker="o", label=f"seed {run['seed']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1-macro")
    ax.set_title("+M1+M2 DistilBERT: validation macro per seed")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "expe_m1_m2_val_curves.pdf")
    fig.savefig(OUT / "expe_m1_m2_val_curves.png")
    plt.close(fig)


def main() -> None:
    fig_ablation_ladder()
    fig_backbone_comparison()
    fig_label_support()
    fig_error_pairs()
    fig_per_class_delta()
    fig_training_curves()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
