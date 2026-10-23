#!/usr/bin/env python3
"""Figures for the defense deck (French labels, large fonts), drawn from archived campaign JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "reference/artifacts"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

NAVY, GREEN, ORANGE, GRAY = "#14284B", "#2F7A5A", "#D9822B", "#7A8798"
BLUE1, BLUE2, BLUE3 = "#5B7BA6", "#86A3C7", "#B4C4DA"

plt.rcParams.update(
    {
        "font.family": ["Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def fr(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def macro(fn: str) -> tuple[float, float]:
    d = json.loads((ART / fn).read_text(encoding="utf-8"))["test_aggregate"]["f1_macro"]
    return float(d["mean"]), float(d["std"])


def commas(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def fig_goemotions_cardinality() -> None:
    # Gold-label counts on the test split (same counts as the cardinality table of the report).
    labels = ["1", "2", "3", "4+"]
    n = [4590, 774, 61, 2]
    total = sum(n)
    assert total == 5427
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    bars = ax.bar(labels, [100 * v / total for v in n], color=[GRAY, BLUE1, BLUE1, BLUE1], edgecolor="none")
    for b, v in zip(bars, n):
        pct = 100 * v / total
        txt = f"{fr(pct, 1)} %" if pct >= 1 else f"{fr(pct, 2)} %"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, txt, ha="center", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Nombre d'étiquettes gold par exemple")
    ax.set_ylabel("Part du split test (%)")
    ax.set_title("GoEmotions : multi-étiquetage", fontsize=14, color=NAVY, fontweight="bold")
    commas(ax)
    save(fig, "goemotions_cardinality")


def fig_ablation() -> None:
    specs = [
        ("Step 0\n(flat)", "baseline_plm_distilbert_base_uncased_campaign.json", GRAY),
        ("+M1", "m1_distilbert_base_uncased_campaign.json", ORANGE),
        ("+M1+M2", "m1_m2_distilbert_base_uncased_campaign.json", GREEN),
        ("+M3\nNRC", "m1_m2_m3_nrc_distilbert_base_uncased_campaign.json", BLUE1),
        ("+M3\nSenticNet", "m1_m2_m3_senticnet_distilbert_base_uncased_campaign.json", BLUE2),
        ("+M4\nSenticNet", "m1_m2_m3_senticnet_m4_distilbert_base_uncased_campaign.json", BLUE3),
    ]
    labels, means, stds, cols = [], [], [], []
    for lab, fn, col in specs:
        mu, sd = macro(fn)
        labels.append(lab)
        means.append(mu)
        stds.append(sd)
        cols.append(col)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.bar(x, means, yerr=stds, capsize=5, color=cols, edgecolor="none", error_kw={"elinewidth": 1.6})
    for xi, mu, sd in zip(x, means, stds):
        ax.text(xi, mu + sd + 0.0025, fr(mu), ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.42, 0.54)
    ax.set_ylabel("F1-macro test (axe tronqué)")
    commas(ax)
    save(fig, "ablation_distilbert")


def fig_backbone() -> None:
    rows = [
        ("Step 0 (flat)", "baseline_plm_distilbert_base_uncased_campaign.json", "baseline_plm_roberta_base_campaign.json"),
        ("+M1", "m1_distilbert_base_uncased_campaign.json", "m1_roberta_base_campaign.json"),
        ("+M1+M2", "m1_m2_distilbert_base_uncased_campaign.json", "m1_m2_roberta_base_campaign.json"),
    ]
    x = np.arange(len(rows))
    w = 0.36
    dm, ds, rm, rs = [], [], [], []
    for _, d, r in rows:
        a, b = macro(d)
        c, e = macro(r)
        dm.append(a), ds.append(b), rm.append(c), rs.append(e)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.bar(x - w / 2, dm, w, yerr=ds, capsize=4, color=BLUE2, label="DistilBERT", error_kw={"elinewidth": 1.5})
    ax.bar(x + w / 2, rm, w, yerr=rs, capsize=4, color=NAVY, label="RoBERTa", error_kw={"elinewidth": 1.5})
    for xi, v, s in zip(x - w / 2, dm, ds):
        ax.text(xi, v + s + 0.002, fr(v), ha="center", fontsize=11)
    for xi, v, s in zip(x + w / 2, rm, rs):
        ax.text(xi, v + s + 0.002, fr(v), ha="center", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=12)
    ax.set_ylim(0.44, 0.545)
    ax.set_ylabel("F1-macro test (axe tronqué)")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    commas(ax)
    save(fig, "backbone")


def fig_per_class() -> None:
    data = json.loads((ART / "per_class_ladder_distilbert_seeds_42_123_456.json").read_text(encoding="utf-8"))
    by = {c["configuration"]: c for c in data["configurations"]}
    s0 = {r["label"]: r["f1_mean"] for r in by["Step~0 PLM (flat)"]["per_label"]}
    m2 = {r["label"]: r["f1_mean"] for r in by["+M1+M2"]["per_label"]}
    pick = ["grief", "relief", "desire", "annoyance", "pride", "curiosity", "remorse",
            "disappointment", "caring", "disgust", "embarrassment", "nervousness"]
    x = np.arange(len(pick))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.bar(x - w / 2, [s0[l] for l in pick], w, color=GRAY, label="Step 0 (flat)")
    ax.bar(x + w / 2, [m2[l] for l in pick], w, color=GREEN, label="+M1+M2")
    ax.set_xticks(x)
    ax.set_xticklabels(pick, rotation=40, ha="right", fontsize=10.5)
    ax.set_ylabel("F1 test par émotion")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False, fontsize=11, loc="upper right", ncol=2)
    commas(ax)
    save(fig, "per_class")


def fig_errors() -> None:
    agg = json.loads((ART / "error_analysis_m1_m2_aggregate.json").read_text(encoding="utf-8"))
    pairs = agg["top_confusion_pairs_aggregate"][:10]

    def tr(p: str) -> str:
        w = p.split()
        if w[0] == "FN":  # FN a pred b
            return f"{w[1]} manqué (prédit : {w[3]})"
        return f"{w[1]} prédit à tort (gold : {w[3]})"  # FP a gold b

    labels = [tr(p["pattern"]) for p in pairs][::-1]
    counts = [p["count"] for p in pairs][::-1]
    fig, ax = plt.subplots(figsize=(7.3, 4.1))
    ax.barh(range(len(labels)), counts, color=ORANGE)
    for i, c in enumerate(counts):
        ax.text(c + 3, i, str(c), va="center", fontsize=10.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Occurrences (+M1+M2, 3 graines cumulées)")
    ax.set_xlim(0, max(counts) * 1.12)
    ax.grid(axis="y", visible=False)
    save(fig, "errors")


def main() -> None:
    fig_goemotions_cardinality()
    fig_ablation()
    fig_backbone()
    fig_per_class()
    fig_errors()
    print("Figures written to", OUT)


if __name__ == "__main__":
    main()
