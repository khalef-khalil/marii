#!/usr/bin/env python3
"""Schematic figures for ch.2 (cite original papers; not extracted bitmaps)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "img/lit"
OUT.mkdir(parents=True, exist_ok=True)


def fig_transformer_encoder() -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.0, "Tokens"),
        (2.0, 1.0, "Embedding"),
        (3.7, 1.0, "Encoder × N"),
        (5.8, 1.0, "Self-attention"),
        (7.5, 1.0, "Pool / [CLS]"),
    ]
    for x, y, text in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), 1.4, 0.9, boxstyle="round,pad=0.05", linewidth=1.2, edgecolor="black", facecolor="#E8EEF7"
        )
        ax.add_patch(rect)
        ax.text(x + 0.7, y + 0.45, text, ha="center", va="center", fontsize=9)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 1.4
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1, 1.45), xytext=(x0, 1.45), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("Schéma encodeur Transformer (couches empilées)", fontsize=11)
    fig.savefig(OUT / "lit_transformer_encoder.pdf")
    fig.savefig(OUT / "lit_transformer_encoder.png")
    plt.close(fig)


def fig_han() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    levels = [
        (1, 1, "Mots", "#F5E6D3"),
        (1, 3, "Phrases", "#D6EAF8"),
        (1, 5, "Document", "#D5F5E3"),
    ]
    for x, y, label, color in levels:
        rect = mpatches.FancyBboxPatch(
            (x, y), 8, 1.2, boxstyle="round,pad=0.06", linewidth=1.2, edgecolor="black", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(5, y + 0.6, f"{label} → attention → vecteur de niveau", ha="center", va="center", fontsize=10)
        if y > 1:
            ax.annotate("", xy=(5, y), xytext=(5, y - 0.8), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("Attention hiérarchique document (mot → phrase → document)", fontsize=11)
    fig.savefig(OUT / "lit_han_schematic.pdf")
    fig.savefig(OUT / "lit_han_schematic.png")
    plt.close(fig)


def fig_goemotions_multilabel_stats() -> None:
    """Bar chart from statistics reported in Demszky et al. (2020), table in ch.2."""
    labels = ["1 label", "2 labels", "3 labels", "4+ labels"]
    pct = [83, 15, 2, 0.2]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(labels, pct, color="#8172B3", edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Part des exemples (%)")
    ax.set_title("GoEmotions: multi-étiquetage (Demszky et al., 2020)")
    fig.savefig(OUT / "lit_goemotions_multilabel.pdf")
    fig.savefig(OUT / "lit_goemotions_multilabel.png")
    plt.close(fig)


def main() -> None:
    fig_transformer_encoder()
    fig_han()
    fig_goemotions_multilabel_stats()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
