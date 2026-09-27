#!/usr/bin/env python3
"""Print trainable parameter counts for HAKE-MER stack variants."""

from __future__ import annotations

from hakemer.config import TrainConfig
from hakemer.model import HAKEMER


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    variants = [
        ("Step~0 PLM (flat)", TrainConfig(use_m1=False)),
        ("+M1", TrainConfig(use_m1=True, use_m2=False)),
        ("+M1+M2", TrainConfig(use_m1=True, use_m2=True)),
        (
            "+M1+M2 (sans encodeur inter-émotions)",
            TrainConfig(
                use_m1=True,
                use_m2=True,
                m2_use_inter_emotion_encoder=False,
            ),
        ),
        (
            "+M1+M2+M3 (NRC)",
            TrainConfig(
                use_m1=True,
                use_m2=True,
                use_m3=True,
                lexicon_source="nrc",
            ),
        ),
        (
            "+M1+M2+M3+M4 (SenticNet)",
            TrainConfig(
                use_m1=True,
                use_m2=True,
                use_m3=True,
                use_m4=True,
                lexicon_source="senticnet",
            ),
        ),
    ]
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tex_tabular_snippet import join_tabular_rows

    row_lines: list[str] = []
    for label, cfg in variants:
        model = HAKEMER(cfg)
        n = count_trainable(model)
        row_lines.append(f"{label} & {n:,} \\\\".replace(",", "{,}"))
    text = join_tabular_rows(row_lines).rstrip("\n")
    out = __import__("pathlib").Path(__file__).resolve().parents[1] / "reference" / "artifacts" / "param_counts_snippet.tex"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
