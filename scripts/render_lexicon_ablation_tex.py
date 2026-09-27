#!/usr/bin/env python3
"""LaTeX rows for lexicon zero/shuffle ablation from campaign run summaries."""

from __future__ import annotations

import json
from pathlib import Path


def fmt(m: float) -> str:
    return f"${m:.3f}$".replace(".", "{,}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "reference" / "artifacts" / "m1_m2_m3_senticnet_distilbert_base_uncased_campaign.json"
    if not path.is_file():
        print("Missing SenticNet M3 campaign JSON")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    run = data["runs"][0]
    normal = run["test"]
    lines: list[str] = []
    lines.append(
        f"Normal & {fmt(normal['f1_macro'])} & {fmt(normal['map'])} \\\\"
    )
    lines.append("\\hline")
    for mode, label in (("test_lexicon_zero", "Lexique nul"), ("test_lexicon_shuffle", "Lexique permuté")):
        if mode in run:
            block = run[mode]
            lines.append(f"{label} & {fmt(block['f1_macro'])} & {fmt(block['map'])} \\\\")
            lines.append("\\hline")
        else:
            lines.append(f"{label} & {{---}} & {{---}} \\\\")
            lines.append("\\hline")
    out = root / "reference" / "artifacts" / "lexicon_ablation_snippet.tex"
    body = "\n".join(lines)
    out.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
