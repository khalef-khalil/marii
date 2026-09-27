#!/usr/bin/env python3
"""LaTeX rows for M2 mechanistic control campaigns (per backbone)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tex_tabular_snippet import join_tabular_rows

BACKBONES = (
    ("distilbert_base_uncased", "m2_controls_snippet.tex"),
    ("roberta_base", "m2_controls_roberta_snippet.tex"),
)


def fmt(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data["test_aggregate"]["f1_macro"]
    return f"${m['mean']:.3f} \\pm {m['std']:.3f}$".replace(".", "{,}")


def render_for_slug(art: Path, slug: str) -> list[str]:
    rows = [
        ("M1+M2 (référence)", f"m1_m2_{slug}_campaign.json"),
        ("M1+M2 sans encodeur inter-émotions", f"m1_m2_no_enc_{slug}_campaign.json"),
        ("M1+M2 sans attention phrase", f"m1_m2_no_xattn_{slug}_campaign.json"),
    ]
    row_lines: list[str] = []
    for label, fname in rows:
        path = art / fname
        if not path.is_file():
            row_lines.append(f"{label} & {{---}} \\\\")
        else:
            row_lines.append(f"{label} & {fmt(path)} \\\\")
    return join_tabular_rows(row_lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference" / "artifacts"
    for slug, out_name in BACKBONES:
        body = render_for_slug(art, slug).rstrip("\n")
        out = art / out_name
        out.write_text(body + "\n", encoding="utf-8")
        print(f"=== {out_name} ===")
        print(body)


if __name__ == "__main__":
    main()
