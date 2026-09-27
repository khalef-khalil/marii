#!/usr/bin/env python3
"""LaTeX rows for M2 mechanistic control campaigns."""

from __future__ import annotations

import json
from pathlib import Path


def fmt(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data["test_aggregate"]["f1_macro"]
    return f"${m['mean']:.3f} \\pm {m['std']:.3f}$".replace(".", "{,}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference" / "artifacts"
    slug = "distilbert_base_uncased"
    rows = [
        ("M1+M2 (référence)", f"m1_m2_{slug}_campaign.json"),
        ("M1+M2 sans encodeur inter-émotions", f"m1_m2_no_enc_{slug}_campaign.json"),
        ("M1+M2 sans attention phrase", f"m1_m2_no_xattn_{slug}_campaign.json"),
    ]
    lines: list[str] = []
    for label, fname in rows:
        path = art / fname
        if not path.is_file():
            lines.append(f"{label} & {{---}} \\\\")
        else:
            lines.append(f"{label} & {fmt(path)} \\\\")
        lines.append("\\hline")
    out = art / "m2_controls_snippet.tex"
    body = "\n".join(lines)
    out.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
