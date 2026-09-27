#!/usr/bin/env python3
"""LaTeX rows for lexicon zero/shuffle ablation (mean F1-macro over seeds)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def fmt(m: float, s: float = 0.0) -> str:
    if s > 0:
        return f"${m:.3f} \\pm {s:.3f}$".replace(".", "{,}")
    return f"${m:.3f}$".replace(".", "{,}")


def aggregate_mode(runs: list[dict], mode: str) -> tuple[float, float] | None:
    key = "test" if mode == "normal" else f"test_lexicon_{mode}"
    vals: list[float] = []
    for run in runs:
        if key == "test":
            block = run.get("test")
        else:
            block = run.get(key)
        if block and "f1_macro" in block:
            vals.append(float(block["f1_macro"]))
    if not vals:
        return None
    m = float(statistics.mean(vals))
    s = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return m, s


def resolve_campaign(art: Path, fname: str) -> Path | None:
    candidates = (
        art / fname,
        art / "step_eval_supplements_distilbert" / "campaigns" / fname,
    )
    for path in candidates:
        if not path.is_file():
            continue
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
        if any("test_lexicon_zero" in r for r in runs):
            return path
    for path in candidates:
        if path.is_file():
            return path
    return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference" / "artifacts"
    specs = [
        ("M3 SenticNet", "m1_m2_m3_senticnet_distilbert_base_uncased_campaign.json"),
        ("M3 NRC", "m1_m2_m3_nrc_distilbert_base_uncased_campaign.json"),
        ("M3+M4 NRC", "m1_m2_m3_nrc_m4_distilbert_base_uncased_campaign.json"),
    ]
    lines: list[str] = []
    for stack_label, fname in specs:
        path = resolve_campaign(art, fname)
        if path is None:
            continue
        runs = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
        for mode, row_label in (
            ("normal", "Normal"),
            ("zero", "Lexique nul"),
            ("shuffle", "Lexique permuté"),
        ):
            agg = aggregate_mode(runs, mode)
            label = f"{stack_label} ({row_label})"
            if agg is None:
                lines.append(f"{label} & {{---}} \\\\")
            else:
                m, s = agg
                lines.append(f"{label} & {fmt(m, s)} \\\\")
            lines.append("\\hline")
    out = art / "lexicon_ablation_snippet.tex"
    body = "\n".join(lines) if lines else "% no lexicon ablation data yet\n"
    out.write_text(body + "\n", encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
