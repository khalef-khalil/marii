#!/usr/bin/env python3
"""Emit a LaTeX fragment for chap_04 from error_analysis_m1_m2_aggregate.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def esc(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "reference/artifacts/error_analysis_m1_m2_aggregate.json"
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    if not src.is_file():
        print(f"Missing {src}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = []
    for row in data.get("top_confusion_pairs_aggregate", [])[:10]:
        pat = esc(row["pattern"])
        rows.append(f"{pat} & {int(row['count'])} \\\\ \n\\hline\n")
    body = "".join(rows)

    examples = ""
    for rep in data.get("reports", [])[:1]:
        rare = rep.get("rare_label_examples", {})
        for label in ("grief", "relief", "neutral"):
            cases = rare.get(label, [])[:2]
            for ex in cases:
                gold = ", ".join(ex["gold"])
                pred = ", ".join(ex["pred"])
                text = esc(ex["text"][:120])
                examples += (
                    f"\\item \\textbf{{{label}}} ({ex['kind']}) : "
                    f"« {text}… » ; or : \\{{{gold}\\}} ; prédit : \\{{{pred}\\}}.\n"
                )

    tex = rf"""
% Auto-generated; paste into sec:expe-error-analysis or \input{{error_analysis_snippet.tex}}
\begin{{table}}[H]
\centering
\caption{{Paires d'erreurs les plus fréquentes (+M1+M2, DistilBERT, agrégat sur graines)}}
\label{{tab:expe-error-pairs}}
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{|p{{0.62\textwidth}}|r|}}
\hline
\textbf{{Motif (référence vs prédiction)}} & \textbf{{Occurrences}} \\
\hline
{body}
\end{{tabular}}
\end{{table}}

\paragraph{{Illustrations (extrait).}}
\begin{{itemize}}
{examples}
\end{{itemize}}
"""
    out = root / "error_analysis_snippet.tex"
    out.write_text(tex.strip() + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
