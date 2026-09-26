#!/usr/bin/env python3
"""Emit annexe_per_class.tex from per_class_ladder JSON (no paths in output prose)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt_f1(mean: float | None, std: float) -> str:
    if mean is None:
        return "---"
    return f"${mean:.3f} \\pm {std:.3f}$".replace(".", "{,}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "reference/artifacts/per_class_ladder_distilbert_seeds_42_123_456.json"
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    data = json.loads(src.read_text(encoding="utf-8"))
    by_cfg = {c["configuration"]: c for c in data["configurations"]}
    step0 = by_cfg["Step~0 PLM (flat)"]
    m1 = by_cfg["+M1"]
    m2 = by_cfg["+M1+M2"]

    def row(label: str) -> str:
        def pl(cfg: dict) -> tuple[float | None, float, int]:
            for r in cfg["per_label"]:
                if r["label"] == label:
                    return r.get("f1_mean"), float(r.get("f1_std", 0)), int(r["support"])
            return None, 0.0, 0

        m0, s0, sup = pl(step0)
        m1v, s1, _ = pl(m1)
        m2v, s2, _ = pl(m2)
        return (
            f"{label} & {sup} & {fmt_f1(m0, s0)} & {fmt_f1(m1v, s1)} & {fmt_f1(m2v, s2)} \\\\"
            f"\n\\hline\n"
        )

    labels = [r["label"] for r in step0["per_label"]]
    body = "".join(row(lab) for lab in labels)

    tex = rf"""% Auto-generated from campaign aggregate; do not hand-edit rows.
\section*{{Annexe A.~F1 test par émotion (DistilBERT, 3 graines)}}
\addcontentsline{{toc}}{{section}}{{Annexe A.~F1 test par émotion}}

Tableau complet complétant le tableau~\ref{{tab:expe-per-class-f1}} du
chapitre~\ref{{chap:experimentation}} (même protocole, moyenne $\pm$
écart-type sur les graines $42$, $123$, $456$).

{{\small
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{longtable}}{{|l|r|c|c|c|}}
\hline
\textbf{{Émotion}} & \textbf{{Support}} & \textbf{{Step~0}} & \textbf{{+M1}} & \textbf{{+M1+M2}} \\
\hline
\endhead
{body}
\end{{longtable}}
}}
"""
    out = root / "annexe_per_class.tex"
    out.write_text(tex, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
