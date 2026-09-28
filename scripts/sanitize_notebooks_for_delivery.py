#!/usr/bin/env python3
"""Clean Colab notebooks for encadrant-facing delivery (wording, no internal workflow)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

PFE_BANNER = (
    "**PFE — Mariem El Wedani** · SIIVA · Encadrant académique : Sahbi Bahroun · "
    "2025–2026\n\n"
)

REPO_OLD = 'REPO, WORKDIR = "khalef-khalil/marii", Path("/content/marii")'
REPO_NEW = (
    'import os\n'
    'REPO = os.environ.get("GITHUB_REPO", "khalef-khalil/marii")\n'
    'WORKDIR = Path("/content/marii")'
)

SUBS: list[tuple[str, str]] = [
    (
        "2. Run all cells → download zip → archive in repo (see last cell)",
        "2. Exécuter toutes les cellules, puis télécharger le zip et le notebook exécuté (dernière cellule).",
    ),
    (
        "Run all cells → download zip → archive in repo (see last cell)",
        "Exécuter toutes les cellules, puis télécharger le zip et le notebook exécuté (dernière cellule).",
    ),
    (
        "Download executed **.ipynb** + zip and hand off for report integration.",
        "Télécharger le notebook exécuté (.ipynb) et l’archive zip ; conserver les deux pour la traçabilité des runs.",
    ),
    (
        "Send **zip** + executed **.ipynb** for repo integration.",
        "Télécharger le zip et le notebook exécuté ; les archiver avec les autres campagnes (dossier `journaux_colab/` du livrable).",
    ),
    (
        "2. Send **zip** + executed notebook for repo integration (`merge_archived_metrics_into_campaigns.py` + commit snippets).",
        "2. Télécharger le zip et le notebook exécuté ; les conserver pour reproductibilité.",
    ),
    (
        "Executed **.ipynb** + zip → back to repo / agent for H4 NRC row in the report.",
        "Télécharger le notebook exécuté et le zip ; conserver pour la campagne H4 (branche NRC).",
    ),
    (
        "2. Unzip into `reference/artifacts/` on your machine (or send both to your agent).",
        "2. Conserver le zip et le notebook exécuté (métriques agrégées dans le rapport).",
    ),
    (
        "Do **not** use local smoke-test `runs/` (subsampled eval).",
        "Utiliser le protocole complet (GoEmotions, trois graines), pas d’évaluation sur sous-échantillon.",
    ),
]

COMPARE_RES = [
    re.compile(r"Compare test F1-macro to[^\n]*\n?", re.IGNORECASE),
    re.compile(r"Compare to[^\n]*\n?", re.IGNORECASE),
    re.compile(r"Compare cardinality table to[^\n]*\n?", re.IGNORECASE),
]
COMPARE_FR = (
    "**Référence :** comparer les métriques test au chapitre 4 du rapport "
    "(même protocole : batch 16, LR 5e-5, 4 époques, trois graines).\n"
)


def transform_text(text: str) -> str:
    out = text
    for old, new in SUBS:
        out = out.replace(old, new)
    for pat in COMPARE_RES:
        out = pat.sub(COMPARE_FR, out)
    if REPO_OLD in out:
        out = out.replace(REPO_OLD, REPO_NEW)
    return out


def maybe_add_banner(source: list[str]) -> list[str]:
    if not source:
        return source
    joined = "".join(source)
    if "Mariem El" in joined or "PFE —" in joined:
        return source
    if not joined.lstrip().startswith("#"):
        return source
    # Insert banner after first markdown line (title)
    lines: list[str] = []
    inserted = False
    for i, line in enumerate(source):
        lines.append(line)
        if not inserted and line.startswith("# ") and i + 1 < len(source):
            lines.append("\n")
            lines.append(PFE_BANNER)
            inserted = True
    return lines if inserted else [PFE_BANNER] + source


def process_notebook(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in data.get("cells", []):
        src = cell.get("source")
        if isinstance(src, str):
            new = transform_text(src)
            if cell.get("cell_type") == "markdown" and src.startswith("#"):
                new_lines = maybe_add_banner([new] if "\n" not in new else new.splitlines(keepends=True))
                if isinstance(new_lines[0], str) and len(new_lines) == 1:
                    new = new_lines[0]
                else:
                    new = "".join(new_lines)
            if new != src:
                cell["source"] = new
                changed = True
        elif isinstance(src, list):
            joined = "".join(src)
            new_joined = transform_text(joined)
            if cell.get("cell_type") == "markdown" and joined.lstrip().startswith("#"):
                new_src = maybe_add_banner(
                    new_joined.splitlines(keepends=True) if new_joined else src
                )
                new_joined = "".join(new_src)
            if new_joined != joined:
                cell["source"] = (
                    new_joined.splitlines(keepends=True)
                    if new_joined
                    else src
                )
                changed = True
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def main() -> int:
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not paths:
        print("No notebooks found", file=sys.stderr)
        return 1
    n = sum(1 for p in paths if process_notebook(p))
    print(f"Updated {n}/{len(paths)} notebooks under {NOTEBOOKS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
