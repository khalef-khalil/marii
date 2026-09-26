#!/usr/bin/env python3
"""Ensure every \\cite{key} in report .tex has bib entry + local PDF; run bib PDF check."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    tex_files = [
        p
        for p in root.glob("*.tex")
        if p.name != "main.tex" or True
    ]
    tex_files = [p for p in root.glob("*.tex") if "mini-projet" not in str(p)]
    tex_files += list(root.glob("chap_*.tex"))
    tex_files += [root / "introduction.tex", root / "conclusion.tex"]
    tex_files = list(dict.fromkeys(tex_files))

    cite_keys: set[str] = set()
    for tex in tex_files:
        if not tex.is_file():
            continue
        text = tex.read_text(encoding="utf-8")
        for block in re.findall(r"\\cite\{([^}]+)\}", text):
            for key in block.split(","):
                cite_keys.add(key.strip())

    bib = (root / "biblio.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing_bib = sorted(cite_keys - bib_keys)
    orphan_bib = sorted(bib_keys - cite_keys)

    errors: list[str] = []
    if missing_bib:
        errors.append(f"\\cite keys missing from biblio.bib: {', '.join(missing_bib)}")
    for key in cite_keys:
        m = re.search(rf"@{{\w+\{{{re.escape(key)},[\s\S]*?file\s*=\s*{{([^}}]+)}}", bib)
        if not m:
            m = re.search(
                rf"@\w+\{{{re.escape(key)},[\s\S]*?file\s*=\s*{{([^}}]+)}}",
                bib,
            )
        if not m:
            errors.append(f"{key}: cited but no file= in bib entry")
            continue
        pdf = root / m.group(1)
        if not pdf.is_file():
            errors.append(f"{key}: missing PDF {pdf}")

    rc = subprocess.run(
        [sys.executable, str(root / "scripts" / "verify_bibliography.py")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        errors.append("verify_bibliography.py failed:")
        errors.extend(rc.stderr.strip().splitlines())

    print(f"Cite keys in report: {len(cite_keys)}")
    print(f"Bib entries: {len(bib_keys)} (uncited in body: {len(orphan_bib)})")
    if orphan_bib:
        print(f"  Uncited keys (OK if intentional): {', '.join(orphan_bib[:20])}"
              + (" ..." if len(orphan_bib) > 20 else ""))

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("OK: all citations have verified bib entries and PDFs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
