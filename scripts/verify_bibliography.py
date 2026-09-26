#!/usr/bin/env python3
"""Check every bib entry with file= has a PDF and first-page title keywords match."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    bib = (root / "biblio.bib").read_text(encoding="utf-8")
    entries = re.split(r"\n(?=@)", bib)
    errors: list[str] = []
    for block in entries:
        if not block.strip().startswith("@"):
            continue
        key_m = re.match(r"@\w+\{([^,]+),", block)
        if not key_m:
            continue
        key = key_m.group(1)
        file_m = re.search(r"file\s*=\s*\{([^}]+)\}", block)
        title_m = re.search(r"title\s*=\s*\{([^}]+)\}", block, re.I)
        if not file_m:
            errors.append(f"{key}: no file= field")
            continue
        pdf = root / file_m.group(1)
        if not pdf.is_file():
            errors.append(f"{key}: missing {pdf}")
            continue
        if title_m:
            title = re.sub(r"\s+", " ", title_m.group(1)).lower()
            snippet = (PdfReader(str(pdf)).pages[0].extract_text() or "").lower()
            # Require at least one significant word from title (len>5) to appear in PDF
            words = [w for w in re.findall(r"[a-z]{6,}", title) if w not in {"using", "approach", "learning"}]
            if words and not any(w in snippet for w in words[:3]):
                errors.append(f"{key}: title may not match PDF (check {pdf.name})")
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print(f"OK: {len([b for b in entries if b.strip().startswith('@')])} entries checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
