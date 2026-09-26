#!/usr/bin/env python3
"""Deprecated helper: prefer ACL/JMLR author URLs; always verify with verify_bibliography.py."""

raise SystemExit(
    "Use manual download → reference/sources/{key}.pdf → biblio.bib → "
    "python3 scripts/verify_bibliography.py && python3 scripts/verify_citation_integrity.py"
)
