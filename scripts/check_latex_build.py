#!/usr/bin/env python3
"""Fail CI/local build when LaTeX log contains hard errors or bad table snippets."""

from __future__ import annotations

import re
import sys
from pathlib import Path

LOG_PATTERNS = (
    re.compile(r"! LaTeX Error"),
    re.compile(r"! Misplaced \\noalign"),
    re.compile(r"! Emergency stop"),
)

SNIPPET_DIR = "reference/artifacts"


def check_log(log_path: Path) -> list[str]:
    if not log_path.is_file():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    issues: list[str] = []
    for pat in LOG_PATTERNS:
        for line in text.splitlines():
            if pat.search(line):
                issues.append(line.strip())
    return issues


def check_snippets(root: Path) -> list[str]:
    issues: list[str] = []
    art = root / SNIPPET_DIR
    if not art.is_dir():
        return issues
    for path in sorted(art.glob("*_snippet.tex")):
        lines = [
            ln.rstrip()
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if not lines:
            issues.append(f"{path}: empty snippet")
            continue
        if lines[-1].strip() != r"\hline":
            issues.append(f"{path}: must end with \\hline (closing rule inside snippet)")
        for i, ln in enumerate(lines):
            if ln.strip() == r"\hline":
                continue
            if not ln.rstrip().endswith(r"\\"):
                issues.append(f"{path}:{i + 1}: tabular row must end with \\\\")
    return issues


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snippets-only",
        action="store_true",
        help="Validate artifact snippets only (pre-build).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    log = root / "build" / "main.log"
    issues = check_snippets(root)
    if not args.snippets_only:
        issues.extend(f"log: {line}" for line in check_log(log))
    if issues:
        print("LaTeX build check failed:", file=sys.stderr)
        for item in issues[:40]:
            print(f"  - {item}", file=sys.stderr)
        if len(issues) > 40:
            print(f"  ... and {len(issues) - 40} more", file=sys.stderr)
        return 1
    print("LaTeX build check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
