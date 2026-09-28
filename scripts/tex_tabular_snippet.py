"""Join LaTeX tabular body rows: \\hline between rows, not after the last.

Snippets are \\input inside tabular; the parent line must end with ``\\%''
before \\hline (see chap_04.tex).
"""


def join_tabular_rows(row_lines: list[str]) -> str:
    if not row_lines:
        return ""
    parts: list[str] = []
    last = len(row_lines) - 1
    for i, row in enumerate(row_lines):
        parts.append(row)
        if i < last:
            parts.append("\\hline")
    return "\n".join(parts) + "\n"
