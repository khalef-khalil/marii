"""Join LaTeX tabular body rows: \\hline after every row, including the last.

Snippets are \\input inside tabular/longtable; the parent must not add a
trailing \\hline after \\input (that causes ``Misplaced \\noalign'').
"""


def join_tabular_rows(row_lines: list[str]) -> str:
    if not row_lines:
        return ""
    parts: list[str] = []
    for row in row_lines:
        parts.append(row)
        parts.append("\\hline")
    return "\n".join(parts) + "\n"
