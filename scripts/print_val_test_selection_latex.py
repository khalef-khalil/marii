#!/usr/bin/env python3
"""Aggregate validation vs test F1-macro from campaign JSON files → LaTeX table."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = float(statistics.mean(values))
    s = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return m, s


def fmt(m: float, s: float) -> str:
    return f"${m:.3f} \\pm {s:.3f}$".replace(".", "{,}")


def load_campaign(path: Path) -> tuple[list[float], list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    vals = [float(r["best_val_f1_macro"]) for r in data["runs"]]
    tests = [float(r["test"]["f1_macro"]) for r in data["runs"]]
    return vals, tests


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    art = root / "reference" / "artifacts"
    dist = "distilbert_base_uncased"
    rob = "roberta_base"
    rows_spec = [
        ("Step~0 PLM", f"baseline_plm_{dist}_campaign.json", f"baseline_plm_{rob}_campaign.json"),
        ("+M1", f"m1_{dist}_campaign.json", f"m1_{rob}_campaign.json"),
        ("+M1+M2", f"m1_m2_{dist}_campaign.json", f"m1_m2_{rob}_campaign.json"),
        (
            "+M1+M2 (sans encodeur inter-émotions)",
            f"m1_m2_no_enc_{dist}_campaign.json",
            None,
        ),
        (
            "+M1+M2 (sans attention phrase)",
            f"m1_m2_no_xattn_{dist}_campaign.json",
            None,
        ),
        (
            "+M1+M2+M3 (NRC)",
            f"m1_m2_m3_nrc_{dist}_campaign.json",
            f"m1_m2_m3_nrc_{rob}_campaign.json",
        ),
        (
            "+M1+M2+M3 (SenticNet)",
            f"m1_m2_m3_senticnet_{dist}_campaign.json",
            f"m1_m2_m3_senticnet_{rob}_campaign.json",
        ),
        (
            "+M1+M2+M3+M4 (SenticNet)",
            f"m1_m2_m3_senticnet_m4_{dist}_campaign.json",
            f"m1_m2_m3_senticnet_m4_{rob}_campaign.json",
        ),
        (
            "+M1+M2+M3+M4 (NRC)",
            f"m1_m2_m3_nrc_m4_{dist}_campaign.json",
            None,
        ),
    ]
    lines: list[str] = []
    for label, dist_name, rob_name in rows_spec:
        dist_p = art / dist_name
        if not dist_p.is_file():
            continue
        dv, dt = load_campaign(dist_p)
        dvm, dvs = mean_std(dv)
        dtm, dts = mean_std(dt)
        if rob_name and (art / rob_name).is_file():
            rv, rt = load_campaign(art / rob_name)
            rvm, rvs = mean_std(rv)
            rtm, rts = mean_std(rt)
            lines.append(
                f"{label} & {fmt(dvm, dvs)} & {fmt(dtm, dts)} & "
                f"{fmt(rvm, rvs)} & {fmt(rtm, rts)} \\\\"
            )
        else:
            lines.append(
                f"{label} & {fmt(dvm, dvs)} & {fmt(dtm, dts)} & {{---}} & {{---}} \\\\"
            )
        lines.append("\\hline")
    out = root / "reference" / "artifacts" / "val_test_selection_snippet.tex"
    body = "\n".join(lines)
    out.write_text(body + "\n", encoding="utf-8")
    print(f"Wrote {out}\n")
    print(body)


if __name__ == "__main__":
    main()
