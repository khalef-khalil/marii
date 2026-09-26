#!/usr/bin/env python3
"""Unpack Colab zip of RoBERTa M3/M4 campaign JSON (+ optional metrics) into reference/artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Integrate roberta_m3_m4_campaign.zip from Colab")
    p.add_argument("zip_path", type=Path, help="Downloaded zip path")
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    args = p.parse_args()
    zpath = args.zip_path.expanduser().resolve()
    if not zpath.is_file():
        print(f"Missing {zpath}", file=sys.stderr)
        return 1
    art = args.root / "reference" / "artifacts"
    runs = args.root / "runs"
    art.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    expected = (
        "m1_m2_m3_nrc_roberta_base_campaign.json",
        "m1_m2_m3_senticnet_roberta_base_campaign.json",
        "m1_m2_m3_senticnet_m4_roberta_base_campaign.json",
    )
    found: list[str] = []
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            base = Path(name).name
            payload = zf.read(name)
            if base.endswith("_campaign.json") and "roberta" in base:
                (art / base).write_bytes(payload)
                found.append(base)
            elif base == "metrics.json":
                parent = Path(name).parent.name
                if "roberta" in parent:
                    run_dir = runs / parent
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "metrics.json").write_bytes(payload)
    missing = [s for s in expected if s not in found]
    if missing:
        print("Warning: expected JSON not in zip:", ", ".join(missing), file=sys.stderr)
    for f in found:
        data = json.loads((art / f).read_text(encoding="utf-8"))
        m = data["test_aggregate"]["f1_macro"]
        print(f"{f}: F1-macro test {m['mean']:.4f} ± {m['std']:.4f}")
    print("Run: python3 scripts/render_roberta_ablation_tex.py && ./build.sh build")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
