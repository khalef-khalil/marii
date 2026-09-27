#!/usr/bin/env python3
"""Unpack Colab zip for RoBERTa M2 mechanistic controls into reference/artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Integrate roberta_m2_controls_campaign.zip")
    p.add_argument("zip_path", type=Path)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
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
        "m1_m2_no_enc_roberta_base_campaign.json",
        "m1_m2_no_xattn_roberta_base_campaign.json",
    )
    found: list[str] = []
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            base = Path(name).name
            payload = zf.read(name)
            if base in expected:
                (art / base).write_bytes(payload)
                found.append(base)
            elif base == "metrics.json":
                parent = Path(name).parent.name
                if "roberta" in parent and ("no_enc" in parent or "no_xattn" in parent):
                    run_dir = runs / parent
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "metrics.json").write_bytes(payload)
    if len(found) < len(expected):
        print(f"Warning: expected {expected}, got {found}", file=sys.stderr)
    for f in found:
        data = json.loads((art / f).read_text(encoding="utf-8"))
        m = data["test_aggregate"]["f1_macro"]
        print(f"{f}: F1-macro test {m['mean']:.4f} ± {m['std']:.4f}")
    render = args.root / "scripts" / "render_m2_controls_tex.py"
    subprocess.run([sys.executable, str(render)], check=True, cwd=args.root)
    print("Next: ./build.sh")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
