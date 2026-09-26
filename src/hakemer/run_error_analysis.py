from __future__ import annotations

import argparse
from pathlib import Path

from hakemer.error_analysis import write_error_analysis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Error analysis JSON from a run checkpoint")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = write_error_analysis(
        args.run_dir,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(out)


if __name__ == "__main__":
    main()
