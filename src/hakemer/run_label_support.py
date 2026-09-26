from __future__ import annotations

import argparse
import json
from pathlib import Path

from hakemer.data import load_go_emotions_splits
from hakemer.metrics import go_emotions_label_names


def main() -> None:
    p = argparse.ArgumentParser(description="GoEmotions simplified label counts per split")
    p.add_argument("--artifact-dir", type=Path, default=Path("reference/artifacts"))
    args = p.parse_args()
    names = go_emotions_label_names()
    train, val, test = load_go_emotions_splits()
    out = {"label_names": names, "splits": {}}
    for split_name, split in [("train", train), ("validation", val), ("test", test)]:
        counts = [0] * len(names)
        for row in split:
            for lid in row["labels"]:
                counts[int(lid)] += 1
        out["splits"][split_name] = [
            {"label": names[i], "count": counts[i], "n_examples": len(split)}
            for i in range(len(names))
        ]
    path = args.artifact_dir / "goemotions_simplified_label_counts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
