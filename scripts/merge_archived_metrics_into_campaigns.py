#!/usr/bin/env python3
"""Copy test fields from archived metrics.json into matching campaign JSON runs."""

from __future__ import annotations

import json
from pathlib import Path

FIELDS_ON_RUN = ("test_lexicon_zero", "test_lexicon_shuffle")
TEST_SUBKEYS = ("by_gold_cardinality",)


def run_key(run: dict) -> tuple[int | None, str | None]:
    return run.get("seed"), run.get("output_dir")


def merge_test_block(target: dict, source_test: dict) -> bool:
    changed = False
    for sub in TEST_SUBKEYS:
        if sub in source_test and sub not in target:
            target[sub] = source_test[sub]
            changed = True
    return changed


def find_metrics(
    run: dict,
    metrics_by_key: dict[tuple[int | None, str | None], dict],
    by_basename: dict[str, dict],
) -> dict | None:
    seed, out = run_key(run)
    if out:
        m = metrics_by_key.get((seed, out)) or metrics_by_key.get((seed, Path(out).name))
        if m:
            return m
        base = Path(out).name
        if base in by_basename and (seed is None or by_basename[base].get("_seed") == seed):
            return by_basename[base]
    if seed is not None:
        for base, m in by_basename.items():
            if m.get("_seed") == seed and f"_seed{seed}_" in base:
                return m
    return None


def patch_campaign(
    campaign_path: Path,
    metrics_by_key: dict[tuple[int | None, str | None], dict],
    by_basename: dict[str, dict],
) -> bool:
    data = json.loads(campaign_path.read_text(encoding="utf-8"))
    changed = False
    for run in data.get("runs", []):
        metrics = find_metrics(run, metrics_by_key, by_basename)
        if metrics is None:
            continue
        test_src = metrics.get("test") or {}
        test_dst = run.setdefault("test", {})
        if merge_test_block(test_dst, test_src):
            changed = True
        for field in FIELDS_ON_RUN:
            if field in metrics and field not in run:
                run[field] = metrics[field]
                changed = True
    if changed:
        campaign_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Patched {campaign_path.name}")
    return changed


def index_metrics(root: Path) -> tuple[dict[tuple[int | None, str | None], dict], dict[str, dict]]:
    out: dict[tuple[int | None, str | None], dict] = {}
    by_basename: dict[str, dict] = {}
    for path in sorted(root.glob("reference/artifacts/**/metrics.json")):
        _add_metrics(path, out, by_basename)
    runs = root / "runs"
    if runs.is_dir():
        for path in sorted(runs.glob("**/metrics.json")):
            _add_metrics(path, out, by_basename)
    return out, by_basename


def _add_metrics(
    path: Path,
    out: dict[tuple[int | None, str | None], dict],
    by_basename: dict[str, dict],
) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    out_dir = data.get("output_dir")
    basename = path.parent.name
    seed = None
    name = Path(out_dir).name if out_dir else basename
    if "_seed" in name:
        part = name.split("_seed", 1)[1]
        seed_str = part.split("_", 1)[0]
        if seed_str.isdigit():
            seed = int(seed_str)
    data["_seed"] = seed
    out[(seed, out_dir)] = data
    out[(seed, name)] = data
    by_basename[name] = data
    by_basename[basename] = data


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics_by_key, by_basename = index_metrics(root)
    art = root / "reference" / "artifacts"
    n = 0
    for path in sorted(art.glob("*_campaign.json")):
        if patch_campaign(path, metrics_by_key, by_basename):
            n += 1
    print(f"Updated {n} campaign file(s).")


if __name__ == "__main__":
    main()
