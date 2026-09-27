#!/usr/bin/env python3
"""Compare normal vs zero/shuffle lexicon features at test time for an M3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from hakemer.eval_checkpoint import config_from_run_dir
from hakemer.data import GoEmotionsTorchDataset, load_go_emotions_splits, make_dataloader
from hakemer.model import HAKEMER
from hakemer.train import evaluate, resolve_device, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    config = config_from_run_dir(run_dir)
    if not config.use_m3:
        raise SystemExit("Run directory is not an M3 configuration.")
    set_seed(config.seed)
    dev = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    _, _, test_hf = load_go_emotions_splits()
    ds_kw = dict(
        use_m1=config.use_m1,
        use_m3=config.use_m3,
        lexicon_source=config.lexicon_source,
        lexicon_fusion=config.lexicon_fusion,
        max_phrases=config.max_phrases,
        phrase_max_length=config.phrase_max_length,
    )
    test_ds = GoEmotionsTorchDataset(
        test_hf,
        tokenizer,
        config.max_length,
        config.num_labels,
        config.max_eval_samples,
        **ds_kw,
    )
    loader = make_dataloader(test_ds, args.batch_size, shuffle=False)
    model = HAKEMER(config).to(dev)
    ckpt = run_dir / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    report: dict = {"run_dir": str(run_dir), "modes": {}}
    for mode in (None, "zero", "shuffle"):
        key = "normal" if mode is None else mode
        metrics = evaluate(
            model,
            loader,
            dev,
            threshold=config.decision_threshold,
            lexicon_ablation=mode,
        )
        report["modes"][key] = {k: v for k, v in metrics.items() if k != "per_label"}
    out = args.output or (
        Path("reference/artifacts") / f"lexicon_ablation_{run_dir.name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
