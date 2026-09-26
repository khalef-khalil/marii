from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from hakemer.config import TrainConfig
from hakemer.model import HAKEMER
from hakemer.train import evaluate, resolve_device, set_seed
from hakemer.data import GoEmotionsTorchDataset, load_go_emotions_splits, make_dataloader
from transformers import AutoTokenizer


def config_from_run_dir(run_dir: Path) -> TrainConfig:
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if "learning_rate" in raw and "lr" not in raw:
        raw["lr"] = raw.pop("learning_rate")
    allowed = {f.name for f in fields(TrainConfig)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return TrainConfig(**filtered)


def eval_run_dir(
    run_dir: Path,
    *,
    device: str = "auto",
    batch_size: int = 16,
) -> dict:
    run_dir = run_dir.resolve()
    ckpt = run_dir / "best_model.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    config = config_from_run_dir(run_dir)
    set_seed(config.seed)
    dev = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    _, _, test_hf = load_go_emotions_splits()
    ds_kw = dict(
        use_m1=config.use_m1,
        use_m3=config.use_m3,
        lexicon_source=config.lexicon_source,
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
    loader = make_dataloader(test_ds, batch_size, shuffle=False)
    model = HAKEMER(config).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    metrics = evaluate(
        model, loader, dev, threshold=config.decision_threshold, include_per_label=True
    )
    out = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "seed": config.seed,
        "test": {k: v for k, v in metrics.items() if k != "per_label"},
        "per_label": metrics["per_label"],
    }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test-set eval from a saved run checkpoint")
    p.add_argument("run_dir", type=Path, help="Directory with config.json and best_model.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", type=Path, default=None, help="Write JSON here")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = eval_run_dir(args.run_dir, device=args.device, batch_size=args.batch_size)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
