from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from hakemer.config import TrainConfig
from hakemer.data import GoEmotionsTorchDataset, load_go_emotions_splits, make_dataloader
from hakemer.metrics import logits_to_preds, logits_to_probs, multilabel_scores
from hakemer.model import HAKEMER


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_true = np.vstack(all_labels)
    logits = np.vstack(all_logits)
    y_pred = logits_to_preds(logits, threshold=threshold)
    y_score = logits_to_probs(logits)
    return multilabel_scores(y_true, y_pred, y_score)


def train_loop(config: TrainConfig) -> dict:
    config.validate_flags()
    set_seed(config.seed)
    device = resolve_device(config.device)

    out_dir = Path(config.output_dir) / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({**config.__dict__, "learning_rate": config.learning_rate()}, indent=2),
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.backbone)
    train_hf, val_hf, test_hf = load_go_emotions_splits()
    train_ds = GoEmotionsTorchDataset(
        train_hf, tokenizer, config.max_length, config.num_labels, config.max_train_samples
    )
    val_ds = GoEmotionsTorchDataset(
        val_hf, tokenizer, config.max_length, config.num_labels, config.max_eval_samples
    )
    test_ds = GoEmotionsTorchDataset(
        test_hf, tokenizer, config.max_length, config.num_labels, config.max_eval_samples
    )

    train_loader = make_dataloader(train_ds, config.batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, config.batch_size, shuffle=False)
    test_loader = make_dataloader(test_ds, config.batch_size, shuffle=False)

    model = HAKEMER(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate(),
        weight_decay=config.weight_decay,
        eps=config.adam_epsilon,
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    epochs_without_improve = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        val_metrics = evaluate(
            model, val_loader, device, threshold=config.decision_threshold
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch}: loss={row['train_loss']:.4f} "
            f"val_f1_macro={row['val_f1_macro']:.4f} val_f1_micro={row['val_f1_micro']:.4f} "
            f"val_exact_match={row['val_exact_match']:.4f} val_map={row['val_map']:.4f}"
        )
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.early_stopping_patience:
                print(
                    f"Early stopping: no val F1-macro improvement for "
                    f"{config.early_stopping_patience} epoch(s)."
                )
                break

    best_path = out_dir / "best_model.pt"
    if not best_path.is_file():
        raise RuntimeError("No checkpoint saved; training did not improve validation F1-macro.")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(
        model, test_loader, device, threshold=config.decision_threshold
    )
    summary = {
        "best_val_f1_macro": best_val_f1,
        "test": test_metrics,
        "history": history,
        "output_dir": str(out_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Test F1-macro={test_metrics['f1_macro']:.4f} "
        f"F1-micro={test_metrics['f1_micro']:.4f} "
        f"exact_match={test_metrics['exact_match']:.4f} "
        f"mAP={test_metrics['map']:.4f}  -> {out_dir}"
    )
    return summary


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train HAKE-MER (baseline PLM path)")
    p.add_argument("--backbone", default="distilbert-base-uncased", choices=[
        "distilbert-base-uncased", "roberta-base",
    ])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--decision-threshold", type=float, default=0.5)
    p.add_argument("--early-stopping-patience", type=int, default=1)
    args = p.parse_args()
    return TrainConfig(
        backbone=args.backbone,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        device=args.device,
        decision_threshold=args.decision_threshold,
        early_stopping_patience=args.early_stopping_patience,
    )


def main() -> None:
    train_loop(parse_args())


if __name__ == "__main__":
    main()
