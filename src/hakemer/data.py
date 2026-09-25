from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


def load_go_emotions_splits():
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    return ds["train"], ds["validation"], ds["test"]


class GoEmotionsTorchDataset(Dataset):
    def __init__(
        self,
        hf_split,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        num_labels: int,
        max_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels
        n = len(hf_split) if max_samples is None else min(max_samples, len(hf_split))
        self.rows = hf_split.select(range(n))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.zeros(self.num_labels, dtype=torch.float32)
        for label_id in row["labels"]:
            labels[int(label_id)] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
