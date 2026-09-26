from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from hakemer.lexicon import document_lexicon_vector
from hakemer.segmentation import split_phrases


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
        *,
        use_m1: bool = False,
        use_m3: bool = False,
        lexicon_source: str = "none",
        max_phrases: int = 4,
        phrase_max_length: int = 32,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels
        self.use_m1 = use_m1
        self.use_m3 = use_m3
        self.lexicon_source = lexicon_source
        self.max_phrases = max_phrases
        self.phrase_max_length = phrase_max_length
        n = len(hf_split) if max_samples is None else min(max_samples, len(hf_split))
        self.rows = hf_split.select(range(n))

    def __len__(self) -> int:
        return len(self.rows)

    def _labels_tensor(self, row) -> torch.Tensor:
        labels = torch.zeros(self.num_labels, dtype=torch.float32)
        for label_id in row["labels"]:
            labels[int(label_id)] = 1.0
        return labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        if not self.use_m1:
            enc = self.tokenizer(
                row["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": self._labels_tensor(row),
            }

        phrases = split_phrases(row["text"], self.max_phrases)
        phrase_ids: list[torch.Tensor] = []
        phrase_masks: list[torch.Tensor] = []
        phrase_valid: list[float] = []
        for slot in range(self.max_phrases):
            text = phrases[slot] if slot < len(phrases) else ""
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.phrase_max_length,
                padding="max_length",
                return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
            has_tokens = bool(text.strip()) and mask.sum().item() > 0
            if not has_tokens:
                mask = torch.zeros_like(mask)
            phrase_ids.append(ids)
            phrase_masks.append(mask)
            phrase_valid.append(1.0 if has_tokens else 0.0)

        item = {
            "input_ids": torch.stack(phrase_ids, dim=0),
            "attention_mask": torch.stack(phrase_masks, dim=0),
            "phrase_mask": torch.tensor(phrase_valid, dtype=torch.float32),
            "labels": self._labels_tensor(row),
        }
        if self.use_m3:
            vec = document_lexicon_vector(row["text"], self.lexicon_source)
            item["lexicon_features"] = torch.tensor(vec, dtype=torch.float32)
        return item


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
