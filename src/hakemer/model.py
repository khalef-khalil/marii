from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from hakemer.config import TrainConfig


class AdditiveAttentionPool(nn.Module):
    """Bahdanau-style pooling (eq. attnpool in conception chapter)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=True)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states [B, T, H], attention_mask [B, T]
        scores = self.score(torch.tanh(self.proj(hidden_states))).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        all_pad = attention_mask.sum(dim=-1) == 0
        scores = scores.masked_fill(all_pad.unsqueeze(1), 0.0)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(all_pad.unsqueeze(1), 0.0)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled


class HAKEMER(nn.Module):
    """Flat PLM baseline or M1 hierarchical encoding (M2–M4 not implemented)."""

    def __init__(self, config: TrainConfig):
        super().__init__()
        config.validate_flags()
        self.config = config
        auto_cfg = AutoConfig.from_pretrained(config.backbone)
        self.backbone = AutoModel.from_pretrained(config.backbone, config=auto_cfg)
        hidden = auto_cfg.hidden_size
        dropout_p = getattr(auto_cfg, "hidden_dropout_prob", None)
        if dropout_p is None:
            dropout_p = getattr(auto_cfg, "dropout", 0.1)
        self.dropout = nn.Dropout(dropout_p)
        self.use_m1 = config.use_m1
        if self.use_m1:
            self.word_pool = AdditiveAttentionPool(hidden)
            self.phrase_pool = AdditiveAttentionPool(hidden)
        self.classifier = nn.Linear(hidden, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phrase_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_m1:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            return self.classifier(pooled)

        if phrase_mask is None:
            raise ValueError("phrase_mask is required when use_m1 is True")
        batch_size, num_phrases, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * num_phrases, seq_len)
        flat_mask = attention_mask.reshape(batch_size * num_phrases, seq_len)
        outputs = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
        token_hidden = outputs.last_hidden_state
        phrase_vecs = self.word_pool(token_hidden, flat_mask)
        phrase_vecs = phrase_vecs.view(batch_size, num_phrases, -1)
        doc_vec = self.phrase_pool(phrase_vecs, phrase_mask)
        doc_vec = self.dropout(doc_vec)
        return self.classifier(doc_vec)
