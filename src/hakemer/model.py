from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from hakemer.config import TrainConfig


class HAKEMER(nn.Module):
    """Flat PLM baseline today; M1–M4 gated by flags (stubs raise until implemented)."""

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
        self.classifier = nn.Linear(hidden, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
