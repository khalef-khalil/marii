from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from hakemer.config import TrainConfig
from hakemer.lexicon import lexicon_dim

MHA_NUM_HEADS = 8
EMOTION_ENCODER_LAYERS = 1


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
        scores = self.score(torch.tanh(self.proj(hidden_states))).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        all_pad = attention_mask.sum(dim=-1) == 0
        scores = scores.masked_fill(all_pad.unsqueeze(1), 0.0)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(all_pad.unsqueeze(1), 0.0)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled


class EmotionPhraseCrossAttention(nn.Module):
    """M2 (+ optional M3 lexicon gate) over phrase vectors."""

    def __init__(
        self,
        hidden: int,
        num_labels: int,
        *,
        num_heads: int = MHA_NUM_HEADS,
        use_m3: bool = False,
        lexicon_dim: int = 0,
    ) -> None:
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"hidden size {hidden} must divide num_heads {num_heads}")
        self.num_labels = num_labels
        self.use_m3 = use_m3
        self.emotion_queries = nn.Parameter(torch.empty(num_labels, hidden))
        nn.init.normal_(self.emotion_queries, mean=0.0, std=0.02)
        if use_m3:
            if lexicon_dim < 1:
                raise ValueError("lexicon_dim required when use_m3 is True")
            self.lexicon_proj = nn.Linear(lexicon_dim, hidden, bias=True)
            self.lexicon_gate = nn.Parameter(torch.tensor(0.0))
        self.phrase_attention = nn.MultiheadAttention(
            hidden, num_heads, dropout=0.1, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.emotion_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=EMOTION_ENCODER_LAYERS
        )
        self.label_weight = nn.Parameter(torch.empty(num_labels, hidden))
        self.label_bias = nn.Parameter(torch.zeros(num_labels))
        nn.init.xavier_uniform_(self.label_weight)

    def forward(
        self,
        phrase_vecs: torch.Tensor,
        phrase_mask: torch.Tensor,
        lexicon_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # phrase_vecs [B, M, H], phrase_mask [B, M] (1 = valid phrase)
        batch_size, num_phrases, hidden = phrase_vecs.shape
        num_labels = self.num_labels
        queries = self.emotion_queries.unsqueeze(0).expand(batch_size, num_labels, hidden)
        if self.use_m3:
            if lexicon_features is None:
                raise ValueError("lexicon_features required when M3 is enabled")
            prior = torch.tanh(self.lexicon_proj(lexicon_features))
            queries = queries + self.lexicon_gate * prior.unsqueeze(1)
        queries = queries.reshape(batch_size * num_labels, 1, hidden)
        keys = (
            phrase_vecs.unsqueeze(1)
            .expand(batch_size, num_labels, num_phrases, hidden)
            .reshape(batch_size * num_labels, num_phrases, hidden)
        )
        key_padding = phrase_mask == 0
        key_padding = key_padding.unsqueeze(1).expand(batch_size, num_labels, num_phrases)
        key_padding = key_padding.reshape(batch_size * num_labels, num_phrases)
        attended, _ = self.phrase_attention(
            queries,
            keys,
            keys,
            key_padding_mask=key_padding,
            need_weights=False,
        )
        emotion_repr = attended.squeeze(1).view(batch_size, num_labels, hidden)
        emotion_repr = self.emotion_encoder(emotion_repr)
        logits = (emotion_repr * self.label_weight.unsqueeze(0)).sum(dim=-1) + self.label_bias
        return logits


class HAKEMER(nn.Module):
    """Flat PLM, M1 hierarchical, or M1+M2 emotion-specific phrase attention."""

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
        self.use_m2 = config.use_m2
        self.use_m3 = config.use_m3
        if not self.use_m1:
            self.classifier = nn.Linear(hidden, config.num_labels)
        else:
            self.word_pool = AdditiveAttentionPool(hidden)
            if self.use_m2:
                self.emotion_head = EmotionPhraseCrossAttention(
                    hidden,
                    config.num_labels,
                    use_m3=config.use_m3,
                    lexicon_dim=lexicon_dim(config.lexicon_source) if config.use_m3 else 0,
                )
            else:
                self.phrase_pool = AdditiveAttentionPool(hidden)
                self.classifier = nn.Linear(hidden, config.num_labels)

    def encode_phrases(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_phrases, seq_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * num_phrases, seq_len)
        flat_mask = attention_mask.reshape(batch_size * num_phrases, seq_len)
        outputs = self.backbone(input_ids=flat_ids, attention_mask=flat_mask)
        token_hidden = outputs.last_hidden_state
        phrase_vecs = self.word_pool(token_hidden, flat_mask)
        return phrase_vecs.view(batch_size, num_phrases, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        phrase_mask: torch.Tensor | None = None,
        lexicon_features: torch.Tensor | None = None,
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
        phrase_vecs = self.encode_phrases(input_ids, attention_mask)
        phrase_vecs = self.dropout(phrase_vecs)

        if self.use_m2:
            return self.emotion_head(phrase_vecs, phrase_mask, lexicon_features=lexicon_features)

        doc_vec = self.phrase_pool(phrase_vecs, phrase_mask)
        doc_vec = self.dropout(doc_vec)
        return self.classifier(doc_vec)
