from dataclasses import dataclass
from typing import Literal


LexiconSource = Literal["none", "nrc", "senticnet"]
BackboneName = Literal["roberta-base", "distilbert-base-uncased"]


@dataclass
class TrainConfig:
    """Fine-tuning contract anchored on GoEmotions BERT setup (Demszky et al.)."""

    backbone: BackboneName = "distilbert-base-uncased"
    seed: int = 42
    epochs: int = 4
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.1
    max_length: int = 128
    max_phrases: int = 4
    phrase_max_length: int = 32
    num_labels: int = 28
    use_m1: bool = False
    use_m2: bool = False
    use_m3: bool = False
    use_m4: bool = False
    lexicon_source: LexiconSource = "none"
    output_dir: str = "runs/baseline"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    device: str = "auto"
    decision_threshold: float = 0.5
    early_stopping_patience: int = 0

    def validate_flags(self) -> None:
        if self.use_m2 and not self.use_m1:
            raise ValueError("Module M2 requires M1 (phrase encodings).")
        if self.use_m3 and not (self.use_m1 and self.use_m2):
            raise ValueError("Module M3 requires M1 and M2.")
        if self.use_m3 and self.lexicon_source == "none":
            raise ValueError("Module M3 requires lexicon_source 'nrc' or 'senticnet'.")
        if self.use_m4 and not (self.use_m1 and self.use_m2 and self.use_m3):
            raise ValueError("Module M4 requires M1, M2, and M3.")
        if not self.use_m3 and self.lexicon_source != "none":
            raise ValueError("lexicon_source is only valid when use_m3 is enabled.")
        if self.use_m1 and self.max_phrases < 1:
            raise ValueError("max_phrases must be >= 1 when use_m1 is set.")

    @property
    def run_name(self) -> str:
        slug = self.backbone.split("/")[-1].replace("-", "_")
        if self.use_m4:
            step = f"m1_m2_m3_{self.lexicon_source}_m4"
        elif self.use_m3:
            step = f"m1_m2_m3_{self.lexicon_source}"
        elif self.use_m2:
            step = "m1_m2"
        elif self.use_m1:
            step = "m1"
        else:
            step = "baseline_plm"
        return f"{slug}_seed{self.seed}_{step}"
