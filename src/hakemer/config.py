from dataclasses import dataclass, field
from typing import Literal


LexiconSource = Literal["none", "nrc", "senticnet"]
BackboneName = Literal["roberta-base", "distilbert-base-uncased"]


@dataclass
class TrainConfig:
    backbone: BackboneName = "distilbert-base-uncased"
    seed: int = 42
    epochs: int = 3
    batch_size: int = 16
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.1
    max_length: int = 128
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

    def learning_rate(self) -> float:
        if self.backbone.startswith("roberta"):
            return 2e-5
        return 3e-5

    def validate_flags(self) -> None:
        if any((self.use_m1, self.use_m2, self.use_m3, self.use_m4)):
            raise NotImplementedError(
                "Modules M1–M4 are not implemented yet. Use all use_m* = False for baseline PLM."
            )
        if self.lexicon_source != "none":
            raise NotImplementedError("Lexicon M3 requires use_m3 (not implemented yet).")

    @property
    def run_name(self) -> str:
        parts = [
            self.backbone.split("/")[-1].replace("-", "_"),
            f"seed{self.seed}",
            "baseline_plm",
        ]
        return "_".join(parts)
