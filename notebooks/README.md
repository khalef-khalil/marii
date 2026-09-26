# Colab — Step 0 baseline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khalef-khalil/marii/blob/main/notebooks/baseline_plm_campaign.ipynb)

**Two steps (same as before):**

1. **Phase 1:** Run through **Download DistilBERT zip** → archive as `reference/artifacts/baseline_plm_distilbert_step0.zip` (and unpack JSON/metrics alongside).
2. **Phase 2:** Uncomment RoBERTa cell, run it, run **Download RoBERTa zip** → archive as `reference/artifacts/baseline_plm_roberta_step0.zip` when ready.

Protocol: batch 16, LR 5e-5, 4 epochs (GoEmotions/Demszky anchor).

## M1 ablation (H1)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khalef-khalil/marii/blob/main/notebooks/m1_campaign.ipynb)

Run `m1_campaign.ipynb` → archive `m1_distilbert_step0.zip` and executed notebook under `reference/training_records/step_m1/`.

## M1+M2 ablation (H2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khalef-khalil/marii/blob/main/notebooks/m1_m2_campaign.ipynb)

Run `m1_m2_campaign.ipynb` → archive `m1_m2_distilbert_step0.zip` under `reference/training_records/step_m1_m2/`.

**Training records:** archive executed Colab notebooks and manifests under [`reference/training_records/`](../reference/training_records/README.md). After updating artifacts, run `python scripts/build_step0_training_record.py` to refresh the Step 0 record notebooks from campaign JSON.
