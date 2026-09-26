# Colab — Step 0 baseline (GoEmotions protocol)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khalef-khalil/marii/blob/main/notebooks/baseline_plm_campaign.ipynb)

**Protocol (chapter 4):** batch **16**, LR **5e-5**, **4 epochs**, best val F1-macro checkpoint, seeds 42 / 123 / 456 — aligned with [GoEmotions](https://arxiv.org/abs/2005.00547) BERT fine-tuning (Demszky et al.).

1. Open the notebook link above.
2. **Runtime → Factory reset runtime** then **GPU**.
3. **Runtime → Run all** (DistilBERT + RoBERTa, ~1.5–3 h on T4).
4. Download `baseline_plm_step0_goemotions_protocol.zip` and add to the project.
