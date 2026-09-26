#!/usr/bin/env bash
# RoBERTa-base: +M3 (NRC, SenticNet) and +M4 (SenticNet), same protocol as DistilBERT H3/H4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
EXTRA=(--epochs 4 --batch-size 16 --lr 5e-5 --early-stopping-patience 0 "$@")
echo "=== RoBERTa-base +M1+M2+M3 (NRC, 3 seeds) ==="
./run_m1_m2_m3_campaign.sh --lexicon nrc --backbone roberta-base "${EXTRA[@]}"
echo "=== RoBERTa-base +M1+M2+M3 (SenticNet, 3 seeds) ==="
./run_m1_m2_m3_campaign.sh --lexicon senticnet --backbone roberta-base "${EXTRA[@]}"
echo "=== RoBERTa-base +M1+M2+M3+M4 (SenticNet, 3 seeds) ==="
./run_m1_m2_m3_m4_campaign.sh --lexicon senticnet --backbone roberta-base "${EXTRA[@]}"
