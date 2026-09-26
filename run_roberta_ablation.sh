#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
EXTRA=(--epochs 4 --batch-size 16 --lr 5e-5 --early-stopping-patience 0 "$@")
echo "=== RoBERTa-base +M1 (3 seeds) ==="
./run_m1_campaign.sh --backbone roberta-base "${EXTRA[@]}"
echo "=== RoBERTa-base +M1+M2 (3 seeds) ==="
./run_m1_m2_campaign.sh --backbone roberta-base "${EXTRA[@]}"
