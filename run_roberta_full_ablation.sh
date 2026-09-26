#!/usr/bin/env bash
# Full RoBERTa ladder: +M1, +M1+M2, +M3 (NRC/SenticNet), +M4 (SenticNet).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
EXTRA=(--epochs 4 --batch-size 16 --lr 5e-5 --early-stopping-patience 0 "$@")
echo "=== RoBERTa +M1 / +M1+M2 ==="
./run_roberta_ablation.sh "${EXTRA[@]}"
echo "=== RoBERTa +M3 / +M4 ==="
./run_roberta_m3_m4_ablation.sh "${EXTRA[@]}"
