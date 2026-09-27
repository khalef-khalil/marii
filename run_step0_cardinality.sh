#!/usr/bin/env bash
# Step~0 only: test by_gold_cardinality on runs/distilbert_base_uncased_seed*_baseline_plm
# Colab: ALLOW_EVAL=1 DEVICE=cuda ./run_step0_cardinality.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
if [[ "${ALLOW_EVAL:-}" != "1" ]]; then
  echo "Refusing: set ALLOW_EVAL=1 (Colab/CUDA recommended)." >&2
  exit 1
fi
DEVICE="${DEVICE:-cuda}"
python3 -m hakemer.run_eval_supplements --device "$DEVICE" --no-merge --glob 'distilbert_base_uncased_seed*_baseline_plm'
python3 scripts/sync_step0_cardinality_supplement.py
