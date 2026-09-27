#!/usr/bin/env bash
# Eval-only: cardinality buckets + M3 lexicon zero/shuffle (needs runs/*/best_model.pt).
# Colab: ALLOW_EVAL=1 DEVICE=cuda ./run_eval_supplements.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
if [[ "${ALLOW_EVAL:-}" != "1" ]]; then
  echo "Refusing eval supplements: set ALLOW_EVAL=1 (Colab/CUDA recommended)." >&2
  exit 1
fi
DEVICE="${DEVICE:-cuda}"
python3 -m hakemer.run_eval_supplements --device "$DEVICE" "$@"
