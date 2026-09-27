#!/usr/bin/env bash
# Full GoEmotions campaigns: run on Colab/GPU only. Never auto-start on a laptop.
# Example (Colab): ALLOW_FULL_TRAINING=1 DEVICE=cuda ./run_high_impact_campaigns.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
if [[ "${ALLOW_FULL_TRAINING:-}" != "1" ]]; then
  echo "Refusing to train: set ALLOW_FULL_TRAINING=1 and DEVICE=cuda (or Colab GPU)." >&2
  echo "This script is not for local Mac/CPU full-data runs." >&2
  exit 1
fi
DEVICE="${DEVICE:-cuda}"
LOG="${ROOT}/reference/training_records/high_impact_campaigns.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== High-impact campaigns started $(date -Iseconds) device=$DEVICE ==="
python3 -m hakemer.run_m1_m2_controls_campaign --variant no_enc --device "$DEVICE"
python3 -m hakemer.run_m1_m2_controls_campaign --variant no_xattn --device "$DEVICE"
python3 -m hakemer.run_m1_m2_m3_m4_campaign --lexicon nrc --device "$DEVICE"
echo "=== High-impact campaigns finished $(date -Iseconds) ==="
