#!/usr/bin/env bash
# M1+M2+M3 with emotion-specific lexicon priors (NRC or SenticNet).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
LEXICON="${1:-nrc}"
shift || true
exec python3 -m hakemer.run_m1_m2_m3_campaign \
  --lexicon "${LEXICON}" \
  --lexicon-fusion emotion_specific \
  "$@"
