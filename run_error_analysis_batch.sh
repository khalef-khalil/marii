#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
shopt -s nullglob
runs=(runs/distilbert_base_uncased_seed*_m1_m2)
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "No distilbert *_m1_m2 run dirs under runs/; train M1+M2 first." >&2
  exit 1
fi
for d in "${runs[@]}"; do
  echo "Error analysis: $d"
  python3 -m hakemer.run_error_analysis "$d"
done
python3 scripts/aggregate_error_analysis.py
python3 scripts/render_error_analysis_tex.py
