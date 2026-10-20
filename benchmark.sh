#!/usr/bin/env bash
# Main benchmark, built-in or custom models, identical fixed evaluation samples.
# Prerequisite: prepare CLEAN58 and fixed splits first; see README.md.
# Metrics: C-index, Brier, AUROC, AP, log loss, ECE, mean risk and calibration gap;
# raw + validation-Platt, overall / stage / region / region-stage, per seed + SD.
# Outputs: outputs/benchmark/<protocol>/<window>/{summary.json,summary.csv,per_seed.csv}
# Per-model directories contain configs, checkpoints, metrics and prediction NPZs.
# Example: ./benchmark.sh --models riskpool dyn_deephit --device cuda:0
# Custom:  ./benchmark.sh --custom-model examples/custom_model.py:Model --model-name logistic
# Inspect all options: ./benchmark.sh --help
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python}" -m survcancel_benchmark.cli benchmark "$@"
