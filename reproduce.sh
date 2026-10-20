#!/usr/bin/env bash
# Four-region experiments (R1-R4), seeds 42 / 123 / 456:
# Prerequisite: prepare CLEAN58, splits and landmarks first; see README.md.
#  1. ID: DeepHit, DynDeepHit, SurvTRACE, TraCeR, Cox, RSF, RiskPool.
#  2. LORO R1-R4 + global chronological: RiskPool, DynDeepHit, TraCeR, RSF.
#  3. Regional chronological: DynDeepHit + TraCeR, separately for each region.
#  4. RiskPool history / feature restrictions.
#  5. Five prediction windows: RiskPool + DynDeepHit; label frequencies.
#  6. Raw probability metrics and validation-only stage-wise Platt for all runs.
#  7. MMD / Wasserstein diagnostics, ID random-split control, inference timings.
#     Distance landmark preparation is repeated for seeds 42 / 123 / 456.
# Outputs: outputs/reproducibility/{id,loro_R1,...,chrono_global,chrono_R1,...,
#          representation,horizon,distances,inference,loro_macro}/
# Every training group: summary.csv / per_seed.csv / summary.json; seed subfolders
# hold configs, checkpoints, metrics and validation/test prediction NPZs.
# Inspect the complete execution matrix: ./reproduce.sh --dry-run
# Help: ./reproduce.sh --help
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python}" -m survcancel_benchmark.cli reproduce "$@"
