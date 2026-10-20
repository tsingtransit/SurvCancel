#!/usr/bin/env bash
# Build CLEAN58, order splits and fixed evaluation landmarks before experiments.
set -euo pipefail
PACKAGE_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RELEASE_DIR=${1:-data/hf_survcancel_release}
WORKDIR=${2:-workdir}
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
# Relative input/output paths use the caller's cwd. Additional flags override defaults.
exec "${PYTHON_BIN:-${PYTHON:-python}}" "$PACKAGE_ROOT/survcancel_benchmark/prepare_clean58.py" \
  --release-dir "$RELEASE_DIR" \
  --output-dir "$WORKDIR" \
  --protocols all \
  "$@"
