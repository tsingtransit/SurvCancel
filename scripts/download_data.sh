#!/usr/bin/env bash
# Download the dataset separately from experiment execution.
set -euo pipefail
REPO_ID=${1:-survcancel-anonymous/SurvCancel}
TARGET_DIR=${2:-data/hf_survcancel_release}
REVISION=${3:-51ab808d26a98d1e982c84e581a9c0fc66063c80}
"${PYTHON_BIN:-${PYTHON:-python}}" - "$REPO_ID" "$TARGET_DIR" "$REVISION" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id, target_dir, revision = sys.argv[1:]
Path(target_dir).mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=target_dir, revision=revision)
print(f"Dataset is ready at {target_dir}")
PY
