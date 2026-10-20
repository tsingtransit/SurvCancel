#!/usr/bin/env python3
"""Evaluate an existing checkpoint on fixed landmarks; never trains a model."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


def validate_landmark_identity(data_path: Path, sample_path: Path) -> None:
    """Reject indices from a different data reconstruction before inference."""
    with np.load(data_path, allow_pickle=False) as data, np.load(sample_path, allow_pickle=False) as samples:
        indices = samples["order_index"]
        tau_index = samples["tau_step_index"]
        order_ids = data["order_ids"]
        times, mask = data["snapshot_times"], data["seq_mask"]
        if not len(indices) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Expected nonempty integer landmark order indices.")
        if np.any((indices < 0) | (indices >= len(order_ids))):
            raise ValueError("Landmark order index is outside the supplied base tensor.")
        if not np.array_equal(order_ids[indices], samples["order_id"]):
            raise ValueError("Landmark order IDs do not match the base tensor.")
        if not np.issubdtype(tau_index.dtype, np.integer) or np.any((tau_index < 0) | (tau_index >= times.shape[1])):
            raise ValueError("Landmark snapshot index is outside the supplied sequences.")
        if not np.all(mask[indices, tau_index] > .5):
            raise ValueError("Landmark snapshot points to sequence padding.")
        if not np.allclose(times[indices, tau_index], samples["tau_time"], rtol=0, atol=1e-5):
            raise ValueError("Landmark time does not match its supplied snapshot.")
        if not np.isin(samples["stage"], [0, 1]).all():
            raise ValueError("Expected benchmark stages 0/1.")

