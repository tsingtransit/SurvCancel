"""Custom models implement Model(config), fit(train, val), predict_batch(batch)."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_model_class(reference):
    module_name, separator, class_name = reference.rpartition(":")
    if not separator or not class_name or not module_name:
        raise ValueError("Expected --custom-model package.module:Class or /path/model.py:Class")
    if module_name.endswith(".py"):
        path = Path(module_name).resolve(strict=True)
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("survcancel_user_model", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not callable(cls):
        raise TypeError(f"{reference} is not a model class.")
    return cls


class Predictor:
    """Give prediction methods only the causal input fields, without labels."""
    def __init__(self, model):
        self.model = model

    def predict_batch(self, batch):
        fields = ("X", "history_mask", "snapshot_times", "stage", "region_id",
                  "tau_time", "tau_step_index", "H_steps")
        inputs = {key: np.array(batch[key], copy=True) for key in fields}
        visible = inputs["history_mask"] > 0.5
        inputs["X"][~visible] = 0.0
        inputs["snapshot_times"][~visible] = 0.0
        inputs["seq_mask"] = inputs["history_mask"].copy()
        result = self.model.predict_batch(inputs)
        if not isinstance(result, dict) or "p_cancel_window" not in result:
            raise ValueError("predict_batch must return {'p_cancel_window': probabilities}.")
        raw = result["p_cancel_window"]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        p = np.asarray(raw, dtype=np.float32)
        if p.shape != np.asarray(batch["stage"]).shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("p_cancel_window must have shape (batch_size,), with finite values in [0,1].")
        return {"p_cancel_window": p}


class InputLoader:
    """Keep training labels while masking future and padded input snapshots."""
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            batch = {key: np.array(value, copy=True) for key, value in batch.items()}
            visible = batch["history_mask"] > 0.5
            batch["X"][~visible] = 0.0
            batch["snapshot_times"][~visible] = 0.0
            batch["seq_mask"] = batch["history_mask"].copy()
            yield batch
