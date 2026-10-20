"""Example: stage-specific logistic prediction at the current landmark.

Run: ./benchmark.sh --custom-model examples/custom_model.py:Model --model-name logistic
The runner supplies the complete config, train/validation loaders, and causal
NumPy batches. Return one cancellation probability per batch row. Checkpoints,
fixed test predictions, all metrics, and validation calibration are automatic.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import sys
from joblib.externals import cloudpickle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from survbench.training import isolate_loader_generators, random_state, restore_random_state


class Model:
    def __init__(self, config):
        self.config = config
        seed = config["runtime"]["seed"]
        self.scaler = StandardScaler()
        self.models = [SGDClassifier(loss="log_loss", alpha=1e-4, random_state=seed + s) for s in (0, 1)]

    @staticmethod
    def features(batch):
        return np.asarray(batch["X"])[np.arange(len(batch["stage"])), batch["tau_step_index"]]

    def fit(self, train_loader, val_loader, *, checkpoint_path=None, resume=False):
        isolate_loader_generators(train_loader.loader, val_loader.loader)
        checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        start_epoch = 0
        if resume and checkpoint_path is not None and checkpoint_path.is_file():
            with checkpoint_path.open("rb") as handle:
                state = cloudpickle.load(handle)
            self.__dict__.update(state["model"].__dict__)
            start_epoch = state["next_epoch"]
            restore_random_state(state["random_state"])
        # Learn scaling exclusively from training inputs.
        if start_epoch == 0:
            for batch in train_loader:
                self.scaler.partial_fit(self.features(batch))
        epochs = int(self.config["model"].get("epochs", 5))
        for epoch in range(start_epoch, epochs):
            train_loader.dataset.set_epoch(epoch)
            for batch in train_loader:
                x = self.scaler.transform(self.features(batch))
                for stage, model in enumerate(self.models):
                    mask = batch["stage"] == stage
                    if mask.any():
                        model.partial_fit(x[mask], batch["y_cancel_window"][mask], classes=[0, 1])
            if checkpoint_path is not None:
                cloudpickle.register_pickle_by_value(sys.modules[type(self).__module__])
                temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
                with temporary.open("wb") as handle:
                    cloudpickle.dump(dict(model=self, next_epoch=epoch + 1, random_state=random_state()), handle)
                temporary.replace(checkpoint_path)
        return self

    def predict_batch(self, batch):
        x = self.scaler.transform(self.features(batch))
        p = np.empty(len(x), dtype=np.float32)
        for stage, model in enumerate(self.models):
            mask = batch["stage"] == stage
            if mask.any():
                if not hasattr(model, "classes_"):
                    raise ValueError(f"Training data contained no stage-{stage} landmarks.")
                p[mask] = model.predict_proba(x[mask])[:, 1]
        return {"p_cancel_window": p}
