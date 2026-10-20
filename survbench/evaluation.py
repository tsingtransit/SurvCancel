"""Shared checkpoint inference and prediction artifacts for the benchmark."""
from __future__ import annotations

from typing import Any
import numpy as np

from survbench.metrics import compute_window_metrics

PREDICTION_FIELDS = (
    "order_id", "order_index", "stage", "region_id", "tau_time",
    "y_cancel_window", "first_event_type", "time_to_first_event_in_window",
)


def predict_loader(model: Any, loader: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in PREDICTION_FIELDS}
    arrays["p_cancel_window"] = []
    for batch in loader:
        prediction = model.predict_batch(batch)
        for key in PREDICTION_FIELDS:
            arrays[key].append(np.asarray(batch[key]))
        arrays["p_cancel_window"].append(np.asarray(prediction["p_cancel_window"], dtype=np.float32))
    if not arrays["p_cancel_window"]:
        raise ValueError("The evaluation set has no landmarks.")
    return {key: np.concatenate(parts) for key, parts in arrays.items()}


def prediction_metrics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    return compute_window_metrics(
        arrays["y_cancel_window"], arrays["p_cancel_window"], arrays["stage"],
        arrays["time_to_first_event_in_window"], arrays["first_event_type"],
        region_id=arrays.get("region_id"),
    )


def save_window_predictions(model: Any, loader: Any, path: Any) -> None:
    np.savez_compressed(path, **predict_loader(model, loader))


def load_neural_checkpoint(path: Any, model_key: str, *, device: str = "cpu",
                           config: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """Strictly load a trusted benchmark checkpoint without fitting a model."""
    import torch
    from survbench.models import StageDeepHit, StageSurvTrace, StageTracer

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint["config"]
    if config is not None:
        # Parameter shapes alone do not detect altered history or feature masks.
        for section in ("model", "window", "train_dataset", "eval_dataset"):
            original = dict(saved.get(section, {}))
            override = dict(config.get(section, {}))
            if section == "model":
                for key in ("device", "verbose"):
                    original.pop(key, None)
                    override.pop(key, None)
            if original != override:
                raise ValueError(f"--config changes checkpoint scientific settings in {section}; use the saved configuration.")
    cfg = config if config is not None else saved
    kwargs = dict(cfg["model"])
    window = cfg.get("window", {})
    step = float(window.get("time_step_seconds", 10))
    hmax = int(max(float(window.get("unmatched_h_seconds", 60)),
                   float(window.get("matched_h_seconds", 30))) / step)
    kwargs.update(device=device, verbose=False, h_max=hmax)
    if model_key in ("deephit", "dyn_deephit", "riskpool"):
        cls = StageDeepHit
        kwargs["time_step_seconds"] = step
        encoder = kwargs.get("encoder_type", "rnn_att")
        tau_only = bool(cfg.get("train_dataset", {}).get("tau_only", False))
        valid = ((model_key == "riskpool" and encoder in {"survcancel", "mh_att_pool"})
                 or (model_key == "deephit" and encoder == "rnn_att" and tau_only)
                 or (model_key == "dyn_deephit" and encoder == "rnn_att" and not tau_only))
        if not valid:
            raise ValueError(f"Model key {model_key} disagrees with saved encoder/history settings.")
    elif model_key == "survtrace":
        cls = StageSurvTrace
    elif model_key == "tracer":
        cls = StageTracer
    else:
        raise ValueError(f"Unknown neural model: {model_key}")
    model = cls(**kwargs)
    state = checkpoint["model"]
    model.model.load_state_dict(state["model_state"], strict=True)
    model.feature_mean = np.asarray(state["feature_mean"], dtype=np.float32)
    model.feature_std = np.asarray(state["feature_std"], dtype=np.float32)
    model.model.eval()
    return model, cfg
