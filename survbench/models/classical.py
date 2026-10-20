"""Landmark Cox and random survival forest competing-risk baselines.

Each model uses 174 causal landmark features: last/mean/max of the 57
CLEAN58 state fields, plus landmark time, visible length, and visible span.
Cancellation and progression are fitted separately within each stage and
combined as a cancellation cumulative-incidence function at prediction time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

STAGE_0 = 0
STAGE_1 = 1
CANCEL = 2
PROGRESS = 1
GRID_STEP_SECONDS = 10.0

def load_split(path: Path, name: str) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return np.asarray(payload[f"{name}_indices"], dtype=np.int64)


def valid_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask > 0.5)


def sample_stage_0(
    times: np.ndarray,
    mask: np.ndarray,
    match_time: float,
    cancel_time: float,
    pickup_time: float,
    rng: np.random.Generator,
) -> int | None:
    valid = valid_indices(mask)
    ends = [x for x in (match_time, cancel_time, pickup_time) if x >= 0]
    if ends:
        valid = valid[times[valid] < min(ends)]
    return None if valid.size == 0 else int(rng.choice(valid))


def sample_stage_1(
    times: np.ndarray,
    mask: np.ndarray,
    match_time: float,
    cancel_time: float,
    pickup_time: float,
    rng: np.random.Generator,
    decay: float,
) -> int | None:
    if match_time < 0:
        return None
    valid = valid_indices(mask)
    ends = [x for x in (cancel_time, pickup_time) if x >= 0]
    keep = times[valid] >= match_time
    if ends:
        keep &= times[valid] < min(ends)
    valid = valid[keep]
    if valid.size == 0:
        return None
    weights = np.exp(-decay * np.arange(valid.size, dtype=np.float64))
    weights /= weights.sum()
    return int(valid[int(rng.choice(valid.size, p=weights))])


def window_target(
    stage: int,
    tau: float,
    match_time: float,
    pickup_time: float,
    cancel_time: float,
    unmatched_h_seconds: float = 60.0,
    matched_h_seconds: float = 30.0,
) -> tuple[int, int, int, int, float, float]:
    horizon = unmatched_h_seconds if stage == STAGE_0 else matched_h_seconds
    end = tau + horizon
    cancel = cancel_time if tau < cancel_time <= end else -1.0
    raw_progress = match_time if stage == STAGE_0 else pickup_time
    progress = raw_progress if tau < raw_progress <= end else -1.0
    if cancel < 0 and progress < 0:
        return 0, 0, 0, 0, -1.0, horizon
    if cancel >= 0 and (progress < 0 or cancel < progress):
        return CANCEL, 1, 0, CANCEL, cancel - tau, horizon
    return PROGRESS, 0, 1, PROGRESS, progress - tau, horizon


def build_train_samples(
    data: Any,
    split_indices: np.ndarray,
    seed: int,
    matched_decay: float,
    unmatched_h_seconds: float = 60.0,
    matched_h_seconds: float = 30.0,
    time_step_seconds: float = GRID_STEP_SECONDS,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, list[Any]] = {
        "order_index": [],
        "order_id": [],
        "stage": [],
        "tau_time": [],
        "tau_step_index": [],
        "H_steps": [],
        "label_3way": [],
        "y_cancel_window": [],
        "y_progress_window": [],
        "first_event_type": [],
        "time_to_first_event_in_window": [],
    }
    for oi_raw in split_indices:
        oi = int(oi_raw)
        times = data["snapshot_times"][oi]
        mask = data["seq_mask"][oi]
        match = float(data["match_time"][oi])
        pickup = float(data["pickup_time"][oi])
        cancel = float(data["cancel_time"][oi])
        taus = (
            sample_stage_0(times, mask, match, cancel, pickup, rng),
            sample_stage_1(times, mask, match, cancel, pickup, rng, matched_decay),
        )
        for stage, tau_index in enumerate(taus):
            if tau_index is None:
                continue
            tau = float(times[tau_index])
            label, yc, yp, first, event_time, horizon = window_target(
                stage, tau, match, pickup, cancel, unmatched_h_seconds, matched_h_seconds
            )
            out["order_index"].append(oi)
            out["order_id"].append(int(data["order_ids"][oi]))
            out["stage"].append(stage)
            out["tau_time"].append(tau)
            out["tau_step_index"].append(tau_index)
            out["H_steps"].append(int(round(horizon / time_step_seconds)))
            out["label_3way"].append(label)
            out["y_cancel_window"].append(yc)
            out["y_progress_window"].append(yp)
            out["first_event_type"].append(first)
            out["time_to_first_event_in_window"].append(event_time)
    dtypes = {
        "order_index": np.int64,
        "order_id": np.int64,
        "stage": np.int8,
        "tau_time": np.float32,
        "tau_step_index": np.int32,
        "H_steps": np.int32,
        "label_3way": np.int8,
        "y_cancel_window": np.int8,
        "y_progress_window": np.int8,
        "first_event_type": np.int8,
        "time_to_first_event_in_window": np.float32,
    }
    return {key: np.asarray(value, dtype=dtypes[key]) for key, value in out.items()}


def load_eval_samples(path: Path, limit: int | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        stop = len(loaded["stage"]) if limit is None else min(limit, len(loaded["stage"]))
        return {key: np.asarray(loaded[key][:stop]) for key in loaded.files}


def build_features(data: Any, samples: dict[str, np.ndarray]) -> np.ndarray:
    """Summarize all visible state history; omit CLEAN58 elapsed increment x[0]."""
    x_all = data["X"]
    if x_all.ndim != 3 or x_all.shape[2] != 58:
        raise ValueError("Classical baselines require CLEAN58 tensors of shape [orders, steps, 58].")
    mask_all = data["seq_mask"]
    times_all = data["snapshot_times"]
    n = len(samples["stage"])
    features = np.empty((n, 57 * 3 + 3), dtype=np.float32)
    for row, (oi_raw, tau_idx_raw, tau_raw) in enumerate(
        zip(
            samples["order_index"],
            samples["tau_step_index"],
            samples["tau_time"],
            strict=True,
        )
    ):
        oi, tau_idx, tau = int(oi_raw), int(tau_idx_raw), float(tau_raw)
        visible = (mask_all[oi] > 0.5) & (times_all[oi] <= tau + 1e-6)
        # tau_idx is retained in the condition as a protocol consistency check.
        if tau_idx < 0 or tau_idx >= visible.size or not visible[tau_idx]:
            raise ValueError(f"Invalid landmark for order={oi}, tau_idx={tau_idx}")
        idx = np.flatnonzero(visible)
        hist = x_all[oi, idx, 1:].astype(np.float32, copy=False)
        features[row, :57] = hist[-1]
        features[row, 57:114] = hist.mean(axis=0)
        features[row, 114:171] = hist.max(axis=0)
        features[row, 171:] = (
            tau,
            float(idx.size),
            float(times_all[oi, idx[-1]] - times_all[oi, idx[0]]),
        )
    if not np.isfinite(features).all():
        raise ValueError("Non-finite landmark features.")
    return features


def durations_and_events(
    samples: dict[str, np.ndarray], mask: np.ndarray, cause: int,
    time_step_seconds: float = GRID_STEP_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = samples["H_steps"][mask].astype(np.float64) * time_step_seconds
    first_time = samples["time_to_first_event_in_window"][mask].astype(np.float64)
    duration = np.where(first_time > 0, first_time, horizon)
    event = samples["first_event_type"][mask] == cause
    # Survival libraries require strictly positive follow-up.
    return np.maximum(duration, 1e-3), event


def choose_stage_rows(
    samples: dict[str, np.ndarray],
    stage: int,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    if maximum is not None and maximum <= 0:
        raise ValueError("train_cap_per_stage must be positive or None.")
    rows = np.flatnonzero(samples["stage"] == stage)
    if maximum is not None and rows.size > maximum:
        rng = np.random.default_rng(seed + stage * 10_000)
        rows = np.sort(rng.choice(rows, size=maximum, replace=False))
    return rows


def cumulative_incidence(cancel_hazard: np.ndarray, progress_hazard: np.ndarray) -> np.ndarray:
    dc = np.diff(np.column_stack([np.zeros(cancel_hazard.shape[0]), cancel_hazard]), axis=1)
    dp = np.diff(np.column_stack([np.zeros(progress_hazard.shape[0]), progress_hazard]), axis=1)
    dc = np.maximum(dc, 0.0)
    dp = np.maximum(dp, 0.0)
    survival = np.ones(cancel_hazard.shape[0], dtype=np.float64)
    cif = np.zeros(cancel_hazard.shape[0], dtype=np.float64)
    for col in range(dc.shape[1]):
        total = dc[:, col] + dp[:, col]
        event_probability = -np.expm1(-total)
        allocation = np.divide(dc[:, col], total, out=np.zeros_like(total), where=total > 0)
        cif += survival * event_probability * allocation
        survival *= np.exp(-total)
    return np.clip(cif, 0.0, 1.0).astype(np.float32)


class CoxCompetingRisk:
    def __init__(self, alpha: float = 100.0, time_step_seconds: float = GRID_STEP_SECONDS) -> None:
        self.alpha = alpha
        self.time_step_seconds = time_step_seconds
        self.scalers: dict[int, StandardScaler] = {}
        self.keep: dict[int, np.ndarray] = {}
        self.models: dict[tuple[int, int], CoxPHSurvivalAnalysis] = {}

    def fit(self, x: np.ndarray, samples: dict[str, np.ndarray], maximum: int, seed: int) -> None:
        for stage in (STAGE_0, STAGE_1):
            rows = choose_stage_rows(samples, stage, maximum, seed)
            if not rows.size:
                raise ValueError(f"No training landmarks for stage {stage}.")
            scaler = StandardScaler()
            xs = scaler.fit_transform(x[rows])
            keep = np.std(xs, axis=0) > 1e-8
            xs = xs[:, keep]
            self.scalers[stage], self.keep[stage] = scaler, keep
            stage_mask = np.zeros(len(samples["stage"]), dtype=bool)
            stage_mask[rows] = True
            for cause in (CANCEL, PROGRESS):
                duration, event = durations_and_events(samples, stage_mask, cause, self.time_step_seconds)
                model = CoxPHSurvivalAnalysis(alpha=self.alpha, n_iter=100, tol=1e-8)
                model.fit(xs, Surv.from_arrays(event, duration))
                self.models[(stage, cause)] = model

    @staticmethod
    def hazards(model: CoxPHSurvivalAnalysis, x: np.ndarray, grid: np.ndarray) -> np.ndarray:
        baseline_fn = model.cum_baseline_hazard_
        indices = np.searchsorted(baseline_fn.x, grid, side="right") - 1
        baseline = np.zeros(grid.size, dtype=np.float64)
        valid = indices >= 0
        baseline[valid] = np.asarray(baseline_fn.y)[indices[valid]]
        linear = np.clip(x @ model.coef_, -30.0, 30.0)
        return np.exp(linear)[:, None] * baseline[None, :]

    def predict(self, x: np.ndarray, samples: dict[str, np.ndarray]) -> np.ndarray:
        output = np.zeros(x.shape[0], dtype=np.float32)
        for stage in (STAGE_0, STAGE_1):
            for h_steps in np.unique(samples["H_steps"][samples["stage"] == stage]):
                mask = (samples["stage"] == stage) & (samples["H_steps"] == h_steps)
                xs = self.scalers[stage].transform(x[mask])[:, self.keep[stage]]
                grid = np.arange(1, int(h_steps) + 1) * self.time_step_seconds
                hc = self.hazards(self.models[(stage, CANCEL)], xs, grid)
                hp = self.hazards(self.models[(stage, PROGRESS)], xs, grid)
                output[mask] = cumulative_incidence(hc, hp)
        return output


class RSFCompetingRisk:
    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_leaf: int,
        max_features: str | float,
        n_jobs: int,
        seed: int,
        time_step_seconds: float = GRID_STEP_SECONDS,
    ) -> None:
        self.time_step_seconds = time_step_seconds
        self.kwargs = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "n_jobs": n_jobs,
            "random_state": seed,
            "low_memory": False,
        }
        self.models: dict[tuple[int, int], RandomSurvivalForest] = {}

    def fit(self, x: np.ndarray, samples: dict[str, np.ndarray], maximum: int, seed: int) -> None:
        for stage in (STAGE_0, STAGE_1):
            rows = choose_stage_rows(samples, stage, maximum, seed)
            if not rows.size:
                raise ValueError(f"No training landmarks for stage {stage}.")
            stage_mask = np.zeros(len(samples["stage"]), dtype=bool)
            stage_mask[rows] = True
            for cause in (CANCEL, PROGRESS):
                duration, event = durations_and_events(samples, stage_mask, cause, self.time_step_seconds)
                model = RandomSurvivalForest(**self.kwargs)
                model.set_params(random_state=int(self.kwargs["random_state"]) + stage * 10 + cause)
                model.fit(x[rows], Surv.from_arrays(event, duration))
                self.models[(stage, cause)] = model

    @staticmethod
    def hazards(
        model: RandomSurvivalForest, x: np.ndarray, grid: np.ndarray, batch_size: int = 4096
    ) -> np.ndarray:
        result = np.zeros((x.shape[0], grid.size), dtype=np.float64)
        indices = np.searchsorted(model.unique_times_, grid, side="right") - 1
        valid = indices >= 0
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            raw = model.predict_cumulative_hazard_function(x[start:stop], return_array=True)
            if np.any(valid):
                result[start:stop, valid] = raw[:, indices[valid]]
        return result

    def predict(self, x: np.ndarray, samples: dict[str, np.ndarray]) -> np.ndarray:
        output = np.zeros(x.shape[0], dtype=np.float32)
        for stage in (STAGE_0, STAGE_1):
            for h_steps in np.unique(samples["H_steps"][samples["stage"] == stage]):
                mask = (samples["stage"] == stage) & (samples["H_steps"] == h_steps)
                grid = np.arange(1, int(h_steps) + 1) * self.time_step_seconds
                hc = self.hazards(self.models[(stage, CANCEL)], x[mask], grid)
                hp = self.hazards(self.models[(stage, PROGRESS)], x[mask], grid)
                output[mask] = cumulative_incidence(hc, hp)
        return output


def prediction_payload(data: Any, samples: dict[str, np.ndarray], prediction: np.ndarray) -> dict[str, np.ndarray]:
    """Keep landmark identities beside each predicted probability."""
    payload = {key: np.asarray(value) for key, value in samples.items()}
    payload["p_cancel_window"] = np.asarray(prediction, dtype=np.float32)
    raw_regions = np.asarray(data["region_ids"])
    values = np.sort(np.unique(raw_regions))
    # Preserve prepared 0..3 region IDs; map other IDs to sorted indices.
    if set(values.tolist()).issubset({0, 1, 2, 3}):
        regions = raw_regions
    else:
        lookup = {value: index for index, value in enumerate(values)}
        regions = np.asarray([lookup[value] for value in raw_regions], dtype=np.int32)
    payload["region_id"] = np.asarray(regions[samples["order_index"]], dtype=np.int32)
    return payload


def load_checkpoint(path: str | Path) -> CoxCompetingRisk | RSFCompetingRisk:
    """Load a saved Cox or RSF competing-risk model."""
    model = joblib.load(path)
    if not isinstance(model, (CoxCompetingRisk, RSFCompetingRisk)):
        raise TypeError("Checkpoint does not contain a Cox/RSF competing-risk model.")
    return model


def predict_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any] | None,
    base_npz_path: str | Path,
    sample_path: str | Path,
    *,
    model_key: str | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate saved weights on prepared landmarks without fitting anything."""
    model = load_checkpoint(checkpoint_path)
    if model_key is not None:
        expected = {"cox": CoxCompetingRisk, "rsf": RSFCompetingRisk}
        if model_key not in expected or not isinstance(model, expected[model_key]):
            raise ValueError("Requested model key does not match the saved classical model type.")
    if config is not None:
        configured_step = float(config.get("window", {}).get("time_step_seconds", GRID_STEP_SECONDS))
        if configured_step != model.time_step_seconds:
            raise ValueError("Configured time grid differs from the saved model.")
    with np.load(base_npz_path, allow_pickle=True) as loaded:
        data = {key: np.asarray(loaded[key]) for key in ("X", "seq_mask", "snapshot_times", "region_ids")}
    samples = load_eval_samples(Path(sample_path))
    prediction = model.predict(build_features(data, samples), samples)
    return prediction_payload(data, samples, prediction)
