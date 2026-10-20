"""Dynamic training and evaluation datasets for survbench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


UNMATCHED_STAGE = 0
MATCHED_STAGE = 1

LABEL_STAY = 0
LABEL_PROGRESS = 1
LABEL_CANCEL = 2

DEFAULT_TIME_STEP_SECONDS = 10.0
DEFAULT_UNMATCHED_H_SECONDS = 60.0
DEFAULT_MATCHED_H_SECONDS = 30.0


def _seconds_to_steps(horizon_seconds: float, time_step_seconds: float) -> int:
    steps = horizon_seconds / time_step_seconds
    rounded = int(round(steps))
    if abs(steps - rounded) > 1e-8:
        raise ValueError(
            f"Invalid horizon {horizon_seconds} for time step {time_step_seconds}; "
            "the ratio must be an integer."
        )
    if rounded <= 0:
        raise ValueError(f"Horizon must be positive, got {horizon_seconds}")
    return rounded


def load_split_indices(split_path: str | Path, split_name: str) -> list[int]:
    """Load order indices for a given split."""
    split_path = Path(split_path)
    with split_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    key = f"{split_name}_indices"
    if key not in payload:
        raise KeyError(f"Split file missing '{key}'")
    return list(payload[key])


@dataclass
class _DynamicBaseData:
    order_ids: np.ndarray
    X: np.ndarray
    seq_mask: np.ndarray
    snapshot_times: np.ndarray
    region_ids: np.ndarray
    event_time: np.ndarray
    end_type: np.ndarray
    match_time: np.ndarray
    pickup_time: np.ndarray
    cancel_time: np.ndarray
    last_meas_time: np.ndarray
    region: np.ndarray
    order_date: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path, preload: bool = True) -> "_DynamicBaseData":
        """Load the base dynamic sequence dataset.

        `preload=True` eagerly reads arrays into memory. On Linux, this works well
        with `DataLoader(num_workers>0)` because worker processes can share the
        read-only pages after `fork`.
        """
        loaded = np.load(Path(path), allow_pickle=True)

        def maybe_copy(name: str) -> np.ndarray:
            arr = loaded[name]
            return np.array(arr) if preload else arr

        return cls(
            order_ids=maybe_copy("order_ids"),
            X=maybe_copy("X"),
            seq_mask=maybe_copy("seq_mask"),
            snapshot_times=maybe_copy("snapshot_times"),
            region_ids=maybe_copy("region_ids"),
            event_time=maybe_copy("event_time"),
            end_type=maybe_copy("end_type"),
            match_time=maybe_copy("match_time"),
            pickup_time=maybe_copy("pickup_time"),
            cancel_time=maybe_copy("cancel_time"),
            last_meas_time=maybe_copy("last_meas_time"),
            region=maybe_copy("region"),
            order_date=maybe_copy("order_date"),
        )

    def __len__(self) -> int:
        return int(self.order_ids.shape[0])


def _valid_snapshot_indices(seq_mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(seq_mask > 0.5)


def _build_history_mask(snapshot_times: np.ndarray, seq_mask: np.ndarray, tau_time: float) -> np.ndarray:
    visible = (seq_mask > 0.5) & (snapshot_times <= tau_time + 1e-6)
    return visible.astype(np.float32, copy=False)


def collate_dynamic_samples(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Stack fixed-shape dynamic samples without converting them to torch in workers."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    batch: dict[str, np.ndarray] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        first = values[0]
        if isinstance(first, np.ndarray):
            batch[key] = np.stack(values, axis=0)
        else:
            batch[key] = np.asarray(values)
    return batch


def _sample_unmatched_tau(
    snapshot_times: np.ndarray,
    seq_mask: np.ndarray,
    match_time: float,
    cancel_time: float,
    pickup_time: float,
    rng: np.random.Generator,
) -> int | None:
    valid_idx = _valid_snapshot_indices(seq_mask)
    if valid_idx.size == 0:
        return None

    stage_end_candidates = [t for t in (match_time, cancel_time, pickup_time) if t >= 0.0]
    if stage_end_candidates:
        stage_end = min(stage_end_candidates)
        candidate_idx = valid_idx[snapshot_times[valid_idx] < stage_end]
    else:
        candidate_idx = valid_idx

    if candidate_idx.size == 0:
        return None
    return int(rng.choice(candidate_idx))


def _sample_matched_tau(
    snapshot_times: np.ndarray,
    seq_mask: np.ndarray,
    match_time: float,
    cancel_time: float,
    pickup_time: float,
    rng: np.random.Generator,
    matched_sampling: str,
    matched_decay: float,
) -> int | None:
    if match_time < 0.0:
        return None

    valid_idx = _valid_snapshot_indices(seq_mask)
    if valid_idx.size == 0:
        return None

    stage_end_candidates = [t for t in (cancel_time, pickup_time) if t >= 0.0]
    if stage_end_candidates:
        stage_end = min(stage_end_candidates)
        candidate_idx = valid_idx[(snapshot_times[valid_idx] >= match_time) & (snapshot_times[valid_idx] < stage_end)]
    else:
        candidate_idx = valid_idx[snapshot_times[valid_idx] >= match_time]

    if candidate_idx.size == 0:
        return None
    if matched_sampling == "uniform" or candidate_idx.size == 1:
        return int(rng.choice(candidate_idx))

    rel_pos = np.arange(candidate_idx.size, dtype=np.float64)
    weights = np.exp(-matched_decay * rel_pos)
    weights /= weights.sum()
    return int(candidate_idx[int(rng.choice(candidate_idx.size, p=weights))])


def _build_window_targets(
    stage: int,
    tau_time: float,
    match_time: float,
    pickup_time: float,
    cancel_time: float,
    unmatched_h_seconds: float,
    matched_h_seconds: float,
    time_step_seconds: float,
) -> tuple[int, int, int, int, float, int]:
    horizon = unmatched_h_seconds if stage == UNMATCHED_STAGE else matched_h_seconds
    h_steps = _seconds_to_steps(horizon, time_step_seconds)
    window_end = tau_time + horizon

    cancel_in_window = cancel_time if tau_time < cancel_time <= window_end else -1.0
    if stage == UNMATCHED_STAGE:
        progress_in_window = match_time if tau_time < match_time <= window_end else -1.0
    else:
        progress_in_window = pickup_time if tau_time < pickup_time <= window_end else -1.0

    if cancel_in_window < 0.0 and progress_in_window < 0.0:
        return LABEL_STAY, 0, 0, LABEL_STAY, -1.0, h_steps
    if cancel_in_window >= 0.0 and (progress_in_window < 0.0 or cancel_in_window < progress_in_window):
        return LABEL_CANCEL, 1, 0, LABEL_CANCEL, cancel_in_window - tau_time, h_steps
    return LABEL_PROGRESS, 0, 1, LABEL_PROGRESS, progress_in_window - tau_time, h_steps


class TrainDynamicDataset:
    """Online-sampling training dataset for stage-conditioned dynamic prediction.

    Designed for multi-worker `torch.utils.data.DataLoader`. The base sequence
    arrays can be eagerly loaded once in the parent process and then shared
    across worker processes on Linux via `fork`.
    """

    def __init__(
        self,
        data_path: str | Path,
        split_path: str | Path,
        split_name: str = "train",
        epoch_size: int | None = None,
        seed: int = 42,
        matched_sampling: str = "exp",
        matched_decay: float = 0.2,
        unmatched_h_seconds: float = DEFAULT_UNMATCHED_H_SECONDS,
        matched_h_seconds: float = DEFAULT_MATCHED_H_SECONDS,
        time_step_seconds: float = DEFAULT_TIME_STEP_SECONDS,
        preload: bool = True,
        tau_only: bool = False,
    ) -> None:
        self.base = _DynamicBaseData.from_npz(data_path, preload=preload)
        self.order_indices = np.asarray(load_split_indices(split_path, split_name), dtype=np.int64)
        if self.order_indices.size == 0:
            raise ValueError(f"No orders found for split '{split_name}'")

        self.epoch_size = int(epoch_size) if epoch_size is not None else int(self.order_indices.size)
        self.seed = int(seed)
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.matched_sampling = matched_sampling
        self.matched_decay = float(matched_decay)
        self.unmatched_h_seconds = float(unmatched_h_seconds)
        self.matched_h_seconds = float(matched_h_seconds)
        self.time_step_seconds = float(time_step_seconds)
        self.tau_only = tau_only

    @property
    def epoch(self) -> int:
        return int(self._shared_epoch.item())

    @epoch.setter
    def epoch(self, epoch: int) -> None:
        self._shared_epoch.fill_(int(epoch))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def _make_rng(self, idx: int) -> np.random.Generator:
        # Stable per-sample seed that changes across epochs.
        epoch_seed = ((self.epoch + 1) * 6364136223846793005) & ((1 << 64) - 1)
        seed = int(self.seed) ^ epoch_seed ^ (idx + 1)
        return np.random.default_rng(seed % (2**63 - 1))

    def _select_order_index(self, idx: int, rng: np.random.Generator) -> int:
        base_pos = idx % self.order_indices.size
        if self.epoch_size <= self.order_indices.size:
            return int(self.order_indices[base_pos])
        return int(self.order_indices[int(rng.integers(0, self.order_indices.size))])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = self._make_rng(idx)
        order_index = self._select_order_index(idx, rng)

        snapshot_times = self.base.snapshot_times[order_index]
        seq_mask = self.base.seq_mask[order_index]
        match_time = float(self.base.match_time[order_index])
        pickup_time = float(self.base.pickup_time[order_index])
        cancel_time = float(self.base.cancel_time[order_index])

        unmatched_tau_idx = _sample_unmatched_tau(
            snapshot_times=snapshot_times,
            seq_mask=seq_mask,
            match_time=match_time,
            cancel_time=cancel_time,
            pickup_time=pickup_time,
            rng=rng,
        )
        matched_tau_idx = _sample_matched_tau(
            snapshot_times=snapshot_times,
            seq_mask=seq_mask,
            match_time=match_time,
            cancel_time=cancel_time,
            pickup_time=pickup_time,
            rng=rng,
            matched_sampling=self.matched_sampling,
            matched_decay=self.matched_decay,
        )

        available_stages: list[tuple[int, int]] = []
        if unmatched_tau_idx is not None:
            available_stages.append((UNMATCHED_STAGE, unmatched_tau_idx))
        if matched_tau_idx is not None:
            available_stages.append((MATCHED_STAGE, matched_tau_idx))
        if not available_stages:
            raise RuntimeError(f"Order {order_index} has no valid unmatched or matched landmark.")

        stage, tau_idx = available_stages[int(rng.integers(0, len(available_stages)))]
        tau_time = float(snapshot_times[tau_idx])
        history_mask = _build_history_mask(snapshot_times, seq_mask, tau_time)
        if self.tau_only:
            tau_only_mask = np.zeros_like(history_mask)
            tau_only_mask[tau_idx] = 1.0
            history_mask = tau_only_mask
        label_3way, y_cancel, y_progress, first_event_type, time_to_first, h_steps = _build_window_targets(
            stage=stage,
            tau_time=tau_time,
            match_time=match_time,
            pickup_time=pickup_time,
            cancel_time=cancel_time,
            unmatched_h_seconds=self.unmatched_h_seconds,
            matched_h_seconds=self.matched_h_seconds,
            time_step_seconds=self.time_step_seconds,
        )

        sample = {
            "order_index": np.int64(order_index),
            "order_id": np.int64(self.base.order_ids[order_index]),
            "X": self.base.X[order_index].astype(np.float32, copy=False),
            "seq_mask": seq_mask.astype(np.float32, copy=False),
            "history_mask": history_mask,
            "snapshot_times": snapshot_times.astype(np.float32, copy=False),
            "region_id": np.int32(self.base.region_ids[order_index]),
            "stage": np.int8(stage),
            "tau_time": np.float32(tau_time),
            "tau_step_index": np.int32(tau_idx),
            "H_steps": np.int32(h_steps),
            "label_3way": np.int8(label_3way),
            "y_cancel_window": np.int8(y_cancel),
            "y_progress_window": np.int8(y_progress),
            "first_event_type": np.int8(first_event_type),
            "time_to_first_event_in_window": np.float32(time_to_first),
        }
        return sample


class EvalDynamicDataset:
    """Fixed-sample evaluation dataset for validation or test."""

    def __init__(
        self,
        data_path: str | Path,
        sample_path: str | Path,
        unmatched_h_seconds: float = DEFAULT_UNMATCHED_H_SECONDS,
        matched_h_seconds: float = DEFAULT_MATCHED_H_SECONDS,
        time_step_seconds: float = DEFAULT_TIME_STEP_SECONDS,
        preload: bool = True,
        tau_only: bool = False,
    ) -> None:
        self.base = _DynamicBaseData.from_npz(data_path, preload=preload)
        samples = np.load(Path(sample_path), allow_pickle=True)
        self.order_index = np.array(samples["order_index"])
        self.order_id = np.array(samples["order_id"])
        self.stage = np.array(samples["stage"])
        self.tau_time = np.array(samples["tau_time"])
        self.tau_step_index = np.array(samples["tau_step_index"])
        self.H_steps = np.array(samples["H_steps"])
        self.label_3way = np.array(samples["label_3way"])
        self.y_cancel_window = np.array(samples["y_cancel_window"])
        self.y_progress_window = np.array(samples["y_progress_window"])
        self.first_event_type = np.array(samples["first_event_type"])
        self.time_to_first = np.array(samples["time_to_first_event_in_window"])
        self.unmatched_h_steps = _seconds_to_steps(float(unmatched_h_seconds), float(time_step_seconds))
        self.matched_h_steps = _seconds_to_steps(float(matched_h_seconds), float(time_step_seconds))
        self.tau_only = tau_only

        expected_h_steps = np.where(self.stage.astype(np.int64) == UNMATCHED_STAGE, self.unmatched_h_steps, self.matched_h_steps)
        if not np.array_equal(self.H_steps.astype(np.int64), expected_h_steps.astype(np.int64)):
            raise ValueError(
                f"Eval sample horizon steps do not match configured horizons in {sample_path}. "
                "Rebuild eval samples for the requested window."
            )

    def __len__(self) -> int:
        return int(self.order_index.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        order_index = int(self.order_index[idx])
        tau_time = float(self.tau_time[idx])
        seq_mask = self.base.seq_mask[order_index]
        snapshot_times = self.base.snapshot_times[order_index]
        history_mask = _build_history_mask(snapshot_times, seq_mask, tau_time)
        if self.tau_only:
            tau_only_mask = np.zeros_like(history_mask)
            tau_only_mask[int(self.tau_step_index[idx])] = 1.0
            history_mask = tau_only_mask

        sample = {
            "order_index": np.int64(order_index),
            "order_id": np.int64(self.order_id[idx]),
            "X": self.base.X[order_index].astype(np.float32, copy=False),
            "seq_mask": seq_mask.astype(np.float32, copy=False),
            "history_mask": history_mask,
            "snapshot_times": snapshot_times.astype(np.float32, copy=False),
            "region_id": np.int32(self.base.region_ids[order_index]),
            "stage": np.int8(self.stage[idx]),
            "tau_time": np.float32(self.tau_time[idx]),
            "tau_step_index": np.int32(self.tau_step_index[idx]),
            "H_steps": np.int32(self.H_steps[idx]),
            "label_3way": np.int8(self.label_3way[idx]),
            "y_cancel_window": np.int8(self.y_cancel_window[idx]),
            "y_progress_window": np.int8(self.y_progress_window[idx]),
            "first_event_type": np.int8(self.first_event_type[idx]),
            "time_to_first_event_in_window": np.float32(self.time_to_first[idx]),
        }
        return sample
