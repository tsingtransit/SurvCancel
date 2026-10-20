"""Window-level benchmark metrics for SurvCancel experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


LABEL_CANCEL = 2
REGION_NAMES = {0: "R1", 1: "R2", 2: "R3", 3: "R4"}


@dataclass
class _FenwickTree:
    values: np.ndarray

    @classmethod
    def zeros(cls, size: int) -> "_FenwickTree":
        return cls(np.zeros(size + 1, dtype=np.int64))

    def add(self, index: int, amount: int = 1) -> None:
        index += 1
        while index < self.values.size:
            self.values[index] += amount
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        if index < 0:
            return 0
        index += 1
        total = 0
        while index > 0:
            total += int(self.values[index])
            index -= index & -index
        return total

    def range_sum(self, left: int, right: int) -> int:
        if right < left:
            return 0
        return self.prefix_sum(right) - self.prefix_sum(left - 1)

    @property
    def total(self) -> int:
        return self.prefix_sum(self.values.size - 2)


def window_c_index(pred_risk: np.ndarray, first_time: np.ndarray, first_event: np.ndarray) -> float:
    """Cause-specific C-index for windowed cancellation risk.

    A cancellation instance is comparable with non-cancellation instances and
    with cancellation instances that cancel later in the benchmark window.
    """
    pred = np.asarray(pred_risk, dtype=np.float64)
    time = np.asarray(first_time, dtype=np.float64)
    event = np.asarray(first_event, dtype=np.int64)
    cancel_mask = event == LABEL_CANCEL
    if pred.size == 0 or not np.any(cancel_mask):
        return float("nan")

    concordant = 0.0
    comparable = 0.0

    cancel_risk = pred[cancel_mask]
    cancel_time = time[cancel_mask]
    non_cancel_risk = np.sort(pred[~cancel_mask])

    if non_cancel_risk.size:
        left = np.searchsorted(non_cancel_risk, cancel_risk, side="left")
        right = np.searchsorted(non_cancel_risk, cancel_risk, side="right")
        comparable += float(non_cancel_risk.size * cancel_risk.size)
        concordant += float(left.sum()) + 0.5 * float((right - left).sum())

    finite = cancel_time >= 0.0
    if np.any(finite):
        finite_risk = cancel_risk[finite]
        finite_time = cancel_time[finite]
        unique_risks = np.unique(finite_risk)
        risk_rank = np.searchsorted(unique_risks, finite_risk)
        tree = _FenwickTree.zeros(unique_risks.size)

        for current_time in sorted(np.unique(finite_time), reverse=True):
            group = finite_time == current_time
            group_ranks = risk_rank[group]
            for rank in group_ranks:
                less = tree.prefix_sum(int(rank) - 1)
                equal = tree.range_sum(int(rank), int(rank))
                concordant += float(less) + 0.5 * float(equal)
                comparable += float(tree.total)
            for rank in group_ranks:
                tree.add(int(rank))

    if comparable == 0.0:
        return float("nan")
    return float(concordant / comparable)


def _basic_metrics(
    y_cancel: np.ndarray,
    p_cancel: np.ndarray,
    first_time: np.ndarray,
    first_event: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_samples": float(y_cancel.size),
        "cancel_rate": float(np.mean(y_cancel)) if y_cancel.size else 0.0,
        "avg_pred_cancel": float(np.mean(p_cancel)) if p_cancel.size else 0.0,
        "brier_cancel": float(np.mean((p_cancel - y_cancel) ** 2)) if y_cancel.size else 0.0,
    }
    ci = window_c_index(p_cancel, first_time, first_event)
    metrics["c_index_cancel"] = float(ci) if not np.isnan(ci) else -1.0
    metrics["pred_cancel_gap"] = abs(metrics["avg_pred_cancel"] - metrics["cancel_rate"])
    return metrics


def _add_prefixed(metrics: dict[str, float], prefix: str, values: dict[str, float]) -> None:
    for key, value in values.items():
        metrics[f"{prefix}_{key}"] = value


def _iter_regions(region_id: np.ndarray | None) -> Iterable[tuple[str, np.ndarray]]:
    if region_id is None:
        return []
    rid = np.asarray(region_id, dtype=np.int64)
    return [
        (REGION_NAMES.get(int(region), f"R{int(region) + 1}"), rid == int(region))
        for region in sorted(np.unique(rid).tolist())
    ]


def compute_window_metrics(
    y_cancel: np.ndarray,
    p_cancel: np.ndarray,
    stage: np.ndarray,
    first_time: np.ndarray,
    first_event: np.ndarray,
    region_id: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute overall, stage-wise, region-wise, and region-stage metrics."""
    y = np.asarray(y_cancel, dtype=np.float32)
    p = np.asarray(p_cancel, dtype=np.float32)
    s = np.asarray(stage, dtype=np.int64)
    t = np.asarray(first_time, dtype=np.float32)
    e = np.asarray(first_event, dtype=np.int64)
    r = None if region_id is None else np.asarray(region_id, dtype=np.int64)

    metrics = _basic_metrics(y, p, t, e)

    for stage_id in (0, 1):
        mask = s == stage_id
        if mask.any():
            _add_prefixed(metrics, f"stage_{stage_id}", _basic_metrics(y[mask], p[mask], t[mask], e[mask]))

    for region_name, region_mask in _iter_regions(r):
        if not region_mask.any():
            continue
        _add_prefixed(metrics, f"region_{region_name}", _basic_metrics(y[region_mask], p[region_mask], t[region_mask], e[region_mask]))
        for stage_id in (0, 1):
            mask = region_mask & (s == stage_id)
            if mask.any():
                _add_prefixed(metrics, f"region_{region_name}_stage_{stage_id}", _basic_metrics(y[mask], p[mask], t[mask], e[mask]))

    return metrics
