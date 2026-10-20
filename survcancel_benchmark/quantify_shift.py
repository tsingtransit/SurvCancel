#!/usr/bin/env python3
"""Quantify regional and temporal CLEAN58 distribution shift.

The analysis uses one deterministic landmark per eligible order and lifecycle
stage.  It compares source-domain and target-domain CLEAN58 vectors after
source-only standardization, excludes the explicit region identifier, controls
sample size, and reports:

* RBF maximum mean discrepancy (MMD);
* a same-source null MMD control;
* mean and median marginal 1-Wasserstein distances;
* group-wise Wasserstein distances and the largest-shifting features.

The primary comparisons are the public four-region LORO and global
chronological protocols.  Random-split and within-region chronological
comparisons are included as controls/diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance


PUBLIC_REGIONS = ("R1", "R2", "R3", "R4")
REGION_LABELS = {region: region for region in PUBLIC_REGIONS}
REGION_FEATURE_INDEX = 10
FEATURE_INDICES = np.asarray([i for i in range(58) if i != REGION_FEATURE_INDEX], dtype=np.int64)

FEATURE_NAMES = (
    "time_step_increment",
    "passenger_count",
    "pickup_x",
    "pickup_y",
    "dropoff_x",
    "dropoff_y",
    "trip_dx",
    "trip_dy",
    "trip_euclidean",
    "trip_manhattan",
    "region_id",
    "weekday_sin",
    "weekday_cos",
    "time_of_day_sin",
    "time_of_day_cos",
    "weekend",
    "focal_elapsed_time",
    "status_pre_match",
    "status_post_match",
    "status_in_service",
    "status_other",
    "has_matched_vehicle",
    "active_order_count",
    "active_pre_match_count",
    "active_post_match_count",
    "active_in_service_count",
    "active_other_count",
    "active_total_passengers",
    "active_mean_passengers",
    "active_mean_elapsed",
    "active_max_elapsed",
    "active_matched_fraction",
    "active_pickup_distance_min",
    "active_pickup_distance_mean",
    "active_pickup_distance_median",
    "active_pickup_distance_max",
    "active_dropoff_distance_mean",
    "active_trip_distance_mean",
    "active_same_passenger_fraction",
    "available_vehicle_count",
    "available_capacity_sum",
    "available_capacity_mean",
    "available_capacity_min",
    "available_capacity_max",
    "vehicle_pickup_distance_min",
    "vehicle_pickup_distance_mean",
    "vehicle_pickup_distance_median",
    "vehicle_pickup_distance_max",
    "vehicle_dropoff_distance_mean",
    "nearest_vehicle_capacity",
    "nearest_vehicle_distance_1",
    "nearest_vehicle_distance_2",
    "nearest_vehicle_distance_3",
    "supply_demand_ratio",
    "capacity_demand_ratio",
    "feasible_vehicle_fraction",
    "nearest_feasible_vehicle_distance",
    "trip_to_nearest_vehicle_ratio",
)

FEATURE_GROUPS = {
    "time_step": (0,),
    "order_trip_time": tuple(range(1, 10)) + tuple(range(11, 16)),
    "focal_state": tuple(range(16, 22)),
    "active_orders": tuple(range(22, 39)),
    "vehicles": tuple(range(39, 53)),
    "supply_feasibility": tuple(range(53, 58)),
}

STAGE_NAMES = {0: "stage0", 1: "stage1"}

@dataclass(frozen=True)
class Comparison:
    name: str
    family: str
    source_indices: np.ndarray
    target_indices: np.ndarray
    source_description: str
    target_description: str
    region: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--landmark-seed", type=int, default=42)
    parser.add_argument("--analysis-seed", type=int, default=20260726)
    parser.add_argument("--sample-size", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--bandwidth-subsample", type=int, default=1000)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--chrono-train-end", default="2025-12-31")
    parser.add_argument("--chrono-test-start", default="2026-01-13")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["id", "loro", "chrono_global", "chrono_region"],
        choices=["id", "loro", "chrono_global", "chrono_region"],
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_snapshot_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask > 0.5)


def sample_stage_landmarks(
    *,
    order_index: int,
    snapshot_times: np.ndarray,
    seq_mask: np.ndarray,
    match_time: float,
    cancel_time: float,
    pickup_time: float,
    seed: int,
    matched_decay: float = 0.2,
) -> tuple[int | None, int | None]:
    """Use the uniform/exponential policy with a deterministic per-order RNG.

    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(order_index)]))
    valid = valid_snapshot_indices(seq_mask)
    if valid.size == 0:
        return None, None

    unmatched_ends = [t for t in (match_time, cancel_time, pickup_time) if t >= 0.0]
    unmatched_candidates = (
        valid[snapshot_times[valid] < min(unmatched_ends)]
        if unmatched_ends
        else valid
    )
    unmatched = (
        None
        if unmatched_candidates.size == 0
        else int(rng.choice(unmatched_candidates))
    )

    matched: int | None = None
    if match_time >= 0.0:
        matched_ends = [t for t in (cancel_time, pickup_time) if t >= 0.0]
        if matched_ends:
            matched_candidates = valid[
                (snapshot_times[valid] >= match_time)
                & (snapshot_times[valid] < min(matched_ends))
            ]
        else:
            matched_candidates = valid[snapshot_times[valid] >= match_time]
        if matched_candidates.size:
            rel = np.arange(matched_candidates.size, dtype=np.float64)
            weights = np.exp(-matched_decay * rel)
            weights /= weights.sum()
            matched = int(
                matched_candidates[
                    int(rng.choice(matched_candidates.size, p=weights))
                ]
            )
    return unmatched, matched


def build_landmark_cache(
    data: np.lib.npyio.NpzFile,
    cache_path: Path,
    seed: int,
) -> dict[str, np.ndarray]:
    regions = np.asarray(data["region"]).astype(str)
    public_orders = np.flatnonzero(np.isin(regions, PUBLIC_REGIONS))
    seq_mask = np.asarray(data["seq_mask"])
    snapshot_times = np.asarray(data["snapshot_times"])
    match_time = np.asarray(data["match_time"])
    pickup_time = np.asarray(data["pickup_time"])
    cancel_time = np.asarray(data["cancel_time"])

    order_out: list[int] = []
    stage_out: list[int] = []
    tau_idx_out: list[int] = []
    for pos, order_index in enumerate(public_orders.tolist(), start=1):
        unmatched, matched = sample_stage_landmarks(
            order_index=order_index,
            snapshot_times=snapshot_times[order_index],
            seq_mask=seq_mask[order_index],
            match_time=float(match_time[order_index]),
            cancel_time=float(cancel_time[order_index]),
            pickup_time=float(pickup_time[order_index]),
            seed=seed,
        )
        for stage, tau_idx in ((0, unmatched), (1, matched)):
            if tau_idx is not None:
                order_out.append(order_index)
                stage_out.append(stage)
                tau_idx_out.append(tau_idx)
        if pos % 25000 == 0:
            print(f"[landmarks] sampled {pos:,}/{public_orders.size:,} orders", flush=True)

    order_arr = np.asarray(order_out, dtype=np.int64)
    stage_arr = np.asarray(stage_out, dtype=np.int8)
    tau_arr = np.asarray(tau_idx_out, dtype=np.int32)
    print("[landmarks] loading CLEAN58 tensor", flush=True)
    x = np.asarray(data["X"])
    vectors = x[order_arr, tau_arr][:, FEATURE_INDICES].astype(np.float32, copy=False)
    if not np.all(np.isfinite(vectors)):
        raise ValueError("Non-finite CLEAN58 landmark values detected")

    payload = {
        "order_index": order_arr,
        "stage": stage_arr,
        "tau_step_index": tau_arr,
        "features": vectors,
        "feature_indices": FEATURE_INDICES,
        "landmark_seed": np.asarray(seed, dtype=np.int64),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(
        f"[landmarks] saved {vectors.shape[0]:,} x {vectors.shape[1]} vectors to {cache_path}",
        flush=True,
    )
    return payload


def load_or_build_landmarks(
    data: np.lib.npyio.NpzFile,
    cache_path: Path,
    seed: int,
    force: bool,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and not force:
        loaded = np.load(cache_path, allow_pickle=False)
        cached_seed = int(np.asarray(loaded["landmark_seed"]).item())
        if cached_seed != seed:
            raise ValueError(
                f"Landmark cache seed is {cached_seed}, requested {seed}; use --force-landmarks"
            )
        return {key: np.asarray(loaded[key]) for key in loaded.files}
    return build_landmark_cache(data, cache_path, seed)


def build_comparisons(
    data: np.lib.npyio.NpzFile,
    split: dict[str, Any],
    train_end_date: str = "2025-12-31",
    test_start_date: str = "2026-01-13",
) -> list[Comparison]:
    regions = np.asarray(data["region"]).astype(str)
    dates = np.asarray(data["order_date"]).astype(str)
    public = np.isin(regions, PUBLIC_REGIONS)
    train = np.asarray(split["train_indices"], dtype=np.int64)
    test = np.asarray(split["test_indices"], dtype=np.int64)
    public_train = train[public[train]]
    public_test = test[public[test]]

    comparisons = [
        Comparison(
            name="id_random",
            family="id",
            source_indices=public_train,
            target_indices=public_test,
            source_description="public-four ID train orders",
            target_description="public-four ID test orders",
        )
    ]
    for region in PUBLIC_REGIONS:
        comparisons.append(
            Comparison(
                name=f"loro_{REGION_LABELS[region]}",
                family="loro",
                source_indices=public_train[regions[public_train] != region],
                target_indices=public_test[regions[public_test] == region],
                source_description=f"ID-train orders from the other three public regions",
                target_description=f"held-out {REGION_LABELS[region]} ID-test orders",
                region=region,
            )
        )

    all_indices = np.arange(regions.shape[0], dtype=np.int64)
    chrono_train = all_indices[public & (dates != "") & (dates <= train_end_date)]
    chrono_test = all_indices[public & (dates >= test_start_date)]
    comparisons.append(
        Comparison(
            name="chrono_global",
            family="chrono_global",
            source_indices=chrono_train,
            target_indices=chrono_test,
            source_description=f"public-four orders through {train_end_date}",
            target_description=f"public-four orders from {test_start_date}",
        )
    )

    for region in PUBLIC_REGIONS:
        region_indices = all_indices[regions == region]
        ordered = region_indices[np.lexsort((region_indices, dates[region_indices]))]
        n = ordered.size
        train_end = int(round(0.6 * n))
        test_start = int(round(0.8 * n))
        comparisons.append(
            Comparison(
                name=f"chrono_{REGION_LABELS[region]}",
                family="chrono_region",
                source_indices=ordered[:train_end],
                target_indices=ordered[test_start:],
                source_description=f"earliest 60% of {REGION_LABELS[region]} orders",
                target_description=f"latest 20% of {REGION_LABELS[region]} orders",
                region=region,
            )
        )
    return comparisons


def map_order_sets(
    landmark_order: np.ndarray,
    source_orders: np.ndarray,
    target_orders: np.ndarray,
    stage: int,
    landmark_stage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stage_mask = landmark_stage == stage
    source_rows = np.flatnonzero(stage_mask & np.isin(landmark_order, source_orders))
    target_rows = np.flatnonzero(stage_mask & np.isin(landmark_order, target_orders))
    return source_rows, target_rows


def standardize_source(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = source.astype(np.float64).mean(axis=0)
    std = source.astype(np.float64).std(axis=0)
    active = np.isfinite(std) & (std > 1e-8)
    if active.sum() < 2:
        raise ValueError("Fewer than two nonconstant features in source landmarks")
    source_z = (source[:, active].astype(np.float64) - mean[active]) / std[active]
    target_z = (target[:, active].astype(np.float64) - mean[active]) / std[active]
    return source_z, target_z, active, std


def median_bandwidth(
    source: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    max_points: int,
) -> float:
    n_each = max(2, min(max_points // 2, source.shape[0], target.shape[0]))
    x = source[rng.choice(source.shape[0], n_each, replace=False)]
    y = target[rng.choice(target.shape[0], n_each, replace=False)]
    distances = pdist(np.vstack([x, y]), metric="euclidean")
    positive = distances[distances > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def rbf_mmd_from_distances(
    d_xx: np.ndarray,
    d_yy: np.ndarray,
    d_xy: np.ndarray,
    sigma: float,
) -> tuple[float, float]:
    denom = 2.0 * sigma * sigma
    mmd2 = float(
        np.exp(-d_xx / denom).mean()
        + np.exp(-d_yy / denom).mean()
        - 2.0 * np.exp(-d_xy / denom).mean()
    )
    return math.sqrt(max(mmd2, 0.0)), mmd2


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
    }


def analyze_stratum(
    source: np.ndarray,
    target: np.ndarray,
    *,
    feature_indices: np.ndarray,
    sample_size: int,
    repeats: int,
    bandwidth_subsample: int,
    seed: int,
    top_features: int,
) -> dict[str, Any]:
    source_z, target_z, active_mask, raw_std = standardize_source(source, target)
    active_indices = feature_indices[active_mask]
    rng = np.random.default_rng(seed)
    sigma = median_bandwidth(source_z, target_z, rng, bandwidth_subsample)
    n = min(sample_size, source_z.shape[0] // 2, target_z.shape[0])
    if n < 100:
        raise ValueError(
            f"Insufficient landmarks for controlled analysis: source={source_z.shape[0]}, "
            f"target={target_z.shape[0]}, effective n={n}"
        )

    metrics: dict[str, list[float]] = {
        "mmd": [],
        "mmd2": [],
        "mmd_half_bandwidth": [],
        "mmd_double_bandwidth": [],
        "null_mmd": [],
        "wasserstein_mean": [],
        "wasserstein_median": [],
        "null_wasserstein_mean": [],
    }
    per_feature_w: list[np.ndarray] = []

    for _ in range(repeats):
        source_idx = rng.choice(source_z.shape[0], 2 * n, replace=False)
        target_idx = rng.choice(target_z.shape[0], n, replace=False)
        x = source_z[source_idx[:n]]
        x_null = source_z[source_idx[n:]]
        y = target_z[target_idx]

        d_xx = cdist(x, x, metric="sqeuclidean")
        d_yy = cdist(y, y, metric="sqeuclidean")
        d_xy = cdist(x, y, metric="sqeuclidean")
        mmd, mmd2 = rbf_mmd_from_distances(d_xx, d_yy, d_xy, sigma)
        metrics["mmd"].append(mmd)
        metrics["mmd2"].append(mmd2)
        metrics["mmd_half_bandwidth"].append(
            rbf_mmd_from_distances(d_xx, d_yy, d_xy, sigma * 0.5)[0]
        )
        metrics["mmd_double_bandwidth"].append(
            rbf_mmd_from_distances(d_xx, d_yy, d_xy, sigma * 2.0)[0]
        )

        d_nn = cdist(x_null, x_null, metric="sqeuclidean")
        d_xn = cdist(x, x_null, metric="sqeuclidean")
        metrics["null_mmd"].append(
            rbf_mmd_from_distances(d_xx, d_nn, d_xn, sigma)[0]
        )

        feature_w = np.asarray(
            [wasserstein_distance(x[:, j], y[:, j]) for j in range(x.shape[1])],
            dtype=np.float64,
        )
        null_feature_w = np.asarray(
            [
                wasserstein_distance(x[:, j], x_null[:, j])
                for j in range(x.shape[1])
            ],
            dtype=np.float64,
        )
        per_feature_w.append(feature_w)
        metrics["wasserstein_mean"].append(float(feature_w.mean()))
        metrics["wasserstein_median"].append(float(np.median(feature_w)))
        metrics["null_wasserstein_mean"].append(float(null_feature_w.mean()))

    feature_w_mean = np.stack(per_feature_w).mean(axis=0)
    ranked = np.argsort(feature_w_mean)[::-1]
    top = [
        {
            "index": int(active_indices[j]),
            "name": FEATURE_NAMES[int(active_indices[j])],
            "wasserstein": float(feature_w_mean[j]),
            "source_raw_sd": float(raw_std[np.flatnonzero(active_mask)[j]]),
        }
        for j in ranked[:top_features]
    ]
    by_index = {
        int(feature_index): float(feature_w_mean[pos])
        for pos, feature_index in enumerate(active_indices)
    }
    group_values: dict[str, dict[str, float | int]] = {}
    for group, indices in FEATURE_GROUPS.items():
        values = [by_index[index] for index in indices if index in by_index]
        group_values[group] = {
            "mean_wasserstein": float(np.mean(values)) if values else float("nan"),
            "median_wasserstein": float(np.median(values)) if values else float("nan"),
            "n_active_features": len(values),
        }

    return {
        "n_source": int(source.shape[0]),
        "n_target": int(target.shape[0]),
        "sample_size_per_domain": int(n),
        "n_active_features": int(active_mask.sum()),
        "excluded_constant_feature_indices": feature_indices[~active_mask].astype(int).tolist(),
        "rbf_bandwidth": sigma,
        "rbf_bandwidth_rule": "median pooled pairwise Euclidean distance after source-only z-scoring",
        "metrics": {key: summarize(value) for key, value in metrics.items()},
        "group_wasserstein": group_values,
        "top_shifted_features": top,
    }


def balanced_overall(
    source_s0: np.ndarray,
    source_s1: np.ndarray,
    target_s0: np.ndarray,
    target_s1: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create large 50/50 stage pools; repeated metric sampling happens later."""
    rng = np.random.default_rng(seed)
    per_stage = min(
        source_s0.shape[0],
        source_s1.shape[0],
        target_s0.shape[0],
        target_s1.shape[0],
    )
    # Retain all feasible information but equalize stage composition in each domain.
    source = np.vstack(
        [
            source_s0[rng.choice(source_s0.shape[0], per_stage, replace=False)],
            source_s1[rng.choice(source_s1.shape[0], per_stage, replace=False)],
        ]
    )
    target = np.vstack(
        [
            target_s0[rng.choice(target_s0.shape[0], per_stage, replace=False)],
            target_s1[rng.choice(target_s1.shape[0], per_stage, replace=False)],
        ]
    )
    # Shuffle so each repeated sample remains stage-balanced in expectation.
    return source[rng.permutation(source.shape[0])], target[rng.permutation(target.shape[0])]


def markdown_table(rows: list[list[str]], header: list[str]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in header[1:]]) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def write_reports(payload: dict[str, Any], output_dir: Path) -> None:
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = finite_json(payload)
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    rows = []
    for comparison, result in payload["results"].items():
        for stratum, scored in result["strata"].items():
            for metric, values in scored.get("metrics", {}).items():
                rows.append(dict(comparison=comparison, stratum=stratum, metric=metric, **values))
    if rows:
        with (output_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.sample_size < 100:
        raise ValueError("--sample-size must be at least 100")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if args.bandwidth_subsample < 4:
        raise ValueError("--bandwidth-subsample must be at least 4")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    print(f"[load] {args.data}", flush=True)
    data = np.load(args.data, allow_pickle=True)
    split = load_json(args.split)
    regions = np.asarray(data["region"]).astype(str)
    if not np.isin(regions, PUBLIC_REGIONS).all():
        raise ValueError("Use a public CLEAN58 file with region labels R1--R4.")
    cache_path = args.output_dir / f"landmark_vectors_seed{args.landmark_seed}.npz"
    landmarks = load_or_build_landmarks(
        data, cache_path, args.landmark_seed, False
    )
    comparisons = [
        item
        for item in build_comparisons(data, split, args.chrono_train_end, args.chrono_test_start)
        if item.family in set(args.families)
    ]

    landmark_order = landmarks["order_index"]
    landmark_stage = landmarks["stage"]
    features = landmarks["features"]
    feature_indices = landmarks["feature_indices"]
    results: dict[str, Any] = {}

    for comparison_index, comparison in enumerate(comparisons):
        print(f"[analyze] {comparison.name}", flush=True)
        stage_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        strata: dict[str, Any] = {}
        for stage in (0, 1):
            source_rows, target_rows = map_order_sets(
                landmark_order,
                comparison.source_indices,
                comparison.target_indices,
                stage,
                landmark_stage,
            )
            source = features[source_rows]
            target = features[target_rows]
            stage_arrays[stage] = (source, target)
            strata[STAGE_NAMES[stage]] = analyze_stratum(
                source,
                target,
                feature_indices=feature_indices,
                sample_size=args.sample_size,
                repeats=args.repeats,
                bandwidth_subsample=args.bandwidth_subsample,
                seed=args.analysis_seed + comparison_index * 101 + stage,
                top_features=args.top_features,
            )

        source_overall, target_overall = balanced_overall(
            stage_arrays[0][0],
            stage_arrays[1][0],
            stage_arrays[0][1],
            stage_arrays[1][1],
            sample_size=args.sample_size,
            seed=args.analysis_seed + comparison_index * 101 + 50,
        )
        strata["overall_balanced"] = analyze_stratum(
            source_overall,
            target_overall,
            feature_indices=feature_indices,
            sample_size=args.sample_size,
            repeats=args.repeats,
            bandwidth_subsample=args.bandwidth_subsample,
            seed=args.analysis_seed + comparison_index * 101 + 51,
            top_features=args.top_features,
        )
        results[comparison.name] = {
            "family": comparison.family,
            "source_description": comparison.source_description,
            "target_description": comparison.target_description,
            "n_source_orders": int(comparison.source_indices.size),
            "n_target_orders": int(comparison.target_indices.size),
            "region": comparison.region,
            "strata": strata,
        }

    payload = {
        "status": "complete",
        "data_path": str(args.data.resolve()),
        "split_path": str(args.split.resolve()),
        "public_regions": list(PUBLIC_REGIONS),
        "region_labels": REGION_LABELS,
        "region_feature_excluded": REGION_FEATURE_INDEX,
        "feature_indices_analyzed": feature_indices.astype(int).tolist(),
        "landmark_seed": args.landmark_seed,
        "analysis_seed": args.analysis_seed,
        "sample_size_requested": args.sample_size,
        "repeats": args.repeats,
        "families": args.families,
        "method": {
            "landmark": "one deterministic per-order/per-stage landmark; stage0 uniform, stage1 exponential decay 0.2",
            "scaling": "source-domain landmark mean/std per comparison and stratum",
            "overall": "50/50 stage-balanced pools; repeated uniform subsamples are balanced in expectation",
            "mmd": "biased nonnegative RBF MMD; median bandwidth; equal sample size",
            "mmd_sensitivity": "also evaluated at 0.5x and 2x median bandwidth",
            "wasserstein": "mean/median empirical 1D Wasserstein over active standardized features",
            "uncertainty": "95% percentile interval over repeated equal-size subsamples",
            "null": "same-size disjoint source-vs-source subsamples",
        },
        "results": results,
        "chronological_cutoffs": {"train_end": args.chrono_train_end, "test_start": args.chrono_test_start},
    }
    data.close()
    write_reports(payload, args.output_dir)
    print(f"[done] {args.output_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
