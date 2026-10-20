#!/usr/bin/env python3
"""Prepare Clean58 benchmark artifacts from the Hugging Face SurvCancel JSONL release."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

UNMATCHED_STAGE = 0
MATCHED_STAGE = 1
LABEL_STAY = 0
LABEL_PROGRESS = 1
LABEL_CANCEL = 2
TIME_STEP_SECONDS = 10.0
UNMATCHED_H_SECONDS = 60.0
MATCHED_H_SECONDS = 30.0
WINDOWS = {
    "h060_030": (60.0, 30.0),
    "h060_060": (60.0, 60.0),
    "h120_060": (120.0, 60.0),
    "h300_180": (300.0, 180.0),
    "h600_300": (600.0, 300.0),
}
DEFAULT_WINDOW = "h060_030"
REGION_FILES = ("train-R1.jsonl", "train-R2.jsonl", "train-R3.jsonl", "train-R4.jsonl")
PROTOCOL_NAMES = ("id", "loro_R1", "loro_R2", "loro_R3", "loro_R4", "chrono_global", "chrono_R1", "chrono_R2", "chrono_R3", "chrono_R4")
CHRONO_TRAIN_END = "2025-12-31"
CHRONO_VAL_START = "2026-01-01"
CHRONO_VAL_END = "2026-01-12"
CHRONO_TEST_START = "2026-01-13"


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def _euclidean_distance(x1: Any, y1: Any, x2: Any, y2: Any) -> float:
    return float(np.hypot(_safe_float(x2) - _safe_float(x1), _safe_float(y2) - _safe_float(y1)))


def _manhattan_distance(x1: Any, y1: Any, x2: Any, y2: Any) -> float:
    return float(abs(_safe_float(x2) - _safe_float(x1)) + abs(_safe_float(y2) - _safe_float(y1)))


def _stat_or_zero(values: list[float], reducer) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return 0.0 if arr.size == 0 else float(reducer(arr))


def _encode_cyclical(value: float, period: float) -> list[float]:
    angle = 2.0 * np.pi * float(value) / float(period)
    return [float(np.sin(angle)), float(np.cos(angle))]


def _encode_status_one_hot(status: str | None) -> list[float]:
    status = status or ""
    return [
        1.0 if status == "pre_match" else 0.0,
        1.0 if status == "post_match" else 0.0,
        1.0 if status == "in_service" else 0.0,
        1.0 if status not in {"pre_match", "post_match", "in_service"} else 0.0,
    ]


def _stable_int_id(text: Any) -> int:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _extract_order_static_part(order: dict[str, Any], region_numeric: int) -> list[float]:
    pickup_x = _safe_float(order.get("pickup_x", 0))
    pickup_y = _safe_float(order.get("pickup_y", 0))
    dropoff_x = _safe_float(order.get("dropoff_x", 0))
    dropoff_y = _safe_float(order.get("dropoff_y", 0))
    pax_num = _safe_float(order.get("pax_num", 0))
    trip_dx = dropoff_x - pickup_x
    trip_dy = dropoff_y - pickup_y
    trip_distance = _euclidean_distance(pickup_x, pickup_y, dropoff_x, dropoff_y)
    trip_manhattan = _manhattan_distance(pickup_x, pickup_y, dropoff_x, dropoff_y)

    time_feats = [0.0, 0.0, 0.0, 0.0, 0.0]
    snapshots = order.get("snapshots", []) or []
    if snapshots:
        dt = _parse_dt(snapshots[0].get("snapshot_ts"))
        if dt is not None:
            weekday = dt.weekday()
            seconds_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
            time_feats = [
                *_encode_cyclical(weekday, 7.0),
                *_encode_cyclical(seconds_of_day, 24.0 * 3600.0),
                1.0 if weekday >= 5 else 0.0,
            ]

    return [
        pax_num,
        pickup_x,
        pickup_y,
        dropoff_x,
        dropoff_y,
        trip_dx,
        trip_dy,
        trip_distance,
        trip_manhattan,
        float(region_numeric),
        *time_feats,
    ]


def _snapshot_features(snapshot: dict[str, Any], order: dict[str, Any]) -> list[float]:
    pickup_x = _safe_float(order.get("pickup_x", 0))
    pickup_y = _safe_float(order.get("pickup_y", 0))
    dropoff_x = _safe_float(order.get("dropoff_x", 0))
    dropoff_y = _safe_float(order.get("dropoff_y", 0))
    pax_num = _safe_float(order.get("pax_num", 0))
    trip_distance = _euclidean_distance(pickup_x, pickup_y, dropoff_x, dropoff_y)

    elapsed_time = _safe_float(snapshot.get("elapsed_time", 0.0))
    status = snapshot.get("status_at_t", "")
    has_matched = 1.0 if snapshot.get("matched_vehicle_id") is not None else 0.0

    active_orders = snapshot.get("active_orders_list", []) or []
    n_active = float(len(active_orders))
    n_pre = n_post = n_service = n_other = 0.0
    total_pax = matched_count = same_pax = 0.0
    elapsed_vals: list[float] = []
    pickup_dists: list[float] = []
    dropoff_dists: list[float] = []
    trip_dists: list[float] = []

    for active in active_orders:
        active_status = active.get("status_at_t", "")
        if active_status == "pre_match":
            n_pre += 1.0
        elif active_status == "post_match":
            n_post += 1.0
        elif active_status == "in_service":
            n_service += 1.0
        else:
            n_other += 1.0
        active_pax = _safe_float(active.get("pax_num", 0))
        total_pax += active_pax
        same_pax += 1.0 if active_pax == pax_num else 0.0
        matched_count += 1.0 if active.get("matched_vehicle_id") is not None else 0.0
        elapsed_vals.append(_safe_float(active.get("elapsed_time", 0.0)))
        pickup_dists.append(_euclidean_distance(pickup_x, pickup_y, active.get("pickup_x", 0), active.get("pickup_y", 0)))
        dropoff_dists.append(_euclidean_distance(dropoff_x, dropoff_y, active.get("dropoff_x", 0), active.get("dropoff_y", 0)))
        trip_dists.append(_euclidean_distance(active.get("pickup_x", 0), active.get("pickup_y", 0), active.get("dropoff_x", 0), active.get("dropoff_y", 0)))

    mean_pax = total_pax / n_active if n_active > 0 else 0.0
    matched_ratio = matched_count / n_active if n_active > 0 else 0.0
    same_pax_ratio = same_pax / n_active if n_active > 0 else 0.0

    vehicles = snapshot.get("available_vehicles_list", []) or []
    n_vehicles = float(len(vehicles))
    capacities: list[float] = []
    vehicle_pickup_dists: list[float] = []
    vehicle_dropoff_dists: list[float] = []
    feasible_dists: list[float] = []
    sorted_pairs: list[tuple[float, float]] = []
    for vehicle in vehicles:
        capacity = _safe_float(vehicle.get("remaining_capacity", 0))
        capacities.append(capacity)
        pickup_dist = _euclidean_distance(pickup_x, pickup_y, vehicle.get("vehicle_x", 0), vehicle.get("vehicle_y", 0))
        vehicle_pickup_dists.append(pickup_dist)
        vehicle_dropoff_dists.append(_euclidean_distance(dropoff_x, dropoff_y, vehicle.get("vehicle_x", 0), vehicle.get("vehicle_y", 0)))
        sorted_pairs.append((pickup_dist, capacity))
        if capacity >= pax_num:
            feasible_dists.append(pickup_dist)

    sorted_pairs.sort(key=lambda item: item[0])
    nearest_capacity = float(sorted_pairs[0][1]) if sorted_pairs else 0.0
    topk = [pair[0] for pair in sorted_pairs[:3]]
    while len(topk) < 3:
        topk.append(0.0)

    capacity_sum = float(np.sum(capacities)) if capacities else 0.0
    supply_demand = n_vehicles / max(n_active, 1.0) if n_active > 0 else n_vehicles
    capacity_demand = capacity_sum / max(total_pax, 1.0) if n_active > 0 else capacity_sum
    enough_capacity = float(np.mean(np.asarray(capacities, dtype=np.float32) >= pax_num)) if capacities else 0.0
    nearest_dist = topk[0]
    trip_to_nearest = trip_distance / max(nearest_dist, 1.0) if nearest_dist > 0 else 0.0

    return [
        elapsed_time,
        *_encode_status_one_hot(status),
        has_matched,
        n_active,
        n_pre,
        n_post,
        n_service,
        n_other,
        total_pax,
        mean_pax,
        _stat_or_zero(elapsed_vals, np.mean),
        _stat_or_zero(elapsed_vals, np.max),
        matched_ratio,
        _stat_or_zero(pickup_dists, np.min),
        _stat_or_zero(pickup_dists, np.mean),
        _stat_or_zero(pickup_dists, np.median),
        _stat_or_zero(pickup_dists, np.max),
        _stat_or_zero(dropoff_dists, np.mean),
        _stat_or_zero(trip_dists, np.mean),
        same_pax_ratio,
        n_vehicles,
        capacity_sum,
        _stat_or_zero(capacities, np.mean),
        _stat_or_zero(capacities, np.min),
        _stat_or_zero(capacities, np.max),
        _stat_or_zero(vehicle_pickup_dists, np.min),
        _stat_or_zero(vehicle_pickup_dists, np.mean),
        _stat_or_zero(vehicle_pickup_dists, np.median),
        _stat_or_zero(vehicle_pickup_dists, np.max),
        _stat_or_zero(vehicle_dropoff_dists, np.mean),
        nearest_capacity,
        *topk,
        supply_demand,
        capacity_demand,
        enough_capacity,
        _stat_or_zero(feasible_dists, np.min),
        trip_to_nearest,
    ]


def _extract_sequence(order: dict[str, Any], region_numeric: int, max_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.zeros((max_length, 58), dtype=np.float32)
    mask = np.zeros(max_length, dtype=np.float32)
    times = np.zeros(max_length, dtype=np.float32)
    static = _extract_order_static_part(order, region_numeric)
    snapshots = order.get("snapshots", []) or []
    prev_elapsed = 0.0
    last_time = 0.0
    for idx, snapshot in enumerate(snapshots[:max_length]):
        elapsed = _safe_float(snapshot.get("elapsed_time", 0.0))
        x[idx, 0] = elapsed - prev_elapsed
        prev_elapsed = elapsed
        x[idx, 1:16] = static
        x[idx, 16:] = _snapshot_features(snapshot, order)
        mask[idx] = 1.0
        times[idx] = elapsed
        last_time = elapsed
    return x, mask, times, last_time


def _infer_event_time(order: dict[str, Any]) -> float:
    snapshots = order.get("snapshots", []) or []
    return _safe_float(snapshots[-1].get("elapsed_time", 0.0)) if snapshots else 0.0


def _infer_match_time(order: dict[str, Any]) -> float:
    for snapshot in order.get("snapshots", []) or []:
        if snapshot.get("status_at_t") != "pre_match" or snapshot.get("matched_vehicle_id") is not None:
            return _safe_float(snapshot.get("elapsed_time", 0.0))
    return -1.0


def _infer_pickup_time(order: dict[str, Any]) -> float:
    for snapshot in order.get("snapshots", []) or []:
        if snapshot.get("status_at_t") == "in_service":
            return _safe_float(snapshot.get("elapsed_time", 0.0))
    return _infer_event_time(order) if order.get("end_type") == "completed" else -1.0


def _infer_cancel_time(order: dict[str, Any]) -> float:
    return _infer_event_time(order) if order.get("end_type") in {"pre_match_cancel", "post_match_cancel"} else -1.0


def _region_files(release_dir: Path) -> list[tuple[str, Path]]:
    files = []
    for idx, name in enumerate(REGION_FILES, start=1):
        path = release_dir / name
        if path.exists():
            files.append((f"R{idx}", path))
    if files:
        return files

    sample_path = release_dir / "sample_records.jsonl"
    if sample_path.exists():
        return [("R1", sample_path)]
    nested_sample_path = release_dir / "sample" / "sample_records.jsonl"
    if nested_sample_path.exists():
        return [("R1", nested_sample_path)]

    raise FileNotFoundError(f"No train-R*.jsonl or sample_records.jsonl files found in {release_dir}")


def _iter_records(release_dir: Path, limit_orders: int | None) -> Iterator[tuple[str, dict[str, Any], str, int]]:
    """Stream JSONL orders without retaining the entire release in memory."""
    count = 0
    for region_name, path in _region_files(release_dir):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    order = json.loads(line)
                    record_region = region_name
                    if path.name == "sample_records.jsonl":
                        record_region = order.get("_sample_source_region", region_name)
                        if record_region not in {"R1", "R2", "R3", "R4"}:
                            raise ValueError(f"Invalid sample source region: {record_region!r}")
                    yield record_region, order, path.relative_to(release_dir).as_posix(), line_number
                    count += 1
                    if limit_orders is not None and count >= limit_orders:
                        return


def build_clean58(release_dir: Path, output: Path, max_length: int, limit_orders: int | None) -> None:
    if max_length <= 0 or (limit_orders is not None and limit_orders <= 0):
        raise ValueError("max_length and limit_orders must be positive.")
    # Count physical nonempty records first, then parse one order at a time.
    # Only the compact numeric arrays and public order identifiers stay in RAM.
    n = 0
    for _, path in _region_files(release_dir):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    n += 1
                    if limit_orders is not None and n >= limit_orders:
                        break
        if limit_orders is not None and n >= limit_orders:
            break
    if n == 0:
        raise ValueError("No orders found in release JSONL files.")

    x = np.zeros((n, max_length, 58), dtype=np.float32)
    seq_mask = np.zeros((n, max_length), dtype=np.float32)
    snapshot_times = np.zeros((n, max_length), dtype=np.float32)
    order_ids = np.zeros(n, dtype=np.int64)
    region_ids = np.zeros(n, dtype=np.int32)
    event_time = np.zeros(n, dtype=np.float32)
    match_time = np.full(n, -1.0, dtype=np.float32)
    pickup_time = np.full(n, -1.0, dtype=np.float32)
    cancel_time = np.full(n, -1.0, dtype=np.float32)
    last_meas_time = np.zeros(n, dtype=np.float32)
    end_type = np.empty(n, dtype="<U32")
    region = np.empty(n, dtype="<U16")
    order_date = np.empty(n, dtype="<U16")
    public_order_ids: list[str] = []
    source_files: list[str] = []
    source_line_numbers = np.zeros(n, dtype=np.int64)
    seen_ids: set[int] = set()

    region_to_id = {f"R{i}": i - 1 for i in range(1, 5)}
    records = _iter_records(release_dir, limit_orders)
    iterator = tqdm(enumerate(records), total=n, desc="Building Clean58") if tqdm else enumerate(records)
    for idx, (region_name, order, source_file, line_number) in iterator:
        region_numeric = region_to_id.get(region_name, 0)
        xi, mi, ti, lmt = _extract_sequence(order, region_numeric, max_length)
        x[idx] = xi
        seq_mask[idx] = mi
        snapshot_times[idx] = ti
        if "order_id" not in order:
            raise ValueError(f"Missing public order_id in {region_name}, record {idx}.")
        public_id = str(order["order_id"])
        # The same public order string can recur across regions/days. Identify
        # each released record, rather than merging records by that string.
        identity = json.dumps([region_name, source_file, line_number, public_id], separators=(",", ":"))
        internal_id = _stable_int_id(identity)
        if internal_id in seen_ids:
            raise ValueError(f"Internal-ID hash collision at released record {idx}.")
        seen_ids.add(internal_id)
        public_order_ids.append(public_id)
        source_files.append(source_file)
        source_line_numbers[idx] = line_number
        order_ids[idx] = internal_id
        region_ids[idx] = region_numeric
        event_time[idx] = _infer_event_time(order)
        match_time[idx] = _infer_match_time(order)
        pickup_time[idx] = _infer_pickup_time(order)
        cancel_time[idx] = _infer_cancel_time(order)
        last_meas_time[idx] = lmt
        end_type[idx] = str(order.get("end_type", ""))
        region[idx] = region_name
        snapshots = order.get("snapshots", []) or []
        order_date[idx] = str(snapshots[0].get("snapshot_ts", ""))[:10] if snapshots else ""

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        order_ids=order_ids,
        public_order_ids=np.asarray(public_order_ids, dtype=str),
        source_files=np.asarray(source_files, dtype=str),
        source_line_numbers=source_line_numbers,
        X=x,
        seq_mask=seq_mask,
        snapshot_times=snapshot_times,
        region_ids=region_ids,
        event_time=event_time,
        end_type=end_type,
        match_time=match_time,
        pickup_time=pickup_time,
        cancel_time=cancel_time,
        last_meas_time=last_meas_time,
        region=region,
        order_date=order_date,
    )
    print(f"Saved Clean58 sequences to {output}")
    print(f"  orders={n:,} X={x.shape} snapshots={int(seq_mask.sum()):,}")


def _stratify_key(region: str, end_type: str, match_time: float) -> str:
    return f"{region}|{end_type}|matched={int(match_time >= 0.0)}"


def _allocate(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    raw = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64) * n
    counts = np.floor(raw).astype(int)
    for idx in np.argsort(-(raw - counts))[: n - int(counts.sum())]:
        counts[idx] += 1
    for split_idx in (1, 2):
        if counts[split_idx] == 0 and n >= 3:
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[split_idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def build_split(data_path: Path, output: Path, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    data = np.load(data_path, allow_pickle=True)
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, (reg, etype, mt) in enumerate(zip(data["region"], data["end_type"], data["match_time"])):
        grouped[_stratify_key(str(reg), str(etype), float(mt))].append(idx)
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(grouped):
        arr = np.asarray(grouped[key], dtype=np.int64)
        rng.shuffle(arr)
        n_train, n_val, n_test = _allocate(len(arr), train_ratio, val_ratio, test_ratio)
        splits["train"].extend(arr[:n_train].tolist())
        splits["val"].extend(arr[n_train:n_train+n_val].tolist())
        splits["test"].extend(arr[n_train+n_val:n_train+n_val+n_test].tolist())
    for key in splits:
        splits[key] = sorted(splits[key])
    payload = {
        "split_name": "clean58_release_split",
        "split_seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "stratification_keys": ["region", "end_type", "has_matched"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_orders": int(len(data["order_ids"])),
    }
    for key in ["train", "val", "test"]:
        idx = splits[key]
        payload[f"{key}_indices"] = idx
        payload[f"{key}_order_ids"] = data["order_ids"][idx].astype(np.int64).tolist()
        payload[f"n_{key}_orders"] = len(idx)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved split to {output}")
    print(f"  train={len(splits['train']):,} val={len(splits['val']):,} test={len(splits['test']):,}")


def _allocate_flexible(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    raw = np.asarray(ratios, dtype=np.float64) * n
    counts = np.floor(raw).astype(int)
    remainder = n - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def _stratified_indices(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    seed: int,
    ratios: tuple[float, float, float],
    force_eval_splits: bool,
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    index_set = set(int(i) for i in indices.tolist())
    for idx, (reg, etype, mt) in enumerate(zip(data["region"], data["end_type"], data["match_time"])):
        if idx in index_set:
            grouped[_stratify_key(str(reg), str(etype), float(mt))].append(idx)

    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(grouped):
        arr = np.asarray(grouped[key], dtype=np.int64)
        rng.shuffle(arr)
        if force_eval_splits:
            n_train, n_val, n_test = _allocate(len(arr), *ratios)
        else:
            n_train, n_val, n_test = _allocate_flexible(len(arr), ratios)
        splits["train"].extend(arr[:n_train].tolist())
        splits["val"].extend(arr[n_train:n_train + n_val].tolist())
        splits["test"].extend(arr[n_train + n_val:n_train + n_val + n_test].tolist())
    for key in splits:
        splits[key] = sorted(splits[key])
    return splits


def _split_payload(
    data: np.lib.npyio.NpzFile,
    protocol: str,
    seed: int,
    splits: dict[str, list[int]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "split_name": protocol,
        "split_seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_orders": int(len(data["order_ids"])),
        **metadata,
    }
    for key in ["train", "val", "test"]:
        idx = sorted(int(i) for i in splits.get(key, []))
        payload[f"{key}_indices"] = idx
        payload[f"{key}_order_ids"] = data["order_ids"][idx].astype(np.int64).tolist()
        payload[f"n_{key}_orders"] = len(idx)
    return payload


def _write_split_payload(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved split to {output}")
    print(
        f"  train={payload['n_train_orders']:,} "
        f"val={payload['n_val_orders']:,} test={payload['n_test_orders']:,}"
    )


def _protocol_paths(output_dir: Path, protocol: str, window: str = DEFAULT_WINDOW) -> tuple[Path, Path, Path]:
    base = output_dir / "protocols" / protocol
    if window != DEFAULT_WINDOW:
        base = base / window
    return base / "order_split.json", base / "dynamic_val_samples.npz", base / "dynamic_test_samples.npz"


def _build_id_split(data: np.lib.npyio.NpzFile, seed: int) -> tuple[dict[str, list[int]], dict[str, Any]]:
    indices = np.arange(len(data["order_ids"]), dtype=np.int64)
    splits = _stratified_indices(data, indices, seed, (0.70, 0.15, 0.15), force_eval_splits=True)
    metadata = {
        "protocol": "in_distribution",
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "stratification_keys": ["region", "end_type", "has_matched"],
    }
    return splits, metadata


def _build_loro_split(data: np.lib.npyio.NpzFile, seed: int, heldout_region: str,
                      base_split: dict[str, list[int]] | None = None) -> tuple[dict[str, list[int]], dict[str, Any]]:
    region = np.asarray(data["region"]).astype(str)
    if base_split is None:
        base_split, _ = _build_id_split(data, seed)
    splits = {}
    for part in ("train", "val", "test"):
        indices = np.asarray(base_split[part], dtype=np.int64)
        keep = region[indices] == heldout_region
        splits[part] = sorted(indices[keep if part == "test" else ~keep].tolist())
    metadata = {
        "protocol": "leave_one_region_out",
        "heldout_region": heldout_region,
        "base_protocol": "id",
        "test_subset": "heldout_region_id_test_orders",
    }
    return splits, metadata


def id_splits_for_loro(data, output_dir, seed, window=DEFAULT_WINDOW):
    """Use the saved ID assignment when available, including standalone LORO preparation."""
    for candidate in dict.fromkeys((_protocol_paths(output_dir, "id", window)[0],
                                    _protocol_paths(output_dir, "id", DEFAULT_WINDOW)[0])):
        if candidate.is_file():
            payload = json.loads(candidate.read_text())
            if payload.get("split_seed") != seed:
                raise ValueError("ID and LORO preparation must use the same split seed.")
            return {part: payload[f"{part}_indices"] for part in ("train", "val", "test")}
    return _build_id_split(data, seed)[0]


def _build_chrono_split(data: np.lib.npyio.NpzFile) -> tuple[dict[str, list[int]], dict[str, Any]]:
    dates = np.asarray(data["order_date"]).astype(str)
    indices = np.arange(len(dates), dtype=np.int64)
    train_idx = indices[(dates != "") & (dates <= CHRONO_TRAIN_END)]
    val_idx = indices[(dates >= CHRONO_VAL_START) & (dates <= CHRONO_VAL_END)]
    test_idx = indices[dates >= CHRONO_TEST_START]
    splits = {
        "train": train_idx.astype(np.int64).tolist(),
        "val": val_idx.astype(np.int64).tolist(),
        "test": test_idx.astype(np.int64).tolist(),
    }

    for key in splits:
        splits[key] = sorted(splits[key])
    metadata = {
        "protocol": "global_chronological",
        "split_key": "order_date",
        "train_end": CHRONO_TRAIN_END,
        "val_start": CHRONO_VAL_START,
        "val_end": CHRONO_VAL_END,
        "test_start": CHRONO_TEST_START,
    }
    return splits, metadata


def _build_region_chrono_split(data, region):
    dates = np.asarray(data["order_date"]).astype(str)
    indices = np.flatnonzero(np.asarray(data["region"]).astype(str) == region)
    if np.any(dates[indices] == ""):
        raise ValueError(f"Missing order dates in {region}")
    ordered = indices[np.lexsort((indices, dates[indices]))]
    n = len(ordered)
    nt, nv = int(round(n * 0.6)), int(round(n * 0.2))
    nt = min(max(nt, 1), n - 2) if n >= 3 else max(n - 1, 0)
    nv = min(max(nv, 1), n - nt - 1) if n - nt >= 2 else max(n - nt, 0)
    return {"train": ordered[:nt].tolist(), "val": ordered[nt:nt+nv].tolist(),
            "test": ordered[nt+nv:].tolist()}, {
                "protocol": "region_chronological", "region": region,
                "split_key": "order_date", "ratios": [0.6, 0.2, 0.2]}


def _validate_or_skip(protocol: str, splits: dict[str, list[int]], strict: bool) -> bool:
    missing = [key for key in ("train", "val", "test") if len(splits.get(key, [])) == 0]
    if not missing:
        return True
    message = f"Protocol {protocol} has empty split(s): {', '.join(missing)}"
    if strict:
        raise ValueError(message)
    print(f"[skip] {message}")
    return False


def build_protocol_artifacts(
    data_path: Path,
    output_dir: Path,
    protocols: list[str],
    seed: int,
    matched_decay: float,
    strict: bool,
    windows: list[str] | None = None,
) -> None:
    windows = list(dict.fromkeys(windows or [DEFAULT_WINDOW]))
    with np.load(data_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in (
            "order_ids", "region", "end_type", "match_time", "pickup_time",
            "cancel_time", "order_date", "snapshot_times", "seq_mask",
        )}
    id_splits = None
    for protocol in protocols:
        if protocol == "id":
            splits, metadata = _build_id_split(data, seed)
            id_splits = splits
        elif protocol.startswith("loro_R"):
            if id_splits is None:
                id_splits = (_build_id_split(data, seed)[0] if "id" in protocols
                             else id_splits_for_loro(data, output_dir, seed, windows[0]))
            splits, metadata = _build_loro_split(data, seed, protocol.removeprefix("loro_"), id_splits)
        elif protocol == "chrono_global":
            splits, metadata = _build_chrono_split(data)
        elif protocol.startswith("chrono_R"):
            splits, metadata = _build_region_chrono_split(data, protocol.removeprefix("chrono_"))
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        if not _validate_or_skip(protocol, splits, strict):
            continue

        metadata.update(
            identity_scheme="blake2b63(region,source_file,source_line_number,public_order_id)",
            eval_seed=seed,
            matched_sampling="exp",
            matched_decay=matched_decay,
        )
        for window in windows:
            unmatched_h, matched_h = WINDOWS[window]
            window_metadata = dict(metadata, window=window, unmatched_h_seconds=unmatched_h,
                                   matched_h_seconds=matched_h, time_step_seconds=TIME_STEP_SECONDS)
            payload = _split_payload(data, protocol, seed, splits, window_metadata)
            split_path, val_path, test_path = _protocol_paths(output_dir, protocol, window)
            _write_split_payload(payload, split_path)
            build_eval_samples(data_path, split_path, "val", val_path, seed, matched_decay,
                               unmatched_h, matched_h, base_data=data)
            build_eval_samples(data_path, split_path, "test", test_path, seed, matched_decay,
                               unmatched_h, matched_h, base_data=data)


def _expand_protocols(protocols: list[str]) -> tuple[list[str], bool]:
    if "all" not in protocols:
        return protocols, True
    expanded = list(PROTOCOL_NAMES)
    extra = [item for item in protocols if item != "all"]
    for item in extra:
        if item not in expanded:
            expanded.append(item)
    return expanded, False


def _steps(seconds: float) -> int:
    return int(round(seconds / TIME_STEP_SECONDS))


def _valid_idx(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask > 0.5)


def _sample_unmatched(times: np.ndarray, mask: np.ndarray, match: float, cancel: float, pickup: float, rng: np.random.Generator) -> int | None:
    valid = _valid_idx(mask)
    ends = [t for t in (match, cancel, pickup) if t >= 0]
    candidates = valid[times[valid] < min(ends)] if ends else valid
    return None if candidates.size == 0 else int(rng.choice(candidates))


def _sample_matched(times: np.ndarray, mask: np.ndarray, match: float, cancel: float, pickup: float, rng: np.random.Generator, decay: float) -> int | None:
    if match < 0:
        return None
    valid = _valid_idx(mask)
    ends = [t for t in (cancel, pickup) if t >= 0]
    candidates = valid[(times[valid] >= match) & (times[valid] < min(ends))] if ends else valid[times[valid] >= match]
    if candidates.size == 0:
        return None
    rel = np.arange(candidates.size, dtype=np.float64)
    weights = np.exp(-decay * rel)
    weights /= weights.sum()
    return int(candidates[int(rng.choice(candidates.size, p=weights))])


def _label(stage: int, tau: float, match: float, pickup: float, cancel: float,
           unmatched_h_seconds: float = UNMATCHED_H_SECONDS,
           matched_h_seconds: float = MATCHED_H_SECONDS) -> tuple[int, int, int, int, float, int]:
    horizon = unmatched_h_seconds if stage == UNMATCHED_STAGE else matched_h_seconds
    end = tau + horizon
    cancel_t = cancel if tau < cancel <= end else -1.0
    progress_raw = match if stage == UNMATCHED_STAGE else pickup
    progress_t = progress_raw if tau < progress_raw <= end else -1.0
    h_steps = _steps(horizon)
    if cancel_t < 0 and progress_t < 0:
        return LABEL_STAY, 0, 0, LABEL_STAY, -1.0, h_steps
    if cancel_t >= 0 and (progress_t < 0 or cancel_t < progress_t):
        return LABEL_CANCEL, 1, 0, LABEL_CANCEL, cancel_t - tau, h_steps
    return LABEL_PROGRESS, 0, 1, LABEL_PROGRESS, progress_t - tau, h_steps


def build_eval_samples(data_path: Path, split_path: Path, split_name: str, output: Path,
                       seed: int, matched_decay: float,
                       unmatched_h_seconds: float = UNMATCHED_H_SECONDS,
                       matched_h_seconds: float = MATCHED_H_SECONDS,
                       base_data: dict[str, np.ndarray] | None = None) -> None:
    if base_data is None:
        with np.load(data_path, allow_pickle=False) as loaded:
            # NPZ fields are not cached by NumPy: read metadata once, not once
            # per order, and never load X just to construct landmark labels.
            data = {key: loaded[key] for key in (
                "order_ids", "snapshot_times", "seq_mask", "match_time", "pickup_time", "cancel_time",
            )}
    else:
        data = base_data
    split = json.loads(split_path.read_text(encoding="utf-8"))[f"{split_name}_indices"]
    rng = np.random.default_rng(seed)
    out: dict[str, list[Any]] = {k: [] for k in ["order_index", "order_id", "stage", "tau_time", "tau_step_index", "H_steps", "label_3way", "y_cancel_window", "y_progress_window", "first_event_type", "time_to_first_event_in_window"]}
    for order_index in split:
        times = data["snapshot_times"][order_index]
        mask = data["seq_mask"][order_index]
        match = float(data["match_time"][order_index])
        pickup = float(data["pickup_time"][order_index])
        cancel = float(data["cancel_time"][order_index])
        for stage, tau_idx in [
            (UNMATCHED_STAGE, _sample_unmatched(times, mask, match, cancel, pickup, rng)),
            (MATCHED_STAGE, _sample_matched(times, mask, match, cancel, pickup, rng, matched_decay)),
        ]:
            if tau_idx is None:
                continue
            tau = float(times[tau_idx])
            label, y_cancel, y_progress, first_event, time_to_first, h_steps = _label(
                stage, tau, match, pickup, cancel, unmatched_h_seconds, matched_h_seconds)
            out["order_index"].append(int(order_index))
            out["order_id"].append(int(data["order_ids"][order_index]))
            out["stage"].append(stage)
            out["tau_time"].append(tau)
            out["tau_step_index"].append(tau_idx)
            out["H_steps"].append(h_steps)
            out["label_3way"].append(label)
            out["y_cancel_window"].append(y_cancel)
            out["y_progress_window"].append(y_progress)
            out["first_event_type"].append(first_event)
            out["time_to_first_event_in_window"].append(time_to_first)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        order_index=np.asarray(out["order_index"], dtype=np.int64),
        order_id=np.asarray(out["order_id"], dtype=np.int64),
        stage=np.asarray(out["stage"], dtype=np.int8),
        tau_time=np.asarray(out["tau_time"], dtype=np.float32),
        tau_step_index=np.asarray(out["tau_step_index"], dtype=np.int32),
        H_steps=np.asarray(out["H_steps"], dtype=np.int32),
        label_3way=np.asarray(out["label_3way"], dtype=np.int8),
        y_cancel_window=np.asarray(out["y_cancel_window"], dtype=np.int8),
        y_progress_window=np.asarray(out["y_progress_window"], dtype=np.int8),
        first_event_type=np.asarray(out["first_event_type"], dtype=np.int8),
        time_to_first_event_in_window=np.asarray(out["time_to_first_event_in_window"], dtype=np.float32),
    )
    print(f"Saved {split_name} eval samples to {output}: {len(out['stage']):,} instances")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Clean58 benchmark artifacts from a SurvCancel JSONL release.")
    parser.add_argument("--release-dir", type=Path, required=True, help="Directory containing train-R1.jsonl ... train-R4.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("workdir"))
    parser.add_argument("--max-length", type=int, default=120)
    parser.add_argument("--limit-orders", type=int, default=None, help="Small debug subset; omit for full benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--matched-decay", type=float, default=0.2)
    parser.add_argument(
        "--windows", nargs="+", choices=list(WINDOWS), default=list(WINDOWS),
        help="Stage-0/stage-1 horizon pairs to prepare. Default: all five paper windows.",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["id"],
        choices=["all", *PROTOCOL_NAMES],
        help="Use 'all' for ID, LORO R1-R4, global chronological, and regional chronological R1-R4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.matched_decay) or args.matched_decay < 0:
        raise ValueError("matched-decay must be finite and nonnegative.")
    processed = args.output_dir / "processed"
    data_path = processed / "dynamic_order_sequences_clean58.npz"
    build_clean58(args.release_dir, data_path, args.max_length, args.limit_orders)
    protocols, strict = _expand_protocols(args.protocols)
    build_protocol_artifacts(data_path, args.output_dir, protocols, args.seed, args.matched_decay, strict, args.windows)
    manifest = {
        "source_files": [path.relative_to(args.release_dir).as_posix() for _, path in _region_files(args.release_dir)],
        "identity_scheme": "blake2b63(region,source_file,source_line_number,public_order_id)",
        "retained_identity_fields": ["public_order_ids", "source_files", "source_line_numbers"],
        "max_length": args.max_length, "limit_orders": args.limit_orders,
        "split_seed": args.seed, "landmark_seed": args.seed,
        "matched_decay": args.matched_decay,
        "windows": {name: WINDOWS[name] for name in dict.fromkeys(args.windows)},
        "requested_protocols": protocols,
        "generated_protocol_windows": [
            {"protocol": protocol, "window": window,
             "split": _protocol_paths(args.output_dir, protocol, window)[0].relative_to(args.output_dir).as_posix()}
            for protocol in protocols for window in dict.fromkeys(args.windows)
            if _protocol_paths(args.output_dir, protocol, window)[0].exists()
        ],
    }
    (args.output_dir / "PREPARATION.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Done. Artifacts:")
    print(f"  data:  {data_path}")
    print(f"  protocols: {', '.join(protocols)}")
    print(f"  windows: {', '.join(args.windows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
