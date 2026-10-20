#!/usr/bin/env python3
"""Measure neural inference on actual prepared landmarks.

Two scopes are reported separately: standardized resident-tensor forward, and
NumPy-to-device transfer/standardization/forward/probability-to-CPU. Neither is
production end-to-end latency. Database access, JSONL parsing, CLEAN58 feature
construction, networking, queuing, and dispatch decisions are excluded.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from survbench.evaluation import load_neural_checkpoint

def prepare_input_cache(
    *,
    data_path: Path,
    sample_path: Path,
    output: Path,
    size: int,
    seed: int,
) -> None:
    samples = np.load(sample_path, allow_pickle=False)
    stage = np.asarray(samples["stage"], dtype=np.int8)
    size = min(size, stage.size)
    if size == 0:
        raise ValueError("The sample file contains no landmarks.")
    rng = np.random.default_rng(seed)
    per_stage = size // 2
    selected_parts = []
    for stage_value in (0, 1):
        candidates = np.flatnonzero(stage == stage_value)
        n = min(per_stage, candidates.size)
        selected_parts.append(rng.choice(candidates, n, replace=False))
    selected = np.concatenate(selected_parts)
    if selected.size < size:
        remaining = np.setdiff1d(np.arange(stage.size), selected, assume_unique=False)
        selected = np.concatenate(
            [selected, rng.choice(remaining, size - selected.size, replace=False)]
        )
    selected = selected[rng.permutation(selected.size)]

    order_index = np.asarray(samples["order_index"][selected], dtype=np.int64)
    tau_step_index = np.asarray(samples["tau_step_index"][selected], dtype=np.int64)
    base = np.load(data_path, allow_pickle=True)
    print("[cache] loading compressed CLEAN58 sequence tensor", flush=True)
    x = np.asarray(base["X"])[order_index].astype(np.float32, copy=False)
    seq_mask = np.asarray(base["seq_mask"])[order_index].astype(np.float32, copy=False)
    snapshot_times = np.asarray(base["snapshot_times"])[order_index].astype(
        np.float32, copy=False
    )
    tau_time = np.asarray(samples["tau_time"][selected], dtype=np.float32)
    history_mask = (
        (seq_mask > 0.5) & (snapshot_times <= tau_time[:, None] + 1e-6)
    ).astype(np.float32)
    region_id = np.asarray(base["region_ids"])[order_index].astype(np.int64)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=x,
        seq_mask=seq_mask,
        history_mask=history_mask,
        stage=np.asarray(samples["stage"][selected], dtype=np.int64),
        tau_step_index=tau_step_index,
        H_steps=np.asarray(samples["H_steps"][selected], dtype=np.int64),
        region_id=region_id,
        snapshot_times=snapshot_times,
        source_order_index=order_index,
        source_sample_row=selected.astype(np.int64),
        cache_seed=np.asarray(seed, dtype=np.int64),
    )
    samples.close()
    base.close()
    print(f"[cache] saved {selected.size:,} supplied landmarks to {output}", flush=True)


def batch_from_cache(
    cache: dict[str, np.ndarray], batch_size: int, *, history_length: int | None = None,
) -> dict[str, np.ndarray]:
    n = int(cache["X"].shape[0])
    candidates = np.arange(n, dtype=np.int64)
    # The representative history is only for single-request latency. Throughput
    # uses the original cache with its full distribution of history lengths.
    if batch_size == 1 and history_length is not None:
        lengths = cache["history_mask"].sum(axis=1).astype(int)
        candidates = np.flatnonzero(lengths == history_length)
        if not candidates.size:
            raise ValueError("No cached landmark has the requested single-request history length; increase --cache-size or use another length.")
    if not candidates.size:
        raise ValueError("The input cache contains no landmarks.")
    indices = candidates[np.arange(batch_size, dtype=np.int64) % candidates.size]
    return {
        key: np.asarray(value[indices])
        for key, value in cache.items()
        if key
        in {
            "X",
            "seq_mask",
            "history_mask",
            "stage",
            "tau_step_index",
            "H_steps",
            "region_id",
            "snapshot_times",
            "source_sample_row",
        }
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_timings(milliseconds: list[float], batch_size: int) -> dict[str, float | int]:
    values = np.asarray(milliseconds, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "batch_size": batch_size,
        "repeats": int(values.size),
        "mean_ms_per_batch": mean_ms,
        "sd_ms_per_batch": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p50_ms_per_batch": float(np.quantile(values, 0.50)),
        "p95_ms_per_batch": float(np.quantile(values, 0.95)),
        "p99_ms_per_batch": float(np.quantile(values, 0.99)),
        "mean_ms_per_sample": mean_ms / batch_size,
        "samples_per_second": batch_size * 1000.0 / mean_ms,
    }


def time_callable(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    batch_size: int,
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        synchronize(device)
        timings = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter_ns()
            function()
            synchronize(device)
            timings.append((time.perf_counter_ns() - start) / 1e6)
    return summarize_timings(timings, batch_size)


def prepare_resident_batch(
    wrapper: Any,
    raw: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    device = wrapper.device
    x = torch.as_tensor(raw["X"], dtype=torch.float32, device=device)
    seq_mask = torch.as_tensor(raw["seq_mask"], dtype=torch.float32, device=device)
    mean = torch.as_tensor(wrapper.feature_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(wrapper.feature_std, dtype=torch.float32, device=device)
    x_scaled = x.clone()
    x_scaled[:, :, 1:] = (x_scaled[:, :, 1:] - mean[1:]) / std[1:]
    drop_indices = getattr(wrapper, "feature_drop_indices", [])
    if drop_indices:
        x_scaled[:, :, list(drop_indices)] = 0.0
    x_scaled *= seq_mask.unsqueeze(-1)
    return {
        "X": x_scaled,
        "history_mask": torch.as_tensor(
            raw["history_mask"], dtype=torch.float32, device=device
        ),
        "stage": torch.as_tensor(raw["stage"], dtype=torch.long, device=device),
        "tau_step_index": torch.as_tensor(
            raw["tau_step_index"], dtype=torch.long, device=device
        ),
        "H_steps": torch.as_tensor(
            raw["H_steps"], dtype=torch.long, device=device
        ),
        "snapshot_times": torch.as_tensor(raw["snapshot_times"], dtype=torch.float32, device=device),
        "region_id": torch.as_tensor(
            raw["region_id"], dtype=torch.long, device=device
        ),
    }


def forward_resident(wrapper: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    kwargs = {"x": batch["X"], "history_mask": batch["history_mask"],
              "stage": batch["stage"], "h_steps": batch["H_steps"], "dropout_rate": 0.0}
    if hasattr(wrapper.model, "encoder_type"):
        kwargs.update(tau_step_index=batch["tau_step_index"], region_id=batch["region_id"])
    else:
        kwargs["snapshot_times"] = batch["snapshot_times"]
    output = wrapper.model(**kwargs)
    steps = torch.arange(wrapper.h_max, device=wrapper.device).unsqueeze(0)
    valid = steps < batch["H_steps"].unsqueeze(1)
    return (output["p_cancel"] * valid).sum(dim=1)


def local_pipeline(wrapper: Any, raw: dict[str, np.ndarray]) -> np.ndarray:
    resident = prepare_resident_batch(wrapper, raw)
    return forward_resident(wrapper, resident).cpu().numpy()


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def hardware_payload(devices: list[str], cpu_threads: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cpu": cpu_name(),
        "cpu_threads_configured": cpu_threads,
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            payload["gpus"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    payload["requested_devices"] = devices
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["deephit", "dyn_deephit", "riskpool", "survtrace", "tracer"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional YAML/JSON configuration override.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", "--batch-sizes", dest="batch_sizes", nargs="+", type=int, default=[1, 1024])
    parser.add_argument("--single-history-length", "--history-length", dest="history_length", type=int,
                        help="Select a real history of this length for batch-1 latency only; throughput uses the unfiltered cache.")
    parser.add_argument("--cache-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-seed", type=int, default=20260726)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--single-repeats", type=int, default=150, help="Repetitions for batch-1 latency.")
    parser.add_argument("--batch-repeats", type=int, default=50, help="Repetitions for batch throughput.")
    parser.add_argument("--repeats", type=int, help="Override repetitions for every batch size.")
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    positive = args.batch_sizes + [args.cache_size, args.single_repeats, args.batch_repeats, args.cpu_threads]
    if args.repeats is not None:
        positive.append(args.repeats)
    if min(positive) < 1 or args.warmup < 0:
        raise ValueError("Batch/cache sizes, repeats and CPU threads must be positive; warmup must be nonnegative.")
    if args.history_length is not None and args.history_length < 1:
        raise ValueError("--history-length must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    config = yaml.safe_load(args.config.read_text()) if args.config else None
    wrapper, cfg = load_neural_checkpoint(args.checkpoint, args.model, device=args.device, config=config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_path = args.output_dir / "landmark_inputs.npz"
    prepare_input_cache(data_path=args.data, sample_path=args.samples, output=cache_path,
                        size=args.cache_size, seed=args.cache_seed)
    with np.load(cache_path, allow_pickle=False) as loaded:
        cache = {key: np.asarray(loaded[key]) for key in loaded.files}
    if cfg.get("train_dataset", {}).get("tau_only", False):
        cache["history_mask"][:] = 0
        cache["history_mask"][np.arange(len(cache["X"])), cache["tau_step_index"]] = 1
    lengths = cache["history_mask"].sum(axis=1).astype(int)
    payload = {
        "status": "complete",
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "parameters": sum(p.numel() for p in wrapper.model.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in wrapper.model.parameters()),
        "hardware": hardware_payload([args.device], args.cpu_threads),
        "input": {"data": str(args.data.resolve()), "samples": str(args.samples.resolve()),
                  "n_cached_landmarks_used": len(lengths), "allocated_sequence_length": int(cache["X"].shape[1]),
                  "feature_dimension": int(cache["X"].shape[2]), "visible_history_min": int(lengths.min()),
                  "visible_history_median": float(np.median(lengths)), "visible_history_max": int(lengths.max()),
                  "stage_counts": {str(stage): int(np.count_nonzero(cache["stage"] == stage)) for stage in (0, 1)},
                  "selection": "seeded real landmarks, balanced stages where available; batch rows cycle if needed",
                  "single_request_history_length": args.history_length},
        "settings": {"warmup": args.warmup, "repeats": args.repeats,
                     "single_repeats": args.repeats if args.repeats is not None else args.single_repeats,
                     "batch_repeats": args.repeats if args.repeats is not None else args.batch_repeats,
                     "seed": args.seed, "cache_seed": args.cache_seed},
        "measurement_scope": {
            "model_forward": "standardized tensors resident on device through cancellation-window probability",
            "local_pipeline": "NumPy arrays through device transfer, on-device standardization, forward and cancellation probability copied to CPU",
            "excluded": ["database/log access", "JSONL parsing", "CLEAN58 construction", "networking", "queueing", "dispatch logic"],
            "not_production_end_to_end": True,
        },
        "results": [],
    }
    for size in args.batch_sizes:
        raw = batch_from_cache(cache, size, history_length=args.history_length)
        resident = prepare_resident_batch(wrapper, raw)
        repeats = args.repeats if args.repeats is not None else (
            args.single_repeats if size == 1 else args.batch_repeats)
        timings = {}
        for scope, function in (("model_forward", lambda: forward_resident(wrapper, resident)),
                                ("local_pipeline", lambda: local_pipeline(wrapper, raw))):
            timings[scope] = time_callable(function, device=device, warmup=args.warmup,
                                           repeats=repeats, batch_size=size)
        visible_lengths = raw["history_mask"].sum(axis=1).astype(int)
        payload["results"].append({"batch_size": size,
                                  "input": {"history_length_filter": args.history_length if size == 1 else None,
                                            "visible_history_min": int(visible_lengths.min()),
                                            "visible_history_max": int(visible_lengths.max()),
                                            "source_sample_rows": raw["source_sample_row"].tolist()},
                                  **timings})
    output = args.output_dir / "benchmark.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved measurements to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
