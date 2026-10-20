#!/usr/bin/env python3
"""Train landmark Cox or RSF from the benchmark's common YAML configuration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from survbench.metrics import compute_window_metrics
from survbench.training import prepare_run_directory, atomic_json_save
from survbench.models.classical import (
    CoxCompetingRisk,
    RSFCompetingRisk,
    build_features,
    build_train_samples,
    choose_stage_rows,
    load_eval_samples,
    load_split,
    prediction_payload,
)


def save_json(path: Path, payload: Any) -> None:
    atomic_json_save(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def resolve_configuration(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float], int]:
    model = dict(cfg.get("model", {}))
    name = model.get("model")
    if name not in {"cox", "rsf"}:
        raise ValueError("model.model must be 'cox' or 'rsf'.")
    cap = int(model.get("train_cap_per_stage", 50_000 if name == "cox" else 30_000))
    if cap <= 0:
        raise ValueError("model.train_cap_per_stage must be positive.")
    window = {"unmatched_h_seconds": 60.0, "matched_h_seconds": 30.0, "time_step_seconds": 10.0}
    window.update({key: float(value) for key, value in cfg.get("window", {}).items() if key in window})
    step = window["time_step_seconds"]
    if step <= 0:
        raise ValueError("The time step must be positive.")
    for key in ("unmatched_h_seconds", "matched_h_seconds"):
        ratio = window[key] / step
        if ratio <= 0 or not np.isclose(ratio, round(ratio)):
            raise ValueError("Each window must be a positive whole number of time steps.")
    return model, window, cap


def main() -> int:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    model_cfg, window, cap = resolve_configuration(cfg)
    data_cfg = cfg["data"]
    train_cfg = cfg.get("train_dataset", {})
    runtime = cfg.get("runtime", {})
    seed = int(runtime.get("seed", 42))
    name = str(model_cfg["model"])
    run_name = str(runtime.get("run_name", f"{name}_s{seed}"))
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("runtime.run_name must be a single directory name.")
    run_dir = Path(cfg["output"]["root_dir"]) / run_name
    for key in ("train_data_path", "split_path", "val_sample_path"):
        if not Path(data_cfg[key]).is_file():
            raise FileNotFoundError(data_cfg[key])
    if data_cfg.get("test_sample_path") and not Path(data_cfg["test_sample_path"]).is_file():
        raise FileNotFoundError(data_cfg["test_sample_path"])
    evaluation_samples = {}
    for part in ("val", "test"):
        sample_path = data_cfg.get(f"{part}_sample_path")
        if not sample_path:
            continue
        samples = load_eval_samples(Path(sample_path))
        expected_h = np.where(samples["stage"] == 0, window["unmatched_h_seconds"], window["matched_h_seconds"])
        if not np.allclose(samples["H_steps"] * window["time_step_seconds"], expected_h):
            raise ValueError(f"{part} landmark windows do not match the configuration.")
        evaluation_samples[part] = samples
    run_dir = prepare_run_directory(cfg)
    save_json(run_dir / "config.json", cfg)
    started_total = time.monotonic()
    timing: dict[str, Any] = {}

    started = time.monotonic()
    with np.load(data_cfg["train_data_path"], allow_pickle=True) as loaded:
        # Materialize once: repeatedly reading an NpzFile would decompress X
        # for every landmark. Only fields needed by the classical protocol load.
        keys = ("X", "seq_mask", "snapshot_times", "region_ids", "order_ids", "match_time", "pickup_time", "cancel_time")
        data = {key: np.asarray(loaded[key]) for key in keys}
    timing["load_base_seconds"] = time.monotonic() - started
    train_samples = build_train_samples(
        data,
        load_split(Path(data_cfg["split_path"]), train_cfg.get("split_name", "train")),
        seed,
        float(train_cfg.get("matched_decay", 0.2)),
        **window,
    )
    # One seeded landmark per eligible stage/order, followed by the stated
    # stage-specific cap, is the classical training protocol.
    np.savez_compressed(run_dir / "train_samples.npz", **train_samples)
    fit_rows = {f"stage_{stage}": choose_stage_rows(train_samples, stage, cap, seed) for stage in (0, 1)}
    np.savez_compressed(run_dir / "fit_rows.npz", **fit_rows)
    started = time.monotonic()
    train_x = build_features(data, train_samples)
    timing["train_features_seconds"] = time.monotonic() - started
    if name == "cox":
        model = CoxCompetingRisk(float(model_cfg.get("alpha", 100.0)), window["time_step_seconds"])
    else:
        max_features = model_cfg.get("max_features", "sqrt")
        if isinstance(max_features, str) and max_features not in {"sqrt", "log2"}:
            max_features = float(max_features)
        model = RSFCompetingRisk(
            n_estimators=int(model_cfg.get("n_estimators", 100)),
            max_depth=model_cfg.get("max_depth", 12),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 100)),
            max_features=max_features,
            n_jobs=int(model_cfg.get("n_jobs", 12)),
            seed=seed,
            time_step_seconds=window["time_step_seconds"],
        )
    started = time.monotonic()
    model.fit(train_x, train_samples, cap, seed)
    timing["fit_seconds"] = time.monotonic() - started
    joblib.dump(model, run_dir / "model.joblib.tmp", compress=3)
    (run_dir / "model.joblib.tmp").replace(run_dir / "model.joblib")
    metrics = {}
    for part, samples in evaluation_samples.items():
        started = time.monotonic()
        features = build_features(data, samples)
        timing[f"{part}_features_seconds"] = time.monotonic() - started
        started = time.monotonic()
        prediction = model.predict(features, samples)
        timing[f"predict_{part}_seconds"] = time.monotonic() - started
        payload = prediction_payload(data, samples, prediction)
        np.savez_compressed(run_dir / f"{part}_predictions.npz", **payload)
        metrics[part] = compute_window_metrics(
            payload["y_cancel_window"], payload["p_cancel_window"], payload["stage"],
            payload["time_to_first_event_in_window"], payload["first_event_type"],
            region_id=payload["region_id"],
        )
        timing[f"n_{part}_samples"] = len(prediction)
        del features
    timing.update({"total_seconds": time.monotonic() - started_total, "n_train_samples": len(train_x),
                   "n_features": train_x.shape[1], "fit_rows_by_stage": {key: len(value) for key, value in fit_rows.items()}})
    save_json(run_dir / "metrics.json", metrics)
    save_json(run_dir / "timing.json", timing)
    sources = (Path(__file__), ROOT / "survbench/models/classical.py", ROOT / "survbench/metrics.py")
    save_json(run_dir / "provenance.json", {
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "python": sys.version,
        "packages": {package: importlib.metadata.version(package) for package in ("numpy", "scipy", "scikit-learn", "scikit-survival", "joblib")},
    })
    print(json.dumps({"run_dir": str(run_dir), "metrics": metrics, "timing": timing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
