"""Train StageTracer from YAML config and save all artifacts into one run directory."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from survbench.training import prepare_run_directory, atomic_torch_save, atomic_json_save
from survbench.data import EvalDynamicDataset, TrainDynamicDataset, collate_dynamic_samples
from survbench.evaluation import save_window_predictions
from survbench.metrics import compute_window_metrics as compute_window_metric_arrays
from survbench.models import StageTracer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StageTracer with config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "models" / "tracer.yaml",
        help="YAML config path.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(cfg: dict[str, Any]) -> Path:
    return prepare_run_directory(cfg)


def _loader_kwargs(num_workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    data_cfg = cfg["data"]
    train_dataset_cfg = cfg["train_dataset"]
    eval_dataset_cfg = cfg["eval_dataset"]
    loader_cfg = cfg["loader"]
    runtime_cfg = cfg["runtime"]
    window_cfg = cfg.get("window", {})
    unmatched_h_seconds = float(window_cfg.get("unmatched_h_seconds", 60.0))
    matched_h_seconds = float(window_cfg.get("matched_h_seconds", 30.0))
    time_step_seconds = float(window_cfg.get("time_step_seconds", 10.0))

    train_ds = TrainDynamicDataset(
        data_path=data_cfg["train_data_path"],
        split_path=data_cfg["split_path"],
        split_name=train_dataset_cfg.get("split_name", "train"),
        epoch_size=train_dataset_cfg.get("epoch_size"),
        seed=runtime_cfg["seed"],
        matched_sampling=train_dataset_cfg.get("matched_sampling", "exp"),
        matched_decay=train_dataset_cfg.get("matched_decay", 0.35),
        unmatched_h_seconds=unmatched_h_seconds,
        matched_h_seconds=matched_h_seconds,
        time_step_seconds=time_step_seconds,
        preload=bool(train_dataset_cfg.get("preload", True)),
    )
    val_ds = EvalDynamicDataset(
        data_path=data_cfg["train_data_path"],
        sample_path=data_cfg["val_sample_path"],
        unmatched_h_seconds=unmatched_h_seconds,
        matched_h_seconds=matched_h_seconds,
        time_step_seconds=time_step_seconds,
        preload=bool(eval_dataset_cfg.get("preload", True)),
    )
    test_ds = None
    test_sample_path = data_cfg.get("test_sample_path")
    if test_sample_path:
        test_ds = EvalDynamicDataset(
            data_path=data_cfg["train_data_path"],
                sample_path=test_sample_path,
            unmatched_h_seconds=unmatched_h_seconds,
            matched_h_seconds=matched_h_seconds,
            time_step_seconds=time_step_seconds,
            preload=bool(eval_dataset_cfg.get("preload", True)),
        )

    common_loader_kwargs = _loader_kwargs(
        num_workers=int(loader_cfg["train_num_workers"]),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", True)),
        prefetch_factor=int(loader_cfg.get("prefetch_factor", 2)),
    )
    eval_loader_kwargs = _loader_kwargs(
        num_workers=int(loader_cfg["eval_num_workers"]),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", True)),
        prefetch_factor=int(loader_cfg.get("prefetch_factor", 2)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(loader_cfg["train_batch_size"]),
        shuffle=False,
        collate_fn=collate_dynamic_samples,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(loader_cfg["eval_batch_size"]),
        shuffle=False,
        collate_fn=collate_dynamic_samples,
        **eval_loader_kwargs,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(loader_cfg["eval_batch_size"]),
            shuffle=False,
            collate_fn=collate_dynamic_samples,
            **eval_loader_kwargs,
        )
    return train_loader, val_loader, test_loader


def compute_window_metrics(model: StageTracer, loader: DataLoader) -> dict[str, float]:
    y_true: list[np.ndarray] = []
    p_cancel: list[np.ndarray] = []
    stage: list[np.ndarray] = []
    region: list[np.ndarray] = []
    time_to_first: list[np.ndarray] = []
    first_event_type: list[np.ndarray] = []

    for batch in loader:
        pred = model.predict_batch(batch)
        y_true.append(np.asarray(batch["y_cancel_window"], dtype=np.float32))
        p_cancel.append(np.asarray(pred["p_cancel_window"], dtype=np.float32))
        stage.append(np.asarray(batch["stage"], dtype=np.int32))
        region.append(np.asarray(batch["region_id"], dtype=np.int32))
        time_to_first.append(np.asarray(batch["time_to_first_event_in_window"], dtype=np.float32))
        first_event_type.append(np.asarray(batch["first_event_type"], dtype=np.int32))

    y = np.concatenate(y_true, axis=0)
    p = np.concatenate(p_cancel, axis=0)
    s = np.concatenate(stage, axis=0)
    r = np.concatenate(region, axis=0)
    t = np.concatenate(time_to_first, axis=0)
    e = np.concatenate(first_event_type, axis=0)
    return compute_window_metric_arrays(y, p, s, t, e, region_id=r)


def save_json(path: Path, payload: Any) -> None:
    atomic_json_save(path, payload)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["runtime"]["seed"])
    set_global_seed(seed)

    torch_num_threads = cfg["runtime"].get("torch_num_threads")
    if torch_num_threads:
        torch.set_num_threads(int(torch_num_threads))

    run_dir = make_run_dir(cfg)
    print(f"Run directory: {run_dir}")

    save_json(run_dir / "config.json", cfg)
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    train_loader, val_loader, test_loader = build_loaders(cfg)

    model_cfg = cfg["model"]
    model = StageTracer(
        x_dim=int(model_cfg["x_dim"]),
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        num_layers=int(model_cfg["num_layers"]),
        dim_feedforward=int(model_cfg["dim_feedforward"]),
        h_dim_fc=int(model_cfg["h_dim_fc"]),
        num_layers_cs=int(model_cfg["num_layers_cs"]),
        stage_emb_dim=int(model_cfg["stage_emb_dim"]),
        horizon_emb_dim=int(model_cfg["horizon_emb_dim"]),
        h_max=int(model_cfg.get("h_max", 6)),
        dropout_rate=float(model_cfg["dropout_rate"]),
        learning_rate=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
        epochs=int(model_cfg["epochs"]),
        patience=int(model_cfg["patience"]),
        max_gradient_norm=float(model_cfg["max_gradient_norm"]),
        cancel_loss_weight=float(model_cfg.get("cancel_loss_weight", 1.0)),
        progress_loss_weight=float(model_cfg.get("progress_loss_weight", 1.0)),
        stay_loss_weight=float(model_cfg.get("stay_loss_weight", 1.0)),
        unmatched_stage_loss_weight=float(model_cfg.get("unmatched_stage_loss_weight", 1.0)),
        matched_stage_loss_weight=float(model_cfg.get("matched_stage_loss_weight", 1.0)),
        device=str(model_cfg["device"]),
        verbose=bool(model_cfg.get("verbose", True)),
    )

    model.fit(train_loader, val_loader, checkpoint_path=run_dir / "training_state.pt",
              resume=bool(cfg["runtime"].get("resume", False)))
    save_json(run_dir / "history.json", model.history)

    val_metrics = compute_window_metrics(model, val_loader)
    metrics: dict[str, Any] = {"val": val_metrics}
    save_window_predictions(model, val_loader, run_dir / "val_predictions.npz")
    if test_loader is not None:
        metrics["test"] = compute_window_metrics(model, test_loader)
        save_window_predictions(model, test_loader, run_dir / "test_predictions.npz")
    save_json(run_dir / "metrics.json", metrics)

    checkpoint = {
        "config": cfg,
        "model": model.state_dict(),
    }
    atomic_torch_save(checkpoint, run_dir / "model.pt")

    print("Validation metrics:")
    print(json.dumps(val_metrics, indent=2))
    if "test" in metrics:
        print("Test metrics:")
        print(json.dumps(metrics["test"], indent=2))
    print(f"Saved artifacts to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
