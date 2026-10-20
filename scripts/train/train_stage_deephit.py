"""Train StageDeepHit from YAML config and save all artifacts into one run directory."""

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
from survbench.metrics import compute_window_metrics as compute_window_metric_arrays
from survbench.models import StageDeepHit


def _window_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    window_cfg = cfg.get("window", {})
    return {
        "unmatched_h_seconds": float(window_cfg.get("unmatched_h_seconds", 60.0)),
        "matched_h_seconds": float(window_cfg.get("matched_h_seconds", 30.0)),
        "time_step_seconds": float(window_cfg.get("time_step_seconds", 10.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StageDeepHit with config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "models" / "deephit.yaml",
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
    window_cfg = _window_cfg(cfg)

    train_ds = TrainDynamicDataset(
        data_path=data_cfg["train_data_path"],
        split_path=data_cfg["split_path"],
        split_name=train_dataset_cfg.get("split_name", "train"),
        epoch_size=train_dataset_cfg.get("epoch_size"),
        seed=runtime_cfg["seed"],
        matched_sampling=train_dataset_cfg.get("matched_sampling", "exp"),
        matched_decay=train_dataset_cfg.get("matched_decay", 0.35),
        unmatched_h_seconds=window_cfg["unmatched_h_seconds"],
        matched_h_seconds=window_cfg["matched_h_seconds"],
        time_step_seconds=window_cfg["time_step_seconds"],
        preload=bool(train_dataset_cfg.get("preload", True)),
        tau_only=bool(train_dataset_cfg.get("tau_only", False)),
    )
    val_ds = EvalDynamicDataset(
        data_path=data_cfg["train_data_path"],
        sample_path=data_cfg["val_sample_path"],
        unmatched_h_seconds=window_cfg["unmatched_h_seconds"],
        matched_h_seconds=window_cfg["matched_h_seconds"],
        time_step_seconds=window_cfg["time_step_seconds"],
        preload=bool(eval_dataset_cfg.get("preload", True)),
        tau_only=bool(train_dataset_cfg.get("tau_only", False)),
    )
    test_ds = None
    test_sample_path = data_cfg.get("test_sample_path")
    if test_sample_path:
        test_ds = EvalDynamicDataset(
            data_path=data_cfg["train_data_path"],
                sample_path=test_sample_path,
            unmatched_h_seconds=window_cfg["unmatched_h_seconds"],
            matched_h_seconds=window_cfg["matched_h_seconds"],
            time_step_seconds=window_cfg["time_step_seconds"],
            preload=bool(eval_dataset_cfg.get("preload", True)),
            tau_only=bool(train_dataset_cfg.get("tau_only", False)),
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


def compute_window_metrics(model: StageDeepHit, loader: DataLoader) -> dict[str, float]:
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


def save_window_predictions(model: StageDeepHit, loader: DataLoader, path: Path) -> None:
    fields = (
        "order_id",
        "order_index",
        "stage",
        "region_id",
        "tau_time",
        "y_cancel_window",
        "first_event_type",
        "time_to_first_event_in_window",
    )
    arrays: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    arrays["p_cancel_window"] = []
    for batch in loader:
        prediction = model.predict_batch(batch)
        for field in fields:
            arrays[field].append(np.asarray(batch[field]))
        arrays["p_cancel_window"].append(
            np.asarray(prediction["p_cancel_window"], dtype=np.float32)
        )
    np.savez_compressed(
        path,
        **{field: np.concatenate(parts, axis=0) for field, parts in arrays.items()},
    )


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
    window_cfg = _window_cfg(cfg)
    h_max = int(
        max(
            window_cfg["unmatched_h_seconds"] / window_cfg["time_step_seconds"],
            window_cfg["matched_h_seconds"] / window_cfg["time_step_seconds"],
        )
    )

    model_cfg = cfg["model"]
    model = StageDeepHit(
        x_dim=int(model_cfg["x_dim"]),
        h_dim_rnn=int(model_cfg["h_dim_rnn"]),
        h_dim_fc=int(model_cfg["h_dim_fc"]),
        num_layers_rnn=int(model_cfg["num_layers_rnn"]),
        num_layers_att=int(model_cfg["num_layers_att"]),
        num_layers_cs=int(model_cfg["num_layers_cs"]),
        stage_emb_dim=int(model_cfg["stage_emb_dim"]),
        rnn_type=str(model_cfg["rnn_type"]),
        encoder_type=str(model_cfg.get("encoder_type", "rnn_att")),
        head_type=str(model_cfg.get("head_type", "pmf")),
        h_max=h_max,
        time_step_seconds=window_cfg["time_step_seconds"],
        dropout_rate=float(model_cfg["dropout_rate"]),
        learning_rate=float(model_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
        epochs=int(model_cfg["epochs"]),
        patience=int(model_cfg["patience"]),
        max_gradient_norm=float(model_cfg["max_gradient_norm"]),
        device=str(model_cfg["device"]),
        verbose=bool(model_cfg.get("verbose", True)),
        cancel_weight=float(model_cfg.get("cancel_weight", 1.0)),
        ranking_weight=float(model_cfg.get("ranking_weight", 0.0)),
        aux_cancel_weight=float(model_cfg.get("aux_cancel_weight", 0.0)),
        use_rel_time_pe=bool(model_cfg.get("use_rel_time_pe", False)),
        num_shared_layers=int(model_cfg.get("num_shared_layers", 1)),
        region_emb_dim=int(model_cfg.get("region_emb_dim", 0)),
        num_regions=int(model_cfg.get("num_regions", 200)),
        use_delta_features=bool(model_cfg.get("use_delta_features", False)),
        use_interaction=bool(model_cfg.get("use_interaction", True)),
        history_policy=str(model_cfg.get("history_policy", "strict_prior")),
        route_length_weight=float(model_cfg.get("route_length_weight", 0.0)),
        route_diversity_weight=float(model_cfg.get("route_diversity_weight", 0.0)),
        route_balance_weight=float(model_cfg.get("route_balance_weight", 0.0)),
        feature_drop_indices=list(model_cfg.get("feature_drop_indices", [])),
    )

    model.fit(train_loader, val_loader, checkpoint_path=run_dir / "training_state.pt",
              resume=bool(cfg["runtime"].get("resume", False)))

    history = model.history
    save_json(run_dir / "history.json", history)

    val_metrics = compute_window_metrics(model, val_loader)
    metrics: dict[str, Any] = {"val": val_metrics}
    save_window_predictions(model, val_loader, run_dir / "val_predictions.npz")
    if test_loader is not None:
        metrics["test"] = compute_window_metrics(model, test_loader)
    save_json(run_dir / "metrics.json", metrics)
    if test_loader is not None:
        save_window_predictions(model, test_loader, run_dir / "test_predictions.npz")

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
