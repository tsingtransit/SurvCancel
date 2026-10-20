"""Train a custom model using the benchmark's fixed data and evaluation."""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from joblib.externals import cloudpickle
import numpy as np
import torch

from scripts.train.train_stage_deephit import build_loaders, load_config, make_run_dir, save_json, set_global_seed
from survbench.evaluation import predict_loader, prediction_metrics
from survcancel_benchmark.custom import InputLoader, Predictor, load_model_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg["runtime"]["seed"])
    torch.set_num_threads(int(cfg["runtime"].get("torch_num_threads", 8)))
    model_class = load_model_class(cfg["custom_model"])
    run = make_run_dir(cfg)
    save_json(run / "config.json", cfg)
    train, val, test = build_loaders(cfg)
    model = model_class(cfg)
    fit_parameters = inspect.signature(model.fit).parameters
    options = {}
    accepts_options = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in fit_parameters.values())
    if "checkpoint_path" in fit_parameters or accepts_options:
        options["checkpoint_path"] = run / "custom_training_state.pkl"
    if "resume" in fit_parameters or accepts_options:
        options["resume"] = bool(cfg["runtime"].get("resume", False))
    model.fit(InputLoader(train), InputLoader(val), **options)
    predictor = Predictor(model)
    metrics = {}
    for name, loader in (("val", val), ("test", test)):
        if loader is None:
            continue
        arrays = predict_loader(predictor, loader)
        np.savez_compressed(run / f"{name}_predictions.npz", **arrays)
        metrics[name] = prediction_metrics(arrays)
    # Serialize file-based plugins by value so the checkpoint is self-contained.
    cloudpickle.register_pickle_by_value(sys.modules[model_class.__module__])
    with (run / "model.pkl.tmp").open("wb") as handle:
        cloudpickle.dump(model, handle)
    (run / "model.pkl.tmp").replace(run / "model.pkl")
    save_json(run / "metrics.json", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
