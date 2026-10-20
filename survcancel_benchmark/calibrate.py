#!/usr/bin/env python3
"""Summarize raw and validation-calibrated predictions across saved runs."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from survbench.calibration import probability_metrics, validation_platt


def load(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    for key in ("order_index", "stage", "tau_time", "p_cancel_window", "y_cancel_window",
                "first_event_type", "time_to_first_event_in_window"):
        if key not in arrays:
            raise ValueError(f"{path} lacks {key}; export full prediction metadata first.")
    probability_metrics(arrays)
    if not np.isin(arrays["stage"], [0, 1]).all():
        raise ValueError("Only benchmark stages 0/1 are supported.")
    return arrays



def scientific_configuration(config):
    """Settings that must be shared before treating runs as independent seeds."""
    model = {key: value for key, value in config["model"].items()
             if key not in {"device", "verbose"}}
    window = {"unmatched_h_seconds": 60.0, "matched_h_seconds": 30.0,
              "time_step_seconds": 10.0, **config.get("window", {})}
    # I/O placement and eager loading do not change the statistical experiment.
    datasets = {section: {key: value for key, value in config.get(section, {}).items()
                          if key not in {"preload", "seed"} and not key.endswith("_path")}
                for section in ("train_dataset", "eval_dataset")}
    return {"model": model, "window": window, **datasets}


def register_run_configuration(run, references):
    """Read the saved seed and reject mislabeled or incompatible run groups."""
    path = run / "config.json"
    if not path.is_file():
        raise ValueError(f"{run} lacks config.json; calibration requires the saved model/window/dataset configuration and runtime.seed to verify independent seed runs.")
    config = json.loads(path.read_text(encoding="utf-8"))
    seed = config.get("runtime", {}).get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{path} must contain an integer runtime.seed.")
    if not isinstance(config.get("model"), dict):
        raise ValueError(f"{path} must contain the saved model configuration.")
    match = re.fullmatch(r"(.+)_s(\d+)", run.name)
    label = match[1] if match else run.name
    if match and int(match[2]) != seed:
        raise ValueError(f"Directory seed {match[2]} does not match config.json runtime.seed={seed}: {run}")
    current = scientific_configuration(config)
    if label in references:
        different = [key for key in current if references[label][key] != current[key]]
        if different:
            raise ValueError(f"Cannot aggregate different scientific configurations as seeds of {label}: {', '.join(different)}. Analyze the configurations as separate groups.")
    else:
        references[label] = current
    return label, seed

