"""Common prediction scoring and independent-seed aggregation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from survbench.calibration import probability_metrics, validation_platt
from .calibrate import load, register_run_configuration


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_groups(arrays, probability=None):
    p = np.asarray(arrays["p_cancel_window"] if probability is None else probability)
    masks = {"overall": np.ones(len(p), dtype=bool)}
    for stage in (0, 1):
        masks[f"stage_{stage}"] = arrays["stage"] == stage
    for region in sorted(np.unique(arrays["region_id"])):
        name = f"region_R{int(region) + 1}"
        masks[name] = arrays["region_id"] == region
        for stage in (0, 1):
            masks[f"{name}_stage_{stage}"] = masks[name] & (arrays["stage"] == stage)
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        subset = {key: value[mask] for key, value in arrays.items()}
        row = probability_metrics(subset, p[mask])
        row["cancel_count"] = int(np.sum(subset["y_cancel_window"]))
        row["cancel_rate"] = float(np.mean(subset["y_cancel_window"]))
        row["avg_pred_cancel"] = float(np.mean(p[mask], dtype=np.float64))
        row["pred_cancel_gap"] = abs(row["avg_pred_cancel"] - row["cancel_rate"])
        result[name] = row
    return result


def validate_prediction_samples(arrays, sample_path):
    with np.load(sample_path, allow_pickle=False) as samples:
        for key in ("order_index", "order_id", "stage", "tau_time", "y_cancel_window",
                    "first_event_type", "time_to_first_event_in_window"):
            if key not in arrays or not np.array_equal(arrays[key], samples[key]):
                raise ValueError(f"Prediction metadata differs from fixed landmarks: {sample_path}: {key}")
    if "region_id" not in arrays or not np.isin(arrays["region_id"], [0, 1, 2, 3]).all():
        raise ValueError("Expected region_id 0, 1, 2, or 3.")


def aggregate(rows):
    collected = defaultdict(list)
    seeds = defaultdict(set)
    for row in rows:
        group = (row["model"], row["probabilities"], row["group"], row["metric"])
        if row["seed"] in seeds[group]:
            raise ValueError(f"Duplicate seed in {group}")
        seeds[group].add(row["seed"])
        if row["value"] is not None:
            collected[group].append(float(row["value"]))
        else:
            collected[group]
    result = []
    for (model, probabilities, group, metric), values in sorted(collected.items()):
        result.append(dict(model=model, probabilities=probabilities, group=group,
                           metric=metric, mean=mean(values) if values else None,
                           sample_sd=stdev(values) if len(values) > 1 else None,
                           n_seeds=len(values)))
    return result


def summarize(output_dir, model_runs):
    output_dir = Path(output_dir)
    rows, run_rows, configurations, references, protocols = [], [], {}, {}, set()
    for model, runs in model_runs.items():
        for run in runs:
            run = Path(run)
            _, seed = register_run_configuration(run, configurations)
            config = json.loads((run / "config.json").read_text())
            protocols.add(config["runtime"]["protocol"])
            val, test = load(run / "val_predictions.npz"), load(run / "test_predictions.npz")
            for split, arrays in (("val", val), ("test", test)):
                validate_prediction_samples(arrays, config["data"][f"{split}_sample_path"])
                key = (model, split)
                identity_fields = ("order_index", "order_id", "stage", "region_id", "tau_time",
                                   "y_cancel_window", "first_event_type", "time_to_first_event_in_window")
                if key in references:
                    for field in identity_fields:
                        if not np.array_equal(references[key][field], arrays[field]):
                            raise ValueError(f"Landmarks differ across seeds: {model}: {field}")
                else:
                    references[key] = {field: arrays[field] for field in identity_fields}
            if np.intersect1d(val["order_index"], test["order_index"]).size:
                raise ValueError(f"Validation and test orders overlap: {run}")
            calibrated, params = validation_platt(val, test)
            scored = {"raw": score_groups(test), "calibrated": score_groups(test, calibrated)}
            payload = dict(model=model, seed=seed, scores=scored, platt_parameters=params)
            write_json(run / "evaluation.json", payload)
            np.savez_compressed(run / "test_calibrated_predictions.npz", **{**test, "p_cancel_window": calibrated})
            run_rows.append(dict(model=model, seed=seed, run_dir=str(run)))
            for kind, groups in scored.items():
                for group, metrics in groups.items():
                    for metric, value in metrics.items():
                        if metric != "reliability_bins":
                            rows.append(dict(model=model, seed=seed, probabilities=kind,
                                             group=group, metric=metric, value=value))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    write_json(output_dir / "summary.json", dict(runs=run_rows, metrics=summary))
    write_csv(output_dir / "per_seed.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    if protocols == {"id"}:
        summarize_region_macro(output_dir / "region_macro", rows)
        if {"RiskPool", "DynDeepHit"} <= set(model_runs):
            for field in references[("RiskPool", "test")]:
                if not np.array_equal(references[("RiskPool", "test")][field], references[("DynDeepHit", "test")][field]):
                    raise ValueError(f"Paired model comparison requires identical test landmarks: {field}")
            summarize_paired(output_dir, rows)
    print(f"[results] {output_dir / 'summary.csv'}")
    return rows


def summarize_loro(output_dir, regional_rows):
    """Average four held-out regions within each seed before taking seed SD."""
    collected = defaultdict(dict)
    for region, rows in regional_rows.items():
        for row in rows:
            if row["group"].startswith("region_") or row["metric"] in {"n", "cancel_count"}:
                continue
            key = tuple(row[k] for k in ("model", "seed", "probabilities", "group", "metric"))
            collected[key][region] = row["value"]
    rows = []
    for key, values in sorted(collected.items()):
        if set(values) != {"R1", "R2", "R3", "R4"}:
            raise ValueError("LORO macro summaries require all four regions.")
        value = mean(values.values()) if all(v is not None for v in values.values()) else None
        rows.append(dict(zip(("model", "seed", "probabilities", "group", "metric"), key), value=value))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", {"metrics": aggregate(rows)})
    write_csv(output_dir / "per_seed.csv", rows)
    write_csv(output_dir / "summary.csv", aggregate(rows))
    return rows


def summarize_region_macro(output_dir, rows):
    """Equal-region ID means are formed within each seed, before aggregation."""
    regional = defaultdict(list)
    for row in rows:
        for region in ("R1", "R2", "R3", "R4"):
            prefix = f"region_{region}"
            if row["group"] == prefix or row["group"].startswith(prefix + "_stage_"):
                group = row["group"][len(prefix):].lstrip("_") or "overall"
                regional[region].append({**row, "group": group})
    if set(regional) == {"R1", "R2", "R3", "R4"}:
        return summarize_loro(output_dir, regional)
    return []


def summarize_paired(output_dir, rows, left="RiskPool", right="DynDeepHit"):
    """Subtract matching seed metrics first; SD is computed on those differences."""
    indexed = {left: {}, right: {}}
    for row in rows:
        if row["model"] not in indexed or row["metric"] != "c_index_cancel":
            continue
        key = tuple(row[field] for field in ("seed", "probabilities", "group", "metric"))
        if key in indexed[row["model"]]:
            raise ValueError("Duplicate seed metric in paired model comparison.")
        indexed[row["model"]][key] = row["value"]
    if indexed[left].keys() != indexed[right].keys():
        raise ValueError("Paired model comparison requires matching seeds and metric groups.")
    differences = []
    for key, value in sorted(indexed[left].items()):
        other = indexed[right][key]
        differences.append(dict(model=f"{left}-{right}",
                                **dict(zip(("seed", "probabilities", "group", "metric"), key)),
                                value=value - other if value is not None and other is not None else None))
    summary = aggregate(differences)
    write_json(Path(output_dir) / "paired_summary.json", {"left": left, "right": right, "metrics": summary})
    write_csv(Path(output_dir) / "paired_summary.csv", summary)
    write_csv(Path(output_dir) / "paired_per_seed.csv", differences)
    return differences


def summarize_distance_seeds(output_dir, seed_results):
    """Keep landmark-preparation sensitivity separate from training-run SDs."""
    if set(seed_results) != {42, 123, 456}:
        raise ValueError("Distance sensitivity requires preparation seeds 42, 123, 456.")
    payloads = {seed: json.loads(Path(path).read_text()) for seed, path in seed_results.items()}
    reference = payloads[42]
    for seed, payload in payloads.items():
        if payload["landmark_seed"] != seed:
            raise ValueError("Distance result has a different landmark preparation seed.")
        for field in ("data_path", "split_path", "analysis_seed", "sample_size_requested", "repeats", "families", "method"):
            if payload[field] != reference[field]:
                raise ValueError(f"Distance sensitivity configurations differ: {field}")
        if payload["results"].keys() != reference["results"].keys():
            raise ValueError("Distance comparisons differ between preparation seeds.")
    rows = []
    for comparison, result in reference["results"].items():
        for payload in payloads.values():
            if payload["results"][comparison]["strata"].keys() != result["strata"].keys():
                raise ValueError("Distance strata differ between preparation seeds.")
        for stratum in result["strata"]:
            for metric in ("mmd", "wasserstein_mean"):
                values = {seed: payload["results"][comparison]["strata"][stratum]["metrics"][metric]["mean"]
                          for seed, payload in payloads.items()}
                if not all(value is not None and np.isfinite(value) for value in values.values()):
                    raise ValueError("Expected finite distance means for every preparation seed.")
                rows.append(dict(comparison=comparison, stratum=stratum, metric=metric,
                                 seed42=values[42], seed123=values[123], seed456=values[456],
                                 max_abs_change_from_seed42=max(abs(v - values[42]) for v in values.values()),
                                 range_across_seeds=max(values.values()) - min(values.values())))
    overall = [row for row in rows if row["stratum"] == "overall_balanced"]
    maximum = {metric: {key: max(row[key] for row in overall if row["metric"] == metric)
                        for key in ("max_abs_change_from_seed42", "range_across_seeds")}
               for metric in ("mmd", "wasserstein_mean")}
    write_json(Path(output_dir) / "preparation_sensitivity.json",
               {"reference_seed": 42, "landmark_seeds": [42, 123, 456],
                "overall_balanced_max_changes": maximum, "comparisons": rows})
    write_csv(Path(output_dir) / "preparation_sensitivity.csv", rows)
    return rows
