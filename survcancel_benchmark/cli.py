"""The benchmark and complete four-region experiment pipelines."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from uuid import uuid4

from . import engine
from .reporting import summarize, summarize_loro, summarize_distance_seeds, write_csv, write_json

ROOT = engine.RELEASE_ROOT
EXPERIMENTS = ("main", "shifts", "regional", "representation", "horizons", "distances", "inference")
SHIFT_MODELS = ("riskpool", "dyn_deephit", "tracer", "rsf")
DISTANCE_SEEDS = (42, 123, 456)


def parser_for(mode):
    parser = argparse.ArgumentParser(
        prog="reproduce.sh" if mode == "reproduce" else "benchmark.sh",
        description=("Run the four-region experiment suite." if mode == "reproduce" else
                     "Train built-in or custom models and score the complete benchmark."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Outputs:
  plan.json: experiment list, model/seed combinations, commands and output paths.
  Each experiment: summary.csv, summary.json, per_seed.csv.
  Each run: config.json, metrics.json, evaluation.json, model checkpoint,
            val_predictions.npz, test_predictions.npz, test_calibrated_predictions.npz.
  Metrics: C-index, Brier, AUROC, AP, log loss, 10-bin equal-mass ECE,
           mean predicted risk, cancellation rate, absolute calibration gap, counts.
  Raw and validation-Platt scores: overall, each stage, each region, region-stage.
  CSV summaries give means and sample SD across independent seeds.

Install dependencies: python -m pip install -r requirements.txt
Prepare data first: see README.md. Experiments only read prepared --workdir files.

Examples:
  ./reproduce.sh --device cuda:0
  ./reproduce.sh --dry-run
  ./benchmark.sh --models riskpool dyn_deephit --device cpu --seeds 42
  ./benchmark.sh --custom-model examples/custom_model.py:Model --model-name logistic
  ./benchmark.sh --models riskpool --config path/to/model.yaml
  PYTHON=/path/to/python ./benchmark.sh --help

Custom model contract (see examples/custom_model.py):
  Model(config); fit(train_loader, val_loader); predict_batch(batch)
  predict_batch returns {'p_cancel_window': one probability per row}.
  Inputs are NumPy arrays; future/padded snapshots are masked. Prediction batches
  contain no labels. Training uses the same windows and splits as built-in models.
""")
    parser.add_argument("--workdir", type=Path, default=ROOT / "workdir", help="Existing prepared CLEAN58 and protocol files; see README.md.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / ("reproducibility" if mode == "reproduce" else "benchmark"))
    parser.add_argument("--device", default="cuda:0", help="Training/inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--resume", action="store_true", help="Reuse complete matching runs and recover interrupted training from saved epoch checkpoints.")
    parser.add_argument("--dry-run", action="store_true", help="Write plan/configs and print commands without training or requiring prepared data.")
    if mode == "reproduce":
        parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS),
                            help="Default: all experiments; see reproduce.sh for the experiment list.")
    else:
        parser.add_argument("--models", nargs="+", choices=list(engine.MODEL_SPECS))
        parser.add_argument("--protocol", choices=engine.PROTOCOLS, default="id")
        parser.add_argument("--window", choices=list(engine.WINDOWS), default="h060_030")
        parser.add_argument("--config", type=Path, help="YAML override for exactly one model.")
        parser.add_argument("--custom-model", help="package.module:Class or /path/model.py:Class; replaces --models.")
        parser.add_argument("--model-name", default="custom", help="Output name for --custom-model.")
    return parser


def normalize_custom(reference):
    module, sep, cls = reference.rpartition(":")
    if not sep or not module or not cls:
        raise ValueError("--custom-model requires module:Class or file.py:Class")
    if module.endswith(".py"):
        path = Path(module).resolve(strict=True)
        return str(path) + ":" + cls, engine.input_identity(path)
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise ValueError(f"Cannot import custom model module {module}")
    return reference, engine.input_identity(Path(spec.origin)) if spec.origin else {"module": module}


def training_groups(args, mode):
    """Share full-history controls with ID; train every window separately."""
    groups = []

    def add(name, experiment="main", models=None, protocol="id", window="h060_030", conditions=None):
        group_args = argparse.Namespace(**vars(args))
        group_args.experiment = experiment
        group_args.protocol, group_args.window = protocol, window
        group_args.models = list(models or ["riskpool"])
        group_args.conditions = conditions
        group_args.config = getattr(args, "config", None) if mode == "benchmark" else None
        group_args.custom_model = getattr(args, "custom_model", None) if mode == "benchmark" else None
        group_args.custom_source = getattr(args, "custom_source", None)
        group_args.conda_env = None
        group_args.skip_existing = args.resume
        group_args.allow_missing_data = args.dry_run
        output = args.output_dir / name
        specs = engine.selected_specs(group_args) if not group_args.custom_model else [
            (args.model_name, engine.ModelSpec(args.model_name, args.config or ROOT / "configs/custom.yaml",
                                             ROOT / "scripts/train/train_custom.py"))]
        runs = []
        for key, spec in specs:
            run_root = output
            if mode == "reproduce" and key == "riskpool" and experiment == "representation":
                run_root = args.output_dir / "id"
            for seed in args.seeds:
                config_path, run_dir = engine.run_paths(key, seed, group_args, run_root)
                runs.append(dict(key=key, model=spec.name, seed=seed, run_dir=run_dir,
                                 config_path=config_path, root=run_root, spec=spec))
        groups.append(dict(name=name, args=group_args, runs=runs, output=output))

    if mode == "benchmark":
        add(f"{args.protocol}/{args.window}", models=args.models or list(engine.MODEL_SPECS),
            protocol=args.protocol, window=args.window)
        return groups
    selected = set(args.experiments)
    # ID provides shift comparisons, full-history controls and inference checkpoints.
    if selected & {"main", "shifts", "representation", "inference"}:
        if "main" in selected:
            models = list(engine.MODEL_SPECS)
        else:
            needed = set(SHIFT_MODELS) if "shifts" in selected else {"riskpool"}
            if "inference" in selected:
                needed.add("dyn_deephit")
            models = [model for model in engine.MODEL_SPECS if model in needed]
        add("id", models=models)
    if "shifts" in selected:
        for protocol in ("loro_R1", "loro_R2", "loro_R3", "loro_R4", "chrono_global"):
            add(protocol, models=SHIFT_MODELS, protocol=protocol)
    if "regional" in selected:
        for region in ("R1", "R2", "R3", "R4"):
            add("chrono_" + region, models=["dyn_deephit", "tracer"], protocol="chrono_" + region)
    if "representation" in selected:
        add("representation", experiment="representation", conditions=list(engine.REPRESENTATIONS))
    if "horizons" in selected:
        for window in engine.WINDOWS:
            add("horizon/" + window, experiment="horizon", models=["riskpool", "dyn_deephit"], window=window)
    return groups


def extra_commands(args, mode):
    if mode != "reproduce":
        return []
    commands = []
    data = args.workdir / "processed/dynamic_order_sequences_clean58.npz"
    if "distances" in args.experiments:
        for seed in DISTANCE_SEEDS:
            output = args.output_dir / "distances" / f"seed{seed}"
            commands.append((f"distances_seed{seed}", output / "results.json", [
                sys.executable, "-m", "survcancel_benchmark.quantify_shift",
                "--data", str(data), "--split", str(args.workdir / "protocols/id/order_split.json"),
                "--output-dir", str(output), "--landmark-seed", str(seed),
                "--families", "id", "loro", "chrono_global"]))
    if "inference" in args.experiments:
        for model in ("riskpool", "dyn_deephit"):
            output = args.output_dir / "inference" / model
            commands.append(("inference_" + model, output / "benchmark.json", [
                sys.executable, "-m", "survcancel_benchmark.benchmark_inference", "--model", model,
                "--checkpoint", str(args.output_dir / "id" / f"{model}_s{args.seeds[0]}" / "model.pt"),
                "--data", str(data), "--samples", str(args.workdir / "protocols/id/dynamic_test_samples.npz"),
                "--output-dir", str(output), "--device", args.device, "--batch-sizes", "1", "1024",
                "--single-history-length", "36", "--warmup", "20",
                "--single-repeats", "150", "--batch-repeats", "50"]))
    return commands


def check_prepared_data(args, groups, extras):
    """Validate existing inputs without downloading or modifying data."""
    import numpy as np
    data_path = args.workdir / "processed/dynamic_order_sequences_clean58.npz"
    pairs = {(group["args"].protocol, group["args"].window) for group in groups}
    if extras:
        pairs.add(("id", "h060_030"))
    expected = {data_path}
    for protocol, window in pairs:
        expected.update(engine.protocol_paths(args.workdir, protocol, window))
    missing = sorted(str(path) for path in expected if not path.is_file())
    if missing:
        shown = "\n  ".join(missing[:8])
        remaining = f"\n  ... and {len(missing) - 8} more files" if len(missing) > 8 else ""
        command = shlex.join(["bash", str(ROOT / "scripts/prepare_clean58.sh"),
                              str(ROOT / "data/hf_survcancel_release"), str(args.workdir)])
        raise FileNotFoundError(f"Missing prepared data:\n  {shown}{remaining}\n"
                                f"Prepare data first (see README.md):\n  {command}\n"
                                "Replace the raw-data path with your downloaded dataset directory.")
    with np.load(data_path, allow_pickle=False) as data:
        if set(data["region"].tolist()) != {"R1", "R2", "R3", "R4"}:
            raise ValueError("The benchmark requires CLEAN58 data from all four regions R1-R4.")
        loro_pairs = [(protocol, window) for protocol, window in pairs if protocol.startswith("loro_")]
        if loro_pairs:
            from .prepare_clean58 import _build_loro_split, id_splits_for_loro
            metadata = {key: data[key] for key in ("order_ids", "region", "end_type", "match_time")}
            for protocol, window in sorted(loro_pairs):
                split_path = engine.protocol_paths(args.workdir, protocol, window)[0]
                saved = json.loads(split_path.read_text())
                seed = saved["split_seed"]
                base = id_splits_for_loro(metadata, args.workdir, seed, window)
                expected_split, _ = _build_loro_split(metadata, seed, protocol.removeprefix("loro_"), base)
                if any(sorted(saved[f"{part}_indices"]) != expected_split[part] for part in ("train", "val", "test")):
                    raise ValueError(f"LORO must retain ID train/validation assignments and test only held-out ID-test orders: {split_path}. "
                                     "Prepare the data again with scripts/prepare_clean58.sh; see README.md.")


def export_frequencies(args, groups):
    import numpy as np
    rows = []
    for group in groups:
        if group["args"].experiment != "horizon":
            continue
        window = group["args"].window
        sample_path = engine.protocol_paths(args.workdir, "id", window)[2]
        with np.load(sample_path, allow_pickle=False) as samples:
            stages, y = samples["stage"], samples["y_cancel_window"]
            events = samples["first_event_type"]
        for name, mask in (("overall", np.ones(len(y), dtype=bool)), ("stage_0", stages == 0), ("stage_1", stages == 1)):
            rows.append(dict(window=window, group=name, n=int(mask.sum()),
                             cancel_count=int(y[mask].sum()), progress_count=int(np.sum(events[mask] == 1)),
                             stay_count=int(np.sum(events[mask] == 0)), cancel_rate=float(np.mean(y[mask]))))
    if rows:
        write_csv(args.output_dir / "horizon/frequencies.csv", rows)
        write_json(args.output_dir / "horizon/frequencies.json", rows)


def export_dataset_statistics(args, groups):
    import numpy as np
    rows = []
    with np.load(args.workdir / "processed/dynamic_order_sequences_clean58.npz", allow_pickle=False) as data:
        regions, end_type = data["region"], data["end_type"]
        lengths = data["seq_mask"].sum(axis=1)
        matched = data["match_time"] >= 0
        for name, mask in [("overall", np.ones(len(regions), dtype=bool))] + [
                (region, regions == region) for region in ("R1", "R2", "R3", "R4")]:
            if mask.any():
                rows.append(dict(region=name, n_orders=int(mask.sum()), n_snapshots=int(lengths[mask].sum()),
                                 mean_sequence_length=float(np.mean(lengths[mask])),
                                 pre_match_cancels=int(np.sum(end_type[mask] == "pre_match_cancel")),
                                 post_match_cancels=int(np.sum(end_type[mask] == "post_match_cancel")),
                                 completed=int(np.sum(end_type[mask] == "completed")),
                                 matched_orders=int(np.sum(matched[mask]))))
    write_json(args.output_dir / "dataset_statistics.json", rows)
    write_csv(args.output_dir / "dataset_statistics.csv", rows)
    distributions = []
    for protocol, window in sorted({(g["args"].protocol, g["args"].window) for g in groups}):
        split = json.loads(engine.protocol_paths(args.workdir, protocol, window)[0].read_text())
        for part, sample_path in zip(("val", "test"), engine.protocol_paths(args.workdir, protocol, window)[1:]):
            with np.load(sample_path, allow_pickle=False) as samples:
                stage, event = samples["stage"], samples["first_event_type"]
            for label, mask in (("overall", np.ones(len(stage), dtype=bool)), ("stage_0", stage == 0), ("stage_1", stage == 1)):
                distributions.append(dict(protocol=protocol, window=window, split=part, group=label,
                                          train_orders=split["n_train_orders"], val_orders=split["n_val_orders"],
                                          test_orders=split["n_test_orders"], n_landmarks=int(mask.sum()),
                                          cancel_count=int(np.sum(event[mask] == 2)),
                                          progress_count=int(np.sum(event[mask] == 1)), stay_count=int(np.sum(event[mask] == 0))))
    write_json(args.output_dir / "landmark_distribution.json", distributions)
    write_csv(args.output_dir / "landmark_distribution.csv", distributions)


def run_extra(args, name, marker, command):
    if args.dry_run:
        print("[dry-run]", shlex.join(command))
        return
    inputs = {}
    for flag in ("--checkpoint", "--data", "--samples", "--split"):
        if flag in command:
            inputs[flag] = engine.input_identity(Path(command[command.index(flag) + 1]))
    expected = {"command": command, "inputs": inputs}
    request = marker.parent / "request.json"
    saved_request = None
    if request.is_file():
        try:
            saved_request = json.loads(request.read_text())
        except (ValueError, OSError):
            pass
    if args.resume and saved_request is not None and saved_request != expected:
        raise ValueError(f"Existing diagnostic does not match this invocation: {marker.parent}")
    if marker.is_file() and args.resume:
        # Diagnostics are inexpensive compared with training; their saved arguments
        # must still match the requested pipeline before reuse.
        if saved_request == expected:
            try:
                json.loads(marker.read_text())
            except (ValueError, OSError):
                pass
            else:
                print(f"[reuse] {marker}")
                return
    if marker.parent.exists():
        if not args.resume:
            raise FileExistsError(f"Choose a new diagnostic output directory: {marker.parent}")
        preserved = marker.parent.with_name(marker.parent.name + ".interrupted-" + uuid4().hex[:8])
        marker.parent.rename(preserved)
        print(f"[resume] restarting diagnostic; previous artifacts: {preserved}")
    log_path = args.output_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", shlex.join(command))
    with log_path.open("a" if args.resume else "w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        if proc.wait():
            raise RuntimeError(f"{name} failed: {log_path}")
    if not marker.is_file():
        raise RuntimeError(f"Missing diagnostic result: {marker}")
    from survbench.training import atomic_json_save
    atomic_json_save(marker.parent / "request.json", expected)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"benchmark", "reproduce"}:
        raise SystemExit("Use ./benchmark.sh or ./reproduce.sh")
    mode = sys.argv[1]
    parser = parser_for(mode)
    args = parser.parse_args(sys.argv[2:])
    for key in ("workdir", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if not args.seeds or len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        parser.error("--seeds requires distinct nonnegative integers.")
    if mode == "benchmark":
        if args.config:
            args.config = args.config.expanduser().resolve(strict=True)
        if args.custom_model:
            if args.models:
                parser.error("Use --models or --custom-model.")
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.model_name):
                parser.error("--model-name must be a simple model identifier.")
            args.custom_model, args.custom_source = normalize_custom(args.custom_model)
        elif args.config and (not args.models or len(args.models) != 1):
            parser.error("--config requires exactly one --models value, or --custom-model.")
        if args.models and len(set(args.models)) != len(args.models):
            parser.error("--models must be distinct.")
        if not args.custom_model and args.window != "h060_030" and set(args.models or engine.MODEL_SPECS) & {"cox", "rsf"}:
            parser.error("Cox/RSF configurations support --window h060_030.")
    groups = training_groups(args, mode)
    extras = extra_commands(args, mode)
    if not args.dry_run:
        try:
            check_prepared_data(args, groups, extras)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
    plan = {"regions": ["R1", "R2", "R3", "R4"], "seeds": args.seeds,
            "workdir": str(args.workdir), "output_dir": str(args.output_dir),
            "experiments": [], "diagnostics": []}
    for group in groups:
        jobs = []
        for run in group["runs"]:
            jobs.append(dict(model=run["model"], seed=run["seed"], run_dir=str(run["run_dir"]),
                             command=engine.command(run["config_path"], run["spec"].trainer, None)))
        plan["experiments"].append(dict(name=group["name"], protocol=group["args"].protocol,
                                       window=group["args"].window, output=str(group["output"]), runs=jobs))
    plan["diagnostics"] = [dict(name=name, result=str(marker), command=command) for name, marker, command in extras]
    plan["n_training_runs"] = len({job["run_dir"] for group in plan["experiments"] for job in group["runs"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "plan.json", plan)
    print(f"[plan] {plan['n_training_runs']} training runs; outputs: {args.output_dir}")
    if not args.dry_run:
        export_dataset_statistics(args, groups)
    completed, regional_rows = set(), {}
    for group in groups:
        runs_by_model = {}
        for run in group["runs"]:
            engine.run_one(run["key"], run["spec"], run["seed"], group["args"], run["root"],
                           reuse_completed=run["run_dir"] in completed)
            completed.add(run["run_dir"])
            runs_by_model.setdefault(run["model"], []).append(run["run_dir"])
        if not args.dry_run:
            rows = summarize(group["output"], runs_by_model)
            if group["name"].startswith("loro_"):
                regional_rows[group["name"].removeprefix("loro_")] = rows
    if not args.dry_run:
        if len(regional_rows) == 4:
            summarize_loro(args.output_dir / "loro_macro", regional_rows)
        export_frequencies(args, groups)
    for name, marker, command in extras:
        run_extra(args, name, marker, command)
    if not args.dry_run and mode == "reproduce" and "distances" in args.experiments:
        summarize_distance_seeds(args.output_dir / "distances", {
            seed: args.output_dir / "distances" / f"seed{seed}" / "results.json" for seed in DISTANCE_SEEDS})
    print(f"[done] {args.output_dir / 'plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
