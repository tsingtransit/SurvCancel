#!/usr/bin/env python3
"""Train the public CLEAN58 benchmark and summarize independent seed runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {
    "h060_030": (60, 30), "h060_060": (60, 60),
    "h120_060": (120, 60), "h300_180": (300, 180), "h600_300": (600, 300),
}
PROTOCOLS = ("id", "loro_R1", "loro_R2", "loro_R3", "loro_R4", "chrono_global", "chrono_R1", "chrono_R2", "chrono_R3", "chrono_R4")

@dataclass(frozen=True)
class ModelSpec:
    name: str
    base_config: Path
    trainer: Path


def _spec(key: str, name: str, trainer: str = "train_stage_deephit.py") -> ModelSpec:
    return ModelSpec(name, RELEASE_ROOT / f"configs/models/{key}.yaml", RELEASE_ROOT / "scripts/train" / trainer)

MODEL_SPECS = {
    "deephit": _spec("deephit", "DeepHit"),
    "dyn_deephit": _spec("dyn_deephit", "DynDeepHit"),
    "survtrace": _spec("survtrace", "SurvTRACE", "train_stage_survtrace.py"),
    "tracer": _spec("tracer", "TraCeR", "train_stage_tracer.py"),
    "cox": _spec("cox", "Cox", "train_classical.py"),
    "rsf": _spec("rsf", "RSF", "train_classical.py"),
    "riskpool": _spec("riskpool", "RiskPool"),
}
REPRESENTATIONS = {
    "full": ("RiskPool", "models/riskpool.yaml"),
    "current_only_attention": ("Current-only pooling", "representations/current_only_attention.yaml"),
    "no_fine_spatial": ("No fine spatial proxies", "representations/no_fine_spatial.yaml"),
    "no_local_context": ("No local context", "representations/no_local_context.yaml"),
    "lifecycle_plus_context": ("Lifecycle + context", "representations/lifecycle_plus_context.yaml"),
    "lifecycle_only": ("Lifecycle only", "representations/lifecycle_only.yaml"),
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def protocol_paths(workdir: Path, protocol: str, window: str = "h060_030") -> tuple[Path, Path, Path]:
    protocol_dir = workdir / "protocols" / protocol
    if window != "h060_030": protocol_dir = protocol_dir / window
    return (protocol_dir / "order_split.json", protocol_dir / "dynamic_val_samples.npz", protocol_dir / "dynamic_test_samples.npz")


def ensure_protocol_artifacts(workdir: Path, protocol: str, window: str = "h060_030") -> tuple[Path, Path, Path]:
    paths = protocol_paths(workdir, protocol, window)
    expected = (workdir / "processed/dynamic_order_sequences_clean58.npz", *paths)
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared data: " + ", ".join(missing)
                                + ". Prepare data with scripts/prepare_clean58.sh first; see README.md.")
    return paths


def selected_specs(args: argparse.Namespace) -> list[tuple[str, ModelSpec]]:
    if args.experiment == "representation":
        table = REPRESENTATIONS
        return [("riskpool" if key == "full" else "riskpool_" + key,
                 ModelSpec(table[key][0], RELEASE_ROOT / "configs" / table[key][1], MODEL_SPECS["riskpool"].trainer))
                for key in args.conditions]
    return [(key, ModelSpec(MODEL_SPECS[key].name, args.config or MODEL_SPECS[key].base_config, MODEL_SPECS[key].trainer)) for key in args.models]


def build_config(spec_key: str, spec: ModelSpec, seed: int, args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    cfg = load_yaml(spec.base_config)
    split, val, test = (protocol_paths(args.workdir, args.protocol, args.window) if args.allow_missing_data else ensure_protocol_artifacts(args.workdir, args.protocol, args.window))
    run_name = f"{spec_key}_s{seed}" if args.protocol == "id" else f"{args.protocol}_{spec_key}_s{seed}"
    cfg["data"] = {"train_data_path": str(args.workdir / "processed/dynamic_order_sequences_clean58.npz"),
                   "split_path": str(split), "val_sample_path": str(val), "test_sample_path": str(test)}
    cfg.setdefault("runtime", {}).update(seed=seed, run_name=run_name, protocol=args.protocol, experiment=args.experiment, window=args.window)
    cfg["output"] = {"root_dir": str(output_root)}
    h0, h1 = WINDOWS[args.window]
    cfg["window"] = {"unmatched_h_seconds": float(h0), "matched_h_seconds": float(h1), "time_step_seconds": 10.0}
    cfg.setdefault("model", {})["device"] = args.device
    if spec_key in {"survtrace", "tracer"}: cfg["model"]["h_max"] = max(h0, h1) // 10
    if getattr(args, "custom_model", None):
        cfg["custom_model"] = args.custom_model
        cfg["custom_source"] = getattr(args, "custom_source", None)
    cfg["input_identity"] = {key: input_identity(Path(value)) for key, value in cfg["data"].items()}
    return cfg


@lru_cache(maxsize=256)
def _input_identity(path: str, size: int, mtime_ns: int) -> dict[str, Any]:
    result: dict[str, Any] = {"path": path, "size": size, "mtime_ns": mtime_ns}
    # Hash compact splits/landmarks. For large tensors, record stat identity
    # without rereading gigabytes for each independently launched seed group.
    if size <= 64 * 1024 * 1024:
        h = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""): h.update(block)
        result["sha256"] = h.hexdigest()
    return result


def input_identity(path: Path) -> dict[str, Any]:
    if not path.is_file(): return {"path": str(path), "missing": True}
    stat = path.stat()
    return _input_identity(str(path), stat.st_size, stat.st_mtime_ns)


def run_paths(spec_key: str, seed: int, args: argparse.Namespace, output_root: Path) -> tuple[Path, Path]:
    name = f"{spec_key}_s{seed}" if args.protocol == "id" else f"{args.protocol}_{spec_key}_s{seed}"
    return output_root / "_generated_configs" / f"{name}.yaml", output_root / name


def generated_config(spec_key: str, spec: ModelSpec, seed: int, args: argparse.Namespace, output_root: Path) -> tuple[Path, Path]:
    paths = run_paths(spec_key, seed, args, output_root)
    write_yaml(paths[0], build_config(spec_key, spec, seed, args, output_root))
    return paths


def assert_reusable(run_dir: Path, expected: dict[str, Any]) -> None:
    saved_path = run_dir / "config.json"
    if not saved_path.is_file():
        raise ValueError(f"Cannot verify existing run without saved config.json: {run_dir}")
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    fields = ("model", "window", "data", "train_dataset", "eval_dataset", "loader", "input_identity", "custom_model", "custom_source")
    changed = [key for key in fields if saved.get(key, {}) != expected.get(key, {})]
    for key in ("seed", "torch_num_threads", "protocol"):
        if saved.get("runtime", {}).get(key) != expected.get("runtime", {}).get(key): changed.append("runtime." + key)
    if changed:
        raise ValueError(f"Existing run does not match requested configuration/inputs ({', '.join(changed)}): {run_dir}. Choose a new output directory; the saved run has not been changed.")


def command(config_path: Path, trainer: Path, conda_env: str | None) -> list[str]:
    if conda_env: return ["conda", "run", "-n", conda_env, "python", str(trainer), "--config", str(config_path)]
    return [sys.executable, str(trainer), "--config", str(config_path)]


def run_one(spec_key: str, spec: ModelSpec, seed: int, args: argparse.Namespace, output_root: Path, reuse_completed: bool = False) -> Path:
    expected = build_config(spec_key, spec, seed, args, output_root)
    config_path, run_dir = run_paths(spec_key, seed, args, output_root)
    checkpoint = ("model.pkl" if spec.trainer.name == "train_custom.py" else
                  "model.joblib" if spec.trainer.name == "train_classical.py" else "model.pt")
    restarting = False
    if run_dir.exists() and not args.dry_run and (args.skip_existing or reuse_completed):
        if any(path.name != "config.json.tmp" for path in run_dir.iterdir()):
            assert_reusable(run_dir, expected)
        if run_complete(run_dir, checkpoint):
            print(f"[reuse] {run_dir}")
            return run_dir
        restarting = True
        expected["runtime"]["resume"] = True
        print(f"[resume] interrupted run: {run_dir}")
    if run_dir.exists() and not args.dry_run:
        if not restarting:
            raise FileExistsError(f"Run directory already exists: {run_dir}; use --resume or choose a new output directory.")
    write_yaml(config_path, expected)
    cmd = command(config_path, spec.trainer, args.conda_env)
    print("[dry-run]" if args.dry_run else "[train]", shlex.join(cmd))
    if args.dry_run: return run_dir
    log_path = output_root / "logs" / f"{run_dir.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a" if restarting else "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout: print(line, end=""); log.write(line)
        code = proc.wait()
    if code != 0: raise RuntimeError(f"Training failed: {run_dir}; see {log_path}")
    if not run_complete(run_dir, checkpoint):
        raise RuntimeError(f"Training finished without complete artifacts: {run_dir}")
    from survbench.training import atomic_json_save
    atomic_json_save(run_dir / "completed.json", {"status": "complete", "seed": seed})
    return run_dir


def run_complete(run_dir: Path, checkpoint: str) -> bool:
    required = ("config.json", "metrics.json", "val_predictions.npz", "test_predictions.npz", checkpoint)
    if any(not (run_dir / name).is_file() or not (run_dir / name).stat().st_size for name in required):
        return False
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if not all(isinstance(metrics.get(split), dict) and metrics[split] for split in ("val", "test")):
            return False
        for name in ("val_predictions.npz", "test_predictions.npz") + ((checkpoint,) if checkpoint.endswith(".pt") else ()):
            with zipfile.ZipFile(run_dir / name) as archive:
                if archive.testzip() is not None:
                    return False
    except (ValueError, OSError, zipfile.BadZipFile):
        return False
    return True
