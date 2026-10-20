"""Atomic epoch checkpoints and reproducible training recovery."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def random_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def isolate_loader_generators(*loaders):
    # Worker startup must not advance the model's dropout RNG after a restart.
    for index, loader in enumerate(loaders):
        if loader is not None and hasattr(loader, "generator") and loader.generator is None:
            loader.generator = torch.Generator().manual_seed((torch.initial_seed() + index + 1) % (2**63 - 1))


class EpochCheckpoint:
    def __init__(self, path=None, resume=False):
        self.path = None if path is None else Path(path)
        self.resume = resume

    def restore(self, trainer, *loaders):
        isolate_loader_generators(*loaders)
        if not self.resume or self.path is None or not self.path.is_file():
            return 0, float("inf"), None, 0
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        if state["trainer_class"] != type(trainer).__name__:
            raise ValueError("Training checkpoint belongs to a different model class.")
        trainer.model.load_state_dict(state["model"])
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.feature_mean = state["feature_mean"]
        trainer.feature_std = state["feature_std"]
        for name in ("_feature_mean_tensor", "_feature_std_tensor"):
            if hasattr(trainer, name):
                setattr(trainer, name, None)
        trainer.history = state["history"]
        for loader, saved in zip(loaders, state["loader_generators"]):
            if loader is not None and saved is not None:
                loader.generator.set_state(saved)
        restore_random_state(state["random_state"])
        print(f"[resume] next epoch {state['next_epoch'] + 1}: {self.path}", flush=True)
        return state["next_epoch"], state["best_val_loss"], state["best_state"], state["bad_epochs"]

    def save(self, trainer, next_epoch, best_val_loss, best_state, bad_epochs, *loaders):
        if self.path is None:
            return
        atomic_torch_save(dict(trainer_class=type(trainer).__name__, next_epoch=next_epoch,
                               model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                               feature_mean=trainer.feature_mean, feature_std=trainer.feature_std,
                               history=trainer.history, best_val_loss=best_val_loss,
                               best_state=best_state, bad_epochs=bad_epochs, random_state=random_state(),
                               loader_generators=[loader.generator.get_state() if loader is not None and loader.generator is not None else None
                                                  for loader in loaders]), self.path)


def prepare_run_directory(cfg):
    """Create a run, or reopen a matching interrupted run without deleting artifacts."""
    from survcancel_benchmark.engine import assert_reusable
    from datetime import datetime
    name = cfg["runtime"].get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run = Path(cfg["output"]["root_dir"]) / name
    if run.exists():
        if not cfg["runtime"].get("resume", False):
            raise FileExistsError(run)
        if any(path.name != "config.json.tmp" for path in run.iterdir()):
            assert_reusable(run, cfg)
    else:
        run.mkdir(parents=True, exist_ok=False)
    return run


def atomic_json_save(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)
