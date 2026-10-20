"""Stage-conditioned SurvTRACE-style baseline adapted from the original field-transformer design."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from survbench.training import EpochCheckpoint

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


_EPSILON = 1e-8
_TIME_STEP_SECONDS = 10.0
_DEFAULT_H_MAX = 6
_NUM_STAGE = 2
LABEL_STAY = 0
LABEL_PROGRESS = 1
LABEL_CANCEL = 2


class _NumericFieldEmbedding(nn.Module):
    """Numerical field embedding in the style of the original SurvTRACE implementation."""

    def __init__(self, num_fields: int, d_model: int, dropout_rate: float) -> None:
        super().__init__()
        self.field_embeddings = nn.Parameter(torch.randn(1, num_fields, d_model))
        nn.init.normal_(self.field_embeddings, mean=0.0, std=0.02)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        embeddings = x_num.unsqueeze(-1) * self.field_embeddings
        return self.dropout(embeddings)


class _FieldTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int, dropout_rate: float) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, field_tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(field_tokens)


class _CauseSpecificHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers_cs: int, out_dim: int, dropout_rate: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers_cs):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StageSurvTraceNet(nn.Module):
    """Field-transformer encoder plus stage-conditioned fixed-window hazard head."""

    def __init__(
        self,
        x_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        h_dim_fc: int = 256,
        num_layers_cs: int = 2,
        stage_emb_dim: int = 16,
        horizon_emb_dim: int = 8,
        h_max: int = _DEFAULT_H_MAX,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if h_max < 1:
            raise ValueError("h_max must be positive")
        self.x_dim = x_dim
        self.num_fields = x_dim - 1
        self.d_model = d_model
        self.h_max = int(h_max)

        self.field_embed = _NumericFieldEmbedding(self.num_fields, d_model, dropout_rate)
        self.encoder = _FieldTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        self.stage_emb = nn.Embedding(_NUM_STAGE, stage_emb_dim)
        self.horizon_emb = nn.Embedding(self.h_max + 1, horizon_emb_dim)
        self.tau_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.field_flat_proj = nn.Sequential(
            nn.Linear(self.num_fields * d_model, h_dim_fc),
            nn.ReLU(),
        )

        shared_input_dim = h_dim_fc + d_model + stage_emb_dim + horizon_emb_dim + 1
        self.shared_fc = nn.Sequential(
            nn.Linear(shared_input_dim, h_dim_fc),
            nn.ReLU(),
        )

        self.cancel_head = _CauseSpecificHead(h_dim_fc, h_dim_fc, num_layers_cs, self.h_max, dropout_rate)
        self.progress_head = _CauseSpecificHead(h_dim_fc, h_dim_fc, num_layers_cs, self.h_max, dropout_rate)
        self.stay_head = _CauseSpecificHead(h_dim_fc, h_dim_fc, num_layers_cs, self.h_max, dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        history_mask: torch.Tensor,
        stage: torch.Tensor,
        h_steps: torch.Tensor,
        snapshot_times: torch.Tensor,
        dropout_rate: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        visible_mask = history_mask > 0.5
        lengths = visible_mask.sum(dim=1).clamp(min=1).long()
        last_idx = (lengths - 1).clamp(min=0)

        x_last = x[torch.arange(batch_size, device=x.device), last_idx, 1:]
        tau_time = snapshot_times[torch.arange(batch_size, device=x.device), last_idx]
        delta_last = x[torch.arange(batch_size, device=x.device), last_idx, 0]
        tau_features = torch.stack(
            [tau_time / 600.0, torch.log1p(tau_time / 10.0), delta_last / 600.0],
            dim=1,
        )

        field_tokens = self.field_embed(x_last)
        encoded_fields = self.encoder(field_tokens)
        encoded_flat = encoded_fields.flatten(start_dim=1)
        field_summary = self.field_flat_proj(encoded_flat)
        tau_summary = self.tau_proj(tau_features)

        stage_emb = self.stage_emb(stage.long())
        h_steps_clamped = h_steps.long().clamp(min=1, max=self.h_max)
        horizon_emb = self.horizon_emb(h_steps_clamped)
        h_steps_norm = (h_steps.float() / float(self.h_max)).unsqueeze(1)

        combined = torch.cat([field_summary, tau_summary, stage_emb, horizon_emb, h_steps_norm], dim=1)
        if dropout_rate > 0:
            combined = F.dropout(combined, p=dropout_rate, training=self.training)
        shared = self.shared_fc(combined)

        cancel_logits = self.cancel_head(shared)
        progress_logits = self.progress_head(shared)
        stay_logits = self.stay_head(shared)

        step_positions = torch.arange(self.h_max, device=x.device).unsqueeze(0)
        valid_steps = step_positions < h_steps.long().unsqueeze(1)
        invalid_mask = ~valid_steps
        cancel_logits = cancel_logits.masked_fill(invalid_mask, -1e9)
        progress_logits = progress_logits.masked_fill(invalid_mask, -1e9)
        stay_logits = stay_logits.masked_fill(invalid_mask, 0.0)

        step_logits = torch.stack([cancel_logits, progress_logits, stay_logits], dim=2)
        hazard_probs = F.softmax(step_logits, dim=2)
        haz_cancel = hazard_probs[:, :, 0]
        haz_progress = hazard_probs[:, :, 1]
        haz_stay = hazard_probs[:, :, 2]

        survival_prev = torch.cat(
            [
                torch.ones(batch_size, 1, device=x.device, dtype=haz_stay.dtype),
                torch.cumprod(haz_stay[:, :-1].clamp_min(_EPSILON), dim=1),
            ],
            dim=1,
        )
        p_cancel = survival_prev * haz_cancel
        p_progress = survival_prev * haz_progress
        p_stay = torch.cumprod(haz_stay.clamp_min(_EPSILON), dim=1)[:, -1]

        return {
            "hazard_probs": hazard_probs,
            "haz_cancel": haz_cancel,
            "haz_progress": haz_progress,
            "haz_stay": haz_stay,
            "p_cancel": p_cancel,
            "p_progress": p_progress,
            "p_stay": p_stay,
            "step_logits": step_logits,
        }


def _standardize_fit(X: np.ndarray, seq_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_flat = X.reshape(-1, X.shape[2])
    mask_flat = seq_mask.reshape(-1)
    valid = X_flat[mask_flat > 0.5]

    feature_mean = np.zeros(X.shape[2], dtype=np.float32)
    feature_std = np.ones(X.shape[2], dtype=np.float32)
    if len(valid) > 0:
        feature_mean[1:] = valid[:, 1:].mean(axis=0)
        feature_std[1:] = valid[:, 1:].std(axis=0)
        feature_std[feature_std < 1e-8] = 1.0

    X_scaled = X.copy()
    X_scaled[:, :, 1:] = (X_scaled[:, :, 1:] - feature_mean[1:]) / feature_std[1:]
    X_scaled *= seq_mask[:, :, np.newaxis]
    return X_scaled, feature_mean, feature_std


def _compute_feature_stats_from_sequences(X: np.ndarray, seq_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_dim = X.shape[2]
    feature_mean = np.zeros(x_dim, dtype=np.float32)
    feature_std = np.ones(x_dim, dtype=np.float32)

    valid = seq_mask > 0.5
    if not np.any(valid):
        return feature_mean, feature_std

    features = X[:, :, 1:]
    valid_count = float(valid.sum())
    mean = (features * valid[:, :, None]).sum(axis=(0, 1)) / max(valid_count, 1.0)
    centered = (features - mean[None, None, :]) * valid[:, :, None]
    var = (centered ** 2).sum(axis=(0, 1)) / max(valid_count, 1.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0

    feature_mean[1:] = mean.astype(np.float32)
    feature_std[1:] = std.astype(np.float32)
    return feature_mean, feature_std


def _standardize_predict(X: np.ndarray, seq_mask: np.ndarray, feature_mean: np.ndarray, feature_std: np.ndarray) -> np.ndarray:
    X_scaled = X.copy()
    X_scaled[:, :, 1:] = (X_scaled[:, :, 1:] - feature_mean[1:]) / feature_std[1:]
    X_scaled *= seq_mask[:, :, np.newaxis]
    return X_scaled


def _target_index_from_batch(
    batch: dict[str, torch.Tensor], h_max: int
) -> torch.Tensor:
    label_3way = batch["label_3way"].long()
    time_to_first = batch["time_to_first_event_in_window"].float()

    step_idx = torch.clamp(torch.ceil(time_to_first / _TIME_STEP_SECONDS).long() - 1, min=0, max=h_max - 1)
    target = torch.full_like(label_3way, fill_value=2 * h_max)

    cancel_mask = label_3way == LABEL_CANCEL
    progress_mask = label_3way == LABEL_PROGRESS
    target[cancel_mask] = step_idx[cancel_mask]
    target[progress_mask] = h_max + step_idx[progress_mask]
    return target


def stage_survtrace_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    event_weights: dict[str, float] | None = None,
    stage_weights: dict[int, float] | None = None,
) -> torch.Tensor:
    h_max = int(output["p_cancel"].shape[1])
    target = _target_index_from_batch(batch, h_max=h_max)
    label_3way = batch["label_3way"].long()
    stage = batch["stage"].long()
    log_p_cancel = torch.log(output["p_cancel"].clamp_min(_EPSILON))
    log_p_progress = torch.log(output["p_progress"].clamp_min(_EPSILON))
    log_p_stay = torch.log(output["p_stay"].clamp_min(_EPSILON))

    target_log_prob = log_p_stay.clone()

    cancel_mask = target < h_max
    progress_mask = (target >= h_max) & (target < 2 * h_max)
    if cancel_mask.any():
        cancel_idx = target[cancel_mask]
        target_log_prob[cancel_mask] = log_p_cancel[cancel_mask, cancel_idx]
    if progress_mask.any():
        progress_idx = target[progress_mask] - h_max
        target_log_prob[progress_mask] = log_p_progress[progress_mask, progress_idx]

    sample_weight = torch.ones_like(target_log_prob)
    if event_weights:
        sample_weight = torch.where(
            label_3way == LABEL_CANCEL,
            sample_weight * float(event_weights.get("cancel", 1.0)),
            sample_weight,
        )
        sample_weight = torch.where(
            label_3way == LABEL_PROGRESS,
            sample_weight * float(event_weights.get("progress", 1.0)),
            sample_weight,
        )
        sample_weight = torch.where(
            label_3way == LABEL_STAY,
            sample_weight * float(event_weights.get("stay", 1.0)),
            sample_weight,
        )
    if stage_weights:
        for stage_id, weight in stage_weights.items():
            sample_weight = torch.where(stage == int(stage_id), sample_weight * float(weight), sample_weight)

    weighted_nll = -target_log_prob * sample_weight
    return weighted_nll.sum() / sample_weight.sum().clamp_min(_EPSILON)


@dataclass
class StageSurvTraceMetrics:
    loss: float
    n_samples: int


class StageSurvTrace:
    """Training wrapper for a SurvTRACE-style field-transformer baseline."""

    def __init__(
        self,
        x_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        h_dim_fc: int = 256,
        num_layers_cs: int = 2,
        stage_emb_dim: int = 16,
        horizon_emb_dim: int = 8,
        h_max: int = _DEFAULT_H_MAX,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        epochs: int = 20,
        patience: int = 5,
        max_gradient_norm: float = 1.0,
        cancel_loss_weight: float = 1.0,
        progress_loss_weight: float = 1.0,
        stay_loss_weight: float = 1.0,
        unmatched_stage_loss_weight: float = 1.0,
        matched_stage_loss_weight: float = 1.0,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        self.x_dim = x_dim
        self.h_max = int(h_max)
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.max_gradient_norm = max_gradient_norm
        self.event_loss_weights = {
            "cancel": float(cancel_loss_weight),
            "progress": float(progress_loss_weight),
            "stay": float(stay_loss_weight),
        }
        self.stage_loss_weights = {
            0: float(unmatched_stage_loss_weight),
            1: float(matched_stage_loss_weight),
        }
        self.device = torch.device(device)
        self.verbose = verbose

        self.model = StageSurvTraceNet(
            x_dim=x_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            h_dim_fc=h_dim_fc,
            num_layers_cs=num_layers_cs,
            stage_emb_dim=stage_emb_dim,
            horizon_emb_dim=horizon_emb_dim,
            h_max=self.h_max,
            dropout_rate=dropout_rate,
        ).to(self.device)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.history: list[dict[str, float | int]] = []

    def _prepare_numpy_stats(self, train_loader: Any) -> None:
        dataset = getattr(train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "base") and hasattr(dataset, "order_indices"):
            order_indices = np.asarray(dataset.order_indices, dtype=np.int64)
            X = np.asarray(dataset.base.X[order_indices], dtype=np.float32)
            seq_mask = np.asarray(dataset.base.seq_mask[order_indices], dtype=np.float32)
            self.feature_mean, self.feature_std = _compute_feature_stats_from_sequences(X, seq_mask)
            return

        xs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for batch in train_loader:
            xs.append(np.asarray(batch["X"], dtype=np.float32))
            masks.append(np.asarray(batch["seq_mask"], dtype=np.float32))
        X = np.concatenate(xs, axis=0)
        seq_mask = np.concatenate(masks, axis=0)
        _, self.feature_mean, self.feature_std = _standardize_fit(X, seq_mask)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        assert self.feature_mean is not None and self.feature_std is not None
        X = np.asarray(batch["X"], dtype=np.float32)
        seq_mask = np.asarray(batch["seq_mask"], dtype=np.float32)
        X_scaled = _standardize_predict(X, seq_mask, self.feature_mean, self.feature_std)

        return {
            "X": torch.as_tensor(X_scaled, device=self.device),
            "seq_mask": torch.as_tensor(seq_mask, device=self.device),
            "history_mask": torch.as_tensor(np.asarray(batch["history_mask"], dtype=np.float32), device=self.device),
            "snapshot_times": torch.as_tensor(np.asarray(batch["snapshot_times"], dtype=np.float32), device=self.device),
            "stage": torch.as_tensor(np.asarray(batch["stage"]), device=self.device),
            "H_steps": torch.as_tensor(np.asarray(batch["H_steps"]), device=self.device),
            "label_3way": torch.as_tensor(np.asarray(batch["label_3way"]), device=self.device),
            "y_cancel_window": torch.as_tensor(np.asarray(batch["y_cancel_window"]), device=self.device),
            "y_progress_window": torch.as_tensor(np.asarray(batch["y_progress_window"]), device=self.device),
            "time_to_first_event_in_window": torch.as_tensor(
                np.asarray(batch["time_to_first_event_in_window"], dtype=np.float32),
                device=self.device,
            ),
        }

    def _run_epoch(self, loader: Any, training: bool) -> StageSurvTraceMetrics:
        total_loss = 0.0
        total_n = 0
        self.model.train(mode=training)

        for batch in loader:
            tensor_batch = self._move_batch(batch)
            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                output = self.model(
                    x=tensor_batch["X"],
                    history_mask=tensor_batch["history_mask"],
                    stage=tensor_batch["stage"],
                    h_steps=tensor_batch["H_steps"],
                    snapshot_times=tensor_batch["snapshot_times"],
                    dropout_rate=self.dropout_rate if training else 0.0,
                )
                loss = stage_survtrace_loss(
                    output,
                    tensor_batch,
                    event_weights=self.event_loss_weights,
                    stage_weights=self.stage_loss_weights,
                )

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_gradient_norm)
                self.optimizer.step()

            batch_size = int(tensor_batch["X"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_n += batch_size

        return StageSurvTraceMetrics(loss=total_loss / max(total_n, 1), n_samples=total_n)

    def fit(self, train_loader: Any, val_loader: Any | None = None, *, checkpoint_path=None, resume=False) -> "StageSurvTrace":
        if self.verbose:
            print("Preparing feature statistics...")
        self._prepare_numpy_stats(train_loader)
        if self.verbose:
            print("Feature statistics ready. Starting training...")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        bad_epochs = 0
        self.history = []

        checkpoint = EpochCheckpoint(checkpoint_path, resume)
        start_epoch, best_val_loss, best_state, bad_epochs = checkpoint.restore(self, train_loader, val_loader)
        epoch_iter = range(start_epoch, self.epochs) if bad_epochs < self.patience else range(0)
        if self.verbose and tqdm is not None:
            epoch_iter = tqdm(epoch_iter, desc="StageSurvTrace")

        for epoch in epoch_iter:
            dataset = getattr(train_loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

            train_metrics = self._run_epoch(train_loader, training=True)
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, training=False)
                current_val = val_metrics.loss
            else:
                val_metrics = None
                current_val = train_metrics.loss

            if current_val < best_val_loss:
                best_val_loss = current_val
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1

            record: dict[str, float | int] = {
                "epoch": epoch + 1,
                "train_loss": float(train_metrics.loss),
                "train_n": int(train_metrics.n_samples),
                "best_val_loss": float(best_val_loss),
                "bad_epochs": int(bad_epochs),
            }
            if val_loader is not None and val_metrics is not None:
                record["val_loss"] = float(val_metrics.loss)
                record["val_n"] = int(val_metrics.n_samples)
            self.history.append(record)
            checkpoint.save(self, epoch + 1, best_val_loss, best_state, bad_epochs, train_loader, val_loader)

            if self.verbose:
                if tqdm is not None and hasattr(epoch_iter, "set_postfix_str"):
                    if val_metrics is None:
                        epoch_iter.set_postfix_str(f"train={train_metrics.loss:.4f}")
                    else:
                        epoch_iter.set_postfix_str(
                            f"train={train_metrics.loss:.4f} "
                            f"val={val_metrics.loss:.4f} best={best_val_loss:.4f} bad={bad_epochs}"
                        )
                else:
                    if val_metrics is None:
                        print(f"epoch={epoch + 1} train_loss={train_metrics.loss:.4f}")
                    else:
                        print(
                            f"epoch={epoch + 1} train_loss={train_metrics.loss:.4f} "
                            f"val_loss={val_metrics.loss:.4f} best_val={best_val_loss:.4f} bad={bad_epochs}"
                        )

            if bad_epochs >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_state": self.model.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "x_dim": self.x_dim,
            "h_max": self.h_max,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "max_gradient_norm": self.max_gradient_norm,
            "event_loss_weights": self.event_loss_weights,
            "stage_loss_weights": self.stage_loss_weights,
            "history": self.history,
        }

    @torch.no_grad()
    def predict_batch(self, batch: dict[str, Any]) -> dict[str, np.ndarray]:
        tensor_batch = self._move_batch(batch)
        self.model.eval()
        output = self.model(
            x=tensor_batch["X"],
            history_mask=tensor_batch["history_mask"],
            stage=tensor_batch["stage"],
            h_steps=tensor_batch["H_steps"],
            snapshot_times=tensor_batch["snapshot_times"],
            dropout_rate=0.0,
        )
        p_cancel = output["p_cancel"].cpu().numpy()
        p_progress = output["p_progress"].cpu().numpy()
        p_stay = output["p_stay"].cpu().numpy()
        H_steps = np.asarray(batch["H_steps"], dtype=np.int32)
        cancel_window = np.array([p_cancel[i, :H_steps[i]].sum() for i in range(len(H_steps))], dtype=np.float32)
        progress_window = np.array([p_progress[i, :H_steps[i]].sum() for i in range(len(H_steps))], dtype=np.float32)
        return {
            "p_cancel": p_cancel,
            "p_progress": p_progress,
            "p_stay": p_stay,
            "p_cancel_window": cancel_window,
            "p_progress_window": progress_window,
        }
