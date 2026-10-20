"""Stage-conditioned TraCeR-style baseline with factorized temporal/covariate attention."""

from __future__ import annotations

import copy
import math
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

from survbench.models.stage_survtrace import (
    _compute_feature_stats_from_sequences,
    _standardize_fit,
    _standardize_predict,
)


_EPSILON = 1e-8
_DEFAULT_H_MAX = 6
_NUM_STAGE = 2


class _SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(self.d_model, 1))
        )
        pe = torch.zeros(length, self.d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class _TimeAwareCovariateEmbedding(nn.Module):
    """Embed scalar covariates with observation-time awareness, following the TraCeR skeleton."""

    def __init__(self, num_covariates: int, d_model: int, dropout_rate: float) -> None:
        super().__init__()
        self.value_scale = nn.Parameter(torch.ones(num_covariates))
        self.value_bias = nn.Parameter(torch.zeros(num_covariates))
        self.covariate_emb = nn.Embedding(num_covariates, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.observed_proj = nn.Linear(1, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.stale_token = nn.Parameter(torch.randn(1, 1, num_covariates, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        features: torch.Tensor,
        delta_t: torch.Tensor,
        snapshot_times: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, num_covariates = features.shape
        cov_idx = torch.arange(num_covariates, device=features.device)

        scaled = features * self.value_scale.view(1, 1, -1) + self.value_bias.view(1, 1, -1)
        value_embed = self.value_proj(scaled.unsqueeze(-1))
        cov_embed = self.covariate_emb(cov_idx).view(1, 1, num_covariates, -1)

        visible_mask = history_mask > 0.5
        observed_embed = self.observed_proj(history_mask.unsqueeze(-1)).unsqueeze(2)
        visible_times = torch.where(visible_mask, snapshot_times, torch.zeros_like(snapshot_times))
        prev_visible_times = F.pad(visible_times[:, :-1], (1, 0))
        prev_visible_times = torch.cummax(prev_visible_times, dim=1).values
        time_since_observed = torch.where(
            visible_mask,
            (snapshot_times - prev_visible_times).clamp_min(0.0),
            torch.zeros_like(snapshot_times),
        )

        time_input = torch.stack(
            [
                time_since_observed / 600.0,
                torch.log1p(time_since_observed / 10.0),
                delta_t / 600.0,
                torch.log1p(delta_t / 10.0),
                snapshot_times / 600.0,
                torch.log1p(snapshot_times / 10.0),
            ],
            dim=-1,
        )
        time_embed = self.time_proj(time_input).unsqueeze(2)
        tokens = self.norm(value_embed + cov_embed + observed_embed + time_embed)
        tokens = history_mask.unsqueeze(-1).unsqueeze(-1) * tokens + (
            1.0 - history_mask.unsqueeze(-1).unsqueeze(-1)
        ) * self.stale_token
        tokens = self.dropout(tokens)

        token_mask = history_mask.unsqueeze(-1).expand(batch_size, seq_len, num_covariates)
        tokens = tokens.masked_fill(token_mask.unsqueeze(-1) <= 0.5, 0.0)
        return tokens, token_mask


class _FactorizedAttentionBlock(nn.Module):
    """Apply temporal attention first, then covariate attention, on factorized tokens."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout_rate: float) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.covariate_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate, batch_first=True)
        self.covariate_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, tokens: torch.Tensor, history_mask: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_covariates, d_model = tokens.shape

        temporal_in = tokens.permute(0, 2, 1, 3).reshape(batch_size * num_covariates, seq_len, d_model)
        temporal_pad = ~(history_mask > 0.5)
        temporal_pad = temporal_pad.unsqueeze(1).expand(-1, num_covariates, -1).reshape(batch_size * num_covariates, seq_len)
        temporal_out, _ = self.temporal_attn(
            temporal_in, temporal_in, temporal_in, key_padding_mask=temporal_pad, need_weights=False,
        )
        temporal_out = self.temporal_norm(temporal_in + self.dropout(temporal_out))
        tokens = temporal_out.reshape(batch_size, num_covariates, seq_len, d_model).permute(0, 2, 1, 3)

        cov_in = tokens.reshape(batch_size * seq_len, num_covariates, d_model)
        cov_pad = ~(token_mask > 0.5).reshape(batch_size * seq_len, num_covariates)
        cov_out, _ = self.covariate_attn(cov_in, cov_in, cov_in, key_padding_mask=cov_pad, need_weights=False)
        cov_out = self.covariate_norm(cov_in + self.dropout(cov_out))
        tokens = cov_out.reshape(batch_size, seq_len, num_covariates, d_model)

        ffn_out = self.ffn(tokens)
        tokens = self.ffn_norm(tokens + self.dropout(ffn_out))
        return tokens.masked_fill(token_mask.unsqueeze(-1) <= 0.5, 0.0)


class _QuerySummarizer(nn.Module):
    """Learned query attends over all (time, covariate) tokens to build a global summary."""

    def __init__(self, d_model: int, nhead: int, dropout_rate: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_rate, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_covariates, d_model = tokens.shape
        memory = tokens.reshape(batch_size, seq_len * num_covariates, d_model)
        memory_mask = ~(token_mask > 0.5).reshape(batch_size, seq_len * num_covariates)
        query = self.query.expand(batch_size, -1, -1)
        attended, _ = self.attn(query, memory, memory, key_padding_mask=memory_mask, need_weights=False)
        return self.norm(attended.squeeze(1))


class StageTracerNet(nn.Module):
    """TraCeR-style factorized encoder with cause-specific hazard heads."""

    def __init__(
        self,
        x_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
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
        self.num_covariates = x_dim - 1
        self.d_model = d_model
        self.h_max = int(h_max)

        self.time_cov_embed = _TimeAwareCovariateEmbedding(
            num_covariates=self.num_covariates,
            d_model=d_model,
            dropout_rate=dropout_rate,
        )
        self.position_encoding = _SinusoidalPositionEncoding(d_model)
        self.blocks = nn.ModuleList(
            [
                _FactorizedAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.summarizer = _QuerySummarizer(d_model=d_model, nhead=nhead, dropout_rate=dropout_rate)

        self.stage_emb = nn.Embedding(_NUM_STAGE, stage_emb_dim)
        self.horizon_emb = nn.Embedding(self.h_max + 1, horizon_emb_dim)
        self.last_proj = nn.Linear(self.num_covariates, d_model)
        self.token_pool_proj = nn.Linear(self.num_covariates * d_model, d_model)

        shared_input_dim = d_model * 3 + stage_emb_dim + horizon_emb_dim + 1
        self.shared_fc = nn.Sequential(
            nn.Linear(shared_input_dim, h_dim_fc),
            nn.GELU(),
        )
        self.cancel_head = self._make_head(h_dim_fc, num_layers_cs)
        self.progress_head = self._make_head(h_dim_fc, num_layers_cs)
        self.stay_head = self._make_head(h_dim_fc, num_layers_cs)
        self.cancel_out = nn.Linear(h_dim_fc, self.h_max)
        self.progress_out = nn.Linear(h_dim_fc, self.h_max)
        self.stay_out = nn.Linear(h_dim_fc, self.h_max)

    @staticmethod
    def _make_head(hidden_dim: int, num_layers_cs: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for _ in range(num_layers_cs):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        history_mask: torch.Tensor,
        stage: torch.Tensor,
        h_steps: torch.Tensor,
        snapshot_times: torch.Tensor,
        covariate_mask: torch.Tensor | None = None,
        time_since_observed: torch.Tensor | None = None,
        dropout_rate: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        del covariate_mask, time_since_observed
        batch_size, seq_len, _ = x.shape
        features = x[:, :, 1:]
        delta_t = x[:, :, 0]

        tokens, token_mask = self.time_cov_embed(features, delta_t, snapshot_times, history_mask)
        pos = self.position_encoding(seq_len, x.device).view(1, seq_len, 1, self.d_model)
        hidden = (tokens + pos).masked_fill(token_mask.unsqueeze(-1) <= 0.5, 0.0)

        for block in self.blocks:
            hidden = block(hidden, history_mask, token_mask)

        summary_query = self.summarizer(hidden, token_mask)
        token_pool = hidden.sum(dim=1).reshape(batch_size, self.num_covariates * self.d_model)
        token_pool = self.token_pool_proj(token_pool)

        lengths = history_mask.sum(dim=1).clamp(min=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        x_last = features[torch.arange(batch_size, device=x.device), last_idx]
        last_summary = self.last_proj(x_last)

        stage_emb = self.stage_emb(stage.long())
        h_steps_clamped = h_steps.long().clamp(min=1, max=self.h_max)
        horizon_emb = self.horizon_emb(h_steps_clamped)
        h_steps_norm = (h_steps.float() / float(self.h_max)).unsqueeze(1)

        combined = torch.cat([summary_query, token_pool, last_summary, stage_emb, horizon_emb, h_steps_norm], dim=1)
        if dropout_rate > 0:
            combined = F.dropout(combined, p=dropout_rate, training=self.training)
        shared = self.shared_fc(combined)

        cancel_hidden = self.cancel_head(shared)
        progress_hidden = self.progress_head(shared)
        stay_hidden = self.stay_head(shared)
        if dropout_rate > 0:
            cancel_hidden = F.dropout(cancel_hidden, p=dropout_rate, training=self.training)
            progress_hidden = F.dropout(progress_hidden, p=dropout_rate, training=self.training)
            stay_hidden = F.dropout(stay_hidden, p=dropout_rate, training=self.training)

        cancel_logits = self.cancel_out(cancel_hidden)
        progress_logits = self.progress_out(progress_hidden)
        stay_logits = self.stay_out(stay_hidden)

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


def _target_index_from_batch(
    batch: dict[str, torch.Tensor], h_max: int
) -> torch.Tensor:
    label_3way = batch["label_3way"].long()
    time_to_first = batch["time_to_first_event_in_window"].float()

    step_idx = torch.clamp(torch.ceil(time_to_first / 10.0).long() - 1, min=0, max=h_max - 1)
    target = torch.full_like(label_3way, fill_value=2 * h_max)

    cancel_mask = label_3way == 2
    progress_mask = label_3way == 1
    target[cancel_mask] = step_idx[cancel_mask]
    target[progress_mask] = h_max + step_idx[progress_mask]
    return target


def stage_tracer_loss(
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
        sample_weight = torch.where(label_3way == 2, sample_weight * float(event_weights.get("cancel", 1.0)), sample_weight)
        sample_weight = torch.where(label_3way == 1, sample_weight * float(event_weights.get("progress", 1.0)), sample_weight)
        sample_weight = torch.where(label_3way == 0, sample_weight * float(event_weights.get("stay", 1.0)), sample_weight)
    if stage_weights:
        for stage_id, weight in stage_weights.items():
            sample_weight = torch.where(stage == int(stage_id), sample_weight * float(weight), sample_weight)

    weighted_nll = -target_log_prob * sample_weight
    return weighted_nll.sum() / sample_weight.sum().clamp_min(_EPSILON)


def _compute_feature_stats_from_order_indices(
    X: np.ndarray,
    seq_mask: np.ndarray,
    order_indices: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    x_dim = X.shape[2]
    feature_mean = np.zeros(x_dim, dtype=np.float32)
    feature_std = np.ones(x_dim, dtype=np.float32)

    if order_indices.size == 0:
        return feature_mean, feature_std

    feature_sum = np.zeros(x_dim - 1, dtype=np.float64)
    feature_sq_sum = np.zeros(x_dim - 1, dtype=np.float64)
    valid_count = 0.0

    for start in range(0, int(order_indices.size), chunk_size):
        batch_idx = order_indices[start : start + chunk_size]
        X_chunk = np.asarray(X[batch_idx], dtype=np.float32)
        seq_mask_chunk = np.asarray(seq_mask[batch_idx], dtype=np.float32)
        valid = seq_mask_chunk > 0.5
        if not np.any(valid):
            continue

        features = X_chunk[:, :, 1:]
        valid_f = valid[:, :, None]
        feature_sum += (features * valid_f).sum(axis=(0, 1), dtype=np.float64)
        feature_sq_sum += ((features ** 2) * valid_f).sum(axis=(0, 1), dtype=np.float64)
        valid_count += float(valid.sum())

    if valid_count <= 0.0:
        return feature_mean, feature_std

    mean = feature_sum / valid_count
    var = np.maximum(feature_sq_sum / valid_count - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0

    feature_mean[1:] = mean.astype(np.float32)
    feature_std[1:] = std.astype(np.float32)
    return feature_mean, feature_std


@dataclass
class StageTracerMetrics:
    loss: float
    n_samples: int


class StageTracer:
    """Training wrapper for a TraCeR-style stage-conditioned dynamic predictor."""

    def __init__(
        self,
        x_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
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

        self.model = StageTracerNet(
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
            self.feature_mean, self.feature_std = _compute_feature_stats_from_order_indices(
                dataset.base.X,
                dataset.base.seq_mask,
                order_indices,
            )
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
        history_mask = np.asarray(batch["history_mask"], dtype=np.float32)
        snapshot_times = np.asarray(batch["snapshot_times"], dtype=np.float32)
        X_scaled = _standardize_predict(X, seq_mask, self.feature_mean, self.feature_std)

        return {
            "X": torch.as_tensor(X_scaled, device=self.device),
            "seq_mask": torch.as_tensor(seq_mask, device=self.device),
            "history_mask": torch.as_tensor(history_mask, device=self.device),
            "snapshot_times": torch.as_tensor(snapshot_times, device=self.device),
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

    def _run_epoch(self, loader: Any, training: bool, epoch: int | None = None) -> StageTracerMetrics:
        total_loss = 0.0
        total_n = 0
        self.model.train(mode=training)

        batch_iter = loader
        if self.verbose and tqdm is not None:
            phase = "train" if training else "val"
            if epoch is None:
                desc = f"StageTracer {phase}"
            else:
                desc = f"StageTracer {phase} e{epoch + 1}/{self.epochs}"
            batch_iter = tqdm(loader, desc=desc, leave=False)

        for batch in batch_iter:
            tensor_batch = self._move_batch(batch)
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                output = self.model(
                    x=tensor_batch["X"],
                    history_mask=tensor_batch["history_mask"],
                    stage=tensor_batch["stage"],
                    h_steps=tensor_batch["H_steps"],
                    snapshot_times=tensor_batch["snapshot_times"],
                    dropout_rate=self.dropout_rate if training else 0.0,
                )
                loss = stage_tracer_loss(
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
            if self.verbose and tqdm is not None and hasattr(batch_iter, "set_postfix_str"):
                batch_iter.set_postfix_str(f"loss={loss.item():.4f}")

        return StageTracerMetrics(loss=total_loss / max(total_n, 1), n_samples=total_n)

    def fit(self, train_loader: Any, val_loader: Any | None = None, *, checkpoint_path=None, resume=False) -> "StageTracer":
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
            epoch_iter = tqdm(epoch_iter, desc="StageTracer")

        for epoch in epoch_iter:
            dataset = getattr(train_loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

            train_metrics = self._run_epoch(train_loader, training=True, epoch=epoch)
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, training=False, epoch=epoch)
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
