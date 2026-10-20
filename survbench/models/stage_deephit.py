"""Stage-conditioned DeepHit-style model for dynamic fixed-window prediction."""

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
_H_MAX = 6
_NUM_STAGE = 2
LABEL_STAY = 0
LABEL_PROGRESS = 1
LABEL_CANCEL = 2


def _safe_masked_softmax(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Masked softmax that returns exact zeros for an all-masked row."""
    valid = valid_mask.to(dtype=torch.bool)
    masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = F.softmax(masked_scores, dim=dim)
    weights = weights * valid.to(dtype=weights.dtype)
    denominator = weights.sum(dim=dim, keepdim=True)
    return weights / denominator.clamp_min(torch.finfo(weights.dtype).tiny)


class _RNNLastEncoder(nn.Module):
    """RNN encoder that returns the final hidden state summary only."""

    def __init__(
        self,
        x_dim: int,
        h_dim_rnn: int,
        num_layers_rnn: int,
        rnn_type: str = "LSTM",
    ) -> None:
        super().__init__()
        self.h_dim_rnn = h_dim_rnn
        self.num_layers_rnn = num_layers_rnn

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(x_dim, h_dim_rnn, num_layers_rnn, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(x_dim, h_dim_rnn, num_layers_rnn, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

    @property
    def output_dim(self) -> int:
        return self.num_layers_rnn * self.h_dim_rnn

    def forward(self, x_hist: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        raw_lengths = hist_mask.sum(dim=1)
        lengths = raw_lengths.clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(
            x_hist,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.rnn(packed)

        if isinstance(hidden, tuple):
            hidden = hidden[0]
        context = torch.cat([hidden[i] for i in range(self.num_layers_rnn)], dim=1)
        return context * (raw_lengths > 0).to(context.dtype).unsqueeze(1)


class _TemporalAttentionRNN(nn.Module):
    """RNN encoder with temporal attention over visible historical states."""

    def __init__(
        self,
        x_dim: int,
        h_dim_rnn: int,
        num_layers_rnn: int,
        num_layers_att: int,
        h_dim_att: int,
        rnn_type: str = "LSTM",
    ) -> None:
        super().__init__()
        self.h_dim_rnn = h_dim_rnn
        self.num_layers_rnn = num_layers_rnn

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(x_dim, h_dim_rnn, num_layers_rnn, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(x_dim, h_dim_rnn, num_layers_rnn, batch_first=True)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")

        att_input_dim = num_layers_rnn * h_dim_rnn * 2
        layers: list[nn.Module] = []
        in_dim = att_input_dim
        for _ in range(num_layers_att):
            layers.append(nn.Linear(in_dim, h_dim_att))
            layers.append(nn.Tanh())
            in_dim = h_dim_att
        layers.append(nn.Linear(in_dim, 1))
        self.att_net = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return self.num_layers_rnn * self.h_dim_rnn

    def forward(self, x_hist: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x_hist.shape[:2]
        lengths = hist_mask.sum(dim=1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(
            x_hist,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        rnn_out, hidden = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=seq_len)

        h_steps = rnn_out
        if isinstance(hidden, tuple):
            h_last = torch.cat([hidden[0][i] for i in range(self.num_layers_rnn)], dim=1)
        else:
            h_last = torch.cat([hidden[i] for i in range(self.num_layers_rnn)], dim=1)

        h_steps_expanded = h_steps.repeat(1, 1, self.num_layers_rnn)
        h_last_expanded = h_last.unsqueeze(1).expand(-1, seq_len, -1)
        att_input = torch.cat([h_steps_expanded, h_last_expanded], dim=2)
        att_scores = self.att_net(att_input).squeeze(2)
        att_weights = _safe_masked_softmax(att_scores, hist_mask > 0.5, dim=1)
        return (att_weights.unsqueeze(2) * h_steps_expanded).sum(dim=1)


class _TemporalAttentionRNNWithWeights(_TemporalAttentionRNN):
    """RNN temporal attention encoder that also exposes attention weights."""

    def forward(self, x_hist: torch.Tensor, hist_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x_hist.shape[:2]
        lengths = hist_mask.sum(dim=1).clamp(min=1).long()
        packed = nn.utils.rnn.pack_padded_sequence(
            x_hist,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        rnn_out, hidden = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=seq_len)

        h_steps = rnn_out
        if isinstance(hidden, tuple):
            h_last = torch.cat([hidden[0][i] for i in range(self.num_layers_rnn)], dim=1)
        else:
            h_last = torch.cat([hidden[i] for i in range(self.num_layers_rnn)], dim=1)

        h_steps_expanded = h_steps.repeat(1, 1, self.num_layers_rnn)
        h_last_expanded = h_last.unsqueeze(1).expand(-1, seq_len, -1)
        att_input = torch.cat([h_steps_expanded, h_last_expanded], dim=2)
        att_scores = self.att_net(att_input).squeeze(2)
        att_weights = _safe_masked_softmax(att_scores, hist_mask > 0.5, dim=1)
        return (att_weights.unsqueeze(2) * h_steps_expanded).sum(dim=1), att_weights


class _AttentionPoolingEncoder(nn.Module):
    """Lightweight attention pooling directly over visible snapshots."""

    def __init__(self, x_dim: int, hidden_dim: int, num_layers_att: int) -> None:
        super().__init__()
        self.value_proj = nn.Linear(x_dim, hidden_dim)

        layers: list[nn.Module] = []
        in_dim = x_dim
        for _ in range(num_layers_att):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.att_net = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(self, x_hist: torch.Tensor, hist_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        att_scores = self.att_net(x_hist).squeeze(2)
        att_weights = _safe_masked_softmax(att_scores, hist_mask > 0.5, dim=1)
        values = self.value_proj(x_hist)
        return (att_weights.unsqueeze(2) * values).sum(dim=1), att_weights


class _QueryCondAttentionPoolingEncoder(nn.Module):
    """Attention pooling with x_last as query (additive attention)."""

    def __init__(self, x_dim: int, hidden_dim: int, num_layers_att: int) -> None:
        super().__init__()
        self.value_proj = nn.Linear(x_dim, hidden_dim)
        self.query_proj = nn.Linear(x_dim - 1, hidden_dim)
        self.key_proj = nn.Linear(x_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self, x_hist: torch.Tensor, hist_mask: torch.Tensor, x_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query_proj(x_last).unsqueeze(1)  # (B, 1, H)
        keys = self.key_proj(x_hist)  # (B, T, H)
        att_scores = self.score_proj(torch.tanh(query + keys)).squeeze(2)  # (B, T)
        att_weights = _safe_masked_softmax(att_scores, hist_mask > 0.5, dim=1)
        values = self.value_proj(x_hist)
        return (att_weights.unsqueeze(2) * values).sum(dim=1), att_weights


class _MultiHeadAttentionPoolingEncoder(nn.Module):
    """Multi-head scaled dot-product attention pooling with x_last as query."""

    def __init__(self, x_dim: int, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim ** -0.5

        self.query_proj = nn.Linear(x_dim - 1, hidden_dim)
        self.key_proj = nn.Linear(x_dim, hidden_dim)
        self.value_proj = nn.Linear(x_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self, x_hist: torch.Tensor, hist_mask: torch.Tensor, x_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x_hist.shape
        H, D = self.num_heads, self.head_dim

        q = self.query_proj(x_last).view(B, H, 1, D)  # (B, H, 1, D)
        k = self.key_proj(x_hist).view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        v = self.value_proj(x_hist).view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)

        att_scores = (q @ k.transpose(-2, -1)).squeeze(2) * self.scale  # (B, H, T)
        att_weights = _safe_masked_softmax(
            att_scores, (hist_mask > 0.5).unsqueeze(1), dim=2
        )

        context = (att_weights.unsqueeze(3) * v).sum(dim=2)  # (B, H, D)
        context = self.output_proj(context.reshape(B, self.hidden_dim))  # (B, hidden_dim)

        avg_weights = att_weights.mean(dim=1)  # (B, T) — for analysis compatibility
        return context, avg_weights


class _GatedQCMHAttentionPoolingEncoder(nn.Module):
    """Gated combination of additive (qc) and multi-head dot-product attention."""

    def __init__(self, x_dim: int, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim

        # Additive branch (qc)
        self.add_value_proj = nn.Linear(x_dim, hidden_dim)
        self.add_query_proj = nn.Linear(x_dim - 1, hidden_dim)
        self.add_key_proj = nn.Linear(x_dim, hidden_dim)
        self.add_score_proj = nn.Linear(hidden_dim, 1)

        # Multi-head dot-product branch
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mh_query_proj = nn.Linear(x_dim - 1, hidden_dim)
        self.mh_key_proj = nn.Linear(x_dim, hidden_dim)
        self.mh_value_proj = nn.Linear(x_dim, hidden_dim)
        self.mh_output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self, x_hist: torch.Tensor, hist_mask: torch.Tensor, x_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x_hist.shape

        # --- Additive branch ---
        add_q = self.add_query_proj(x_last).unsqueeze(1)
        add_k = self.add_key_proj(x_hist)
        add_scores = self.add_score_proj(torch.tanh(add_q + add_k)).squeeze(2)
        valid_history = hist_mask > 0.5
        add_weights = _safe_masked_softmax(add_scores, valid_history, dim=1)
        add_values = self.add_value_proj(x_hist)
        ctx_add = (add_weights.unsqueeze(2) * add_values).sum(dim=1)

        # --- Multi-head dot-product branch ---
        H, D = self.num_heads, self.head_dim
        q = self.mh_query_proj(x_last).view(B, H, 1, D)
        k = self.mh_key_proj(x_hist).view(B, T, H, D).permute(0, 2, 1, 3)
        v = self.mh_value_proj(x_hist).view(B, T, H, D).permute(0, 2, 1, 3)
        mh_scores = (q @ k.transpose(-2, -1)).squeeze(2) * self.scale
        mh_weights = _safe_masked_softmax(
            mh_scores, valid_history.unsqueeze(1), dim=2
        )
        ctx_mh = (mh_weights.unsqueeze(3) * v).sum(dim=2)
        ctx_mh = self.mh_output_proj(ctx_mh.reshape(B, self.hidden_dim))

        # --- Gated fusion ---
        g = torch.sigmoid(self.gate(torch.cat([ctx_mh, ctx_add], dim=1)))
        context = g * ctx_mh + (1 - g) * ctx_add

        # Average weights from both branches for analysis
        avg_weights = (add_weights + mh_weights.mean(dim=1)) * 0.5
        return context, avg_weights


class _SurvCancelEncoder(nn.Module):
    """Adaptive Temporal Length-Aware Survival encoder.

    Short-sequence branch: MLP on x_last (no history).
    Long-sequence branch: gated dual attention pooling over history.
    Learned gate routes between branches based on sequence length and x_last.
    """

    def __init__(self, x_dim: int, hidden_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Short branch: project x_last directly
        self.short_proj = nn.Sequential(
            nn.Linear(x_dim - 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Long branch: reuse gated QC+MH attention pooling
        self.long_encoder = _GatedQCMHAttentionPoolingEncoder(x_dim, hidden_dim, num_heads)

        # Routing gate: decides short vs long based on x_last + log(seq_len)
        self.route_gate = nn.Sequential(
            nn.Linear(x_dim - 1 + 1, hidden_dim // 4),  # +1 for log(seq_len)
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self, x_hist: torch.Tensor, hist_mask: torch.Tensor, x_last: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x_hist.shape[0]
        raw_seq_len = hist_mask.sum(dim=1)
        seq_len = raw_seq_len.clamp(min=1)
        log_len = torch.log(seq_len + 1).unsqueeze(1)

        ctx_short = self.short_proj(x_last)
        ctx_long, att_weights = self.long_encoder(x_hist, hist_mask, x_last)

        gate_input = torch.cat([x_last, log_len], dim=1)
        g = self.route_gate(gate_input)
        g = g * (raw_seq_len > 0).to(dtype=g.dtype).unsqueeze(1)
        context = (1 - g) * ctx_short + g * ctx_long

        self._last_gate = g
        self._last_log_len = log_len
        self._last_ctx_short = ctx_short
        self._last_ctx_long = ctx_long

        return context, att_weights


class _StageHybridRiskPoolRNNEncoder(nn.Module):
    """Use causal RiskPool in stage 0 and temporal RNN attention in stage 1."""

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        num_layers_rnn: int,
        num_layers_att: int,
        h_dim_att: int,
        rnn_type: str = "LSTM",
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.riskpool = _SurvCancelEncoder(
            x_dim=x_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.rnn = _TemporalAttentionRNNWithWeights(
            x_dim=x_dim,
            h_dim_rnn=hidden_dim,
            num_layers_rnn=num_layers_rnn,
            num_layers_att=num_layers_att,
            h_dim_att=h_dim_att,
            rnn_type=rnn_type,
        )
        rnn_dim = self.rnn.output_dim
        self.rnn_projection = (
            nn.Identity() if rnn_dim == hidden_dim else nn.Linear(rnn_dim, hidden_dim)
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def forward(
        self,
        x_hist: torch.Tensor,
        hist_mask: torch.Tensor,
        x_last: torch.Tensor,
        stage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pool_context, pool_weights = self.riskpool(x_hist, hist_mask, x_last)
        rnn_context, rnn_weights = self.rnn(x_hist, hist_mask)
        rnn_context = self.rnn_projection(rnn_context)
        use_rnn = (stage.long() == 1).unsqueeze(1)
        context = torch.where(use_rnn, rnn_context, pool_context)
        attention_weights = torch.where(use_rnn, rnn_weights, pool_weights)
        return context, attention_weights


class StageDeepHitNet(nn.Module):
    """Backbone + PMF head for stage-conditioned fixed-window prediction."""

    def __init__(
        self,
        x_dim: int,
        h_dim_rnn: int = 100,
        h_dim_fc: int = 100,
        num_layers_rnn: int = 2,
        num_layers_att: int = 2,
        num_layers_cs: int = 2,
        stage_emb_dim: int = 8,
        rnn_type: str = "LSTM",
        encoder_type: str = "rnn_att",
        head_type: str = "pmf",
        h_max: int = _H_MAX,
        use_cancel_aux: bool = False,
        use_rel_time_pe: bool = False,
        num_shared_layers: int = 1,
        region_emb_dim: int = 0,
        num_regions: int = 200,
        use_delta_features: bool = False,
        use_interaction: bool = True,
        history_policy: str = "strict_prior",
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.h_dim_rnn = h_dim_rnn
        self.num_layers_rnn = num_layers_rnn
        self.encoder_type = encoder_type
        self.head_type = head_type
        self.h_max = h_max
        self.snapshot_adapter = None
        self.use_rel_time_pe = use_rel_time_pe
        self.use_delta_features = use_delta_features
        self.use_interaction = use_interaction
        if history_policy not in {"strict_prior", "inclusive_current"}:
            raise ValueError(
                "history_policy must be 'strict_prior' or 'inclusive_current', "
                f"got {history_policy!r}"
            )
        self.history_policy = history_policy

        if stage_emb_dim > 0:
            self.stage_emb = nn.Embedding(_NUM_STAGE, stage_emb_dim)
        else:
            self.stage_emb = None

        if encoder_type == "rnn_att":
            self.encoder = _TemporalAttentionRNN(
                x_dim=x_dim,
                h_dim_rnn=h_dim_rnn,
                num_layers_rnn=num_layers_rnn,
                num_layers_att=num_layers_att,
                h_dim_att=h_dim_fc,
                rnn_type=rnn_type,
            )
        elif encoder_type == "rnn_last":
            self.encoder = _RNNLastEncoder(
                x_dim=x_dim,
                h_dim_rnn=h_dim_rnn,
                num_layers_rnn=num_layers_rnn,
                rnn_type=rnn_type,
            )
        elif encoder_type == "att_pool":
            self.encoder = _AttentionPoolingEncoder(
                x_dim=x_dim,
                hidden_dim=h_dim_rnn,
                num_layers_att=num_layers_att,
            )
        elif encoder_type == "qc_att_pool":
            self.encoder = _QueryCondAttentionPoolingEncoder(
                x_dim=x_dim,
                hidden_dim=h_dim_rnn,
                num_layers_att=num_layers_att,
            )
        elif encoder_type == "mh_att_pool":
            self.encoder = _MultiHeadAttentionPoolingEncoder(
                x_dim=x_dim,
                hidden_dim=h_dim_rnn,
                num_heads=4,
            )
        elif encoder_type == "qc_mh_att_pool":
            self.encoder = _GatedQCMHAttentionPoolingEncoder(
                x_dim=x_dim,
                hidden_dim=h_dim_rnn,
                num_heads=4,
            )
        elif encoder_type == "survcancel":
            self.encoder = _SurvCancelEncoder(
                x_dim=x_dim,
                hidden_dim=h_dim_rnn,
                num_heads=4,
            )
        elif encoder_type == "stage_hybrid_riskpool_rnn":
            self.encoder = _StageHybridRiskPoolRNNEncoder(
                x_dim=x_dim,
                hidden_dim=h_dim_rnn,
                num_layers_rnn=num_layers_rnn,
                num_layers_att=num_layers_att,
                h_dim_att=h_dim_fc,
                rnn_type=rnn_type,
                num_heads=4,
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        x_last_dim = x_dim - 1
        context_dim = self.encoder.output_dim
        shared_input_dim = x_last_dim + context_dim + max(stage_emb_dim, 0) + 1  # +1 H_steps
        if use_interaction:
            shared_input_dim += context_dim  # x_last * context interaction
        if region_emb_dim > 0:
            self.region_emb = nn.Embedding(num_regions, region_emb_dim)
            shared_input_dim += region_emb_dim
        else:
            self.region_emb = None
        if use_delta_features:
            shared_input_dim += x_dim - 1  # delta features same dim as x_last
        shared_layers: list[nn.Module] = []
        in_dim = shared_input_dim
        for _ in range(num_shared_layers):
            shared_layers.append(nn.Linear(in_dim, h_dim_fc))
            shared_layers.append(nn.ReLU())
            in_dim = h_dim_fc
        self.shared_fc = nn.Sequential(*shared_layers)

        # Relative time positional encoding: learned projection from scalar dt to x_dim
        if use_rel_time_pe:
            self.rel_time_pe = nn.Sequential(
                nn.Linear(1, x_dim),
                nn.Tanh(),
            )
        else:
            self.rel_time_pe = None

        if head_type == "pmf":
            self.cancel_head = self._make_head(h_dim_fc, num_layers_cs)
            self.progress_head = self._make_head(h_dim_fc, num_layers_cs)
            self.cancel_out = nn.Linear(h_dim_fc, h_max)
            self.progress_out = nn.Linear(h_dim_fc, h_max)
            self.stay_out = nn.Linear(h_dim_fc, 1)
        elif head_type == "stage_pmf":
            self.stage_cancel_heads = nn.ModuleList([self._make_head(h_dim_fc, num_layers_cs) for _ in range(_NUM_STAGE)])
            self.stage_progress_heads = nn.ModuleList([self._make_head(h_dim_fc, num_layers_cs) for _ in range(_NUM_STAGE)])
            self.stage_cancel_outs = nn.ModuleList([nn.Linear(h_dim_fc, h_max) for _ in range(_NUM_STAGE)])
            self.stage_progress_outs = nn.ModuleList([nn.Linear(h_dim_fc, h_max) for _ in range(_NUM_STAGE)])
            self.stage_stay_outs = nn.ModuleList([nn.Linear(h_dim_fc, 1) for _ in range(_NUM_STAGE)])
        elif head_type == "local":
            self.class_head = self._make_head(h_dim_fc, num_layers_cs)
            self.class_out = nn.Linear(h_dim_fc, 3)
        elif head_type == "hazard":
            self.cancel_head = self._make_head(h_dim_fc, num_layers_cs)
            self.progress_head = self._make_head(h_dim_fc, num_layers_cs)
            self.cancel_out = nn.Linear(h_dim_fc, h_max)
            self.progress_out = nn.Linear(h_dim_fc, h_max)
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        self.cancel_aux_head = nn.Linear(h_dim_fc, 1) if use_cancel_aux else None

    @staticmethod
    def _make_head(hidden_dim: int, num_layers_cs: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = hidden_dim
        for _ in range(num_layers_cs):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        history_mask: torch.Tensor,
        stage: torch.Tensor,
        h_steps: torch.Tensor,
        dropout_rate: float = 0.0,
        region_id: torch.Tensor | None = None,
        tau_step_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        batch_size, max_len, _ = x.shape

        lengths = history_mask.sum(dim=1).clamp(min=1).long()
        if tau_step_index is not None:
            last_idx = tau_step_index.long().clamp(min=0, max=max_len - 1)
        else:
            last_idx = (lengths - 1).clamp(min=0)
        hist_mask = history_mask.clone()
        if self.history_policy == "strict_prior":
            hist_mask[torch.arange(batch_size, device=x.device), last_idx] = 0.0

        # Time encoding uses differences of feature 0 (the CLEAN58 snapshot
        # interval), normalized over visible history rather than elapsed time.
        if self.rel_time_pe is not None:
            tau_time = x[torch.arange(batch_size, device=x.device), last_idx, 0]  # time col of last snapshot
            snap_time = x[:, :, 0]  # (B, T)
            dt = (tau_time.unsqueeze(1) - snap_time).clamp(min=0)  # (B, T)
            # Normalization depends only on the observed history.
            visible_dt = dt * hist_mask
            dt_norm = visible_dt / visible_dt.max(dim=1, keepdim=True).values.clamp(min=1.0)
            pe = self.rel_time_pe(dt_norm.unsqueeze(2))  # (B, T, x_dim)
            x = x + pe * hist_mask.unsqueeze(2)

        x_last = x[torch.arange(batch_size, device=x.device), last_idx, 1:]
        attention_weights = None
        if self.encoder_type == "stage_hybrid_riskpool_rnn":
            context, attention_weights = self.encoder(
                x, hist_mask, x_last, stage
            )
        elif self.encoder_type in ("qc_att_pool", "mh_att_pool", "qc_mh_att_pool", "survcancel"):
            context, attention_weights = self.encoder(x, hist_mask, x_last)
        elif self.encoder_type == "att_pool":
            context, attention_weights = self.encoder(x, hist_mask)
        else:
            context = self.encoder(x, hist_mask)
        if self.stage_emb is not None:
            stage_emb = self.stage_emb(stage.long())
        else:
            stage_emb = x.new_zeros(batch_size, 0)
        h_steps_norm = (h_steps.float() / float(self.h_max)).unsqueeze(1)

        combined_parts = [x_last, context, stage_emb, h_steps_norm]

        if self.use_interaction:
            ctx_dim = context.shape[1]
            x_last_for_interact = x_last[:, :ctx_dim] if x_last.shape[1] >= ctx_dim else F.pad(x_last, (0, ctx_dim - x_last.shape[1]))
            combined_parts.append(x_last_for_interact * context)

        if self.region_emb is not None and region_id is not None:
            combined_parts.append(self.region_emb(region_id.long()))

        if self.use_delta_features:
            # Delta = x_last - x_first (change from first to current snapshot)
            x_first = x[torch.arange(batch_size, device=x.device), 0, 1:]
            delta = x_last - x_first
            combined_parts.append(delta)

        combined = torch.cat(combined_parts, dim=1)
        if dropout_rate > 0:
            combined = F.dropout(combined, p=dropout_rate, training=self.training)
        shared = self.shared_fc(combined)

        step_positions = torch.arange(self.h_max, device=x.device).unsqueeze(0)
        valid_steps = step_positions < h_steps.long().unsqueeze(1)

        if self.head_type == "pmf":
            cancel_hidden = self.cancel_head(shared)
            progress_hidden = self.progress_head(shared)
            if dropout_rate > 0:
                cancel_hidden = F.dropout(cancel_hidden, p=dropout_rate, training=self.training)
                progress_hidden = F.dropout(progress_hidden, p=dropout_rate, training=self.training)

            cancel_logits = self.cancel_out(cancel_hidden)
            progress_logits = self.progress_out(progress_hidden)
            stay_logit = self.stay_out(shared)

            cancel_logits = cancel_logits.masked_fill(~valid_steps, -1e9)
            progress_logits = progress_logits.masked_fill(~valid_steps, -1e9)

            logits = torch.cat([cancel_logits, progress_logits, stay_logit], dim=1)
            probs = F.softmax(logits, dim=1)

            p_cancel = probs[:, : self.h_max]
            p_progress = probs[:, self.h_max : 2 * self.h_max]
            p_stay = probs[:, -1]
            class_logits = None
        elif self.head_type == "stage_pmf":
            stage_long = stage.long()
            p_cancel = x.new_zeros(batch_size, self.h_max)
            p_progress = x.new_zeros(batch_size, self.h_max)
            p_stay = x.new_zeros(batch_size)
            all_logits = x.new_zeros(batch_size, 2 * self.h_max + 1)

            for s_id in range(_NUM_STAGE):
                s_mask = stage_long == s_id
                if not s_mask.any():
                    continue
                s_shared = shared[s_mask]
                s_valid = valid_steps[s_mask]

                c_hidden = self.stage_cancel_heads[s_id](s_shared)
                p_hidden = self.stage_progress_heads[s_id](s_shared)
                if dropout_rate > 0:
                    c_hidden = F.dropout(c_hidden, p=dropout_rate, training=self.training)
                    p_hidden = F.dropout(p_hidden, p=dropout_rate, training=self.training)

                c_logits = self.stage_cancel_outs[s_id](c_hidden)
                p_logits = self.stage_progress_outs[s_id](p_hidden)
                stay_logit = self.stage_stay_outs[s_id](s_shared)

                c_logits = c_logits.masked_fill(~s_valid, -1e9)
                p_logits = p_logits.masked_fill(~s_valid, -1e9)

                s_logits = torch.cat([c_logits, p_logits, stay_logit], dim=1)
                s_probs = F.softmax(s_logits, dim=1)

                p_cancel[s_mask] = s_probs[:, : self.h_max]
                p_progress[s_mask] = s_probs[:, self.h_max : 2 * self.h_max]
                p_stay[s_mask] = s_probs[:, -1]
                all_logits[s_mask] = s_logits

            logits = all_logits
            class_logits = None
        elif self.head_type == "hazard":
            cancel_hidden = self.cancel_head(shared)
            progress_hidden = self.progress_head(shared)
            if dropout_rate > 0:
                cancel_hidden = F.dropout(cancel_hidden, p=dropout_rate, training=self.training)
                progress_hidden = F.dropout(progress_hidden, p=dropout_rate, training=self.training)

            # Hazard rates h(t) = sigmoid(logit) in [0,1]
            cancel_hazard = torch.sigmoid(self.cancel_out(cancel_hidden))  # (B, h_max)
            progress_hazard = torch.sigmoid(self.progress_out(progress_hidden))  # (B, h_max)

            # Overall survival: S(t) = prod_{k<t} (1 - h_cancel(k) - h_progress(k))
            # Clamp combined hazard to avoid negative survival
            combined_hazard = (cancel_hazard + progress_hazard).clamp(max=1.0 - _EPSILON)
            # S(t) for t=0..h_max-1: S(0)=1, S(t) = prod_{k=0}^{t-1} (1 - combined(k))
            log_surv = torch.cumsum(torch.log(1.0 - combined_hazard + _EPSILON), dim=1)
            # Shift: S(0)=1, S(1)=1-h(0), etc.
            log_surv_shifted = F.pad(log_surv[:, :-1], (1, 0), value=0.0)  # (B, h_max)
            surv = torch.exp(log_surv_shifted)

            # PMF: f_cancel(t) = S(t) * h_cancel(t), f_progress(t) = S(t) * h_progress(t)
            p_cancel = surv * cancel_hazard
            p_progress = surv * progress_hazard
            p_cancel = p_cancel.masked_fill(~valid_steps, 0.0)
            p_progress = p_progress.masked_fill(~valid_steps, 0.0)
            p_stay = 1.0 - p_cancel.sum(dim=1) - p_progress.sum(dim=1)
            p_stay = p_stay.clamp(min=0.0)

            # Store hazard logits for loss computation
            logits = torch.cat([self.cancel_out(cancel_hidden), self.progress_out(progress_hidden)], dim=1)
            class_logits = None
        else:
            class_hidden = self.class_head(shared)
            if dropout_rate > 0:
                class_hidden = F.dropout(class_hidden, p=dropout_rate, training=self.training)
            class_logits = self.class_out(class_hidden)
            class_probs = F.softmax(class_logits, dim=1)

            valid_steps_f = valid_steps.float()
            step_den = valid_steps_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            p_cancel = class_probs[:, LABEL_CANCEL : LABEL_CANCEL + 1] * (valid_steps_f / step_den)
            p_progress = class_probs[:, LABEL_PROGRESS : LABEL_PROGRESS + 1] * (valid_steps_f / step_den)
            p_stay = class_probs[:, LABEL_STAY]
            logits = class_logits

        return {
            "head_type": self.head_type,
            "p_cancel": p_cancel,
            "p_progress": p_progress,
            "p_stay": p_stay,
            "logits": logits,
            "class_logits": class_logits,
            "attention_weights": attention_weights,
            "cancel_aux_logit": self.cancel_aux_head(shared).squeeze(1) if self.cancel_aux_head is not None else None,
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
    """Compute feature mean/std on valid sequence positions without materializing sampled batches."""
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
    batch: dict[str, torch.Tensor],
    time_step_seconds: float = _TIME_STEP_SECONDS,
    h_max: int = _H_MAX,
) -> torch.Tensor:
    stage = batch["stage"].long()
    label_3way = batch["label_3way"].long()
    time_to_first = batch["time_to_first_event_in_window"].float()

    step_idx = torch.clamp(torch.ceil(time_to_first / time_step_seconds).long() - 1, min=0, max=h_max - 1)
    target = torch.full_like(label_3way, fill_value=2 * h_max)

    cancel_mask = label_3way == LABEL_CANCEL
    progress_mask = label_3way == LABEL_PROGRESS

    target[cancel_mask] = step_idx[cancel_mask]
    target[progress_mask] = h_max + step_idx[progress_mask]

    # stage currently only affects semantics, not output indexing.
    _ = stage
    return target


def _ranking_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    h_max: int = _H_MAX,
    margin: float = 0.05,
    max_pairs: int = 256,
) -> torch.Tensor:
    """Pairwise margin ranking loss: cancel samples should have higher p_cancel_window."""
    p_cancel = output["p_cancel"]
    H_steps = batch["H_steps"].long()
    # compute p_cancel_window per sample
    mask = torch.arange(h_max, device=p_cancel.device).unsqueeze(0) < H_steps.unsqueeze(1)
    p_cancel_window = (p_cancel * mask.float()).sum(dim=1)

    cancel_mask = batch["label_3way"].long() == LABEL_CANCEL
    if cancel_mask.sum() == 0 or (~cancel_mask).sum() == 0:
        return p_cancel_window.new_tensor(0.0)

    cancel_scores = p_cancel_window[cancel_mask]
    other_scores = p_cancel_window[~cancel_mask]

    n_pairs = min(len(cancel_scores), len(other_scores), max_pairs)
    idx_c = torch.randint(len(cancel_scores), (n_pairs,), device=p_cancel.device)
    idx_o = torch.randint(len(other_scores), (n_pairs,), device=p_cancel.device)

    return F.margin_ranking_loss(
        cancel_scores[idx_c], other_scores[idx_o],
        target=cancel_scores.new_ones(n_pairs), margin=margin,
    )


def _focal_bce_loss(
    logit: torch.Tensor, target: torch.Tensor, gamma: float = 2.0,
) -> torch.Tensor:
    """Binary focal loss from logits."""
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = target * p + (1 - target) * (1 - p)
    return (bce * (1 - pt) ** gamma).mean()


def _survcancel_routing_loss(
    model: nn.Module,
    route_length_weight: float = 0.0,
    route_diversity_weight: float = 0.0,
    route_balance_weight: float = 0.0,
    length_threshold: float = 1.6,  # log(5) ≈ 1.6, so seq_len=5 is the crossover
) -> torch.Tensor:
    """Compute auxiliary routing losses for SurvCancel encoder."""
    encoder = model.encoder if hasattr(model, 'encoder') else None
    if encoder is None or not hasattr(encoder, '_last_gate'):
        return torch.tensor(0.0)

    g = encoder._last_gate  # (B, 1)
    log_len = encoder._last_log_len  # (B, 1)
    ctx_short = encoder._last_ctx_short  # (B, H)
    ctx_long = encoder._last_ctx_long  # (B, H)
    device = g.device
    loss = g.new_tensor(0.0)

    # 1. Length-aware routing: encourage gate to follow sequence length
    if route_length_weight > 0:
        target_gate = torch.sigmoid((log_len - length_threshold) * 3.0)  # smooth step at threshold
        loss = loss + route_length_weight * F.mse_loss(g, target_gate)

    # 2. Diversity: encourage branches to produce different outputs
    if route_diversity_weight > 0:
        cos_sim = F.cosine_similarity(ctx_short, ctx_long, dim=1).mean()
        loss = loss + route_diversity_weight * cos_sim  # minimize similarity

    # 3. Load balancing: prevent gate from collapsing to one branch
    if route_balance_weight > 0:
        mean_gate = g.mean()
        # Encourage mean gate to be around 0.5
        loss = loss + route_balance_weight * (mean_gate - 0.5) ** 2

    return loss


def stage_deephit_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    time_step_seconds: float = _TIME_STEP_SECONDS,
    h_max: int = _H_MAX,
    cancel_weight: float = 1.0,
    ranking_weight: float = 0.0,
    aux_cancel_weight: float = 0.0,
    model: nn.Module | None = None,
    route_length_weight: float = 0.0,
    route_diversity_weight: float = 0.0,
    route_balance_weight: float = 0.0,
) -> torch.Tensor:
    if output["head_type"] in ("pmf", "stage_pmf"):
        target = _target_index_from_batch(batch, time_step_seconds=time_step_seconds, h_max=h_max)
        log_probs = torch.log(output["logits"].softmax(dim=1).clamp_min(_EPSILON))

        if cancel_weight != 1.0:
            w = [cancel_weight] * h_max + [1.0] * h_max + [1.0]
            weight = torch.tensor(w, device=log_probs.device, dtype=log_probs.dtype)
            loss = F.nll_loss(log_probs, target, weight=weight)
        else:
            loss = F.nll_loss(log_probs, target)

        if ranking_weight > 0:
            loss = loss + ranking_weight * _ranking_loss(output, batch, h_max=h_max)

        if aux_cancel_weight > 0 and "cancel_aux_logit" in output:
            y = batch["y_cancel_window"].float()
            loss = loss + aux_cancel_weight * _focal_bce_loss(output["cancel_aux_logit"], y)

        if model is not None:
            loss = loss + _survcancel_routing_loss(model, route_length_weight, route_diversity_weight, route_balance_weight)

        return loss

    if output["head_type"] == "hazard":
        label_3way = batch["label_3way"].long()
        time_to_first = batch["time_to_first_event_in_window"].float()
        step_idx = torch.clamp(torch.ceil(time_to_first / time_step_seconds).long() - 1, min=0, max=h_max - 1)

        # Split logits back into cancel and progress hazard logits
        cancel_logits = output["logits"][:, :h_max]
        progress_logits = output["logits"][:, h_max:]

        B = label_3way.shape[0]
        device = cancel_logits.device

        # Build per-step targets: for each sample, at the event step, the corresponding
        # hazard should be high; at all prior steps, both hazards should be low (survived)
        # For stay samples, all steps should have low hazard

        loss = cancel_logits.new_tensor(0.0)
        n_terms = 0

        for i_step in range(h_max):
            # Mask: samples that are still "at risk" at this step
            # Cancel at step k: at risk for steps 0..k, event at k
            # Progress at step k: at risk for steps 0..k, event at k
            # Stay: at risk for all steps, no event

            cancel_at_step = (label_3way == LABEL_CANCEL) & (step_idx == i_step)
            progress_at_step = (label_3way == LABEL_PROGRESS) & (step_idx == i_step)
            at_risk = (label_3way == LABEL_STAY) | (step_idx >= i_step)

            if not at_risk.any():
                continue

            # Cancel hazard BCE at this step
            c_target = cancel_at_step[at_risk].float()
            c_logit = cancel_logits[at_risk, i_step]
            loss = loss + F.binary_cross_entropy_with_logits(c_logit, c_target, reduction="mean")

            # Progress hazard BCE at this step
            p_target = progress_at_step[at_risk].float()
            p_logit = progress_logits[at_risk, i_step]
            loss = loss + F.binary_cross_entropy_with_logits(p_logit, p_target, reduction="mean")

            n_terms += 2

        if n_terms > 0:
            loss = loss / n_terms

        if cancel_weight != 1.0:
            # Reweight: add extra loss on cancel event steps
            cancel_mask = label_3way == LABEL_CANCEL
            if cancel_mask.any():
                c_idx = step_idx[cancel_mask]
                c_logit = cancel_logits[cancel_mask].gather(1, c_idx.unsqueeze(1)).squeeze(1)
                extra = F.binary_cross_entropy_with_logits(c_logit, torch.ones_like(c_logit), reduction="mean")
                loss = loss + (cancel_weight - 1.0) * extra

        if ranking_weight > 0:
            loss = loss + ranking_weight * _ranking_loss(output, batch, h_max=h_max)

        if aux_cancel_weight > 0 and "cancel_aux_logit" in output:
            y = batch["y_cancel_window"].float()
            loss = loss + aux_cancel_weight * _focal_bce_loss(output["cancel_aux_logit"], y)

        return loss

    return F.cross_entropy(output["class_logits"], batch["label_3way"].long())


@dataclass
class StageDeepHitMetrics:
    loss: float
    n_samples: int


class StageDeepHit:
    """Training wrapper for stage-conditioned fixed-window prediction."""

    def __init__(
        self,
        x_dim: int,
        h_dim_rnn: int = 100,
        h_dim_fc: int = 100,
        num_layers_rnn: int = 2,
        num_layers_att: int = 2,
        num_layers_cs: int = 2,
        stage_emb_dim: int = 8,
        rnn_type: str = "LSTM",
        encoder_type: str = "rnn_att",
        head_type: str = "pmf",
        h_max: int = _H_MAX,
        time_step_seconds: float = _TIME_STEP_SECONDS,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        epochs: int = 20,
        patience: int = 5,
        max_gradient_norm: float = 1.0,
        device: str = "cpu",
        verbose: bool = True,
        cancel_weight: float = 1.0,
        ranking_weight: float = 0.0,
        aux_cancel_weight: float = 0.0,
        use_rel_time_pe: bool = False,
        num_shared_layers: int = 1,
        region_emb_dim: int = 0,
        num_regions: int = 200,
        use_delta_features: bool = False,
        use_interaction: bool = True,
        history_policy: str = "strict_prior",
        **kwargs,
    ) -> None:
        self.x_dim = x_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.max_gradient_norm = max_gradient_norm
        self.device = torch.device(device)
        self.verbose = verbose
        self.h_max = int(h_max)
        self.time_step_seconds = float(time_step_seconds)
        self.cancel_weight = cancel_weight
        self.ranking_weight = ranking_weight
        self.aux_cancel_weight = aux_cancel_weight
        self.route_length_weight = float(kwargs.get('route_length_weight', 0.0))
        self.route_diversity_weight = float(kwargs.get('route_diversity_weight', 0.0))
        self.route_balance_weight = float(kwargs.get('route_balance_weight', 0.0))
        raw_drop_indices = kwargs.get("feature_drop_indices", ())
        self.feature_drop_indices = tuple(
            sorted({int(index) for index in raw_drop_indices})
        )
        if any(index <= 0 or index >= self.x_dim for index in self.feature_drop_indices):
            raise ValueError(
                f"feature_drop_indices must be within 1..{self.x_dim - 1}: "
                f"{self.feature_drop_indices}"
            )

        self.model = StageDeepHitNet(
            x_dim=x_dim,
            h_dim_rnn=h_dim_rnn,
            h_dim_fc=h_dim_fc,
            num_layers_rnn=num_layers_rnn,
            num_layers_att=num_layers_att,
            num_layers_cs=num_layers_cs,
            stage_emb_dim=stage_emb_dim,
            rnn_type=rnn_type,
            encoder_type=encoder_type,
            head_type=head_type,
            h_max=self.h_max,
            use_cancel_aux=aux_cancel_weight > 0,
            use_rel_time_pe=use_rel_time_pe,
            num_shared_layers=num_shared_layers,
            region_emb_dim=region_emb_dim,
            num_regions=num_regions,
            use_delta_features=use_delta_features,
            use_interaction=use_interaction,
            history_policy=history_policy,
        ).to(self.device)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self._feature_mean_tensor: torch.Tensor | None = None
        self._feature_std_tensor: torch.Tensor | None = None
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

    def _ensure_feature_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Feature statistics are not initialized.")
        if self._feature_mean_tensor is None or self._feature_std_tensor is None:
            self._feature_mean_tensor = torch.as_tensor(self.feature_mean, dtype=torch.float32, device=self.device)
            self._feature_std_tensor = torch.as_tensor(self.feature_std, dtype=torch.float32, device=self.device)
        return self._feature_mean_tensor, self._feature_std_tensor

    def _to_tensor(
        self,
        value: Any,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=dtype) if dtype is not None else value.to(device=self.device)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        assert self.feature_mean is not None and self.feature_std is not None
        feature_mean, feature_std = self._ensure_feature_tensors()
        X = self._to_tensor(batch["X"], dtype=torch.float32)
        seq_mask = self._to_tensor(batch["seq_mask"], dtype=torch.float32)
        X_scaled = X.clone()
        X_scaled[:, :, 1:] = (X_scaled[:, :, 1:] - feature_mean[1:]) / feature_std[1:]
        if self.feature_drop_indices:
            X_scaled[:, :, list(self.feature_drop_indices)] = 0.0
        X_scaled = X_scaled * seq_mask.unsqueeze(-1)

        tensor_batch = {
            "X": X_scaled,
            "seq_mask": seq_mask,
            "history_mask": self._to_tensor(batch["history_mask"], dtype=torch.float32),
            "stage": self._to_tensor(batch["stage"]),
            "tau_step_index": self._to_tensor(batch["tau_step_index"]),
            "H_steps": self._to_tensor(batch["H_steps"]),
            "label_3way": self._to_tensor(batch["label_3way"]),
            "y_cancel_window": self._to_tensor(batch["y_cancel_window"]),
            "y_progress_window": self._to_tensor(batch["y_progress_window"]),
            "time_to_first_event_in_window": self._to_tensor(batch["time_to_first_event_in_window"], dtype=torch.float32),
        }
        if "region_id" in batch:
            tensor_batch["region_id"] = self._to_tensor(batch["region_id"], dtype=torch.long)
        return tensor_batch

    def _run_epoch(self, loader: Any, training: bool) -> StageDeepHitMetrics:
        total_loss = 0.0
        total_n = 0

        if training:
            self.model.train()
        else:
            self.model.eval()

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
                    tau_step_index=tensor_batch["tau_step_index"],
                    dropout_rate=self.dropout_rate if training else 0.0,
                    region_id=tensor_batch.get("region_id"),
                )
                loss = stage_deephit_loss(
                    output,
                    tensor_batch,
                    time_step_seconds=self.time_step_seconds,
                    h_max=self.h_max,
                    cancel_weight=self.cancel_weight,
                    ranking_weight=self.ranking_weight,
                    aux_cancel_weight=self.aux_cancel_weight,
                    model=self.model,
                    route_length_weight=self.route_length_weight,
                    route_diversity_weight=self.route_diversity_weight,
                    route_balance_weight=self.route_balance_weight,
                )

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_gradient_norm)
                self.optimizer.step()

            batch_size = int(tensor_batch["X"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_n += batch_size

        return StageDeepHitMetrics(loss=total_loss / max(total_n, 1), n_samples=total_n)

    def fit(self, train_loader: Any, val_loader: Any | None = None, *, checkpoint_path=None, resume=False) -> "StageDeepHit":
        if self.verbose:
            print("Preparing feature statistics...")
        self._prepare_numpy_stats(train_loader)
        self._feature_mean_tensor = None
        self._feature_std_tensor = None
        if self.verbose:
            print("Feature statistics ready. Starting training...")
        self.optimizer = torch.optim.Adam(
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
            epoch_iter = tqdm(epoch_iter, desc="StageDeepHit")

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
            if val_metrics is not None:
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
            "feature_drop_indices": list(self.feature_drop_indices),
            "x_dim": self.x_dim,
            "encoder_type": self.model.encoder_type,
            "history_policy": self.model.history_policy,
            "head_type": self.model.head_type,
            "h_max": self.h_max,
            "time_step_seconds": self.time_step_seconds,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "max_gradient_norm": self.max_gradient_norm,
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
            tau_step_index=tensor_batch["tau_step_index"],
            dropout_rate=0.0,
            region_id=tensor_batch.get("region_id"),
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
            "attention_weights": output["attention_weights"].cpu().numpy() if output["attention_weights"] is not None else None,
        }
