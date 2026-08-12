"""
Model implementations for Churn Prediction Experiment.

Proposed:
  HIFTTransformer — Hospitality-Informed Feature-Tokenizer Transformer
                    Novel components:
                    (A) Booking-Phase Aware Conditioning (BPAC)
                    (B) Semantic Group Attention (SGA)

Baselines:
  VanillaFTTransformer — plain FT-Transformer (no BPAC, no SGA)
  LSTM, GRU            — bidirectional RNN baselines
  MLPChurn             — deep feedforward MLP (no attention, no recurrence)
  CNNChurn             — 1-D convolutional network over feature tokens
  XGBoost              — gradient boosting (GPU)
  LogisticRegression   — linear classifier (interpretable baseline)

All deep models accept (X_num, X_cat) tensors and return logits.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost              import XGBClassifier

from src.config import (
    TRANSFORMER, LSTM_CFG, GRU_CFG,
    XGBOOST_CFG, LR_CFG, SEED
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: ATTENTION SCORES HOOK
# ─────────────────────────────────────────────────────────────────────────────

class AttentionStore:
    """Stores attention weights from the last forward pass (for analysis)."""
    def __init__(self):
        self.weights = []

    def clear(self):
        self.weights.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 1A. NOVEL: BOOKING-PHASE AWARE CONDITIONING (BPAC)
# ─────────────────────────────────────────────────────────────────────────────

class BookingPhaseConditioner(nn.Module):
    """
    Booking-Phase Aware Conditioning (BPAC)  — Novel Component A.

    Hypothesis: cancellation dynamics differ fundamentally across booking horizons.
    Long-horizon bookings (>90 days) are sensitive to deposit policy and pricing;
    short-horizon bookings (<30 days) correlate more with guest history and room
    mismatches.

    Implementation:
      - The (normalised) lead_time feature is mapped to 3 phases via learnable
        soft thresholds (differentiable sigmoid, boundaries initialised to ≈ the
        short/medium/long boundaries in normalised space).
      - A phase-specific embedding is computed as a weighted mixture and then
        applied as a multiplicative residual gate to ALL feature tokens.
      - The boundaries are learned during training, letting the model discover
        the optimal booking-horizon segmentation for churn risk.
    """
    N_PHASES = 3          # short / medium / long horizon

    def __init__(self, d_model: int, init_boundaries: tuple = (-0.5, 0.5)):
        super().__init__()
        # Learnable phase boundaries in normalised feature space
        self.boundaries  = nn.Parameter(
            torch.tensor(list(init_boundaries), dtype=torch.float32)
        )
        self.temperature = 5.0                                # sigmoid sharpness (fixed)
        self.phase_embed = nn.Embedding(self.N_PHASES, d_model)
        self.gate_proj   = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        nn.init.normal_(self.phase_embed.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, lead_time_scaled: torch.Tensor):
        """
        tokens           : (B, seq_len, d_model)  — all tokens including CLS
        lead_time_scaled : (B,)                   — normalised lead_time from X_num
        Returns          : (B, seq_len, d_model)   — phase-conditioned tokens
        """
        lt = lead_time_scaled.unsqueeze(-1)                   # (B, 1)
        b  = torch.sort(self.boundaries).values               # ensure b[0] < b[1]

        # Soft phase membership via differentiable step functions
        p1 = torch.sigmoid(self.temperature * (lt - b[0]))   # P(phase ≥ medium)
        p2 = torch.sigmoid(self.temperature * (lt - b[1]))   # P(phase ≥ long)
        phase_probs = torch.stack(
            [1.0 - p1, p1 - p2, p2], dim=-1
        ).squeeze(1)                                          # (B, 3)

        # Weighted mixture of phase embeddings → multiplicative gate
        phase_emb = phase_probs @ self.phase_embed.weight     # (B, d_model)
        gate       = self.gate_proj(phase_emb).unsqueeze(1)  # (B, 1, d_model)
        return tokens * gate + tokens                         # residual multiplicative gate


# ─────────────────────────────────────────────────────────────────────────────
# 1B. NOVEL: SEMANTIC GROUP ATTENTION (SGA)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticGroupAttention(nn.Module):
    """
    Semantic Group Attention (SGA)  — Novel Component B.

    Two-level hierarchical attention over domain-meaningful feature groups:

    Level 1 — Intra-group self-attention:
      For each semantic feature group (Temporal, Stay Pattern, Guest History,
      Commercial, Property & Room), multi-head self-attention is applied to
      tokens within the group to capture within-domain feature interactions
      (e.g.  lead_time × days_in_waiting_list  within the Temporal group).

    Level 2 — Inter-group cross-attention (CRM dashboard):
      The CLS token queries group-level summary vectors (mean-pooled intra-group
      outputs) via cross-attention, producing group importance weights
      W_g ∈ ℝ^G.  These weights are directly interpretable as a managerial
      risk dashboard: "Which domain facet (timing, guest history, …) drove
      this cancellation prediction?"

    The group importance weights are cached after each evaluation batch for
    post-hoc interpretability analysis.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.intra_attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.inter_attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm_intra  = nn.LayerNorm(d_model)
        self.norm_inter  = nn.LayerNorm(d_model)
        self.drop        = nn.Dropout(dropout)
        self.group_weights_ = None   # cached (B, n_groups) from last forward

    def forward(self, tokens: torch.Tensor, group_indices: list):
        """
        tokens        : (B, seq_len, d_model)  — CLS at position 0
        group_indices : list[list[int]]         — 1-indexed (offset for CLS)
        Returns       : (updated_tokens, group_weights (B, n_groups) or None)
        """
        updated    = tokens.clone()
        group_reps = []

        # ── Level 1: intra-group self-attention ──────────────────────────────
        for idx_list in group_indices:
            if not idx_list:
                continue
            g        = tokens[:, idx_list, :]               # (B, g_size, d)
            g_out, _ = self.intra_attn(g, g, g)
            g_out    = self.norm_intra(g + self.drop(g_out))
            updated[:, idx_list, :] = g_out
            group_reps.append(g_out.mean(dim=1, keepdim=True))  # (B, 1, d)

        if not group_reps:
            return tokens, None

        # ── Level 2: inter-group cross-attention (CLS as query) ──────────────
        group_seq = torch.cat(group_reps, dim=1)            # (B, n_groups, d)
        cls_q     = updated[:, 0:1, :]                      # (B, 1, d)

        inter_out, group_w = self.inter_attn(
            cls_q, group_seq, group_seq,
            need_weights=True, average_attn_weights=True,
        )  # inter_out: (B,1,d)  group_w: (B,1,n_groups)

        group_w = group_w.squeeze(1)                        # (B, n_groups)
        self.group_weights_ = group_w.detach().cpu()

        updated[:, 0:1, :] = self.norm_inter(
            cls_q + self.drop(inter_out)
        )
        return updated, group_w


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PROPOSED MODEL — HI-FTT (Hospitality-Informed Feature-Tokenizer Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Converts:
      - numerical features  → linear projection  → d_model tokens
      - categorical features→ embedding lookup    → d_model tokens
    Concatenates all tokens along sequence dimension.
    """

    def __init__(
        self,
        num_numerical: int,
        cat_dims: list,
        d_model: int,
    ):
        super().__init__()
        self.d_model = d_model

        # Numerical: one linear layer per feature (weight-sharing within feature)
        if num_numerical > 0:
            self.num_proj = nn.Linear(num_numerical, num_numerical * d_model)
        else:
            self.num_proj = None
        self.num_numerical = num_numerical

        # Categorical: one embedding per feature
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(dim + 1, d_model, padding_idx=0)
            for dim in cat_dims
        ])
        self.num_categorical = len(cat_dims)

    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor):
        """
        X_num : (B, num_numerical)
        X_cat : (B, num_categorical)
        Returns: tokens (B, num_numerical + num_categorical, d_model)
        """
        tokens = []

        if self.num_proj is not None and self.num_numerical > 0:
            # (B, num_numerical * d_model) → (B, num_numerical, d_model)
            num_tok = self.num_proj(X_num)
            num_tok = num_tok.view(-1, self.num_numerical, self.d_model)
            tokens.append(num_tok)

        if self.num_categorical > 0:
            cat_tok = torch.stack(
                [emb(X_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)],
                dim=1
            )   # (B, num_categorical, d_model)
            tokens.append(cat_tok)

        return torch.cat(tokens, dim=1)    # (B, N_tokens, d_model)


class HIFTTransformer(nn.Module):
    """
    Hospitality-Informed Feature-Tokenizer Transformer (HI-FTT)  — Proposed Model.

    Extends the standard Feature-Tokenizer Transformer with two novel
    domain-specific components for hotel churn prediction:

    (A) Booking-Phase Aware Conditioning (BPAC):
        Discretises the lead_time feature into short / medium / long booking
        horizons via learnable soft boundaries, then applies phase-specific
        multiplicative gating to all feature tokens.  Captures the empirical
        observation that cancellation risk dynamics differ fundamentally across
        booking horizons.

    (B) Semantic Group Attention (SGA):
        Organises the 35 hotel features into 5 domain-meaningful groups
        (Temporal, Stay Pattern, Guest History, Commercial, Property & Room).
        Intra-group self-attention captures within-domain interactions;
        CLS-driven inter-group cross-attention produces group importance
        weights W_g ∈ ℝ^5 — a directly interpretable CRM risk dashboard.

    Full pipeline:
        FeatureTokenizer → CLS prepend + PosEnc
        → BPAC → SGA → TransformerEncoder (L layers) → CLS head → σ
    """

    def __init__(
        self,
        num_numerical:   int,
        cat_dims:        list,
        d_model:         int   = None,
        nhead:           int   = None,
        num_layers:      int   = None,
        dim_feedforward: int   = None,
        dropout:         float = None,
        attn_dropout:    float = None,
        use_cls_token:   bool  = True,
        use_pos_enc:     bool  = True,
        # ── Novel component parameters ──────────────────────────────────────
        lead_time_idx:   int   = None,   # index of lead_time in X_num
        feature_groups:  list  = None,   # list[list[int]] — 0-indexed feature positions
        group_names:     list  = None,   # human-readable group labels
        use_phase_cond:  bool  = True,   # toggle BPAC (Novel A)
        use_group_attn:  bool  = True,   # toggle SGA  (Novel B)
    ):
        super().__init__()

        d_model         = d_model         or TRANSFORMER["d_model"]
        nhead           = nhead           or TRANSFORMER["nhead"]
        num_layers      = num_layers      or TRANSFORMER["num_layers"]
        dim_feedforward = dim_feedforward or TRANSFORMER["dim_feedforward"]
        dropout         = dropout         if dropout is not None else TRANSFORMER["dropout"]

        self.d_model        = d_model
        self.use_cls_token  = use_cls_token
        self.use_pos_enc    = use_pos_enc
        self.lead_time_idx  = lead_time_idx
        self.feature_groups = feature_groups
        self.group_names    = group_names or (
            [f"G{i}" for i in range(len(feature_groups))] if feature_groups else []
        )
        self.use_phase_cond = use_phase_cond and (lead_time_idx is not None)
        self.use_group_attn = (
            use_group_attn and
            (feature_groups is not None) and
            (len(feature_groups) > 0)
        )

        # ── Feature tokenizer ────────────────────────────────────────────────
        self.tokenizer = FeatureTokenizer(num_numerical, cat_dims, d_model)
        N = num_numerical + len(cat_dims)

        # ── CLS token ────────────────────────────────────────────────────────
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            seq_len = N + 1
        else:
            self.cls_token = None
            seq_len = N

        # ── Positional encoding ──────────────────────────────────────────────
        if use_pos_enc:
            self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        else:
            self.pos_enc = None

        # ── Novel A: Booking-Phase Aware Conditioning ────────────────────────
        self.phase_conditioner = (
            BookingPhaseConditioner(d_model) if self.use_phase_cond else None
        )

        # ── Novel B: Semantic Group Attention ────────────────────────────────
        self.group_attention = (
            SemanticGroupAttention(d_model, nhead, dropout)
            if self.use_group_attn else None
        )

        # ── Transformer encoder (Pre-LN) ─────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,        # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers          = num_layers,
            enable_nested_tensor= False,
        )

        # ── Classification head ──────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()
        self._group_importance = None   # cached mean group weights from eval

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, X_num: torch.Tensor, X_cat: torch.Tensor):
        """Returns logits (B,)."""
        B = X_num.size(0)

        # 1. Feature tokenization
        x = self.tokenizer(X_num, X_cat)           # (B, N, d_model)

        # 2. Prepend CLS + positional encoding
        if self.cls_token is not None:
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)       # (B, N+1, d_model)
        if self.pos_enc is not None:
            x = x + self.pos_enc

        # 3. Novel A: Booking-Phase Aware Conditioning
        if self.phase_conditioner is not None:
            x = self.phase_conditioner(x, X_num[:, self.lead_time_idx])

        # 4. Novel B: Semantic Group Attention
        if self.group_attention is not None:
            cls_offset = 1 if self.cls_token is not None else 0
            shifted    = [[i + cls_offset for i in g] for g in self.feature_groups]
            x, grp_w   = self.group_attention(x, shifted)
            # cache mean importance (eval mode only to avoid overhead)
            if grp_w is not None and not self.training:
                self._group_importance = grp_w.mean(dim=0).detach().cpu()

        # 5. Transformer encoder
        x = self.transformer(x)                    # (B, seq_len, d_model)

        # 6. Pool and classify
        rep    = x[:, 0, :] if self.cls_token is not None else x.mean(dim=1)
        logits = self.head(rep).squeeze(-1)         # (B,)
        return logits

    @property
    def group_importance(self):
        """Mean group importance from last evaluation pass: (n_groups,) or None."""
        return self._group_importance


# Keep TabTransformer as an alias for backward-compatibility (ablation helpers)
TabTransformer = HIFTTransformer


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BASELINE — Bidirectional LSTM
# ─────────────────────────────────────────────────────────────────────────────

class LSTMChurn(nn.Module):
    """
    Bidirectional LSTM for tabular churn prediction.
    Treats each feature as one time-step (feature sequence view).
    """

    def __init__(self, num_numerical: int, cat_dims: list):
        super().__init__()
        d_model   = LSTM_CFG["hidden_size"]
        n_layers  = LSTM_CFG["num_layers"]
        dropout   = LSTM_CFG["dropout"]
        bidir     = LSTM_CFG["bidirectional"]

        self.tokenizer = FeatureTokenizer(num_numerical, cat_dims, d_model)

        self.lstm = nn.LSTM(
            input_size    = d_model,
            hidden_size   = d_model,
            num_layers    = n_layers,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = bidir,
            batch_first   = True,
        )

        out_dim = d_model * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, X_num, X_cat):
        x   = self.tokenizer(X_num, X_cat)    # (B, N, d)
        out, (h, _) = self.lstm(x)
        # Concat last hidden from both directions
        if self.lstm.bidirectional:
            rep = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            rep = h[-1]
        return self.head(rep).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BASELINE — Bidirectional GRU
# ─────────────────────────────────────────────────────────────────────────────

class GRUChurn(nn.Module):
    """Bidirectional GRU — same architecture skeleton as LSTM."""

    def __init__(self, num_numerical: int, cat_dims: list):
        super().__init__()
        d_model   = GRU_CFG["hidden_size"]
        n_layers  = GRU_CFG["num_layers"]
        dropout   = GRU_CFG["dropout"]
        bidir     = GRU_CFG["bidirectional"]

        self.tokenizer = FeatureTokenizer(num_numerical, cat_dims, d_model)

        self.gru = nn.GRU(
            input_size    = d_model,
            hidden_size   = d_model,
            num_layers    = n_layers,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = bidir,
            batch_first   = True,
        )

        out_dim = d_model * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1),
        )

    def forward(self, X_num, X_cat):
        x   = self.tokenizer(X_num, X_cat)
        out, h = self.gru(x)
        if self.gru.bidirectional:
            rep = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            rep = h[-1]
        return self.head(rep).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BASELINE — Deep MLP  (Feature-Tokenizer + Flatten + FFN)
# ─────────────────────────────────────────────────────────────────────────────

class MLPChurn(nn.Module):
    """
    Deep MLP baseline.
    Uses the same FeatureTokenizer as Transformer/LSTM/GRU so feature
    processing is identical; the only difference is the lack of attention
    or recurrence — tokens are flattened and passed through an FFN.
    Architecture: Tokenizer → Flatten → [Linear→LN→GELU→Dropout] × 3 → head
    """

    def __init__(self, num_numerical: int, cat_dims: list):
        super().__init__()
        d_model = TRANSFORMER["d_model"]
        dropout = TRANSFORMER["dropout"]

        self.tokenizer = FeatureTokenizer(num_numerical, cat_dims, d_model)
        n_tokens = num_numerical + len(cat_dims)
        flat_dim = n_tokens * d_model

        self.net = nn.Sequential(
            nn.Flatten(),                                        # (B, N*d)
            nn.Linear(flat_dim, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, X_num, X_cat):
        x = self.tokenizer(X_num, X_cat)    # (B, N, d)
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BASELINE — 1-D CNN  (Feature-Tokenizer + Temporal Convolutions)
# ─────────────────────────────────────────────────────────────────────────────

class CNNChurn(nn.Module):
    """
    1-D Convolutional baseline.
    Treats feature tokens as a sequence and applies 1-D conv layers to
    capture local feature interactions. No global attention — a conv filter
    can only see a local window of features at a time.
    Architecture: Tokenizer → Conv1d × 3 (with residual) → GlobalMaxPool → head
    """

    def __init__(self, num_numerical: int, cat_dims: list):
        super().__init__()
        d_model = TRANSFORMER["d_model"]
        dropout = TRANSFORMER["dropout"]

        self.tokenizer = FeatureTokenizer(num_numerical, cat_dims, d_model)

        # Three conv blocks with increasing receptive field
        self.conv1 = self._conv_block(d_model, d_model,     kernel_size=3, dropout=dropout)
        self.conv2 = self._conv_block(d_model, d_model * 2, kernel_size=3, dropout=dropout)
        self.conv3 = self._conv_block(d_model * 2, d_model, kernel_size=1, dropout=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def _conv_block(in_ch, out_ch, kernel_size, dropout):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, X_num, X_cat):
        x = self.tokenizer(X_num, X_cat)    # (B, N, d)
        x = x.transpose(1, 2)               # (B, d, N) — Conv1d expects (B, C, L)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        rep = x.max(dim=-1).values          # global max pooling over feature dim
        return self.head(rep).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 6–7. TRADITIONAL ML BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def build_xgboost():
    return XGBClassifier(**XGBOOST_CFG)

def build_logistic_regression():
    return LogisticRegression(**LR_CFG)

def build_lightgbm():
    from lightgbm import LGBMClassifier          # optional dependency
    from src.config import LIGHTGBM_CFG
    return LGBMClassifier(**LIGHTGBM_CFG)

def build_catboost():
    from catboost import CatBoostClassifier       # optional dependency
    from src.config import CATBOOST_CFG
    return CatBoostClassifier(**CATBOOST_CFG)


# ─────────────────────────────────────────────────────────────────────────────
# VANILLA FT-TRANSFORMER  (explicit baseline — no BPAC, no SGA)
# ─────────────────────────────────────────────────────────────────────────────

class VanillaFTTransformer(HIFTTransformer):
    """
    Plain Feature-Tokenizer Transformer without the two HI-FTT novel modules.
    Identical to HIFTTransformer with use_phase_cond=False, use_group_attn=False.
    Used as the explicit FT-Transformer baseline in all comparison tables.
    """
    def __init__(self, num_numerical: int, cat_dims: list):
        super().__init__(
            num_numerical  = num_numerical,
            cat_dims       = cat_dims,
            lead_time_idx  = None,
            feature_groups = None,
            group_names    = None,
            use_phase_cond = False,
            use_group_attn = False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str, num_numerical: int, cat_dims: list,
                feature_groups=None, lead_time_idx=None, group_names=None):
    """
    Returns an instantiated model.
    Deep models (Transformer / FTTransformer / LSTM / GRU) are nn.Module.
    Traditional models follow sklearn API.
    """
    name = name.lower()
    if name == "transformer":
        return HIFTTransformer(
            num_numerical  = num_numerical,
            cat_dims       = cat_dims,
            lead_time_idx  = lead_time_idx,
            feature_groups = feature_groups,
            group_names    = group_names,
        )
    elif name in ("fttransformer", "ft_transformer", "vanilla_ft"):
        return VanillaFTTransformer(num_numerical, cat_dims)
    elif name == "lstm":
        return LSTMChurn(num_numerical, cat_dims)
    elif name == "gru":
        return GRUChurn(num_numerical, cat_dims)
    elif name in ("mlp", "mlpchurn"):
        return MLPChurn(num_numerical, cat_dims)
    elif name in ("cnn", "cnn1d", "cnnchurn"):
        return CNNChurn(num_numerical, cat_dims)
    elif name in ("xgboost", "xgb"):
        return build_xgboost()
    elif name in ("lightgbm", "lgbm", "lgb"):
        return build_lightgbm()
    elif name in ("catboost", "cat"):
        return build_catboost()
    elif name in ("logisticregression", "lr", "logistic"):
        return build_logistic_regression()
    else:
        raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION VARIANTS FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_ablation_model(
    variant:       str,
    num_numerical: int,
    cat_dims:      list,
    feature_groups = None,
    lead_time_idx:  int  = None,
    group_names:    list = None,
):
    """
    Creates an HIFTTransformer variant for ablation study.

    Original 8 variants test standard architectural components.
    Two new variants test the novel domain-specific contributions:
      no_phase_cond  — removes BPAC (Booking-Phase Aware Conditioning)
      no_group_attn  — removes SGA  (Semantic Group Attention)
    """
    cfg = dict(
        num_numerical   = num_numerical,
        cat_dims        = cat_dims,
        d_model         = TRANSFORMER["d_model"],
        nhead           = TRANSFORMER["nhead"],
        num_layers      = TRANSFORMER["num_layers"],
        dim_feedforward = TRANSFORMER["dim_feedforward"],
        dropout         = TRANSFORMER["dropout"],
        attn_dropout    = TRANSFORMER["attn_dropout"],
        use_cls_token   = True,
        use_pos_enc     = True,
        # Novel components (enabled in full model)
        lead_time_idx   = lead_time_idx,
        feature_groups  = feature_groups,
        group_names     = group_names,
        use_phase_cond  = True,
        use_group_attn  = True,
    )

    if variant == "full_model":
        pass  # all components enabled

    elif variant == "no_cls_token":
        cfg["use_cls_token"] = False

    elif variant == "no_pos_encoding":
        cfg["use_pos_enc"] = False

    elif variant == "single_head":
        cfg["nhead"]          = 1
        cfg["dim_feedforward"]= 256

    elif variant == "no_feature_emb":
        return NoEmbeddingTransformer(num_numerical, cat_dims)

    elif variant == "shallow_1layer":
        cfg["num_layers"] = 1

    elif variant == "deep_6layer":
        cfg["num_layers"] = 6

    elif variant == "no_dropout":
        cfg["dropout"]     = 0.0
        cfg["attn_dropout"]= 0.0

    # ── Novel ablation variants ───────────────────────────────────────────────
    elif variant == "no_phase_cond":
        cfg["use_phase_cond"] = False     # remove BPAC

    elif variant == "no_group_attn":
        cfg["use_group_attn"] = False     # remove SGA

    else:
        raise ValueError(f"Unknown ablation variant: '{variant}'")

    return HIFTTransformer(**cfg)


class NoEmbeddingTransformer(nn.Module):
    """
    Ablation: No per-feature embedding.
    Concatenates all raw features and linearly projects to a single token,
    then feeds into the Transformer.
    """

    def __init__(self, num_numerical: int, cat_dims: list):
        super().__init__()
        num_cat     = len(cat_dims)
        total_feats = num_numerical + num_cat
        d_model     = TRANSFORMER["d_model"]

        self.proj   = nn.Linear(total_feats, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=TRANSFORMER["nhead"],
            dim_feedforward=TRANSFORMER["dim_feedforward"],
            dropout=TRANSFORMER["dropout"], activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=TRANSFORMER["num_layers"],
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(TRANSFORMER["dropout"]),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, X_num, X_cat):
        x = torch.cat([X_num, X_cat.float()], dim=-1)  # (B, total_feats)
        x = self.proj(x).unsqueeze(1)                  # (B, 1, d_model)
        x = self.transformer(x)
        rep = x[:, 0, :]
        return self.head(rep).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER COUNT UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
