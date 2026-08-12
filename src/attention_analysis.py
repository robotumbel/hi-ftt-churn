"""
Attention Analysis Module — for Transformer interpretability.

Extracts and visualizes:
  1. Per-layer attention weight matrices
  2. Feature-level attention importance (averaged over heads & samples)
  3. CLS attention heatmap (what CLS attends to)
  4. Attention rollout (information flow across layers)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DEVICE, FIG_DIR, VIZ


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION HOOK
# ─────────────────────────────────────────────────────────────────────────────

class AttentionHook:
    """
    Register forward hooks on TransformerEncoderLayer to capture attention weights.
    Usage:
        hook = AttentionHook(model.transformer)
        with torch.no_grad():
            model(X_num, X_cat)
        attn_weights = hook.get_weights()   # list of (B, H, S, S)
        hook.remove()
    """

    def __init__(self, transformer_encoder: nn.TransformerEncoder):
        self.hooks   = []
        self.weights = []  # list of tensors (B, H, S, S)

        for layer in transformer_encoder.layers:
            h = layer.self_attn.register_forward_hook(self._hook_fn)
            self.hooks.append(h)

    def _hook_fn(self, module, input, output):
        # output = (attn_output, attn_weights) when need_weights=True
        # By default TransformerEncoderLayer calls with need_weights=False
        # We need to re-call with need_weights=True
        pass

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.weights.clear()

    def get_weights(self):
        return self.weights


def extract_attention_weights(
    model,
    loader,
    n_samples: int = 200,
    feature_names: List[str] = None,
):
    """
    Extract attention weights from TabTransformer using a custom forward pass
    that captures multi-head attention scores.

    Returns:
        attn_weights: np.ndarray (n_samples, n_layers, n_heads, seq_len, seq_len)
        feature_names: list of token names
    """
    model.eval()
    all_weights = []
    count       = 0

    with torch.no_grad():
        for X_num, X_cat, y in loader:
            if count >= n_samples:
                break
            B = X_num.size(0)
            X_num = X_num.to(DEVICE)
            X_cat = X_cat.to(DEVICE)

            # Tokenize
            x = model.tokenizer(X_num, X_cat)   # (B, N, d)

            # Prepend CLS
            if model.cls_token is not None:
                cls = model.cls_token.expand(B, -1, -1)
                x   = torch.cat([cls, x], dim=1)

            # Add positional encoding
            if model.pos_enc is not None:
                x = x + model.pos_enc

            # Extract attention weights layer by layer
            batch_layer_attn = []
            for layer in model.transformer.layers:
                # self_attn.forward with need_weights=True
                attn_out, attn_w = layer.self_attn(
                    x, x, x, need_weights=True, average_attn_weights=False
                )   # attn_w: (B, H, S, S)
                batch_layer_attn.append(attn_w.cpu().numpy())

                # Still apply the full layer transform for next layer's input
                # (approximate — uses stored output from the layer forward hook)
                x = layer(x)

            # batch_layer_attn: (n_layers, B, H, S, S) → (B, n_layers, H, S, S)
            batch_attn = np.stack(batch_layer_attn, axis=1)  # (B, L, H, S, S)
            all_weights.append(batch_attn)
            count += B

    if not all_weights:
        return None

    attn = np.concatenate(all_weights, axis=0)[:n_samples]  # (N, L, H, S, S)
    return attn


def attention_to_feature_importance(
    attn_weights: np.ndarray,
    use_cls_only: bool = True,
    rollout: bool = False,
) -> np.ndarray:
    """
    Compute feature importance from attention weights.

    attn_weights: (N, L, H, S, S) where S = n_features + 1 (CLS token first)

    Returns:
        importance: (S-1,) = feature importance scores (excluding CLS)
    """
    N, L, H, S, _ = attn_weights.shape

    if rollout:
        # Attention rollout: multiply attention across layers
        eye = np.eye(S)
        roll = eye[np.newaxis, ...]   # (1, S, S)
        for layer_idx in range(L):
            # Average over heads
            attn_l = attn_weights[:, layer_idx, :, :, :].mean(axis=1)  # (N, S, S)
            attn_l = 0.5 * attn_l + 0.5 * eye   # residual
            roll   = np.matmul(roll, attn_l)

        # CLS row: how much CLS attends to each feature
        importance = roll[:, 0, 1:].mean(axis=0)   # (S-1,)

    elif use_cls_only:
        # Take last layer, average over heads, CLS → features attention
        last_layer = attn_weights[:, -1, :, :, :].mean(axis=1)  # (N, S, S)
        cls_attn   = last_layer[:, 0, 1:]                        # (N, S-1)
        importance = cls_attn.mean(axis=0)

    else:
        # All tokens attention: average over all positions and heads
        all_attn   = attn_weights.mean(axis=(0, 2, 3))   # (S,)
        importance = all_attn[1:]                          # remove CLS

    # Normalize
    importance = importance / (importance.sum() + 1e-9)
    return importance


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    feature_names: List[str],
    n_samples_avg: int = 50,
    dataset_name: str = "hotel",
):
    """
    Plot CLS attention to features across all Transformer layers.
    attn_weights: (N, L, H, S, S)
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    N, L, H, S, _ = attn_weights.shape
    n = min(n_samples_avg, N)

    # Average CLS attention per layer over samples and heads
    # Shape: (L, S-1)
    cls_attn = attn_weights[:n, :, :, 0, 1:].mean(axis=(0, 2))  # (L, n_feat)

    feat_labels = feature_names if feature_names else [f"F{i}" for i in range(S-1)]
    layer_labels = [f"Layer {i+1}" for i in range(L)]

    fig, ax = plt.subplots(figsize=(max(10, len(feat_labels)*0.45), L + 1))
    sns.heatmap(
        cls_attn,
        ax        = ax,
        annot     = (len(feat_labels) <= 20),
        fmt       = ".3f",
        cmap      = "Blues",
        xticklabels = feat_labels,
        yticklabels = layer_labels,
        linewidths  = 0.3,
        cbar_kws    = {"label": "Attention Weight (CLS → Feature)"},
    )
    ax.set_title(f"Transformer CLS Attention — {dataset_name.title()} Dataset",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=10)
    fig.tight_layout()

    path = os.path.join(FIG_DIR, f"attention_heatmap_{dataset_name}.{VIZ['fig_format']}")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {path}")


def plot_feature_attention_importance(
    importance: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    dataset_name: str = "hotel",
):
    """Bar chart of feature importance derived from CLS attention."""
    os.makedirs(FIG_DIR, exist_ok=True)
    n = len(importance)
    # Guard: trim to whichever is shorter to prevent index out of range
    if feature_names:
        feat_names = list(feature_names)[:n]
        if len(feat_names) < n:
            feat_names += [f"F{i}" for i in range(len(feat_names), n)]
    else:
        feat_names = [f"F{i}" for i in range(n)]

    # Sort by importance
    idx  = np.argsort(importance)[::-1][:top_n]
    vals = importance[idx]
    labs = [feat_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.4)))
    colors  = ["#1f77b4" if i == 0 else "#aec7e8" for i in range(len(vals))]
    ax.barh(labs[::-1], vals[::-1], color=colors[::-1], edgecolor="black", lw=0.4)
    ax.set_xlabel("Normalized Attention Importance", fontsize=11)
    ax.set_title(f"Feature Importance via Attention — {dataset_name.title()}",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()

    path = os.path.join(FIG_DIR, f"attention_importance_{dataset_name}.{VIZ['fig_format']}")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {path}")


def plot_attention_rollout(
    attn_weights: np.ndarray,
    feature_names: List[str],
    dataset_name: str = "hotel",
):
    """Visualize attention rollout (information flow across layers)."""
    importance_rollout = attention_to_feature_importance(attn_weights, rollout=True)
    importance_cls     = attention_to_feature_importance(attn_weights, use_cls_only=True)

    os.makedirs(FIG_DIR, exist_ok=True)
    feat    = feature_names if feature_names else [f"F{i}" for i in range(len(importance_rollout))]
    n_feat  = len(feat)
    # Clip importance arrays to the number of named features (excludes CLS token)
    importance_rollout = importance_rollout[:n_feat]
    importance_cls     = importance_cls[:n_feat]
    idx     = np.argsort(importance_rollout)[::-1][:20]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, imp, title in zip(axes,
                               [importance_rollout[idx], importance_cls[idx]],
                               ["Attention Rollout", "Last-Layer CLS"]):
        ax.barh([feat[i] for i in idx][::-1], imp[::-1],
                color="#1f77b4", edgecolor="black", lw=0.4)
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"Attention-based Feature Importance — {dataset_name.title()}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIG_DIR, f"attention_rollout_{dataset_name}.{VIZ['fig_format']}")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# HEAD-LEVEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_attention_heads(
    attn_weights: np.ndarray,
    feature_names: List[str],
    dataset_name: str = "hotel",
):
    """
    Analyze each attention head's specialization.
    Returns DataFrame with per-head entropy and top attended features.
    """
    N, L, H, S, _ = attn_weights.shape
    rows = []

    n_feat = len(feature_names)
    for l in range(L):
        for h in range(H):
            # CLS attention to features (averaged over samples)
            cls_head = attn_weights[:, l, h, 0, 1:].mean(axis=0)  # (S-1,)
            cls_head = cls_head[:n_feat]  # clip to valid feature names length
            # Entropy (lower = more focused)
            p = cls_head / (cls_head.sum() + 1e-9)
            entropy = -np.sum(p * np.log2(p + 1e-9))
            # Top 3 features
            top_idx = np.argsort(p)[::-1][:3]
            top_feats = [feature_names[i] if feature_names else f"F{i}" for i in top_idx]

            rows.append({
                "Layer"   : l + 1,
                "Head"    : h + 1,
                "Entropy" : round(float(entropy), 3),
                "MaxAttn" : round(float(p.max()), 4),
                "Top1"    : top_feats[0],
                "Top2"    : top_feats[1] if len(top_feats) > 1 else "",
                "Top3"    : top_feats[2] if len(top_feats) > 2 else "",
            })

    return pd.DataFrame(rows)
