"""
Ablation Study for TabTransformer.

Systematically removes/modifies one component at a time to measure
each component's contribution to the final performance.

Variants tested:
  1. full_model       — all components (proposed)
  2. no_cls_token     — mean-pooling instead of CLS token
  3. no_pos_encoding  — no positional encoding
  4. single_head      — nhead=1 (single-head attention)
  5. no_feature_emb   — flat linear projection (no per-feature tokenization)
  6. shallow_1layer   — only 1 Transformer layer
  7. deep_6layer      — 6 Transformer layers
  8. no_dropout       — no regularization (dropout=0)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config         import ABLATION_VARIANTS, ABLATION_EPOCHS, TRAIN, DEVICE, SEED, TABLE_DIR
from src.models         import build_ablation_model, count_parameters
from src.training       import DeepTrainer
from src.evaluation     import compute_metrics
from src.data_preprocessing import ChurnDataset


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE ABLATION RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_variant(
    variant:       str,
    num_numerical: int,
    cat_dims:      list,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    test_loader:   DataLoader,
    pos_weight:    torch.Tensor,
    epochs:        int  = None,
    verbose:       bool = True,
    feature_groups: list = None,
    lead_time_idx:  int  = None,
    group_names:    list = None,
) -> dict:
    """
    Train one ablation variant and return test metrics + model info.
    feature_groups / lead_time_idx / group_names are forwarded to
    build_ablation_model so that the novel HI-FTT components are
    correctly toggled per variant.
    """
    torch.manual_seed(SEED)
    model = build_ablation_model(
        variant, num_numerical, cat_dims,
        feature_groups=feature_groups,
        lead_time_idx=lead_time_idx,
        group_names=group_names,
    ).to(DEVICE)
    n_param = count_parameters(model)

    if verbose:
        print(f"\n  [Ablation] variant='{variant}'  |  params={n_param:,}")

    trainer = DeepTrainer(model, pos_weight=pos_weight, name=f"ablation_{variant}")
    history = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=verbose)

    probs, labels = trainer.predict_proba(test_loader)
    metrics       = compute_metrics(labels, probs)
    metrics["n_params"]    = n_param
    metrics["best_val_loss"]= trainer.early_stop.best
    metrics["history"]     = history

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# FULL ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study(
    num_numerical:  int,
    cat_dims:       list,
    train_loader:   DataLoader,
    val_loader:     DataLoader,
    test_loader:    DataLoader,
    pos_weight:     torch.Tensor,
    epochs:         int  = None,
    variants:       list = None,
    verbose:        bool = True,
    dataset_name:   str  = "hotel",
    feature_groups: list = None,
    lead_time_idx:  int  = None,
    group_names:    list = None,
) -> dict:
    """
    Run all ablation variants and return a dict of results.
    Each variant is saved immediately after completion so partial
    results are not lost if a later variant crashes.
    """
    import json, traceback
    variants = variants or ABLATION_VARIANTS
    epochs   = epochs or ABLATION_EPOCHS   # use shorter budget for ablation

    os.makedirs(TABLE_DIR, exist_ok=True)
    partial_path = os.path.join(TABLE_DIR, f"ablation_{dataset_name}_partial.json")

    all_results = {}
    for variant in variants:
        try:
            result = run_ablation_variant(
                variant, num_numerical, cat_dims,
                train_loader, val_loader, test_loader,
                pos_weight, epochs=epochs, verbose=verbose,
                feature_groups=feature_groups,
                lead_time_idx=lead_time_idx,
                group_names=group_names,
            )
            all_results[variant] = result

            # Save partial results immediately (exclude non-serialisable history)
            partial_save = {
                v: {k: val for k, val in r.items() if k != "history"}
                for v, r in all_results.items()
            }
            with open(partial_path, "w") as f:
                json.dump(partial_save, f, indent=2, default=float)
            if verbose:
                print(f"  [Ablation] '{variant}' saved (partial: {partial_path})")

        except Exception as e:
            print(f"\n  [Ablation] WARNING: variant '{variant}' failed — {e}")
            traceback.print_exc()
            all_results[variant] = {"error": str(e), "f1_macro": float("nan"),
                                    "roc_auc": float("nan"), "n_params": 0}
            continue   # skip to next variant

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_LABELS = {
    "full_model"      : "Full HI-FTT (proposed)",
    "no_cls_token"    : "w/o CLS Token (mean pool)",
    "no_pos_encoding" : "w/o Positional Encoding",
    "single_head"     : "w/o Multi-head (single head)",
    "no_feature_emb"  : "w/o Feature Tokenization",
    "shallow_1layer"  : "Shallow (1 Layer)",
    "deep_6layer"     : "Deep (6 Layers)",
    "no_dropout"      : "w/o Dropout",
    # Novel component ablations
    "no_phase_cond"   : "w/o Booking-Phase Conditioning (BPAC)",
    "no_group_attn"   : "w/o Semantic Group Attention (SGA)",
}

REPORT_METRICS = [
    "accuracy", "f1_macro", "roc_auc", "pr_auc", "mcc", "kappa",
]

def ablation_to_dataframe(ablation_results: dict) -> pd.DataFrame:
    """
    Convert ablation results dict to a publication-ready DataFrame.
    Computes Δ relative to the full model.
    """
    rows = []
    full = ablation_results.get("full_model", {})
    full_f1 = full.get("f1_macro", np.nan)

    for variant, res in ablation_results.items():
        row = {
            "Variant"   : VARIANT_LABELS.get(variant, variant),
            "#Params"   : f"{res.get('n_params', 0):,}",
        }
        for m in REPORT_METRICS:
            val = res.get(m, np.nan)
            row[m.upper()] = f"{val:.4f}"
        # Delta F1 vs full model
        delta = res.get("f1_macro", np.nan) - full_f1
        row["ΔF1 vs Full"] = f"{delta:+.4f}" if not np.isnan(delta) else "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    # Put full model first
    full_label = VARIANT_LABELS.get("full_model", "Full Model (proposed)")
    df = pd.concat([
        df[df["Variant"] == full_label],
        df[df["Variant"] != full_label],
    ]).reset_index(drop=True)

    return df


def save_ablation_table(ablation_results: dict, dataset_name: str = "hotel"):
    """Save ablation table to CSV and print to console."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    df = ablation_to_dataframe(ablation_results)
    path = os.path.join(TABLE_DIR, f"ablation_{dataset_name}.csv")
    df.to_csv(path, index=False)
    print(f"\n  Ablation table saved → {path}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT IMPORTANCE (attention-based + gradient-based)
# ─────────────────────────────────────────────────────────────────────────────

def gradient_feature_importance(
    model: torch.nn.Module,
    test_loader: DataLoader,
    num_feature_names: list,
    cat_feature_names: list,
    n_samples: int = 1000,
) -> pd.DataFrame:
    """
    Compute gradient-based feature importance (Input × Gradient).
    Returns DataFrame sorted by importance score.
    """
    model.eval()
    importances = []
    count = 0

    for X_num, X_cat, y in test_loader:
        if count >= n_samples:
            break
        X_num = X_num.to(DEVICE).requires_grad_(True)
        X_cat = X_cat.to(DEVICE)

        logits = model(X_num, X_cat)
        loss   = logits.mean()
        loss.backward()

        # Gradient × input (for numerical features)
        grad_imp = (X_num.grad.abs() * X_num.abs()).mean(dim=0).detach().cpu().numpy()
        importances.append(grad_imp)
        count += len(y)

    imp_matrix = np.array(importances).mean(axis=0)
    imp_matrix = imp_matrix / (imp_matrix.sum() + 1e-9)   # normalize

    df = pd.DataFrame({
        "Feature"   : num_feature_names,
        "Importance": imp_matrix,
    }).sort_values("Importance", ascending=False)

    return df
