"""
Convergence Analysis for deep learning models.

Tracks and reports:
  1. Training / validation loss curves
  2. Validation AUC progression
  3. Gradient norm trajectory (stability indicator)
  4. Learning-rate schedule visualization
  5. Effective epoch reached (early stopping point)
  6. Convergence speed metric: epochs to 95% of final AUC
"""

import os
import numpy as np
import pandas as pd

from src.config import FIG_DIR, TABLE_DIR, VIZ


# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE METRICS FROM TRAINING HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def compute_convergence_metrics(history: dict, model_name: str = "") -> dict:
    """
    Extract convergence metrics from a DeepTrainer history dict.

    Parameters
    ----------
    history : { "train_loss": [...], "val_loss": [...], "val_auc": [...], "lr": [...] }

    Returns
    -------
    dict with convergence diagnostics
    """
    train_loss = np.array(history["train_loss"])
    val_loss   = np.array(history["val_loss"])
    val_auc    = np.array(history["val_auc"])
    lr         = np.array(history["lr"])

    epochs     = len(train_loss)

    best_epoch_loss = int(np.argmin(val_loss)) + 1
    best_epoch_auc  = int(np.argmax(val_auc))  + 1
    best_val_loss   = float(val_loss.min())
    best_val_auc    = float(val_auc.max())

    # Overfitting gap (final train vs val loss)
    overfit_gap = float(val_loss[-1] - train_loss[-1])

    # Convergence speed: first epoch where AUC >= 95% of best AUC
    target_auc   = 0.95 * best_val_auc
    speed_epochs = epochs
    for i, auc in enumerate(val_auc):
        if auc >= target_auc:
            speed_epochs = i + 1
            break

    # Stability: std of val loss in last 10% of training
    tail = max(1, epochs // 10)
    stability = float(val_loss[-tail:].std())

    # Convergence slope (linear fit on val_loss); needs ≥2 points
    x = np.arange(epochs)
    if epochs >= 2:
        slope, _ = np.polyfit(x, val_loss, 1)
    else:
        slope = 0.0

    return {
        "model"             : model_name,
        "total_epochs"      : epochs,
        "best_epoch_auc"    : best_epoch_auc,
        "best_epoch_loss"   : best_epoch_loss,
        "best_val_auc"      : round(best_val_auc, 4),
        "best_val_loss"     : round(best_val_loss, 4),
        "final_train_loss"  : round(float(train_loss[-1]), 4),
        "final_val_loss"    : round(float(val_loss[-1]), 4),
        "overfitting_gap"   : round(overfit_gap, 4),
        "speed_epochs_95pct": speed_epochs,
        "tail_stability_std": round(stability, 5),
        "loss_slope"        : round(float(slope), 6),
        "initial_lr"        : round(float(lr[0]), 6),
        "final_lr"          : round(float(lr[-1]), 6),
    }


def summarize_convergence(histories: dict) -> pd.DataFrame:
    """
    histories: { model_name → history_dict }
    Returns a convergence summary DataFrame.
    """
    rows = [compute_convergence_metrics(h, m) for m, h in histories.items()]
    df   = pd.DataFrame(rows).set_index("model")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT NORM ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_gradient_norms(grad_norms: list, window: int = 50) -> dict:
    """
    Analyze gradient norm trajectory from a DeepTrainer.
    grad_norms: list of per-step gradient norms.

    Returns
    -------
    dict with statistics and smoothed trajectory.
    """
    norms = np.array(grad_norms)

    # Smooth with rolling mean
    if len(norms) >= window:
        smoothed = np.convolve(norms, np.ones(window)/window, mode="valid")
    else:
        smoothed = norms

    return {
        "mean"        : float(norms.mean()),
        "std"         : float(norms.std()),
        "max"         : float(norms.max()),
        "min"         : float(norms.min()),
        "pct95"       : float(np.percentile(norms, 95)),
        "exploded"    : bool((norms > 10.0).any()),   # gradient explosion flag
        "vanished"    : bool((norms < 1e-6).any()),   # gradient vanishing flag
        "smoothed"    : smoothed.tolist(),
        "raw"         : norms.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_lr_schedule(lr_history: list) -> dict:
    lr = np.array(lr_history)
    return {
        "initial_lr"  : float(lr[0]),
        "final_lr"    : float(lr[-1]),
        "min_lr"      : float(lr.min()),
        "max_lr"      : float(lr.max()),
        "decay_ratio" : float(lr[-1] / lr[0]) if lr[0] > 0 else 0,
        "schedule"    : lr.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CONVERGENCE TABLE
# ─────────────────────────────────────────────────────────────────────────────

def save_convergence_table(histories: dict, dataset_name: str = "hotel"):
    os.makedirs(TABLE_DIR, exist_ok=True)
    df   = summarize_convergence(histories)
    path = os.path.join(TABLE_DIR, f"convergence_{dataset_name}.csv")
    df.to_csv(path)
    print(f"\n  Convergence table saved → {path}")
    print(df.to_string())
    return df
