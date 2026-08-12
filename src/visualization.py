"""
Publication-quality visualizations for Q1/Q2 journal submission.

Figures generated:
  1.  ROC curve comparison (all models)
  2.  Precision-Recall curve comparison
  3.  Training / validation loss curves
  4.  Convergence comparison (val-AUC over epochs)
  5.  Gradient norm trajectory
  6.  Ablation study bar chart
  7.  McNemar heatmap
  8.  CV box plot (per-metric, all models)
  9.  Feature importance bar chart
  10. Confusion matrices (proposed vs best baseline)
  11. Learning rate schedule
  12. Generalization comparison (hotel vs telco)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
)

from src.config import FIG_DIR, VIZ


# ─── global style ──────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette(VIZ["palette"])
FONT = {"family": "serif", "size": 11}
plt.rc("font", **FONT)

MODEL_COLORS = {
    "Transformer"       : "#1f77b4",
    "LSTM"              : "#ff7f0e",
    "GRU"               : "#2ca02c",
    "XGBoost"           : "#d62728",
    "RandomForest"      : "#9467bd",
    "LightGBM"          : "#8c564b",
    "LogisticRegression": "#e377c2",
}

os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name: str, tight: bool = True):
    if tight:
        fig.tight_layout()
    path = os.path.join(FIG_DIR, f"{name}.{VIZ['fig_format']}")
    fig.savefig(path, dpi=VIZ["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(probs_dict: dict, labels: np.ndarray, dataset_name: str = "hotel"):
    """
    probs_dict: { model_name → prob_array }
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))

    for model, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc     = auc(fpr, tpr)
        color       = MODEL_COLORS.get(model, "gray")
        lw          = 2.5 if model == "Transformer" else 1.5
        ls          = "-"  if model == "Transformer" else "--"
        ax.plot(fpr, tpr, lw=lw, ls=ls, color=color,
                label=f"{model} (AUC={roc_auc:.4f})")

    ax.plot([0,1],[0,1], "k:", lw=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curves — {dataset_name.title()} Dataset", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    _save(fig, f"roc_curves_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRECISION-RECALL CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(probs_dict: dict, labels: np.ndarray, dataset_name: str = "hotel"):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    baseline_pr = labels.mean()

    for model, probs in probs_dict.items():
        prec, rec, _ = precision_recall_curve(labels, probs)
        pr_auc       = auc(rec, prec)
        color        = MODEL_COLORS.get(model, "gray")
        lw           = 2.5 if model == "Transformer" else 1.5
        ls           = "-"  if model == "Transformer" else "--"
        ax.plot(rec, prec, lw=lw, ls=ls, color=color,
                label=f"{model} (AP={pr_auc:.4f})")

    ax.axhline(baseline_pr, color="k", ls=":", lw=1, label=f"Baseline (P={baseline_pr:.3f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves — {dataset_name.title()} Dataset",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    _save(fig, f"pr_curves_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING / VALIDATION LOSS CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(histories: dict, dataset_name: str = "hotel"):
    """histories: { model_name → { train_loss, val_loss, val_auc } }"""
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, h) in zip(axes, histories.items()):
        epochs = range(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], label="Train Loss",  lw=2)
        ax.plot(epochs, h["val_loss"],   label="Val Loss",    lw=2, ls="--")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=9)

    fig.suptitle(f"Training & Validation Loss — {dataset_name.title()}",
                 fontsize=13, fontweight="bold", y=1.01)
    _save(fig, f"loss_curves_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONVERGENCE COMPARISON (val AUC over epochs)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(histories: dict, dataset_name: str = "hotel"):
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, h in histories.items():
        aucs   = h["val_auc"]
        epochs = range(1, len(aucs) + 1)
        color  = MODEL_COLORS.get(name, "gray")
        lw     = 2.5 if name == "Transformer" else 1.5
        ls     = "-"  if name == "Transformer" else "--"
        ax.plot(epochs, aucs, lw=lw, ls=ls, color=color, label=name)

    ax.set_xlabel("Epoch",        fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title(f"Convergence Analysis — {dataset_name.title()}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim([max(0.5, min(h["val_auc"][0] for h in histories.values()) - 0.05), 1.0])

    _save(fig, f"convergence_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. GRADIENT NORM TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradient_norms(grad_analysis: dict, dataset_name: str = "hotel"):
    """
    grad_analysis: { model_name → gradient_analysis_dict (from convergence_analysis) }
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for name, ga in grad_analysis.items():
        smoothed = np.array(ga["smoothed"])
        steps    = range(len(smoothed))
        color    = MODEL_COLORS.get(name, "gray")
        lw       = 2.5 if name == "Transformer" else 1.5
        ax.plot(steps, smoothed, lw=lw, color=color, label=name, alpha=0.85)

    ax.set_xlabel("Training Steps (smoothed)",  fontsize=12)
    ax.set_ylabel("Gradient L2 Norm",           fontsize=12)
    ax.set_title(f"Gradient Norm Trajectory — {dataset_name.title()}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    _save(fig, f"gradient_norms_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. ABLATION STUDY BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_LABELS = {
    "full_model"      : "Full Model\n(proposed)",
    "no_cls_token"    : "w/o CLS\nToken",
    "no_pos_encoding" : "w/o Pos.\nEncoding",
    "single_head"     : "Single\nHead",
    "no_feature_emb"  : "w/o Feature\nTokenization",
    "shallow_1layer"  : "1 Layer\n(Shallow)",
    "deep_6layer"     : "6 Layers\n(Deep)",
    "no_dropout"      : "w/o\nDropout",
}

def plot_ablation(ablation_results: dict, metric: str = "f1_macro",
                  dataset_name: str = "hotel"):
    variants = list(ablation_results.keys())
    values   = [ablation_results[v].get(metric, 0) for v in variants]
    labels   = [ABLATION_LABELS.get(v, v) for v in variants]
    colors   = ["#1f77b4" if v == "full_model" else "#aec7e8" for v in variants]

    fig, ax  = plt.subplots(figsize=(10, 4.5))
    bars     = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8.5
        )

    full_val = ablation_results.get("full_model", {}).get(metric, 0)
    ax.axhline(full_val, color="red", ls="--", lw=1.2, label="Full model baseline")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Ablation Study ({metric.upper()}) — {dataset_name.title()}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim([max(0, min(values) - 0.05), min(1, max(values) + 0.05)])

    _save(fig, f"ablation_{metric}_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. McNEMAR HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_mcnemar_heatmap(p_matrix: pd.DataFrame, dataset_name: str = "hotel"):
    fig, ax = plt.subplots(figsize=(7, 6))
    mask    = p_matrix.values == 1.0   # diagonal

    # pandas ≥2.1 renamed DataFrame.applymap → DataFrame.map
    _elementwise = p_matrix.map if hasattr(p_matrix, "map") else p_matrix.applymap
    annot = _elementwise(
        lambda x: f"{x:.3f}" if x != 1.0 else "—"
    ).values

    sns.heatmap(
        p_matrix.values.astype(float), ax=ax,
        annot=annot, fmt="", cmap="RdYlGn_r",
        vmin=0, vmax=0.1,
        xticklabels=p_matrix.columns,
        yticklabels=p_matrix.index,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "p-value (McNemar's test)"},
    )
    ax.set_title(f"McNemar's Test p-values — {dataset_name.title()}",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)

    _save(fig, f"mcnemar_heatmap_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CROSS-VALIDATION BOX PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_boxplot(cv_results: dict, metric: str = "f1_macro",
                    dataset_name: str = "hotel"):
    """
    cv_results: { model_name → { metric → np.array(k,) } }
    """
    model_names = list(cv_results.keys())
    data        = [cv_results[m][metric] for m in model_names]
    colors_list = [MODEL_COLORS.get(m, "gray") for m in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        data, patch_artist=True, notch=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(metric.replace("_"," ").title(), fontsize=12)
    ax.set_title(f"5-Fold CV Distribution ({metric.upper()}) — {dataset_name.title()}",
                 fontsize=13, fontweight="bold")

    _save(fig, f"cv_boxplot_{metric}_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                             dataset_name: str = "hotel"):
    df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(7, top_n * 0.45 + 1.5))
    ax.barh(df["Feature"][::-1], df["Importance"][::-1],
            color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Normalized Gradient × Input Importance", fontsize=11)
    ax.set_title(f"Top-{top_n} Feature Importance — {dataset_name.title()}",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    _save(fig, f"feature_importance_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(preds_dict: dict, labels: np.ndarray,
                             dataset_name: str = "hotel"):
    """
    preds_dict: { model_name → binary prediction array }
    """
    n = len(preds_dict)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, preds_dict.items()):
        cm     = confusion_matrix(labels, preds)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        annot  = np.array([[f"{v}\n({p:.1%})" for v, p in zip(row, prow)]
                           for row, prow in zip(cm, cm_pct)])

        sns.heatmap(cm, ax=ax, annot=annot, fmt="", cmap="Blues",
                    xticklabels=["No Churn","Churn"],
                    yticklabels=["No Churn","Churn"],
                    linewidths=0.5, linecolor="white",
                    cbar=False)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual",    fontsize=10)

    fig.suptitle(f"Confusion Matrices — {dataset_name.title()}",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, f"confusion_matrices_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. LEARNING RATE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def plot_lr_schedule(histories: dict, dataset_name: str = "hotel"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        lr     = h["lr"]
        epochs = range(1, len(lr) + 1)
        ax.plot(epochs, lr, lw=2, label=name,
                color=MODEL_COLORS.get(name, "gray"))
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(f"Learning Rate Schedule — {dataset_name.title()}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    _save(fig, f"lr_schedule_{dataset_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. GENERALIZATION COMPARISON (hotel vs telco)
# ─────────────────────────────────────────────────────────────────────────────

def plot_generalization(
    hotel_results:  dict,
    telco_results:  dict,
    metric:         str = "f1_macro",
):
    """
    hotel_results / telco_results: { model_name → metric_value }
    """
    models = list(hotel_results.keys())
    hotel_vals = [hotel_results[m] for m in models]
    telco_vals = [telco_results[m] for m in models]

    x    = np.arange(len(models))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, hotel_vals, w, label="Hotel Booking",  color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + w/2, telco_vals, w, label="IBM Telco Churn",color="#ff7f0e", alpha=0.85)

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=7.5, rotation=90
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Generalization Study ({metric.upper()})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim([max(0, min(hotel_vals+telco_vals) - 0.05), 1.02])

    _save(fig, f"generalization_{metric}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. RADAR / SPIDER CHART (all metrics for key models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar_chart(results: dict, dataset_name: str = "hotel"):
    """
    results: { model_name → { metric → float } }
    Plot multi-metric spider chart for key models.
    """
    metrics    = ["accuracy","f1_macro","roc_auc","pr_auc","mcc","g_mean"]
    metric_labels = ["Accuracy","F1","AUC","PR-AUC","MCC","G-Mean"]
    n_metrics  = len(metrics)
    angles     = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for model, res in results.items():
        values  = [res.get(m, 0) for m in metrics]
        values += values[:1]
        color   = MODEL_COLORS.get(model, "gray")
        lw      = 2.5 if model == "Transformer" else 1.5
        ax.plot(angles, values, color=color, lw=lw, label=model)
        ax.fill(angles, values, color=color, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Multi-Metric Radar — {dataset_name.title()}",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    _save(fig, f"radar_chart_{dataset_name}")
