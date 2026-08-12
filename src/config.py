"""
Configuration file for:
  A Transformer-based Approach for Churn Prediction
  in Hotel Customer Relationship Management

All hyperparameters, paths, and experiment settings.
"""

import os
import torch
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = BASE_DIR

# Each run gets its own timestamped folder so results are never overwritten
RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join(BASE_DIR, "results", "runs", RUN_ID)
FIG_DIR    = os.path.join(RESULT_DIR, "figures")
TABLE_DIR  = os.path.join(RESULT_DIR, "tables")
CKPT_DIR   = os.path.join(RESULT_DIR, "checkpoints")

HOTEL_CSV     = os.path.join(DATA_DIR, "hotel_bookings.csv")
TELCO_CSV     = os.path.join(DATA_DIR, "TelcoCustomerChurn.csv")
BANK_CSV      = os.path.join(DATA_DIR, "Bank Customer Churn Prediction.csv")
DATATHON_DIR  = os.path.join(DATA_DIR, "datathon 2019")

# ─────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
SEED = 42

# ─────────────────────────────────────────────────────────────
# DATA SPLITS
# ─────────────────────────────────────────────────────────────
TEST_SIZE  = 0.15   # hold-out test set
VAL_SIZE   = 0.15   # validation (from training set)
CV_FOLDS   = 5      # stratified k-fold for statistical validation

# ─────────────────────────────────────────────────────────────
# PROPOSED TRANSFORMER HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
TRANSFORMER = dict(
    d_model        = 128,
    nhead          = 8,
    num_layers     = 4,
    dim_feedforward= 512,    # d_model * 4
    dropout        = 0.1,
    attn_dropout   = 0.1,
)

# ─────────────────────────────────────────────────────────────
# DEEP LEARNING TRAINING SETTINGS
# ─────────────────────────────────────────────────────────────
TRAIN = dict(
    batch_size     = 512,
    epochs         = 150,
    lr             = 1e-3,
    weight_decay   = 1e-4,
    patience       = 20,          # early stopping patience
    lr_scheduler   = "cosine",    # "cosine" | "step" | "reduce"
    warmup_ratio   = 0.1,         # fraction of epochs used for linear LR warmup
    label_smoothing= 0.05,        # label smoothing for BCE loss
    grad_clip      = 1.0,
    pos_weight     = None,        # auto-computed from class imbalance
)

# ─────────────────────────────────────────────────────────────
# LSTM / GRU HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
LSTM_CFG = dict(
    hidden_size    = 64,
    num_layers     = 2,
    dropout        = 0.3,
    bidirectional  = True,
)

GRU_CFG = dict(
    hidden_size    = 64,
    num_layers     = 2,
    dropout        = 0.3,
    bidirectional  = True,
)

# ─────────────────────────────────────────────────────────────
# TRADITIONAL ML HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
XGBOOST_CFG = dict(
    n_estimators   = 500,
    max_depth      = 6,
    learning_rate  = 0.05,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    use_label_encoder = False,
    eval_metric    = "logloss",
    tree_method    = "hist",
    device         = "cuda" if torch.cuda.is_available() else "cpu",
    random_state   = SEED,
)

LR_CFG = dict(
    C              = 1.0,
    max_iter       = 1000,
    class_weight   = "balanced",
    solver         = "saga",
    n_jobs         = -1,
    random_state   = SEED,
)

# Gradient-boosting baselines requested by reviewers (matched budget to XGBoost).
LIGHTGBM_CFG = dict(
    n_estimators   = 500,
    max_depth      = 6,
    learning_rate  = 0.05,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    num_leaves     = 63,
    class_weight   = "balanced",
    n_jobs         = -1,
    random_state   = SEED,
    verbose        = -1,
)

CATBOOST_CFG = dict(
    iterations     = 500,
    depth          = 6,
    learning_rate  = 0.05,
    l2_leaf_reg    = 3.0,
    loss_function  = "Logloss",
    eval_metric    = "AUC",
    task_type      = "GPU" if torch.cuda.is_available() else "CPU",
    random_state   = SEED,
    verbose        = False,
    allow_writing_files = False,
)

# ─────────────────────────────────────────────────────────────
# ABLATION STUDY VARIANTS
# ─────────────────────────────────────────────────────────────
ABLATION_EPOCHS = 50      # shorter epoch budget for ablation (relative comparison only)

ABLATION_VARIANTS = [
    "full_model",          # proposed HI-FTT (all components)
    "no_cls_token",        # mean-pooling instead of CLS
    "no_pos_encoding",     # remove positional encoding
    "single_head",         # nhead=1 (no multi-head)
    "no_feature_emb",      # raw linear projection (no per-feature embed)
    "shallow_1layer",      # num_layers=1
    "deep_6layer",         # num_layers=6 (over-parameterized)
    "no_dropout",          # dropout=0
    # ── Novel component ablations ──────────────────────────────────────────
    "no_phase_cond",       # remove Booking-Phase Aware Conditioning (BPAC)
    "no_group_attn",       # remove Semantic Group Attention (SGA)
]

# ─────────────────────────────────────────────────────────────
# STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────
STAT = dict(
    alpha          = 0.05,          # significance level
    correction     = "bonferroni",  # multiple comparison correction
)

# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────
VIZ = dict(
    dpi            = 300,
    fig_format     = "png",          # "png" for image output
    style          = "seaborn-v0_8-whitegrid",
    palette        = "colorblind",
)

# ─────────────────────────────────────────────────────────────
# MODEL REGISTRY (used by trainer and evaluator)
# ─────────────────────────────────────────────────────────────
import importlib.util as _ilu
def _have(pkg): return _ilu.find_spec(pkg) is not None

ALL_MODELS = [
    "Transformer",        # proposed HI-FTT
    "FTTransformer",      # vanilla FT-Transformer (no BPAC/SGA)
    "LSTM",               # bidirectional LSTM
    "GRU",                # bidirectional GRU
    "MLP",                # deep feedforward (no attention/recurrence)
    "CNN",                # 1-D convolutional over feature tokens
    "XGBoost",            # gradient boosting
]
# Reviewer-requested GBM baselines — added only when the library is installed.
if _have("lightgbm"):
    ALL_MODELS.append("LightGBM")
if _have("catboost"):
    ALL_MODELS.append("CatBoost")
ALL_MODELS.append("LogisticRegression")   # linear baseline last

DEEP_MODELS = ["Transformer", "FTTransformer", "LSTM", "GRU", "MLP", "CNN"]
TRAD_MODELS = ["XGBoost", "LogisticRegression"] + \
    (["LightGBM"] if _have("lightgbm") else []) + \
    (["CatBoost"] if _have("catboost") else [])
