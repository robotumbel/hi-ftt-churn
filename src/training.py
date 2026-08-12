"""
Training engine for deep learning models (Transformer, LSTM, GRU).
Includes:
  - Training loop with gradient clipping, LR scheduling
  - Early stopping
  - Sklearn-compatible training for traditional models
  - Cross-validation loop for statistical validation
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim               import AdamW
from torch.optim.lr_scheduler  import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp                 import autocast, GradScaler

from src.config import TRAIN, CKPT_DIR, DEVICE, SEED

_AMP_DTYPE = torch.float16    # change to torch.bfloat16 if preferred
_USE_AMP   = DEVICE.type == "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# LABEL SMOOTHING LOSS
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingBCE(nn.Module):
    """
    BCEWithLogitsLoss with label smoothing.
    Smoothed target: y_s = y * (1 - eps) + 0.5 * eps
    Prevents overconfident predictions and improves calibration.
    """
    def __init__(self, smoothing: float = 0.05, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits, targets):
        targets_s = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets_s).mean()


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 1e-4, mode: str = "min"):
        self.patience    = patience
        self.delta       = delta
        self.mode        = mode
        self.best        = np.inf if mode == "min" else -np.inf
        self.counter     = 0
        self.best_state  = None
        self.stopped     = False

    def __call__(self, val_score, model):
        improved = (
            val_score < self.best - self.delta if self.mode == "min"
            else val_score > self.best + self.delta
        )
        if improved:
            self.best       = val_score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────────────────────────────────────
# DEEP MODEL TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class DeepTrainer:
    """
    Trains any nn.Module model that accepts (X_num, X_cat) and returns logits.
    """

    def __init__(
        self,
        model: nn.Module,
        pos_weight: torch.Tensor = None,
        name: str = "model",
    ):
        self.model      = model.to(DEVICE)
        self.name       = name
        self.history    = {"train_loss":[], "val_loss":[], "val_auc":[], "lr":[]}
        self.grad_norms = []     # for convergence analysis

        pw = pos_weight.to(DEVICE) if pos_weight is not None else None
        smoothing = TRAIN.get("label_smoothing", 0.0)
        if smoothing > 0:
            self.criterion = LabelSmoothingBCE(smoothing=smoothing, pos_weight=pw)
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        self.optimizer = AdamW(
            model.parameters(),
            lr           = TRAIN["lr"],
            weight_decay = TRAIN["weight_decay"],
        )

        # LR schedule: linear warmup → cosine annealing
        total_epochs  = TRAIN["epochs"]
        warmup_epochs = max(1, int(total_epochs * TRAIN.get("warmup_ratio", 0.0)))
        cosine_epochs = max(1, total_epochs - warmup_epochs)

        warmup_sched = LinearLR(
            self.optimizer,
            start_factor = 0.01,
            end_factor   = 1.0,
            total_iters  = warmup_epochs,
        )
        cosine_sched = CosineAnnealingLR(
            self.optimizer,
            T_max   = cosine_epochs,
            eta_min = TRAIN["lr"] * 1e-2,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers = [warmup_sched, cosine_sched],
            milestones = [warmup_epochs],
        )

        self.early_stop = EarlyStopping(patience=TRAIN["patience"], mode="min")
        # AMP (automatic mixed precision) for faster GPU training
        self.scaler = GradScaler(enabled=_USE_AMP)

    # ── single epoch ─────────────────────────────────────────────────────────

    def _train_epoch(self, loader):
        self.model.train()
        running_loss    = torch.zeros(1, device=DEVICE)
        n               = 0
        step_grad_norms = []

        for X_num, X_cat, y in loader:
            X_num = X_num.to(DEVICE, non_blocking=True)
            X_cat = X_cat.to(DEVICE, non_blocking=True)
            y     = y.to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE.type, dtype=_AMP_DTYPE, enabled=_USE_AMP):
                logits = self.model(X_num, X_cat)
                loss   = self.criterion(logits, y)

            self.scaler.scale(loss).backward()

            # Unscale before grad clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), TRAIN["grad_clip"]
            )
            step_grad_norms.append(grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.detach() * len(y)
            n += len(y)

        avg_loss = (running_loss / n).item()
        self.grad_norms.extend([g.item() for g in step_grad_norms])
        return avg_loss

    @torch.no_grad()
    def _eval_epoch(self, loader):
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        running_loss = torch.zeros(1, device=DEVICE)
        n = 0
        all_probs, all_labels = [], []
        for X_num, X_cat, y in loader:
            X_num = X_num.to(DEVICE, non_blocking=True)
            X_cat = X_cat.to(DEVICE, non_blocking=True)
            y     = y.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, dtype=_AMP_DTYPE, enabled=_USE_AMP):
                logits = self.model(X_num, X_cat)
                loss   = self.criterion(logits, y)
            probs  = torch.sigmoid(logits.float())  # float32 for stable probs
            running_loss += loss.detach() * len(y)
            n            += len(y)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        avg_loss = (running_loss / n).item()
        auc = roc_auc_score(all_labels, all_probs)
        return avg_loss, auc

    # ── full training loop ────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader, epochs: int = None, verbose: bool = True):
        epochs = epochs or TRAIN["epochs"]

        # CUDA warm-up: full-batch dummy forward pass to compile CUDA kernels
        if DEVICE.type == "cuda":
            self.model.eval()
            try:
                for X_num, X_cat, _ in train_loader:
                    X_num = X_num.to(DEVICE)
                    X_cat = X_cat.to(DEVICE)
                    with torch.no_grad():
                        _ = self.model(X_num, X_cat)
                    break
                torch.cuda.synchronize()
            except Exception:
                pass  # warmup is best-effort

        t0 = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_auc = self._eval_epoch(val_loader)
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(val_auc)
            self.history["lr"].append(lr)

            self.scheduler.step()
            self.early_stop(val_loss, self.model)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"  [{self.name}] Ep {epoch:3d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_auc={val_auc:.4f} | "
                    f"lr={lr:.6f} | "
                    f"patience={self.early_stop.counter}/{self.early_stop.patience}"
                )

            if self.early_stop.stopped:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

        self.early_stop.restore(self.model)
        if verbose:
            print(f"  Training done in {time.time()-t0:.1f}s | "
                  f"best val_loss={self.early_stop.best:.4f}")

        # Save checkpoint
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save(self.model.state_dict(),
                   os.path.join(CKPT_DIR, f"{self.name}.pt"))
        return self.history

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(self, loader):
        """Returns (probs, labels) as numpy arrays."""
        self.model.eval()
        all_probs, all_labels = [], []
        for X_num, X_cat, y in loader:
            X_num = X_num.to(DEVICE, non_blocking=True)
            X_cat = X_cat.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, dtype=_AMP_DTYPE, enabled=_USE_AMP):
                logits = self.model(X_num, X_cat)
            probs  = torch.sigmoid(logits.float())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())
        return np.array(all_probs), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# TRADITIONAL ML TRAINER
# ─────────────────────────────────────────────────────────────────────────────

def train_sklearn_model(model, X_tr, y_tr, X_va=None, y_va=None, name="sklearn"):
    """
    Fit a sklearn-compatible model (XGBoost, RF, LightGBM, LogisticRegression).
    XGBoost and LightGBM support early stopping via eval_set.
    """
    model_name_lower = type(model).__name__.lower()
    t0 = time.time()

    fit_kwargs = {}
    if "xgb" in model_name_lower and X_va is not None:
        fit_kwargs["eval_set"]        = [(X_va, y_va)]
        fit_kwargs["verbose"]         = False

    elif "lgbm" in model_name_lower and X_va is not None:
        fit_kwargs["eval_set"]        = [(X_va, y_va)]
        fit_kwargs["callbacks"]       = []

    model.fit(X_tr, y_tr, **fit_kwargs)
    print(f"  [{name}] Trained in {time.time()-t0:.1f}s")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION LOOP  (for statistical tests)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_deep(
    model_fn,             # callable() → nn.Module
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: np.ndarray,
    cv_splits: list,
    pos_weight: torch.Tensor,
    batch_size: int = 512,
    name: str = "model",
    epochs: int = None,
):
    """
    5-fold cross-validation for a deep model.
    Returns dict of metric arrays (one value per fold).
    """
    from src.data_preprocessing import ChurnDataset
    from torch.utils.data import DataLoader
    from src.evaluation import compute_metrics

    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        print(f"\n  -- {name}  fold {fold+1}/{len(cv_splits)} --")

        X_num_tr = X_num[tr_idx]; X_cat_tr = X_cat[tr_idx]; y_tr = y[tr_idx]
        X_num_va = X_num[va_idx]; X_cat_va = X_cat[va_idx]; y_va = y[va_idx]

        ds_tr = ChurnDataset(X_num_tr, X_cat_tr, y_tr)
        ds_va = ChurnDataset(X_num_va, X_cat_va, y_va)

        tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0)
        va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

        torch.manual_seed(SEED + fold)
        model   = model_fn().to(DEVICE)
        trainer = DeepTrainer(model, pos_weight=pos_weight, name=f"{name}_f{fold+1}")
        trainer.fit(tr_loader, va_loader, epochs=epochs, verbose=False)

        probs, labels = trainer.predict_proba(va_loader)
        metrics = compute_metrics(labels, probs)
        fold_results.append(metrics)

    # Aggregate
    all_keys = fold_results[0].keys()
    return {k: np.array([r[k] for r in fold_results]) for k in all_keys}


def cross_validate_sklearn(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: list,
    name: str = "sklearn",
):
    """
    5-fold cross-validation for sklearn-compatible models.
    Returns dict of metric arrays.
    """
    from src.evaluation import compute_metrics
    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        print(f"  -- {name}  fold {fold+1}/{len(cv_splits)} --")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        np.random.seed(SEED + fold)
        model = model_fn()
        train_sklearn_model(model, X_tr, y_tr, name=name)

        probs  = model.predict_proba(X_va)[:, 1]
        metrics= compute_metrics(y_va, probs)
        fold_results.append(metrics)

    all_keys = fold_results[0].keys()
    return {k: np.array([r[k] for r in fold_results]) for k in all_keys}
