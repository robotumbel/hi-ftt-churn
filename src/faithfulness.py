"""
Faithfulness validation for HI-FTT attention / Semantic Group Attention.

Motivated by the attention-as-explanation debate (Jain & Wallace, 2019;
Wiegreffe & Pinter, 2019): attention weights are not automatically valid
explanations. This module supplies behavioural evidence that the model's
attributions track its actual input dependence:

  1. Group occlusion — neutralise each semantic group's feature tokens and
     measure the resulting test-AUC drop. A large drop means the model truly
     relies on that group. We then rank-correlate the occlusion importances
     with the SGA group-importance weights (Spearman).

  2. Permutation feature importance — model-agnostic reference importance per
     feature (Breiman); rank-correlated against attention-derived importance.

  3. Cross-seed stability — Spearman between group-importance vectors obtained
     under different random seeds (supplied by the caller).

All comparisons are reported with Spearman ρ and its p-value. These are
associational checks on the explanation, not causal claims about churn.
"""

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.config import DEVICE


@torch.no_grad()
def _predict_probs(model, X_num: np.ndarray, X_cat: np.ndarray, batch: int = 1024):
    model.eval()
    probs = []
    Xn = torch.as_tensor(X_num, dtype=torch.float32)
    Xc = torch.as_tensor(X_cat, dtype=torch.long)
    for i in range(0, len(Xn), batch):
        logit = model(Xn[i:i+batch].to(DEVICE), Xc[i:i+batch].to(DEVICE))
        probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(probs)


def group_occlusion(model, X_num, X_cat, y, groups, group_names,
                    num_numerical: int, sga_importance=None) -> pd.DataFrame:
    """
    Neutralise each semantic group (numerical tokens → 0 = standardised mean;
    categorical tokens → index 0) and record the test-AUC drop.

    groups : list[list[int]] token indices over the [numerical + categorical]
             concatenation (same convention as SGA / get_*_feature_groups).
    sga_importance : optional 1-D array of SGA group weights (same order as
             group_names) to rank-correlate against the occlusion importances.
    """
    base_auc = roc_auc_score(y, _predict_probs(model, X_num, X_cat))
    rows = []
    for gname, gidx in zip(group_names, groups):
        Xn = X_num.copy(); Xc = X_cat.copy()
        for t in gidx:
            if t < num_numerical:
                Xn[:, t] = 0.0                      # standardised feature mean
            else:
                c = t - num_numerical
                if 0 <= c < Xc.shape[1]:
                    Xc[:, c] = 0                    # first category level
        occ_auc = roc_auc_score(y, _predict_probs(model, Xn, Xc))
        rows.append({"Group": gname, "AUC_occluded": round(occ_auc, 4),
                     "AUC_drop": round(base_auc - occ_auc, 4), "n_features": len(gidx)})
    df = pd.DataFrame(rows).sort_values("AUC_drop", ascending=False).reset_index(drop=True)
    df.attrs["base_auc"] = round(base_auc, 4)

    if sga_importance is not None and len(sga_importance) == len(group_names):
        imp = pd.Series(np.asarray(sga_importance, dtype=float), index=group_names)
        occ = df.set_index("Group")["AUC_drop"]
        common = [g for g in group_names if g in occ.index]
        rho, p = stats.spearmanr(imp[common].values, occ[common].values)
        df.attrs["spearman_sga_vs_occlusion"] = (float(rho), float(p))
    return df


def permutation_importance(model, X_num, X_cat, y, num_numerical: int,
                           n_repeats: int = 5, seed: int = 42) -> np.ndarray:
    """
    Model-agnostic permutation importance per feature token (AUC decrease when
    the feature is shuffled). Returns (n_tokens,) mean importance.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, _predict_probs(model, X_num, X_cat))
    n_tokens = num_numerical + X_cat.shape[1]
    imp = np.zeros(n_tokens)
    for t in range(n_tokens):
        drops = []
        for _ in range(n_repeats):
            Xn = X_num.copy(); Xc = X_cat.copy()
            perm = rng.permutation(len(y))
            if t < num_numerical:
                Xn[:, t] = Xn[perm, t]
            else:
                Xc[:, t - num_numerical] = Xc[perm, t - num_numerical]
            drops.append(base - roc_auc_score(y, _predict_probs(model, Xn, Xc)))
        imp[t] = float(np.mean(drops))
    return imp


def attention_vs_importance(attn_importance: np.ndarray,
                            perm_importance: np.ndarray) -> dict:
    """Spearman rank correlation between attention-derived and permutation importance."""
    n = min(len(attn_importance), len(perm_importance))
    rho, p = stats.spearmanr(attn_importance[:n], perm_importance[:n])
    return {"spearman_rho": float(rho), "p_value": float(p), "n_features": n}


def cross_seed_stability(importance_by_seed: dict) -> pd.DataFrame:
    """
    Pairwise Spearman between per-seed importance vectors.
    importance_by_seed : {seed: 1-D importance array (aligned order)}
    """
    seeds = list(importance_by_seed)
    rows = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a = np.asarray(importance_by_seed[seeds[i]], dtype=float)
            b = np.asarray(importance_by_seed[seeds[j]], dtype=float)
            m = min(len(a), len(b))
            rho, p = stats.spearmanr(a[:m], b[:m])
            rows.append({"seed_a": seeds[i], "seed_b": seeds[j],
                         "spearman_rho": round(float(rho), 4), "p_value": float(p)})
    return pd.DataFrame(rows)


def run_faithfulness(model, X_num, X_cat, y, groups, group_names,
                     num_numerical, sga_importance=None,
                     attn_importance=None, out_prefix=None,
                     n_perm_repeats: int = 5, seed: int = 42) -> dict:
    """
    Full faithfulness battery for one trained HI-FTT model on a test set.
    Writes {out_prefix}_group_occlusion.csv and {out_prefix}_faithfulness.json
    when out_prefix is given. Returns a summary dict.
    """
    import json, os

    occ = group_occlusion(model, X_num, X_cat, y, groups, group_names,
                          num_numerical, sga_importance=sga_importance)
    summary = {"base_auc": occ.attrs.get("base_auc"),
               "group_occlusion": occ.to_dict(orient="records"),
               "spearman_sga_vs_occlusion": occ.attrs.get("spearman_sga_vs_occlusion")}

    perm = permutation_importance(model, X_num, X_cat, y, num_numerical,
                                  n_repeats=n_perm_repeats, seed=seed)
    summary["permutation_importance"] = perm.tolist()
    if attn_importance is not None:
        summary["attention_vs_permutation"] = attention_vs_importance(
            np.asarray(attn_importance), perm)

    if out_prefix:
        occ.to_csv(f"{out_prefix}_group_occlusion.csv", index=False)
        with open(f"{out_prefix}_faithfulness.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
    return summary
