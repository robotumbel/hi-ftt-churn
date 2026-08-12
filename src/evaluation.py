"""
Evaluation metrics and statistical tests for Q1/Q2 journal quality.

Standard Metrics:
  Accuracy, Precision, Recall, F1 (macro + weighted),
  ROC-AUC, PR-AUC, MCC, Cohen's Kappa, Brier Score,
  G-Mean, Specificity (True Negative Rate)

Business / Churn-Specific Metrics:
  Top Decile Lift (TDL)          — how many more churners found in top 10%
  Capture Rate @ k% (CR@k)       — fraction of churners captured in top k% contact list
  Lift @ k%                      — ratio of precision-at-k to baseline churn rate
  Expected Maximum Profit (EMP)  — profit-based metric (Verbraken et al. 2013)
  Expected Calibration Error (ECE) — probability reliability (calibration)

Statistical Tests:
  McNemar's test       — pairwise binary prediction comparison
  Paired t-test        — cross-validation fold comparison
  Wilcoxon signed-rank — non-parametric paired comparison
  Friedman test        — compare all models across folds
  Nemenyi post-hoc     — multiple comparison after Friedman
  Bonferroni correction
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
    confusion_matrix,
)

from src.config import STAT


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS / CHURN-SPECIFIC METRICS
# ─────────────────────────────────────────────────────────────────────────────

def top_decile_lift(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Top Decile Lift (TDL).
    How many times more churners does the model identify in the top 10%
    of scored customers compared to a random contact strategy?

    TDL = (# churners in top 10%) / (0.10 × total churners)
    TDL = 1.0 → no better than random; TDL = 3.0 → 3× better than random.
    """
    n      = len(y_true)
    n_top  = max(1, int(0.10 * n))
    idx    = np.argsort(y_prob)[::-1][:n_top]
    n_churn_top    = y_true[idx].sum()
    n_churn_total  = y_true.sum()
    if n_churn_total == 0:
        return 0.0
    return float(n_churn_top / (0.10 * n_churn_total))


def capture_rate_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.2) -> float:
    """
    Capture Rate at top-k% (CR@k).
    Fraction of all churners captured when contacting only the top k% of customers.
    Interpretable as: "By reaching k% of our customer base, we identify X% of churners."

    k : fraction of population targeted (e.g. 0.10, 0.20, 0.30)
    """
    n     = len(y_true)
    n_top = max(1, int(k * n))
    idx   = np.argsort(y_prob)[::-1][:n_top]
    return float(y_true[idx].sum() / (y_true.sum() + 1e-9))


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.20) -> float:
    """
    Lift at top-k%.
    Ratio of precision-at-k to the baseline churn rate.
    Lift = 2.5 means the top-k% list is 2.5× more enriched with churners than average.
    """
    n          = len(y_true)
    n_top      = max(1, int(k * n))
    idx        = np.argsort(y_prob)[::-1][:n_top]
    prec_at_k  = y_true[idx].mean()
    baseline   = y_true.mean()
    return float(prec_at_k / (baseline + 1e-9))


def expected_maximum_profit(
    y_true:       np.ndarray,
    y_prob:       np.ndarray,
    clv:          float = 1.0,
    accept_rate:  float = 0.3,
    cost_fraction: float = 0.1,
) -> float:
    """
    Expected Maximum Profit (EMP) — Verbraken et al. (2013).
    Measures the maximum profit achievable by the model at the optimal
    decision threshold, normalised by the total number of customers.

    Parameters
    ----------
    clv           : Customer Lifetime Value (normalised to 1.0 by default)
    accept_rate δ : probability a targeted churner accepts the retention offer
                    (default 0.30 following Verbraken 2013)
    cost_fraction γ : retention offer cost as a fraction of CLV
                    (default 0.10)

    Formula
    -------
    profit(t) = CLV·δ·(1–γ)·TP(t) – CLV·γ·FP(t)
    EMP = max_t [profit(t)] / n

    Returns a per-customer profit value (higher is better).
    """
    n       = len(y_true)
    order   = np.argsort(y_prob)[::-1]
    y_sort  = y_true[order]

    tp_cum  = np.cumsum(y_sort)
    fp_cum  = np.cumsum(1 - y_sort)

    benefit_per_tp = clv * accept_rate * (1.0 - cost_fraction)
    cost_per_fp    = clv * cost_fraction

    profits = benefit_per_tp * tp_cum - cost_per_fp * fp_cum
    return float(profits.max() / n)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Measures how well predicted probabilities match empirical frequencies.
    ECE = Σ_b (|B_b|/n) · |acc(B_b) – conf(B_b)|

    ECE = 0.0 → perfectly calibrated; higher → overconfident or underconfident.
    Essential for CRM deployment where P(churn) must reflect true risk.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 \
               else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute all evaluation metrics given ground-truth labels and predicted probs.

    Parameters
    ----------
    y_true    : binary ground-truth labels
    y_prob    : predicted probabilities for positive class
    threshold : decision threshold (default 0.5)

    Returns
    -------
    dict of metric_name → float
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Derived rates
    specificity  = tn / (tn + fp + 1e-9)   # True Negative Rate
    sensitivity  = tp / (tp + fn + 1e-9)   # same as recall
    g_mean       = np.sqrt(sensitivity * specificity)

    metrics = {
        # ── Standard metrics ─────────────────────────────────────────────────
        "accuracy"          : accuracy_score(y_true, y_pred),
        "precision_macro"   : precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro"      : recall_score(y_true, y_pred, average="macro",    zero_division=0),
        "recall_weighted"   : recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro"          : f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_weighted"       : f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_binary"         : f1_score(y_true, y_pred, average="binary",   zero_division=0),
        "roc_auc"           : roc_auc_score(y_true, y_prob),
        "pr_auc"            : average_precision_score(y_true, y_prob),
        "mcc"               : matthews_corrcoef(y_true, y_pred),
        "kappa"             : cohen_kappa_score(y_true, y_pred),
        "brier_score"       : brier_score_loss(y_true, y_prob),
        "specificity"       : specificity,
        "sensitivity"       : sensitivity,
        "g_mean"            : g_mean,
        # ── Business / churn-specific metrics ────────────────────────────────
        "tdl"               : top_decile_lift(y_true, y_prob),
        "capture_rate_10"   : capture_rate_at_k(y_true, y_prob, k=0.10),
        "capture_rate_20"   : capture_rate_at_k(y_true, y_prob, k=0.20),
        "capture_rate_30"   : capture_rate_at_k(y_true, y_prob, k=0.30),
        "lift_10"           : lift_at_k(y_true, y_prob, k=0.10),
        "lift_20"           : lift_at_k(y_true, y_prob, k=0.20),
        "lift_30"           : lift_at_k(y_true, y_prob, k=0.30),
        "emp"               : expected_maximum_profit(y_true, y_prob),
        "ece"               : expected_calibration_error(y_true, y_prob),
        # ── Confusion matrix components ───────────────────────────────────────
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
    }
    return metrics


def metrics_to_dataframe(results: dict) -> pd.DataFrame:
    """
    results : { model_name → { metric → float } }
    Returns a formatted DataFrame (rows=models, cols=metrics).
    """
    rows = []
    for model_name, m in results.items():
        row = {"Model": model_name}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MCNEMAR'S TEST
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray):
    """
    McNemar's test to compare two classifiers on the same test set.
    Uses the exact binomial or chi-squared formulation.

    Returns
    -------
    dict with: statistic, p_value, b, c, significant
    """
    # Contingency table
    b = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))  # A right, B wrong
    c = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))  # A wrong, B right

    n_discordant = b + c

    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c,
                "significant": False, "note": "No discordant pairs"}

    if n_discordant < 25:
        # Exact binomial test (scipy >= 1.7 uses binomtest)
        from scipy.stats import binomtest
        p_value = binomtest(b, n_discordant, 0.5).pvalue
        statistic = None
    else:
        # Chi-squared with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value   = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic"  : statistic,
        "p_value"    : float(p_value),
        "b"          : int(b),
        "c"          : int(c),
        "significant": p_value < STAT["alpha"],
    }


def pairwise_mcnemar(y_true: np.ndarray, preds_dict: dict) -> pd.DataFrame:
    """
    Run McNemar test for all pairs of models.
    preds_dict: { model_name → binary prediction array }

    Returns
    -------
    DataFrame of p-values (upper triangle) + significance markers
    """
    model_names = list(preds_dict.keys())
    n = len(model_names)
    p_matrix = pd.DataFrame(np.ones((n, n)), index=model_names, columns=model_names)
    sig_matrix = pd.DataFrame("", index=model_names, columns=model_names)

    for a, b in combinations(model_names, 2):
        result = mcnemar_test(y_true, preds_dict[a], preds_dict[b])
        p      = result["p_value"]
        sig    = "*" if p < 0.05 else ("†" if p < 0.10 else "ns")
        p_matrix.loc[a, b]  = p
        p_matrix.loc[b, a]  = p
        sig_matrix.loc[a, b]= sig
        sig_matrix.loc[b, a]= sig

    return p_matrix, sig_matrix


# ─────────────────────────────────────────────────────────────────────────────
# PAIRED T-TEST (cross-validation folds)
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(scores_a: np.ndarray, scores_b: np.ndarray, metric: str = "f1_macro"):
    """
    Paired t-test comparing two models over k-fold results.
    scores_a, scores_b : 1D arrays of length k (one value per fold).

    Returns
    -------
    dict with: statistic, p_value, significant, mean_diff, ci95
    """
    diff = scores_a - scores_b
    t, p = stats.ttest_rel(scores_a, scores_b)
    n    = len(diff)
    se   = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - 0.025, df=n - 1)
    ci   = (diff.mean() - t_crit * se, diff.mean() + t_crit * se)

    return {
        "t_statistic"  : float(t),
        "p_value"      : float(p),
        "significant"  : p < STAT["alpha"],
        "mean_diff"    : float(diff.mean()),
        "std_diff"     : float(diff.std(ddof=1)),
        "ci95"         : (float(ci[0]), float(ci[1])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WILCOXON SIGNED-RANK TEST (non-parametric alternative to paired t-test)
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray):
    """
    Non-parametric Wilcoxon signed-rank test.
    More robust than t-test for small k.
    """
    if np.allclose(scores_a, scores_b):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    stat, p = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
    return {
        "statistic"  : float(stat),
        "p_value"    : float(p),
        "significant": p < STAT["alpha"],
        "mean_diff"  : float((scores_a - scores_b).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FRIEDMAN TEST + NEMENYI POST-HOC
# ─────────────────────────────────────────────────────────────────────────────

def friedman_nemenyi(cv_results: dict, metric: str = "f1_macro"):
    """
    Friedman test to compare all models simultaneously over k-folds.
    If significant, run pairwise Nemenyi post-hoc test.

    Parameters
    ----------
    cv_results : { model_name → { metric → np.array(k,) } }
    metric     : which metric to compare

    Returns
    -------
    dict with Friedman statistic, p_value, and Nemenyi comparison table
    """
    model_names = list(cv_results.keys())
    k_folds     = len(list(cv_results.values())[0][metric])

    # Build matrix (n_models × n_folds)
    score_matrix = np.array([cv_results[m][metric] for m in model_names])  # (M, K)

    stat, p = stats.friedmanchisquare(*[score_matrix[i] for i in range(len(model_names))])

    result = {
        "friedman_stat": float(stat),
        "p_value"       : float(p),
        "significant"   : p < STAT["alpha"],
        "model_means"   : {m: float(cv_results[m][metric].mean()) for m in model_names},
    }

    if p < STAT["alpha"]:
        # Nemenyi test (simplified: use pairwise Wilcoxon with Bonferroni correction)
        pairs   = list(combinations(model_names, 2))
        n_pairs = len(pairs)
        alpha_corrected = STAT["alpha"] / n_pairs  # Bonferroni

        nemenyi = {}
        for a, b in pairs:
            w = wilcoxon_test(cv_results[a][metric], cv_results[b][metric])
            nemenyi[f"{a} vs {b}"] = {
                "p_value"          : w["p_value"],
                "p_corrected"      : min(w["p_value"] * n_pairs, 1.0),  # Bonferroni
                "significant"      : w["p_value"] * n_pairs < STAT["alpha"],
                "mean_diff"        : w["mean_diff"],
            }
        result["nemenyi"] = nemenyi

    return result


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE CV SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def cv_summary_table(cv_results: dict, metrics: list = None) -> pd.DataFrame:
    """
    Build mean±std table from cross-validation results.
    cv_results: { model_name → { metric → array(k,) } }
    """
    if metrics is None:
        metrics = [
            "accuracy", "f1_macro", "roc_auc", "pr_auc",
            "mcc", "kappa", "g_mean",
            "tdl", "capture_rate_20", "lift_20", "emp", "ece",
        ]

    rows = []
    for model, res in cv_results.items():
        row = {"Model": model}
        for m in metrics:
            if m in res:
                vals = res[m]
                row[m] = f"{vals.mean():.4f} ± {vals.std():.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


# ─────────────────────────────────────────────────────────────────────────────
# EFFECT SIZE (Cohen's d)
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Cohen's d effect size between two paired score arrays."""
    diff   = scores_a - scores_b
    return float(diff.mean() / (diff.std(ddof=1) + 1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# FULL STATISTICAL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def full_statistical_comparison(
    cv_results: dict,
    proposed_model: str = "Transformer",
    metric: str = "f1_macro",
) -> pd.DataFrame:
    """
    Compare proposed model against all baselines using:
    - Paired t-test
    - Wilcoxon signed-rank test
    - Cohen's d effect size
    - Bonferroni-corrected significance

    Returns a DataFrame suitable for a journal table.
    """
    baselines   = [m for m in cv_results if m != proposed_model]
    n_tests     = len(baselines)
    alpha_corr  = STAT["alpha"] / n_tests

    rows = []
    proposed_scores = cv_results[proposed_model][metric]

    for baseline in baselines:
        base_scores  = cv_results[baseline][metric]
        tt           = paired_ttest(proposed_scores, base_scores)
        wt           = wilcoxon_test(proposed_scores, base_scores)
        d            = cohens_d(proposed_scores, base_scores)

        # Bonferroni-corrected p
        p_bonf_t = min(tt["p_value"] * n_tests, 1.0)
        p_bonf_w = min(wt["p_value"] * n_tests, 1.0)

        rows.append({
            "Baseline"         : baseline,
            "Proposed Mean"    : f"{proposed_scores.mean():.4f}",
            "Baseline Mean"    : f"{base_scores.mean():.4f}",
            "Mean Diff"        : f"{tt['mean_diff']:+.4f}",
            "t-statistic"      : f"{tt['t_statistic']:.3f}",
            "p-value (t-test)" : f"{tt['p_value']:.4f}",
            "p (Bonferroni)"   : f"{p_bonf_t:.4f}",
            "Sig (t-test)"     : "✓" if p_bonf_t < STAT["alpha"] else "✗",
            "W-statistic"      : f"{wt['statistic']:.3f}",
            "p-value (Wilcoxon)": f"{wt['p_value']:.4f}",
            "Sig (Wilcoxon)"   : "✓" if p_bonf_w < STAT["alpha"] else "✗",
            "Cohen's d"        : f"{d:.3f}",
            "Effect Size"      : (
                "large" if abs(d) >= 0.8 else
                "medium" if abs(d) >= 0.5 else
                "small"
            ),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# AUC SIGNIFICANCE ON THE TEST SET  (DeLong + paired bootstrap)
# ─────────────────────────────────────────────────────────────────────────────
#
# Replaces the earlier headline that conflated a McNemar test on thresholded
# accuracy with a claim about AUC. DeLong (1988; fast algorithm Sun & Xu 2014)
# gives an analytic test for two correlated ROC-AUCs on the SAME test set; the
# paired stratified bootstrap gives a distribution-free CI on the AUC gap.

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_T: np.ndarray, m: int):
    """preds_sorted_T: (k, N) predictions, columns sorted positives-first; m = #positives."""
    n = preds_sorted_T.shape[1] - m
    pos = preds_sorted_T[:, :m]
    neg = preds_sorted_T[:, m:]
    k   = preds_sorted_T.shape[0]

    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_T[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01  = (tz[:, :m] - tx) / n
    v10  = 1.0 - (tz[:, m:] - ty) / m
    sx   = np.cov(v01)
    sy   = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """
    Two-sided DeLong test that AUC(a) == AUC(b) on one test set.
    Returns auc_a, auc_b, z, p_value.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)

    order = (-y_true).argsort(kind="mergesort")   # positives (label 1) first
    m     = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = _fast_delong(preds, m)

    l   = np.array([[1.0, -1.0]])
    var = float(np.asarray(l @ np.atleast_2d(cov) @ l.T).ravel()[0])
    if var <= 1e-16:
        # identical (or perfectly correlated) predictions → no difference
        z, p = 0.0, 1.0
    else:
        z = float(np.abs(aucs[0] - aucs[1]) / np.sqrt(var))
        p = float(2.0 * stats.norm.sf(z))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "auc_diff": float(aucs[0] - aucs[1]), "z": z, "p_value": p}


def paired_bootstrap_auc(y_true, prob_a, prob_b, n_boot: int = 5000,
                         seed: int = 42) -> dict:
    """Stratified paired bootstrap CI + p-value for AUC(a) - AUC(b)."""
    y_true = np.asarray(y_true).astype(int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    rng    = np.random.default_rng(seed)
    pos    = np.where(y_true == 1)[0]
    neg    = np.where(y_true == 0)[0]

    obs = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        bi = np.concatenate([rng.choice(pos, pos.size, replace=True),
                             rng.choice(neg, neg.size, replace=True)])
        yb = y_true[bi]
        diffs[b] = roc_auc_score(yb, prob_a[bi]) - roc_auc_score(yb, prob_b[bi])
    ci  = np.percentile(diffs, [2.5, 97.5])
    p   = float(2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"auc_diff": float(obs), "ci_low": float(ci[0]),
            "ci_high": float(ci[1]), "p_value": min(p, 1.0)}


def auc_significance_table(y_true, probs_dict: dict, proposed: str = "Transformer",
                           n_boot: int = 5000, bonferroni: bool = True) -> pd.DataFrame:
    """
    Proposed-vs-each-baseline AUC comparison on the test set: DeLong + paired
    bootstrap, with an explicit (and correct) Bonferroni family size = #baselines.
    """
    baselines = [m for m in probs_dict if m != proposed]
    n_tests   = len(baselines)
    rows = []
    for b in baselines:
        d  = delong_roc_test(y_true, probs_dict[proposed], probs_dict[b])
        bs = paired_bootstrap_auc(y_true, probs_dict[proposed], probs_dict[b], n_boot=n_boot)
        p_bonf = min(d["p_value"] * n_tests, 1.0) if bonferroni else d["p_value"]
        rows.append({
            "Baseline"        : b,
            "AUC (proposed)"  : round(d["auc_a"], 4),
            "AUC (baseline)"  : round(d["auc_b"], 4),
            "ΔAUC"            : round(d["auc_diff"], 4),
            "DeLong z"        : round(d["z"], 3),
            "DeLong p"        : d["p_value"],
            f"DeLong p (Bonf, k={n_tests})": p_bonf,
            "Bootstrap 95% CI": f"[{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]",
            "Bootstrap p"     : bs["p_value"],
            "Sig (Bonf 0.05)" : "yes" if p_bonf < 0.05 else "no",
        })
    return pd.DataFrame(rows)
