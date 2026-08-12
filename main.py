"""
main.py — Experiment Orchestrator
==================================
A Transformer-based Approach for Churn Prediction
in Hotel Customer Relationship Management

Usage:
  python main.py                          # full experiment on hotel dataset
  python main.py --dataset telco          # generalization on Telco
  python main.py --dataset both           # both datasets
  python main.py --quick                  # fast debug mode (10 epochs)
  python main.py --skip_cv               # skip cross-validation (faster)
  python main.py --skip_ablation         # skip ablation study
"""

import os
import sys

# Force UTF-8 output on Windows to avoid encoding errors with special chars
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import random
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch


class _NumpyEncoder(json.JSONEncoder):
    """Handles numpy scalars (int64, float64, bool_) in json.dump."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DEVICE, SEED, TRAIN, TABLE_DIR, FIG_DIR, RESULT_DIR, RUN_ID,
    TRANSFORMER, ALL_MODELS, DEEP_MODELS, TRAD_MODELS,
    ABLATION_VARIANTS, ABLATION_EPOCHS, CV_FOLDS,
)
from src.data_preprocessing import (
    prepare_dataset, get_hotel_feature_groups,
    get_telco_feature_groups, get_bank_feature_groups,
)
from src.models import build_model, HIFTTransformer, VanillaFTTransformer, MLPChurn, CNNChurn, count_parameters
from src.training import DeepTrainer, train_sklearn_model, cross_validate_deep, cross_validate_sklearn
from src.evaluation import (
    compute_metrics, metrics_to_dataframe, cv_summary_table,
    pairwise_mcnemar, friedman_nemenyi, full_statistical_comparison,
    auc_significance_table,
)
from src.ablation_study import run_ablation_study, save_ablation_table
from src.convergence_analysis import (
    analyze_gradient_norms, save_convergence_table,
)
from src.visualization import (
    plot_roc_curves, plot_pr_curves, plot_loss_curves, plot_convergence,
    plot_gradient_norms, plot_ablation, plot_mcnemar_heatmap,
    plot_cv_boxplot, plot_feature_importance, plot_confusion_matrices,
    plot_lr_schedule, plot_generalization, plot_radar_chart,
)
from src.attention_analysis import (
    extract_attention_weights, attention_to_feature_importance,
    plot_attention_heatmap, plot_feature_attention_importance,
    plot_attention_rollout, analyze_attention_heads,
)
from src.faithfulness import run_faithfulness


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE DATASET EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    dataset:        str   = "hotel",
    epochs:         int   = None,
    skip_cv:        bool  = False,
    skip_ablation:  bool  = False,
    verbose:        bool  = True,
    seed:           int   = None,
    split_mode:     str   = "random",
    tag:            str   = "",
):
    """Full experimental pipeline for one dataset."""
    seed = SEED if seed is None else seed
    # Output-file suffix so multi-seed / alternative-split runs don't overwrite.
    suffix = tag or (f"_seed{seed}" if seed != SEED else "")
    if split_mode != "random":
        suffix += f"_{split_mode}"

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {dataset.upper()} DATASET  |  Device: {DEVICE}  |  "
          f"seed={seed} split={split_mode}")
    print(f"{'='*70}\n")

    set_seed(seed)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,   exist_ok=True)

    epochs = epochs or TRAIN["epochs"]

    # ── 1. DATA PREPARATION ──────────────────────────────────────────────────
    print("[ 1/7 ] Data Preparation ...")
    loaders, meta, cv_splits = prepare_dataset(
        dataset=dataset,
        use_smote=True,
        batch_size=TRAIN["batch_size"],
        return_cv=True,
        seed=seed,
        split_mode=split_mode,
    )

    num_numerical = meta["num_features"]
    cat_dims      = meta["cat_dims"]
    pos_weight    = meta["pos_weight"]
    X_flat_tr     = np.hstack([meta["X_num_tr"], meta["X_cat_tr"].astype(float)])
    X_flat_va     = np.hstack([meta["X_num_va"], meta["X_cat_va"].astype(float)])
    X_flat_te     = np.hstack([meta["X_num_te"], meta["X_cat_te"].astype(float)])
    y_tr          = meta["y_tr"]
    y_va          = meta["y_va"]
    y_te          = meta["y_te"]

    print(f"  Numerical features: {num_numerical}  |  "
          f"Categorical features: {len(cat_dims)}")

    # ── Feature groups + phase-conditioning var for HI-FTT novel components ──
    # Each dataset supplies domain-specific semantic groups (SGA) and a
    # lifecycle "phase" variable (BPAC). Off-hotel modules are now genuinely
    # active — HI-FTT no longer degenerates to the FT-Transformer baseline.
    proc             = meta["processor"]
    feature_groups_list  = None
    feature_group_names  = None
    lead_time_idx        = None   # index of the BPAC conditioning variable in X_num

    if hasattr(proc, "_num_cols"):
        _bin = list(getattr(proc, "_bin_cols", []) or [])
        if dataset == "hotel":
            fg_dict   = get_hotel_feature_groups(proc._num_cols, proc._cat_cols)
            cond_var  = "lead_time"
        elif dataset == "telco":
            fg_dict   = get_telco_feature_groups(proc._num_cols, _bin, proc._cat_cols)
            cond_var  = "TenureinMonths"
        elif dataset == "bank":
            fg_dict   = get_bank_feature_groups(proc._num_cols, _bin, proc._cat_cols)
            cond_var  = "tenure"
        else:
            fg_dict, cond_var = None, None

        if fg_dict:
            feature_groups_list = list(fg_dict.values())
            feature_group_names = list(fg_dict.keys())
            # X_num tensor order = num_cols + bin_cols; conditioning var lives in num_cols
            if cond_var in proc._num_cols:
                lead_time_idx = list(proc._num_cols).index(cond_var)
            print(f"  [HI-FTT] Feature groups : {feature_group_names}")
            print(f"  [HI-FTT] phase-cond var : {cond_var} (idx {lead_time_idx})")

    # ── 2. TRAIN ALL MODELS ───────────────────────────────────────────────────
    print("\n[ 2/7 ] Training All Models ...")

    trained_dl    = {}    # deep learning: name → DeepTrainer
    trained_sk    = {}    # sklearn: name → fitted model
    test_probs    = {}    # name → test prob array
    val_probs     = {}    # name → validation prob array (for threshold tuning)
    test_preds    = {}    # name → binary pred array
    dl_histories  = {}    # name → history dict
    dl_grad_info  = {}    # name → grad norm analysis

    for model_name in ALL_MODELS:
        print(f"\n  > Training {model_name} ...")
        set_seed(seed)

        if model_name in DEEP_MODELS:
            if model_name == "Transformer":
                # Proposed HI-FTT with novel BPAC + SGA components
                model = HIFTTransformer(
                    num_numerical   = num_numerical,
                    cat_dims        = cat_dims,
                    lead_time_idx   = lead_time_idx,
                    feature_groups  = feature_groups_list,
                    group_names     = feature_group_names,
                )
            else:
                # FTTransformer, LSTM, GRU — via factory
                model = build_model(model_name, num_numerical, cat_dims)
            n_param = count_parameters(model)
            print(f"    Parameters: {n_param:,}")

            trainer = DeepTrainer(model, pos_weight=pos_weight, name=f"{dataset}_{model_name}")
            history = trainer.fit(loaders["train"], loaders["val"], epochs=epochs, verbose=verbose)

            probs, labels = trainer.predict_proba(loaders["test"])
            val_probs[model_name] = trainer.predict_proba(loaders["val"])[0]
            dl_histories[model_name] = history
            dl_grad_info[model_name] = analyze_gradient_norms(trainer.grad_norms)
            trained_dl[model_name]   = trainer

        else:
            sk_model = build_model(model_name, num_numerical, cat_dims)
            train_sklearn_model(sk_model, X_flat_tr, y_tr,
                                X_va=X_flat_va, y_va=y_va, name=model_name)
            probs    = sk_model.predict_proba(X_flat_te)[:, 1]
            labels   = y_te
            val_probs[model_name] = sk_model.predict_proba(X_flat_va)[:, 1]
            trained_sk[model_name] = sk_model

        test_probs[model_name] = probs
        test_preds[model_name] = (probs >= 0.5).astype(int)

    # ── 3. TEST SET EVALUATION ────────────────────────────────────────────────
    print("\n[ 3/7 ] Test Set Evaluation ...")

    test_results = {}
    for name, probs in test_probs.items():
        test_results[name] = compute_metrics(y_te, probs)

    results_df = metrics_to_dataframe(test_results)
    display_metrics = [
        "accuracy", "f1_macro", "roc_auc", "pr_auc",
        "mcc", "g_mean", "brier_score",
        "tdl", "capture_rate_20", "lift_20", "emp", "ece",
    ]
    print("\n  === Test Set Results ===")
    print(results_df[[m for m in display_metrics if m in results_df.columns]].to_string())

    results_path = os.path.join(TABLE_DIR, f"test_results_{dataset}{suffix}.csv")
    results_df.to_csv(results_path)
    print(f"\n  Results saved → {results_path}")

    # Persist per-model test prediction arrays + labels for reproducibility and
    # for post-hoc AUC significance (DeLong / paired bootstrap) and faithfulness.
    preds_path = os.path.join(TABLE_DIR, f"preds_{dataset}{suffix}.npz")
    np.savez_compressed(
        preds_path,
        y_true=np.asarray(y_te),
        seed=np.asarray(seed),
        **{f"prob__{m}": np.asarray(p) for m, p in test_probs.items()},
    )
    print(f"  Prediction arrays saved → {preds_path}")

    # Validation-optimised decision thresholds (R2 Major7/Major8): pick the
    # threshold that maximises F1-macro on the VALIDATION set, then report test
    # metrics at that threshold. Default-0.5 metrics remain in results_df above.
    if val_probs:
        thr_rows = []
        for name, vp in val_probs.items():
            grid = np.linspace(0.05, 0.95, 91)
            f1s  = [compute_metrics(y_va, vp, threshold=t)["f1_macro"] for t in grid]
            t_opt = float(grid[int(np.argmax(f1s))])
            m = compute_metrics(y_te, test_probs[name], threshold=t_opt)
            m2 = {"Model": name, "threshold": round(t_opt, 3)}
            m2.update({k: m[k] for k in ("accuracy", "f1_macro", "mcc", "g_mean",
                                         "recall_macro", "precision_macro")
                       if k in m})
            thr_rows.append(m2)
        thr_df = pd.DataFrame(thr_rows)
        thr_path = os.path.join(TABLE_DIR, f"test_results_threshopt_{dataset}{suffix}.csv")
        thr_df.to_csv(thr_path, index=False)
        print(f"  Threshold-optimised results saved → {thr_path}")
        print(thr_df.to_string(index=False))

    # ── 4. STATISTICAL TESTS ─────────────────────────────────────────────────
    print("\n[ 4/7 ] Statistical Tests ...")

    # McNemar's test (on test set)
    p_matrix, sig_matrix = pairwise_mcnemar(y_te, test_preds)
    mcnemar_path = os.path.join(TABLE_DIR, f"mcnemar_{dataset}.csv")
    p_matrix.to_csv(mcnemar_path)
    print("\n  McNemar's p-value matrix:")
    print(p_matrix.round(4).to_string())
    print("\n  Significance (* p<0.05, † p<0.10, ns):")
    print(sig_matrix.to_string())

    # AUC significance on the test set: DeLong + paired bootstrap (proposed vs
    # each baseline). This — not McNemar on thresholded accuracy — is the correct
    # test for the ranking-quality (AUC) claims made in the paper.
    if "Transformer" in test_probs and len(test_probs) > 1:
        print("\n  AUC significance (DeLong + paired bootstrap, proposed vs baselines):")
        auc_sig = auc_significance_table(y_te, test_probs, proposed="Transformer",
                                         n_boot=5000, bonferroni=True)
        auc_sig_path = os.path.join(TABLE_DIR, f"auc_significance_{dataset}.csv")
        auc_sig.to_csv(auc_sig_path, index=False)
        print(auc_sig.to_string(index=False))
        print(f"  AUC significance saved → {auc_sig_path}")

    # Cross-validation (optional)
    cv_results = {}
    if not skip_cv:
        print("\n  Running 5-Fold Cross-Validation for Statistical Validation ...")
        X_num_all = meta["X_num_all"]
        X_cat_all = meta["X_cat_all"]
        y_all     = meta["y_all_tv"]
        X_flat_all = meta["X_all_tv"]

        for model_name in ALL_MODELS:
            print(f"\n  > CV: {model_name}")
            set_seed(seed)
            if model_name in DEEP_MODELS:
                if model_name == "Transformer":
                    def _transformer_fn(
                        _n=num_numerical, _c=cat_dims,
                        _lt=lead_time_idx, _fg=feature_groups_list, _gn=feature_group_names
                    ):
                        return HIFTTransformer(
                            num_numerical=_n, cat_dims=_c,
                            lead_time_idx=_lt, feature_groups=_fg, group_names=_gn,
                        )
                    _model_fn = _transformer_fn
                else:
                    # FTTransformer, LSTM, GRU
                    _model_fn = lambda mn=model_name: build_model(mn, num_numerical, cat_dims)
                cv_results[model_name] = cross_validate_deep(
                    model_fn    = _model_fn,
                    X_num       = X_num_all,
                    X_cat       = X_cat_all,
                    y           = y_all,
                    cv_splits   = cv_splits,
                    pos_weight  = pos_weight,
                    batch_size  = TRAIN["batch_size"],
                    name        = model_name,
                    epochs      = epochs,
                )
            else:
                cv_results[model_name] = cross_validate_sklearn(
                    model_fn  = lambda mn=model_name: build_model(mn, num_numerical, cat_dims),
                    X         = X_flat_all,
                    y         = y_all,
                    cv_splits = cv_splits,
                    name      = model_name,
                )

        # CV summary table
        cv_df = cv_summary_table(cv_results)
        cv_path = os.path.join(TABLE_DIR, f"cv_results_{dataset}.csv")
        cv_df.to_csv(cv_path)
        print("\n  === 5-Fold CV Results (mean ± std) ===")
        print(cv_df.to_string())

        # Friedman test
        print("\n  Running Friedman + Nemenyi Post-hoc Test ...")
        friedman_res = friedman_nemenyi(cv_results, metric="f1_macro")
        print(f"  Friedman χ²={friedman_res['friedman_stat']:.4f}, "
              f"p={friedman_res['p_value']:.4f}, "
              f"significant={friedman_res['significant']}")
        if "nemenyi" in friedman_res:
            print("  Nemenyi post-hoc (Bonferroni-corrected):")
            for pair, r in friedman_res["nemenyi"].items():
                sig = "✓" if r["significant"] else "✗"
                print(f"    {pair}: p={r['p_corrected']:.4f} {sig}")

        # Proposed vs. baselines comparison table
        stat_cmp = full_statistical_comparison(cv_results, proposed_model="Transformer")
        stat_path = os.path.join(TABLE_DIR, f"statistical_comparison_{dataset}.csv")
        stat_cmp.to_csv(stat_path, index=False)
        print(f"\n  Statistical comparison saved → {stat_path}")
        print(stat_cmp.to_string(index=False))

        # Save Friedman results
        friedman_path = os.path.join(TABLE_DIR, f"friedman_{dataset}.json")
        with open(friedman_path, "w") as f:
            json.dump({k: v for k, v in friedman_res.items() if k != "nemenyi"}, f, indent=2, cls=_NumpyEncoder)
        if "nemenyi" in friedman_res:
            with open(friedman_path.replace(".json", "_nemenyi.json"), "w") as f:
                json.dump(friedman_res["nemenyi"], f, indent=2, cls=_NumpyEncoder)

    # ── 5. ABLATION STUDY ────────────────────────────────────────────────────
    # Free GPU cache before ablation (CV leaves residual CUDA allocations)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ablation_results = {}
    if not skip_ablation:
        print("\n[ 5/7 ] Ablation Study ...")
        ablation_results = run_ablation_study(
            num_numerical   = num_numerical,
            cat_dims        = cat_dims,
            train_loader    = loaders["train"],
            val_loader      = loaders["val"],
            test_loader     = loaders["test"],
            pos_weight      = pos_weight,
            epochs          = ABLATION_EPOCHS,
            verbose         = verbose,
            dataset_name    = dataset,
            feature_groups  = feature_groups_list,
            lead_time_idx   = lead_time_idx,
            group_names     = feature_group_names,
        )
        save_ablation_table(ablation_results, dataset_name=dataset)
    else:
        print("\n[ 5/7 ] Ablation Study (SKIPPED)")

    # ── 6. CONVERGENCE ANALYSIS ───────────────────────────────────────────────
    print("\n[ 6/7 ] Convergence Analysis ...")
    if dl_histories:
        save_convergence_table(dl_histories, dataset_name=dataset)

    # ── 7. VISUALIZATIONS ────────────────────────────────────────────────────
    print("\n[ 7/7 ] Generating Figures ...")

    # ROC curves
    plot_roc_curves(test_probs, y_te, dataset_name=dataset)

    # PR curves
    plot_pr_curves(test_probs, y_te, dataset_name=dataset)

    # Loss curves (deep models only)
    if dl_histories:
        plot_loss_curves(dl_histories, dataset_name=dataset)
        plot_convergence(dl_histories, dataset_name=dataset)
        plot_lr_schedule(dl_histories, dataset_name=dataset)

    # Gradient norms
    if dl_grad_info:
        plot_gradient_norms(dl_grad_info, dataset_name=dataset)

    # Ablation
    if ablation_results:
        plot_ablation(ablation_results, metric="f1_macro", dataset_name=dataset)
        plot_ablation(ablation_results, metric="roc_auc",  dataset_name=dataset)

    # McNemar heatmap
    plot_mcnemar_heatmap(p_matrix, dataset_name=dataset)

    # CV box plots
    if cv_results:
        for metric in ["f1_macro", "roc_auc", "mcc"]:
            plot_cv_boxplot(cv_results, metric=metric, dataset_name=dataset)

    # Confusion matrices (proposed vs best baseline)
    key_models = {"Transformer": test_preds["Transformer"]}
    # Best baseline by F1
    best_base = max(
        [m for m in test_results if m != "Transformer"],
        key=lambda m: test_results[m]["f1_macro"]
    )
    key_models[best_base] = test_preds[best_base]
    plot_confusion_matrices(key_models, y_te, dataset_name=dataset)

    # Radar chart
    plot_radar_chart(test_results, dataset_name=dataset)

    # Attention analysis (Transformer interpretability)
    if "Transformer" in trained_dl:
        print("\n  Attention Analysis (Transformer interpretability) ...")
        proc = meta["processor"]
        feature_names = list(proc._num_cols) + list(proc._cat_cols)

        attn_weights = extract_attention_weights(
            trained_dl["Transformer"].model,
            loaders["test"],
            n_samples=200,
        )
        if attn_weights is not None:
            plot_attention_heatmap(attn_weights, feature_names, dataset_name=dataset)

            importance_cls = attention_to_feature_importance(attn_weights, use_cls_only=True)
            plot_feature_attention_importance(
                importance_cls, feature_names, top_n=20, dataset_name=dataset
            )

            plot_attention_rollout(attn_weights, feature_names, dataset_name=dataset)

            head_df = analyze_attention_heads(attn_weights, feature_names, dataset_name=dataset)
            head_path = os.path.join(TABLE_DIR, f"attention_heads_{dataset}.csv")
            head_df.to_csv(head_path, index=False)
            print(f"  Attention head analysis saved -> {head_path}")

        # ── SGA Group Importance (Novel B interpretability) ───────────────────
        transformer_model = trained_dl["Transformer"].model
        if (hasattr(transformer_model, "group_importance") and
                transformer_model.group_importance is not None and
                feature_group_names):
            g_imp = transformer_model.group_importance.numpy()
            g_imp_dict = {
                name: float(w)
                for name, w in zip(feature_group_names, g_imp)
            }
            print("\n  [HI-FTT] Semantic Group Importance (CRM Dashboard):")
            for g, w in sorted(g_imp_dict.items(), key=lambda x: -x[1]):
                print(f"    {g:<20s} : {w:.4f}")

            g_imp_path = os.path.join(TABLE_DIR, f"group_importance_{dataset}.json")
            with open(g_imp_path, "w") as f:
                json.dump(g_imp_dict, f, indent=2, cls=_NumpyEncoder)
            print(f"  Group importance saved -> {g_imp_path}")

        # ── Faithfulness validation (attention-as-explanation checks) ─────────
        if feature_groups_list:
            print("\n  [HI-FTT] Faithfulness validation "
                  "(group occlusion + attention-vs-permutation) ...")
            attn_imp = importance_cls if "importance_cls" in dir() else None
            sga_imp  = (transformer_model.group_importance.numpy()
                        if getattr(transformer_model, "group_importance", None) is not None
                        else None)
            fth = run_faithfulness(
                transformer_model,
                meta["X_num_te"], meta["X_cat_te"], y_te,
                feature_groups_list, feature_group_names, num_numerical,
                sga_importance=sga_imp, attn_importance=attn_imp,
                out_prefix=os.path.join(TABLE_DIR, f"faithfulness_{dataset}{suffix}"),
            )
            occ = fth.get("spearman_sga_vs_occlusion")
            avp = fth.get("attention_vs_permutation")
            print(f"    base AUC={fth.get('base_auc')}  "
                  f"Spearman(SGA,occlusion)={occ}  attn-vs-perm={avp}")

    print(f"\n  All figures saved to: {FIG_DIR}")

    return {
        "test_results"   : test_results,
        "cv_results"     : cv_results,
        "ablation"       : ablation_results,
        "dl_histories"   : dl_histories,
        "dl_grad_info"   : dl_grad_info,
        "test_probs"     : test_probs,
        "test_preds"     : test_preds,
        "y_te"           : y_te,
        "meta"           : meta,
        "p_matrix"       : p_matrix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GENERALIZATION EXPERIMENT (hotel → telco)
# ─────────────────────────────────────────────────────────────────────────────

def run_generalization(hotel_results: dict, epochs: int = None, verbose: bool = True):
    """
    Train on Telco dataset separately and compare results for generalization.
    """
    print(f"\n{'='*70}")
    print(f"  GENERALIZATION: IBM TELCO CHURN  |  Device: {DEVICE}")
    print(f"{'='*70}\n")

    telco_out = run_experiment(
        dataset       = "telco",
        epochs        = epochs,
        skip_cv       = True,
        skip_ablation = True,
        verbose       = verbose,
    )

    # Generalization bar chart (F1 on both datasets)
    hotel_f1 = {m: r["f1_macro"] for m, r in hotel_results["test_results"].items()}
    telco_f1 = {m: r["f1_macro"] for m, r in telco_out["test_results"].items()}
    common   = sorted(set(hotel_f1) & set(telco_f1))

    plot_generalization(
        hotel_results= {m: hotel_f1[m] for m in common},
        telco_results= {m: telco_f1[m] for m in common},
        metric       = "f1_macro",
    )

    # AUC comparison
    hotel_auc = {m: r["roc_auc"] for m, r in hotel_results["test_results"].items()}
    telco_auc = {m: r["roc_auc"] for m, r in telco_out["test_results"].items()}
    plot_generalization(
        hotel_results= {m: hotel_auc[m] for m in common},
        telco_results= {m: telco_auc[m] for m in common},
        metric       = "roc_auc",
    )

    # Save combined table
    hotel_df = pd.DataFrame(hotel_results["test_results"]).T.add_prefix("hotel_")
    telco_df = pd.DataFrame(telco_out["test_results"]).T.add_prefix("telco_")
    combined = hotel_df.join(telco_df, how="outer")
    combined.to_csv(os.path.join(TABLE_DIR, "generalization_comparison.csv"))
    print(f"\n  Generalization table saved → {TABLE_DIR}/generalization_comparison.csv")

    return telco_out


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SEED EXPERIMENT  (mean ± sd over seeds; R2 Major6)
# ─────────────────────────────────────────────────────────────────────────────

def run_multiseed(dataset: str, seeds: list, epochs: int = None,
                  split_mode: str = "random", verbose: bool = True):
    """
    Repeat the full single-dataset experiment over several seeds and report
    mean ± sd per metric per model, plus seed-level paired t-tests of the
    proposed model against each baseline (n = #seeds). CV and ablation are
    skipped on repeats; each seed still saves its own prediction arrays for
    per-seed DeLong / bootstrap.
    """
    from scipy import stats as _st

    METRICS = ["accuracy", "f1_macro", "roc_auc", "pr_auc", "mcc",
               "g_mean", "brier_score", "tdl", "emp", "ece"]
    # per_seed[model][metric] = list over seeds
    per_seed = {}
    for s in seeds:
        out = run_experiment(
            dataset=dataset, epochs=epochs,
            skip_cv=True, skip_ablation=True, verbose=verbose,
            seed=s, split_mode=split_mode,
        )
        for model, res in out["test_results"].items():
            per_seed.setdefault(model, {m: [] for m in METRICS})
            for m in METRICS:
                if m in res:
                    per_seed[model][m].append(float(res[m]))

    # Mean ± sd table
    rows = []
    for model, md in per_seed.items():
        row = {"Model": model}
        for m in METRICS:
            vals = np.array(md[m], dtype=float)
            if vals.size:
                row[f"{m}_mean"] = round(vals.mean(), 4)
                row[f"{m}_std"]  = round(vals.std(ddof=1), 4) if vals.size > 1 else 0.0
                row[m] = f"{vals.mean():.4f}±{vals.std(ddof=1) if vals.size>1 else 0:.4f}"
        rows.append(row)
    ms_df = pd.DataFrame(rows)
    suffix = "" if split_mode == "random" else f"_{split_mode}"
    ms_path = os.path.join(TABLE_DIR, f"multiseed_{dataset}{suffix}.csv")
    ms_df.to_csv(ms_path, index=False)
    print(f"\n  Multi-seed summary ({len(seeds)} seeds) saved → {ms_path}")
    print(ms_df[["Model"] + [m for m in METRICS if m in ms_df.columns]].to_string(index=False))

    # Seed-level paired t-tests: proposed vs each baseline
    proposed = "Transformer"
    if proposed in per_seed and len(seeds) > 1:
        prows = []
        for model, md in per_seed.items():
            if model == proposed:
                continue
            for m in ("roc_auc", "f1_macro", "pr_auc"):
                a = np.array(per_seed[proposed][m]); b = np.array(md[m])
                if a.size == b.size and a.size > 1:
                    t, p = _st.ttest_rel(a, b)
                    prows.append({
                        "Baseline": model, "Metric": m,
                        "Proposed mean": round(a.mean(), 4),
                        "Baseline mean": round(b.mean(), 4),
                        "Mean diff": round((a - b).mean(), 4),
                        "t": round(float(t), 3), "p (paired t, n=%d)" % a.size: float(p),
                    })
        pt_df = pd.DataFrame(prows)
        pt_path = os.path.join(TABLE_DIR, f"multiseed_ttest_{dataset}{suffix}.csv")
        pt_df.to_csv(pt_path, index=False)
        print(f"\n  Seed-level paired t-tests saved → {pt_path}")
        print(pt_df.to_string(index=False))
    return per_seed


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer-based Churn Prediction Experiment"
    )
    parser.add_argument(
        "--dataset", type=str, default="both",
        choices=["hotel", "telco", "bank", "datathon", "both", "all"],
        help="Dataset to use for experiments (default: both)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick debug mode: 10 epochs, skip CV and ablation"
    )
    parser.add_argument(
        "--skip_cv", action="store_true",
        help="Skip cross-validation (much faster)"
    )
    parser.add_argument(
        "--skip_ablation", action="store_true",
        help="Skip ablation study"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce verbosity"
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds for multi-seed runs, e.g. 42,1,2,3,4. "
             "When given, runs mean±sd multi-seed instead of a single run."
    )
    parser.add_argument(
        "--split_mode", type=str, default="random",
        choices=["random", "temporal", "hotelwise"],
        help="Data split: random (default), temporal (hotel time-ordered holdout), "
             "or hotelwise (train City Hotel, test unseen Resort Hotel)."
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.quick:
        args.epochs       = 10
        args.skip_cv      = True
        args.skip_ablation= True
        print("  [QUICK MODE] epochs=10, no CV, no ablation")

    verbose = not args.quiet

    t_start = time.time()

    # Create run directory and save run info
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,   exist_ok=True)
    run_info = {
        "run_id"        : RUN_ID,
        "dataset"       : args.dataset,
        "epochs"        : args.epochs or TRAIN["epochs"],
        "quick_mode"    : args.quick,
        "skip_cv"       : args.skip_cv,
        "skip_ablation" : args.skip_ablation,
        "transformer"   : TRANSFORMER,
        "train_cfg"     : {k: v for k, v in TRAIN.items()},
        "result_dir"    : RESULT_DIR,
    }
    with open(os.path.join(RESULT_DIR, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2, cls=_NumpyEncoder)

    # Print system info
    print(f"\n  Run ID   : {RUN_ID}")
    print(f"  Results  : {RESULT_DIR}")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
    print(f"  Device   : {DEVICE}")

    # ── Multi-seed path ──────────────────────────────────────────────────────
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
        datasets = (["hotel", "telco", "bank"] if args.dataset == "all"
                    else ["hotel", "telco"] if args.dataset == "both"
                    else [args.dataset])
        for ds in datasets:
            sm = args.split_mode if ds == "hotel" else "random"
            print(f"\n########## MULTI-SEED: {ds}  seeds={seeds}  split={sm} ##########")
            run_multiseed(ds, seeds, epochs=args.epochs, split_mode=sm, verbose=verbose)
        t_total = time.time() - t_start
        print(f"\n  MULTI-SEED COMPLETE  |  Total time: {t_total/60:.1f} min")
        return

    hotel_out = None
    telco_out = None
    bank_out  = None

    if args.dataset in ("hotel", "both", "all"):
        hotel_out = run_experiment(
            dataset       = "hotel",
            epochs        = args.epochs,
            skip_cv       = args.skip_cv,
            skip_ablation = args.skip_ablation,
            verbose       = verbose,
            split_mode    = args.split_mode,   # temporal / hotelwise honoured here
        )

    if args.dataset in ("telco", "both", "all"):
        if args.dataset in ("both", "all") and hotel_out is not None:
            telco_out = run_generalization(hotel_out, epochs=args.epochs, verbose=verbose)
        else:
            telco_out = run_experiment(
                dataset       = "telco",
                epochs        = args.epochs,
                skip_cv       = args.skip_cv,
                skip_ablation = args.skip_ablation,
                verbose       = verbose,
            )

    if args.dataset in ("bank", "all"):
        bank_out = run_experiment(
            dataset       = "bank",
            epochs        = args.epochs,
            skip_cv       = args.skip_cv,
            skip_ablation = True,
            verbose       = verbose,
        )

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE  |  Total time: {t_total/60:.1f} min")
    print(f"  Results  → {TABLE_DIR}")
    print(f"  Figures  → {FIG_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
