# HI-FTT: Hospitality-Informed Feature-Tokenizer Transformer

Reproducibility package for the manuscript *"HI-FTT: A Hospitality-Informed
Feature-Tokenizer Transformer for Interpretable Hotel Booking-Cancellation Churn
Prediction"* (under review).

**This repository is anonymised for double-blind peer review.** It contains the
code, the exact data splits, and the per-model test-set prediction arrays needed to
reproduce every table and statistical test in the paper.

## What HI-FTT is

HI-FTT extends the FT-Transformer with two domain modules integrated into the forward
pass:

- **BPAC** (Booking-Phase Aware Conditioning): learnable soft lead-time phase
  boundaries that condition all feature tokens via a multiplicative gate.
- **SGA** (Semantic Group Attention): two-level hierarchical attention over five
  CRM-meaningful feature groups, producing inter-group importance weights that form a
  native CRM risk dashboard.

The honest headline result: HI-FTT **significantly improves the vanilla
FT-Transformer backbone** on ROC-AUC/PR-AUC on the hotel dataset (DeLong Bonferroni
p < 8e-5; five-seed paired t-test p = 0.012), while gradient-boosted trees (XGBoost,
LightGBM) remain the strongest tabular baselines. The domain gain is
hospitality-specific and absent on Telco/Bank.

## Layout

```
src/                     model, data preprocessing, training, evaluation (DeLong,
                         paired bootstrap), ablation, faithfulness, visualization
main.py                  experiment orchestrator (single-run + multi-seed)
data/                    the three public churn datasets used
figures/                 the exact figures used in the paper (regenerable via main.py)
reproducibility/         released artifacts per experiment:
  hotel_multiseed/       5-seed hotel: multiseed_*.csv, per-seed test_results,
                         preds_*.npz (prediction arrays), auc_significance, faithfulness
  telco_multiseed/       5-seed telco
  bank_multiseed/        5-seed bank
  hotel_full/            single-seed hotel with ablation, CV, convergence, Friedman
  hotel_temporal/        temporal-holdout hotel
  hotel_hotelwise/       property-wise (City->Resort) holdout hotel
```

`preds_*.npz` files store `y_true` and per-model test probabilities
(`prob__<Model>`); these are the inputs to the DeLong and paired-bootstrap AUC tests
(`src/evaluation.py`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or your env of choice
pip install -r requirements.txt
```

Tested with Python 3.12/3.14, PyTorch 2.11 (CUDA 12.8), on an NVIDIA RTX 4050.

## Reproducing the results

Main five-seed comparison (mean +/- sd, DeLong + paired bootstrap, per dataset):

```bash
python main.py --dataset hotel --seeds 42,1,2,3,4
python main.py --dataset telco --seeds 42,1,2,3,4
python main.py --dataset bank  --seeds 42,1,2,3,4
```

Full single-seed hotel run (ablation + 5-fold CV + convergence + attention +
faithfulness):

```bash
python main.py --dataset hotel
```

Distribution-shift holdouts (hotel):

```bash
python main.py --dataset hotel --split_mode temporal  --skip_cv --skip_ablation
python main.py --dataset hotel --split_mode hotelwise --skip_cv --skip_ablation
```

Each run writes tables and prediction arrays under `results/runs/<timestamp>/tables/`.

## Datasets

- **Hotel Booking Demand** (`data/hotel_bookings.csv`) — Antonio, de Almeida & Nunes
  (2019), *Data in Brief* 22:41-49. Post-outcome fields (`reservation_status`,
  `reservation_status_date`) are dropped before any fit (see
  `src/data_preprocessing.py`).
- **IBM Telco Customer Churn** (`data/TelcoCustomerChurn.csv`). `SatisfactionScore`
  and other post-hoc/target-derived fields are dropped (leakage).
- **Bank Customer Churn** (`data/Bank Customer Churn Prediction.csv`, Kaggle).

## Notes on integrity

- SMOTE is applied to the training split only, after the train/val/test split.
- All AUC significance uses DeLong + paired stratified bootstrap on a shared test set
  (`src/evaluation.py: delong_roc_test`, `paired_bootstrap_auc`,
  `auc_significance_table`).
- Decision thresholds for the threshold-optimised tables are selected on validation.
