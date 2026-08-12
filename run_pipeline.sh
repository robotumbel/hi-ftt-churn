#!/usr/bin/env bash
# Full experiment pipeline — runs 3 steps sequentially.
# Logs each step to a separate file under results/pipeline_logs/

set -e  # stop on first error

cd "$(dirname "$0")"

LOG_DIR="results/pipeline_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================"
echo " PIPELINE START  |  $(date)"
echo "========================================================"

# ── Step 1: All datasets, no CV, no ablation (fast — test set results + figures)
echo ""
echo "[STEP 1/3] All datasets — test set evaluation + figures"
echo "  Started: $(date)"
python -u main.py --dataset all --skip_cv --skip_ablation \
    2>&1 | tee "$LOG_DIR/step1_all_${TIMESTAMP}.log"
echo "  Finished: $(date)"

# ── Step 2: Hotel — full 5-fold CV + statistical tests (no ablation)
echo ""
echo "[STEP 2/3] Hotel — 5-fold CV + statistical tests"
echo "  Started: $(date)"
python -u main.py --dataset hotel --skip_ablation \
    2>&1 | tee "$LOG_DIR/step2_cv_${TIMESTAMP}.log"
echo "  Finished: $(date)"

# ── Step 3: Hotel — ablation study (no CV, faster)
echo ""
echo "[STEP 3/3] Hotel — ablation study"
echo "  Started: $(date)"
python -u main.py --dataset hotel --skip_cv \
    2>&1 | tee "$LOG_DIR/step3_ablation_${TIMESTAMP}.log"
echo "  Finished: $(date)"

echo ""
echo "========================================================"
echo " PIPELINE COMPLETE  |  $(date)"
echo " Logs -> $LOG_DIR"
echo "========================================================"
