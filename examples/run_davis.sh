#!/usr/bin/env bash
# ================================
# Run DAVIS train + val splits
# ================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ttt3r

cd /home/xiaoang/FF_Reconstruction/categorize_motion_level || exit 1

echo "[INFO] Running DAVIS train split..."
python scripts/run_split.py --config configs/davis_train.json

echo "[INFO] Running DAVIS val split..."
python scripts/run_split.py --config configs/davis_val.json

echo "[DONE] All splits processed."
