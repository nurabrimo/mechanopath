#!/bin/bash
#SBATCH --job-name=mp_fusion
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/fusion_%j.out
# Re-runs fusion + confounding checks with the full-resolution branches.
#   ENCODER=uni REPORT_MODEL_M2=bert_M2 sbatch dcc/06_fusion_and_checks.sbatch
source dcc/config.sh
source activate "$CONDA_ENV"
cd "$MECHANOPATH_ROOT"
python dcc/merge_dcc_outputs.py --encoder "${ENCODER:-uni}"
REPORT_MODEL_M2=${REPORT_MODEL_M2:-tfidf_M2} python src/05_fusion.py
python src/07_confounding.py
python src/08_power.py
