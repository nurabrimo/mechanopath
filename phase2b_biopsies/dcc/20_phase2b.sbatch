#!/bin/bash
#SBATCH -J mp_p2b
#SBATCH -p common
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /work/%u/mechanopath/logs/p2b_%j.out
# Phase 2b: tumour versus matched healthy human biopsies (RT-FDC), full analysis of the compact DCOR table.
#   sbatch dcc/20_phase2b.sbatch
# Needs $OUT_DIR/dcor_compact/{dcor_events_compact.csv.gz,dcor_measurements_full.csv} (dcc/17_dcor_compact.py).
source /work/$USER/mechanopath/code/dcc/config.sh
python -c "import statsmodels, joblib" 2>/dev/null || pip install -q statsmodels joblib
export DCOR_COMPACT=$OUT_DIR/dcor_compact
python -I src/15_mech_biopsy_dcor.py --cells 1000 --bags 12 --perm 1000 --permanova-perm 20000 --jobs "${SLURM_CPUS_PER_TASK:-8}"
python src/17_sim_params.py biopsy
tar -C results -czf "$OUT_DIR/mechanopath_phase2b_results.tar.gz" phase2b
sha256sum "$OUT_DIR/mechanopath_phase2b_results.tar.gz"
