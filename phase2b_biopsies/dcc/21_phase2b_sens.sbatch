#!/bin/bash
#SBATCH -J mp_p2b_sens
#SBATCH -p common
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /work/%u/mechanopath/logs/p2b_sens_%j.out
# Phase 2b sensitivity analyses and plotting tables (stages sens, figdata); run after dcc/20_phase2b.sbatch.
#   sbatch dcc/21_phase2b_sens.sbatch
source /work/$USER/mechanopath/code/dcc/config.sh
export DCOR_COMPACT=$OUT_DIR/dcor_compact
python -I src/15_mech_biopsy_dcor.py --stages sens,figdata --cells 1000 --bags 12 --jobs "${SLURM_CPUS_PER_TASK:-8}"
tar -C results -czf "$OUT_DIR/mechanopath_phase2b_results.tar.gz" phase2b
sha256sum "$OUT_DIR/mechanopath_phase2b_results.tar.gz"
