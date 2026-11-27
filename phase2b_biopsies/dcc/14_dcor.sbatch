#!/bin/bash
#SBATCH -J mp_dcor
#SBATCH -p common
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --array=0-3
#SBATCH -o /work/%u/mechanopath/logs/dcor_%A_%a.out
# Scalar RT-FDC features of the DCOR human tumour/healthy biopsy measurements (no images); 4 shards.
#   sbatch dcc/14_dcor.sbatch ; afterwards (once all tasks finished):  python dcc/06_dcor_extract.py --out-dir $OUT_DIR/dcor --combine
source /work/$USER/mechanopath/code/dcc/config.sh
python dcc/06_dcor_extract.py --out-dir "$OUT_DIR/dcor" --max-events ${MAX_EVENTS:-40000} --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT"
