#!/bin/bash
#SBATCH --job-name=mp_tile
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=logs/tile_%A_%a.out
# Tissue mask + 256-px tile grid at 0.5 um/px for every slide in data/wsi_list.txt.
# 50 array tasks; each handles every 50th slide and skips slides that already have coordinates.
source dcc/config.sh
source activate "$CONDA_ENV"
mapfile -t WSIS < "$MECHANOPATH_ROOT/data/wsi_list.txt"
N=${SLURM_ARRAY_TASK_COUNT:-1}; I=${SLURM_ARRAY_TASK_ID:-0}
for ((i=I; i<${#WSIS[@]}; i+=N)); do
  stem=$(basename "${WSIS[$i]}"); stem=${stem%%.*}
  [ -f "$TILE_DIR/$stem.h5" ] && continue
  python dcc/02_tile_wsi.py --wsi "${WSIS[$i]}" --out-dir "$TILE_DIR" --tile 256 --mpp 0.5 --min-tissue 0.5 \
    || echo "TILE_FAILED $stem"
done
echo TILE_TASK_DONE
