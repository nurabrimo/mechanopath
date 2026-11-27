#!/bin/bash
#SBATCH --job-name=mp_feat
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --requeue
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-majoroslab-gpu-ferc-a-r32-34
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/feat_%A_%a.out
# Tile embeddings on GPU. 8 tasks share the slides that have tile coordinates; slides that already
# have features are skipped, so the job can be resubmitted (also on scavenger-gpu with --requeue).
#   ENCODER=phikon2 sbatch -J mp_feat_phikon2 dcc/03_extract_features.sbatch      (or ENCODER=hipt, -J mp_feat_hipt)
source dcc/config.sh
source activate "$CONDA_ENV"
# A node whose GPU cannot be initialised fails fast; the task resubmits itself with that node excluded (max 4 times)
EXCL_BASE=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-majoroslab-gpu-ferc-a-r32-34
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)"; then
  echo "NO_GPU on $(hostname)"
  R=${MP_RETRY:-0}
  if [ "$R" -lt 4 ]; then
    EX="${MP_EXCLUDE:-$EXCL_BASE},$(hostname)"
    cd "$MECHANOPATH_ROOT" && MP_RETRY=$((R + 1)) MP_EXCLUDE="$EX" NTASKS_TOTAL=${NTASKS_TOTAL:-$SLURM_ARRAY_TASK_COUNT} \
      sbatch -J "$SLURM_JOB_NAME" --array="$SLURM_ARRAY_TASK_ID" --exclude="$EX" "$MECHANOPATH_ROOT/dcc/03_extract_features.sbatch"
  fi
  exit 3
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}       # weights are pre-downloaded once; no parallel downloads
ENCODER=${ENCODER:-phikon2}
mapfile -t H5S < <(ls "$TILE_DIR"/*.h5 2>/dev/null | sort)
N=${NTASKS_TOTAL:-${SLURM_ARRAY_TASK_COUNT:-1}}; I=${SLURM_ARRAY_TASK_ID:-0}
MINE=()
for ((i=I; i<${#H5S[@]}; i+=N)); do MINE+=("${H5S[$i]}"); done
echo "encoder $ENCODER | task $I/$N | slides ${#MINE[@]} | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
[ ${#MINE[@]} -gt 0 ] && python dcc/03_extract_features.py --coords "${MINE[@]}" --encoder "$ENCODER" \
    --out-dir "$FEAT_DIR" --batch 128 --workers 7
echo FEAT_TASK_DONE
