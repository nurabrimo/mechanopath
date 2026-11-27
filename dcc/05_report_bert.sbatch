#!/bin/bash
#SBATCH --job-name=mp_bert
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --requeue
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-majoroslab-gpu-ferc-a-r32-34
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/bert_%A_%a.out
# BiomedBERT on masked reports: 8 tasks = {M2, M1} x {all, x40} x {institution, patient}; 25 fine-tunes each.
# Finished folds are kept on disk, so a pre-empted (scavenger) task resumes where it stopped.
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
      sbatch --array="$SLURM_ARRAY_TASK_ID" --exclude="$EX" "$MECHANOPATH_ROOT/dcc/05_report_bert.sbatch"
  fi
  exit 3
fi
export HF_HUB_OFFLINE=1
cd "$MECHANOPATH_ROOT"
python dcc/05_report_bert.py --task-id "${SLURM_ARRAY_TASK_ID:-0}"
