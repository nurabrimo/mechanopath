#!/bin/bash
#SBATCH --job-name=mp_mil
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --requeue
#SBATCH --exclude=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-majoroslab-gpu-ferc-a-r32-34
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/mil_%A_%a.out
# Patient-level image models (mean-pool LR and attention MIL) on full-resolution embeddings: one array task per
# cohort x CV scheme; task 1 (matched cohort, institutions held out) also fits the full-data model for the attention
# maps. dcc/04b_merge_mil.py (run by dcc/phase_final.sbatch) combines the four parts.
#   ENCODER=phikon2 sbatch -J mp_mil_phikon2 dcc/04_train_mil.sbatch      (or ENCODER=hipt, -J mp_mil_hipt)
source dcc/config.sh
source activate "$CONDA_ENV"
# A node whose GPU cannot be initialised fails fast; the task resubmits itself with that node excluded (max 4 times)
EXCL_BASE=dcc-carlsonlab-gpu-ferc-s-h36-24,dcc-majoroslab-gpu-ferc-a-r32-34
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)"; then
  echo "NO_GPU on $(hostname)"
  R=${MP_RETRY:-0}
  if [ "$R" -lt 4 ]; then
    EX="${MP_EXCLUDE:-$EXCL_BASE},$(hostname)"
    cd "$MECHANOPATH_ROOT" && MP_RETRY=$((R + 1)) MP_EXCLUDE="$EX" \
      sbatch -J "$SLURM_JOB_NAME" --array="$SLURM_ARRAY_TASK_ID" --exclude="$EX" "$MECHANOPATH_ROOT/dcc/04_train_mil.sbatch"
  fi
  exit 3
fi
cd "$MECHANOPATH_ROOT"
# Feature tasks that resubmitted themselves (NO_GPU) are not covered by the original dependency: wait for them
FEATJ=$(squeue -u "$USER" -h -n "mp_feat_${ENCODER:-phikon2}" -o %A | sort -u | paste -sd: -)
if [ -n "$FEATJ" ] && [ "${MP_WAIT:-0}" -lt 8 ]; then
  echo "feature extraction for ${ENCODER:-phikon2} still queued or running ($FEATJ); resubmitting this task after it"
  MP_WAIT=$((${MP_WAIT:-0} + 1)) sbatch -J "$SLURM_JOB_NAME" --array="$SLURM_ARRAY_TASK_ID" --dependency="afterany:$FEATJ" \
    dcc/04_train_mil.sbatch
  exit 0
fi
COMBOS=("all:patient" "all:institution" "x40:patient" "x40:institution")
I=${SLURM_ARRAY_TASK_ID:-1}
ATT=0; [ "$I" = "1" ] && ATT=1
echo "encoder ${ENCODER:-phikon2} | ${COMBOS[$I]} | attention $ATT | $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python dcc/04_train_mil.py --encoder "${ENCODER:-phikon2}" --max-instances 4096 --abmil-repeats 5 \
    --combos "${COMBOS[$I]}" --part "$I" --attention "$ATT"
