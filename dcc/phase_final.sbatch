#!/bin/bash
#SBATCH --job-name=mp_final
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/final_%j.out
# Final analysis, figures and manuscript (CPU). Runs after the image models (04) and BiomedBERT (05) finish:
#   Phase 1 statistics -> Phase 2 mechanics -> plasmode design analysis -> figures -> manuscript.
# Stops before the analysis if a branch is incomplete, so that the pre-specified primary model is never replaced.
source dcc/config.sh
source activate "$CONDA_ENV"
export MPLBACKEND=Agg                     # jobs submitted from a Jupyter session inherit its inline backend
cd "$MECHANOPATH_ROOT"
# Upstream tasks that resubmitted themselves are not covered by the original dependency: run after them
UP=$(squeue -u "$USER" -h -n mp_feat,mp_feat_phikon2,mp_feat_hipt,mp_mil,mp_mil_phikon2,mp_mil_hipt,mp_bert -o %A \
     | sort -u | paste -sd: -)
if [ -n "$UP" ] && [ "${MP_WAIT:-0}" -lt 8 ]; then
  echo "upstream jobs still queued or running ($UP); resubmitting the final analysis after them"
  MP_WAIT=$((${MP_WAIT:-0} + 1)) sbatch --dependency="afterany:$UP" dcc/phase_final.sbatch
  exit 0
fi
python dcc/04b_merge_mil.py phikon2 hipt || { echo "MISSING image-model array tasks - rerun dcc/04_train_mil.sbatch"; exit 1; }
nb=$(ls results/bert/report_oof_bert_*.csv 2>/dev/null | wc -l)
[ -s results/image_oof_phikon2.csv ] || { echo "MISSING results/image_oof_phikon2.csv - run 04_train_mil first"; exit 1; }
[ "$nb" -ge 8 ] || { echo "MISSING BiomedBERT outputs ($nb of 8 tasks) - rerun dcc/05_report_bert.sbatch"; exit 1; }
for f in results/bert/report_folds_bert_*.csv; do
  n=$(cut -d, -f2,3 "$f" | sort -u | wc -l)
  [ "$n" -ge 26 ] || echo "WARNING $f has $((n - 1)) of 25 folds"
done
set -e
python dcc/06_phase1_analysis.py
python src/13_mechanics_phase2.py
python src/14_design_plasmode.py --reps 500 --grid-reps 300
python dcc/08_figures.py
python dcc/09_manuscript.py
python - <<'PY'
import glob, zipfile
pats = ["report/MechanoPath_manuscript.docx", "report/MechanoPath_manuscript.md", "figures/manuscript/*",
        "results/phase1/*", "results/phase2/*", "results/image_metrics_*.csv", "results/image_fold_log_*.csv",
        "results/bert/report_fold_log_bert_*.csv", "results/tissue_mask_validation.csv", "results/slide_tiles_*.csv"]
with zipfile.ZipFile("report/MechanoPath_results_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            z.write(f)
PY
echo "bundle: $MECHANOPATH_ROOT/report/MechanoPath_results_bundle.zip"
echo FINAL_DONE
