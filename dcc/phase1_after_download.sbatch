#!/bin/bash
#SBATCH --job-name=mp_chain
#SBATCH --partition=common
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/chain_%j.out
# Runs once the slide download (and the first feature pass) has finished: lists every complete slide,
# then tiles the remaining slides, extracts Phikon-v2 and HIPT embeddings, trains the image models and, after them
# and BiomedBERT, runs the final analysis, figures and manuscript (dcc/phase_final.sbatch).
# Each step skips slides that are already done; 'afterany' keeps one corrupt slide from blocking the chain.
source dcc/config.sh
source activate "$CONDA_ENV"
cd "$MECHANOPATH_ROOT"
python dcc/02b_make_wsi_list.py
T=$(sbatch --parsable dcc/02_tile_wsi.sbatch)
F1=$(ENCODER=phikon2 sbatch --parsable -J mp_feat_phikon2 --dependency=afterany:$T dcc/03_extract_features.sbatch)
F2=$(ENCODER=hipt sbatch --parsable -J mp_feat_hipt --dependency=afterany:$T dcc/03_extract_features.sbatch)
M1=$(ENCODER=phikon2 sbatch --parsable -J mp_mil_phikon2 --dependency=afterany:$F1 dcc/04_train_mil.sbatch)
M2=$(ENCODER=hipt sbatch --parsable -J mp_mil_hipt --dependency=afterany:$F2 dcc/04_train_mil.sbatch)
B=$(squeue -u "$USER" -h -n mp_bert -o %A | sort -u | paste -sd: -)          # BiomedBERT arrays still queued/running
DEP="afterany:$M1:$M2"; [ -n "$B" ] && DEP="$DEP:$B"
FIN=$(sbatch --parsable --dependency=$DEP dcc/phase_final.sbatch)
echo "tile $T | features phikon2 $F1, hipt $F2 | MIL phikon2 $M1, hipt $M2 | final $FIN (after $DEP)" \
    | tee logs/job_ids_chain.txt
echo CHAIN_SUBMITTED
