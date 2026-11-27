#!/bin/bash
# Phase 1 start-up on DCC, run once from the Jupyter session (compute node):
#   bash ~/mechanopath_full/dcc/step1_start.sh 2>&1 | tail -40
# 1) caches model weights so array jobs run offline, 2) lists completed downloads, 3) submits a first
# tiling + feature pass on those slides, the BiomedBERT report models, and the post-download chain.
set -uo pipefail
cd ~/mechanopath_full
source dcc/config.sh
source activate "$CONDA_ENV"
echo "== node $(hostname) | sbatch $(command -v sbatch) | python $(python -V 2>&1)"
pip install -q python-docx 2>&1 | grep -v -i warning | tail -1
python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
m = AutoModel.from_pretrained("owkin/phikon-v2"); print("phikon-v2 cached:", sum(p.numel() for p in m.parameters()) // 10**6, "M params")
bb = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
AutoTokenizer.from_pretrained(bb); AutoModelForSequenceClassification.from_pretrained(bb, num_labels=1); print("BiomedBERT cached")
PY
mkdir -p data/raw/models
M=data/raw/models
[ -s $M/vit256_small_dino.pth ] || curl -fsSL -o $M/vit256_small_dino.pth \
    https://media.githubusercontent.com/media/mahmoodlab/HIPT/master/HIPT_4K/Checkpoints/vit256_small_dino.pth
[ -s $M/hipt_vision_transformer.py ] || curl -fsSL -o $M/hipt_vision_transformer.py \
    https://raw.githubusercontent.com/mahmoodlab/HIPT/master/HIPT_4K/vision_transformer.py
echo "HIPT weights sha256 $(sha256sum $M/vit256_small_dino.pth | cut -c1-16) (expected 6960cd5a8657dc8b)"
python dcc/02b_make_wsi_list.py
T=$(sbatch --parsable dcc/02_tile_wsi.sbatch)
F1=$(ENCODER=phikon2 sbatch --parsable --dependency=afterany:$T dcc/03_extract_features.sbatch)
F2=$(ENCODER=hipt sbatch --parsable --dependency=afterany:$T dcc/03_extract_features.sbatch)
B=$(sbatch --parsable dcc/05_report_bert.sbatch)
DL=$(squeue -u "$USER" -h -n mp_download -o %i | head -1)
DEP="afterany:$F1:$F2"; [ -n "$DL" ] && DEP="afterany:$DL:$F1:$F2"
C=$(sbatch --parsable --dependency=$DEP dcc/phase1_after_download.sbatch)
echo "first pass: tile $T, features phikon2 $F1, hipt $F2 | BERT $B | download $DL | chain $C (after $DEP)" \
    | tee logs/job_ids_step1.txt
squeue -u "$USER" -o '%.10i %.12j %.10P %.8T %.10M %.6D %R' | head -30
echo STEP1_DONE
