#!/bin/bash
# One-time setup on a DCC login node (run interactively, not as a batch job).
#   bash dcc/00_setup.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"

# 1. code: copy this whole mechanopath/ folder to $MECHANOPATH_ROOT (scp/rsync/git), then:
cd "$MECHANOPATH_ROOT"

# 2. conda environment (Miniconda in your group space keeps it off the 25 GB home quota)
if ! command -v conda >/dev/null; then
  echo "Install Miniconda under /hpc/group/$GROUP/$NETID/miniconda3 first:"
  echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -p /hpc/group/$GROUP/$NETID/miniconda3"
  exit 1
fi
conda env create -f dcc/environment.yml -n "$CONDA_ENV" || conda env update -f dcc/environment.yml -n "$CONDA_ENV"

# 3. GDC data transfer tool (single binary)
mkdir -p "$MECHANOPATH_ROOT/bin" && cd "$MECHANOPATH_ROOT/bin"
if [ ! -x gdc-client ]; then
  echo "Download the Ubuntu gdc-client zip from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool into $PWD and unzip it."
fi

# 4. Gated pathology encoders (optional but recommended): request access to MahmoodLab/UNI on
#    Hugging Face with your Duke account, then on DCC:
#      conda activate $CONDA_ENV && huggingface-cli login
echo "setup done"
