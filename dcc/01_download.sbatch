#!/bin/bash
#SBATCH --job-name=mp_download
#SBATCH --partition=common
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/download_%j.out
source dcc/config.sh
python3 dcc/01_make_gdc_manifest.py --out "$MECHANOPATH_ROOT/data/gdc_manifest_dx.txt" --one-per-patient
"$MECHANOPATH_ROOT/bin/gdc-client" download -m "$MECHANOPATH_ROOT/data/gdc_manifest_dx.txt" -d "$WSI_DIR" -n 8 --retry-amount 5 --wait-time 10
find "$WSI_DIR" -name "*.svs" | sort > "$MECHANOPATH_ROOT/data/wsi_list.txt"
echo "slides downloaded: $(wc -l < "$MECHANOPATH_ROOT/data/wsi_list.txt")"
echo DOWNLOAD_DONE
