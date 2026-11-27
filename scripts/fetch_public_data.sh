#!/usr/bin/env bash
# Downloads the public inputs this repository does not store into data/raw/ (about 0.8 GB):
# TCGA pan-glioma labels, TCGA-Reports, HIPT code + weights, Mechanomics RT-DC tables.
# Slide tiles are separate: download mechanopath_gdc_tiles_partNN.zip from the GitHub release (or regenerate them
# with harvest/gdc_tile_harvester.js) and set UPLOADS in src/02_tiles.py to the folder that holds them.
set -euo pipefail
cd "$(dirname "$0")/.."
R=data/raw
get() { echo "  $(basename "$2")"; curl -fsSL --retry 3 -o "$2" "$1"; }

echo "1/4 TCGA pan-glioma labels (cBioPortal datahub, study lgggbm_tcga_pub)"
mkdir -p "$R/cbioportal/lgggbm_tcga_pub"
for f in data_clinical_patient.txt data_clinical_sample.txt; do
  get "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/lgggbm_tcga_pub/$f" "$R/cbioportal/lgggbm_tcga_pub/$f"
done

echo "2/4 TCGA-Reports (Kefeli & Tatonetti, MIT licence)"
mkdir -p "$R/tcga_reports/extracted"
get https://raw.githubusercontent.com/tatonetti-lab/tcga-path-reports/main/TCGA_Reports.csv.zip "$R/tcga_reports/TCGA_Reports.csv.zip"
unzip -o -q "$R/tcga_reports/TCGA_Reports.csv.zip" -d "$R/tcga_reports/extracted"

echo "3/4 HIPT ViT-S/16 code and DINO weights (non-commercial academic licence, 704 MB)"
mkdir -p "$R/models"
get https://raw.githubusercontent.com/mahmoodlab/HIPT/master/HIPT_4K/vision_transformer.py "$R/models/hipt_vision_transformer.py"
get https://raw.githubusercontent.com/mahmoodlab/HIPT/master/HIPT_4K/hipt_model_utils.py "$R/models/hipt_model_utils.py"
get https://media.githubusercontent.com/media/mahmoodlab/HIPT/master/HIPT_4K/Checkpoints/vit256_small_dino.pth "$R/models/vit256_small_dino.pth"

python3 - "$R" <<'PY'
import hashlib, sys
from pathlib import Path
R = Path(sys.argv[1])
expected = {"models/vit256_small_dino.pth": "6960cd5a8657dc8bb214671aa0c6dbd3f5b698e84386884955836487ddc89e24",
            "tcga_reports/TCGA_Reports.csv.zip": "bbbc0ee0c75d8feb0bb246b4fa40e8e29bba9775d4a9e4f41e0b73a7955e1c12"}
for rel, want in expected.items():
    h = hashlib.sha256()
    with open(R / rel, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    print(f"  {rel}: {'same file as the analysis' if h.hexdigest() == want else 'DIFFERENT from the file used here (upstream changed?)'}")
PY

echo "4/4 Mechanomics RT-DC Set 01, glioblastoma (figshare 10.6084/m9.figshare.28161236, CC BY 4.0)"
mkdir -p "$R/mechanics/tsv"
python3 - "$R/mechanics/tsv" <<'PY'
import io, json, sys, urllib.request, zipfile
from pathlib import Path
out, api, rec = Path(sys.argv[1]), "https://api.figshare.com/v2", 28161236
manual = (f"download MechSet01_Glio_tsv.zip from https://doi.org/10.6084/m9.figshare.{rec} and unzip its .tsv files "
          f"into {out}")

def fetch(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mechanopath-fetch"}), timeout=timeout).read()

try:
    try:
        files = json.loads(fetch(f"{api}/articles/{rec}/files"))
    except Exception:                                   # the DOI may point to a collection
        arts = json.loads(fetch(f"{api}/collections/{rec}/articles?page_size=100"))
        files = [f for a in arts for f in json.loads(fetch(f"{api}/articles/{a['id']}/files"))]
    zips = [f for f in files if f["name"].lower().endswith(".zip") and "tsv" in f["name"].lower()]
    if not zips:
        sys.exit("  no *_tsv.zip in the figshare record: " + manual)
    for f in zips:
        print("  " + f["name"])
        with zipfile.ZipFile(io.BytesIO(fetch(f["download_url"], timeout=600))) as z:
            for n in z.namelist():
                if n.lower().endswith(".tsv") and "__MACOSX" not in n:
                    (out / Path(n).name).write_bytes(z.read(n))
except Exception as e:
    sys.exit(f"  figshare download failed ({e}): " + manual)
print(f"  {len(list(out.glob('*.tsv')))} .tsv files (21 expected)")
PY
echo "done"
