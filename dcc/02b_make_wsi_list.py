"""List the slides whose download is complete (file size equals the size in the GDC manifest).

  source dcc/config.sh && python dcc/02b_make_wsi_list.py

Writes data/wsi_list.txt (one .svs path per line). Partial files from a running gdc-client
download have a smaller size and are left out, so tiling can start before the download ends.
"""
import csv
import os
from pathlib import Path

root = Path(os.environ["MECHANOPATH_ROOT"])
wsi_dir = Path(os.environ["WSI_DIR"])
rows = list(csv.DictReader(open(root / "data" / "gdc_manifest_dx.txt"), delimiter="\t"))
done, partial, missing = [], 0, 0
for r in rows:
    p = wsi_dir / r["id"] / r["filename"]
    if not p.exists():
        missing += 1
    elif p.stat().st_size == int(r["size"]):
        done.append(str(p))
    else:
        partial += 1
(root / "data" / "wsi_list.txt").write_text("".join(f"{p}\n" for p in sorted(done)))
print(f"manifest {len(rows)} | complete {len(done)} | partial {partial} | not yet downloaded {missing}")
