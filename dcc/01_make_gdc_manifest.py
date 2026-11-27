"""Write a gdc-client manifest for TCGA-GBM/LGG FFPE diagnostic slides.

  python dcc/01_make_gdc_manifest.py --out $MECHANOPATH_ROOT/data/gdc_manifest_dx.txt [--one-per-patient]

--one-per-patient reproduces the slide choice used in this project (primary tumour '01',
lowest DX number). Without it, every diagnostic slide is listed (1,704 files, ~1.4 TB);
the MIL script pools all slides of a patient into one bag.
"""
import argparse
import json
import re
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--one-per-patient", action="store_true")
a = ap.parse_args()

filters = {"op": "and", "content": [
    {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-GBM", "TCGA-LGG"]}},
    {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
    {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}}]}
params = urllib.parse.urlencode({"filters": json.dumps(filters), "size": 5000, "format": "json",
                                 "fields": "file_id,file_name,md5sum,file_size,state,cases.submitter_id"})
hits = json.load(urllib.request.urlopen("https://api.gdc.cancer.gov/files?" + params))["data"]["hits"]
rows = []
for h in hits:
    m = re.match(r"^TCGA-\w\w-\w{4}-(\d\d)\w-\d\d-DX(\d+)", h["file_name"])
    rows.append({**h, "patient": h["cases"][0]["submitter_id"], "stype": m.group(1) if m else "NA",
                 "dx": int(m.group(2)) if m else 99})
if a.one_per_patient:
    best = {}
    for r in rows:
        score = (0 if r["stype"] == "01" else 1000) + r["dx"]
        if r["patient"] not in best or score < best[r["patient"]][0]:
            best[r["patient"]] = (score, r)
    rows = [v[1] for v in best.values()]
with open(a.out, "w") as f:
    f.write("id\tfilename\tmd5\tsize\tstate\n")
    for r in rows:
        f.write(f"{r['file_id']}\t{r['file_name']}\t{r['md5sum']}\t{r['file_size']}\t{r.get('state', 'released')}\n")
print(f"wrote {len(rows)} files, {sum(r['file_size'] for r in rows) / 1e12:.2f} TB -> {a.out}")
