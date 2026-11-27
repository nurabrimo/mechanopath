"""Step 0 - data inventory and patient-level linkage.

Builds data/processed/cohort.csv (one row per TCGA glioma patient) recording
which modalities exist for that patient, the label and its source, the
institution (tissue source site) and the WHO-2021-like molecular class, and
creates the shared cross-validation splits.

Inputs
  data/raw/cbioportal/lgggbm_tcga_pub/*          labels + molecular annotations
  data/raw/tcga_reports/extracted/TCGA_Reports.csv   TCGA-Reports (Kefeli et al.)
  data/raw/idc/idc_tcga_glioma_sm_series.csv      IDC slide index (cross-check)
  data/raw/tiles/gdc_manifest.json                GDC slide manifest + tissue source sites
Outputs
  data/processed/cohort.csv, data/processed/splits.csv, results/inventory.csv
"""
import json
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import PROC, RAW, RES, load_cbio_glioma, make_splits, who2021_like  # noqa: E402

lab = load_cbio_glioma()
lab["who2021_like"] = who2021_like(lab)

# --- reports -----------------------------------------------------------------
rep = pd.read_csv(RAW / "tcga_reports/extracted/TCGA_Reports.csv")
rep["patient"] = rep.patient_filename.str.split(".").str[0]
assert rep.patient.is_unique, "expected one report per patient"
lab["has_report"] = lab.patient.isin(rep.patient)

# --- slides (GDC manifest written by the tile harvester) --------------------
gm = json.load(open(RAW / "tiles/gdc_manifest.json"))
sel = pd.DataFrame(gm["selected"])
cases = pd.DataFrame(gm["cases"]).drop_duplicates("patient")
all_dx = pd.DataFrame(gm["all_dx_files"])
lab = lab.merge(cases[["patient", "tss_code", "tss_name"]].rename(columns={"tss_code": "gdc_tss_code"}),
                on="patient", how="left")
lab["institution"] = lab.tss_name.fillna("unknown")
lab["has_dx_slide"] = lab.patient.isin(sel.patient)
nslides = all_dx.groupby("patient").size()
lab["n_dx_slides"] = lab.patient.map(nslides).fillna(0).astype(int)
lab = lab.merge(sel[["patient", "file_id", "file_name"]].rename(columns={"file_id": "slide_file_id", "file_name": "slide_file_name"}),
                on="patient", how="left")

# cross-check project labels: GDC project vs cBioPortal study
chk = sel.merge(lab[["patient", "project"]], on="patient", suffixes=("_gdc", "_cbio"))
mismatch = (chk.project_gdc != chk.project_cbio).sum()
print("GDC vs cBioPortal project mismatches:", mismatch)

# IDC cross-check (same slides are mirrored in IDC as DICOM)
idc = pd.read_csv(RAW / "idc/idc_tcga_glioma_sm_series.csv")
idc_dx = set(idc[idc.SeriesDescription.str.contains("DX", na=False)].PatientID)
lab["in_idc_dx"] = lab.patient.isin(idc_dx)

lab["has_report_and_slide"] = lab.has_report & lab.has_dx_slide
lab["cell_mechanics"] = False   # no patient-linked whole-cell mechanics exist in any public source we found
cols = ["patient", "project", "y_gbm", "institution", "tss_code", "AGE", "SEX", "GRADE",
        "HISTOLOGICAL_DIAGNOSIS", "IDH_STATUS", "IDH_CODEL_SUBTYPE", "TERT_PROMOTER_STATUS",
        "CHR_7_GAIN_CHR_10_LOSS", "MGMT_PROMOTER_STATUS", "who2021_like", "has_report", "has_dx_slide",
        "n_dx_slides", "slide_file_id", "slide_file_name", "in_idc_dx", "has_report_and_slide", "cell_mechanics"]
cohort = lab[cols].copy()
cohort.to_csv(PROC / "cohort.csv", index=False)

inv = []
for name, m in [("labels (pan-glioma)", cohort.patient.notna()), ("pathology report", cohort.has_report),
                ("diagnostic slide", cohort.has_dx_slide), ("report + slide", cohort.has_report_and_slide),
                ("whole-cell mechanics", cohort.cell_mechanics)]:
    d = cohort[m]
    inv.append({"modality": name, "patients": len(d), "GBM": int(d.y_gbm.sum()), "LGG": int((1 - d.y_gbm).sum()),
                "institutions": d.institution.nunique()})
inv = pd.DataFrame(inv)
inv.to_csv(RES / "inventory.csv", index=False)
print(inv.to_string(index=False))

# institutions contributing both classes
it = cohort.groupby("institution").y_gbm.agg(n="size", gbm="sum")
it["lgg"] = it.n - it.gbm
it.sort_values("n", ascending=False).to_csv(RES / "institutions.csv")
print("mixed institutions:", (it[(it.gbm > 0) & (it.lgg > 0)]).to_string())

# shared CV splits over every labelled patient (subsets reuse the same assignment)
splits = make_splits(cohort, group_col="institution")
splits.to_csv(PROC / "splits.csv", index=False)
print("splits:", splits.groupby("scheme").size().to_dict())
