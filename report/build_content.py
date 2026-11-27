"""Builds report/blocks.json: the full MechanoPath technical report as a list of typed blocks
(headings, paragraphs, lists, tables, figures, code). Every number in tables and most numbers in
the text are read from results/ and data/processed/ so the document matches the analysis outputs.
Rendered to .docx by report/render_docx.js."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES, PROC, FIG = ROOT / "results", ROOT / "data" / "processed", ROOT / "figures"
NAT = FIG / "nature"                     # Nature-style figures from src/12_nature_figures.py
from figure_legends import LEGENDS  # noqa: E402

B = []                       # the document
fig_no = [0]
sfig_no = [0]
tab_no = [0]


def h1(t): B.append({"t": "h1", "text": t})
def h2(t): B.append({"t": "h2", "text": t})
def h3(t): B.append({"t": "h3", "text": t})
def p(t): B.append({"t": "p", "text": t})
def bullets(items): B.append({"t": "bullets", "items": items})
def numbered(items): B.append({"t": "numbered", "items": items})
def pagebreak(): B.append({"t": "pagebreak"})
def code(lines, title=None, size=7.5): B.append({"t": "code", "lines": lines, "title": title, "size": size})


def fig(name, supplementary=False, max_in=6.5):
    """Insert a Nature-style figure at its true print width (89-183 mm, capped at the 6.5-in text width).
    Title and legend come from src/figure_legends.py; Figures S1-S9 plot results otherwise shown as tables."""
    jpg = NAT / f"{name}_word.jpg"                # photographic figures are embedded as high-quality JPEG
    path = jpg if jpg.exists() else NAT / f"{name}_300dpi.png"
    im = Image.open(path)
    title, legend = LEGENDS[name]
    counter = sfig_no if supplementary else fig_no
    counter[0] += 1
    lab = f"Figure {'S' if supplementary else ''}{counter[0]}"
    B.append({"t": "img", "path": str(path), "w": im.size[0], "h": im.size[1], "w_in": min(max_in, im.size[0] / 300),
              "caption": f"**{lab} | {title}** {legend}".rstrip(), "alt": f"{lab}. {title} {legend}".strip()})
    return counter[0]


def table(header, rows, widths, caption, fs=8.5, num=()):
    tab_no[0] += 1
    B.append({"t": "table", "header": header, "rows": [[str(c) for c in r] for r in rows], "widths": widths,
              "caption": f"Table {tab_no[0]}. {caption}", "fs": fs, "num": list(num)})
    return tab_no[0]


NAME = {"img_meanpool": "Image, HIPT mean-pool", "img_abmil": "Image, attention MIL", "handcrafted": "Image, tissue features",
        "acq_meta": "Acquisition metadata only", "stain_color": "Stain colour + sharpness only",
        "tfidf_M0": "Report, unmasked M0", "tfidf_M1": "Report, masked M1", "tfidf_M2": "Report, masked M2", "age": "Age only",
        "image_meanpool": "Image, HIPT mean-pool", "image_abmil": "Image, attention MIL", "image_handcrafted": "Image, tissue features",
        "report_M1": "Report, masked M1", "report_M2": "Report, masked M2", "report_M0 (unmasked)": "Report, unmasked M0",
        "acq_meta (nuisance)": "Acquisition metadata only", "stain_color (nuisance)": "Stain colour + sharpness only",
        "late_image+report_M1": "Late fusion: image + report M1", "late_image+report_M2": "Late fusion: image + report M2",
        "late_image+report_M2+age": "Late fusion: image + report M2 + age", "late_image+acq_meta": "Late fusion: image + acquisition metadata",
        "early_image+report_M2": "Early fusion: image + report M2", "three_input_with_mechanics": "Three inputs incl. cell mechanics"}


def nm(x):
    return NAME.get(x, x)


def fnum(x):
    if pd.isna(x):
        return ""
    ax = abs(x)
    return f"{x:,.0f}" if ax >= 1000 else (f"{x:.1f}" if ax >= 10 else f"{x:.3f}")


def f3(x):
    return "" if pd.isna(x) else f"{x:.3f}"


def ci(r):
    return f"{r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]"


# ------------------------------------------------------------------ load results
inv = pd.read_csv(RES / "inventory.csv")
inst = pd.read_csv(RES / "institutions.csv")
rm = pd.read_csv(RES / "report_metrics.csv")
im = pd.read_csv(RES / "image_metrics.csv")
fm = pd.read_csv(RES / "fusion_metrics.csv")
fd = pd.read_csv(RES / "fusion_deltas.csv")
fs_ = pd.read_csv(RES / "fusion_subgroups.csv")
leak = pd.read_csv(RES / "report_leak_audit.csv")
laud = pd.read_csv(RES / "report_label_audit.csv")
t1 = pd.read_csv(RES / "report_top_terms_M1.csv", index_col=0)
t2 = pd.read_csv(RES / "report_top_terms_M2.csv", index_col=0)
scan = pd.read_csv(RES / "confounding_scanner_recognition.csv")
strat = pd.read_csv(RES / "confounding_stratified_auroc.csv")
uf = pd.read_csv(RES / "confounding_university_of_florida.csv")
lrt = pd.read_csv(RES / "confounding_lrt_x40.csv")
pm = pd.read_csv(RES / "mechanics_per_measurement.csv")
bag = pd.read_csv(RES / "mechanics_bag_classification.csv")
vc = pd.read_csv(RES / "mechanics_variance_components.csv")
pw = pd.read_csv(RES / "power_matched_cohort.csv")
co = pd.read_csv(PROC / "cohort.csv")
sl = pd.read_csv(PROC / "slides.csv")
tf = pd.read_csv(PROC / "tile_features.csv")
hp = pd.read_csv(PROC / "handcrafted_patient.csv")
ev = pd.read_csv(PROC / "rtdc_events_gated.csv.gz")


def R(df, model, cohort="all", scheme="patient"):
    r = df[(df.model == model) & (df.cohort == cohort) & (df.scheme == scheme)]
    return r.iloc[0] if len(r) else None


def D(base, model, cohort="all", scheme="institution"):
    r = fd[(fd.baseline == base) & (fd.model == model) & (fd.cohort == cohort) & (fd.scheme == scheme)].iloc[0]
    return r


def pfmt(x):
    return "<0.001" if x < 0.001 else f"{x:.3f}"


fus = R(fm, "late_image+report_M2", "all", "institution")
img_i = R(im, "img_meanpool", "all", "institution")
img_p = R(im, "img_meanpool", "all", "patient")
rep2_i = R(rm, "tfidf_M2", "all", "institution")
rep2_p = R(rm, "tfidf_M2", "all", "patient")
acq_p = R(im, "acq_meta", "all", "patient")
acq40_i = R(im, "acq_meta", "x40", "institution")
img40_i = R(im, "img_meanpool", "x40", "institution")
d_img = D("image_meanpool", "late_image+report_M2")
d_rep = D("report_M2", "late_image+report_M2")
n_match = int(fus.n); n_match_gbm = int(fus.n_gbm)

# ================================================================== TITLE
B.append({"t": "title", "title": "MechanoPath",
          "subtitle": "Glioma classification through histopathology, pathology reports and cell biomechanics",
          "lines": ["Real-data analysis, methods, results and reproducibility guide",
                    "Prepared for Nura Brimo, Department of Biomedical Engineering, Duke University",
                    "Analysis run October 8–9, 2026 · Report dated October 9, 2026"]})
B.append({"t": "toc"})
pagebreak()

# ================================================================== 1 SUMMARY
h1("1. Summary")
p(f"This report documents a full real-data run of the MechanoPath project on public TCGA glioma data: which data "
  f"were used, how they were obtained and linked, every analysis, all results, and how to rerun everything on a "
  f"laptop or on the Duke Compute Cluster (DCC). The target is glioblastoma (TCGA-GBM) versus lower-grade glioma "
  f"(TCGA-LGG).")
bullets([
    f"**Image + report fusion works across sites.** Averaging the image and masked-report predictions gives AUROC "
    f"{fus.auroc:.3f} [{fus.auroc_lo:.3f}, {fus.auroc_hi:.3f}] when whole institutions are held out "
    f"({n_match} patients, {n_match_gbm} GBM): +{d_img.delta_auroc:.3f} over images alone and +{d_rep.delta_auroc:.3f} "
    f"over reports alone (both p < 0.001).",
    f"**Random patient folds flatter every model.** The same models score {img_p.auroc:.3f} (image), "
    f"{rep2_p.auroc:.3f} (masked report) and {R(fm, 'late_image+report_M2', 'all', 'patient').auroc:.3f} (fusion) "
    f"when patients are split at random, because GBM and LGG cases came from different institutions, years and scanners.",
    f"**Scanner shortcut.** Every 20x-scanned slide is a GBM. Acquisition metadata alone (scan date, magnification, "
    f"JPEG quality, software, slide size) reaches {acq_p.auroc:.3f} with random folds. On 40x-scanned slides with "
    f"institutions held out it falls to {acq40_i.auroc:.3f}, while the image model keeps {img40_i.auroc:.3f}.",
    f"**Masked reports mostly carry site templates.** With diagnoses, grades, molecular results and grading criteria "
    f"masked, reports score {rep2_p.auroc:.3f} with random folds but {rep2_i.auroc:.3f} with institutions held out, "
    f"close to age alone ({R(rm, 'age', 'all', 'institution').auroc:.3f}).",
    "**Errors follow molecular class.** Tumours labelled lower-grade that are glioblastoma by WHO 2021 molecular "
    "criteria are called GBM more often than IDH-mutant astrocytomas.",
    "**Cell mechanics could not enter the classifier.** No public source links whole-cell mechanics to these "
    "patients. On public RT-DC data from three glioblastoma lines, growth-factor culture softened cells "
    "(EGF 0.80x, FGFJI 0.72x serum), and the spread of single-cell stiffness did not add information beyond the median; "
    "adding cell size and deformation did.",
    "**What the mechanics claim needs.** Showing a +0.03 AUROC gain over a 0.90 image + report baseline needs roughly "
    "150 matched patients per class (simulation, 80% power).",
])
rows = []
for lab, src, model in [("Report, unmasked (M0)", rm, "tfidf_M0"), ("Report, masked M1", rm, "tfidf_M1"),
                        ("Report, masked M2", rm, "tfidf_M2"), ("Age only", rm, "age"),
                        ("Image, HIPT mean-pool", im, "img_meanpool"), ("Image, attention MIL", im, "img_abmil"),
                        ("Image, hand-crafted tissue features", im, "handcrafted"),
                        ("Acquisition metadata only", im, "acq_meta"), ("Stain colour + sharpness only", im, "stain_color"),
                        ("Fusion, image + report M2 (late)", fm, "late_image+report_M2"),
                        ("Fusion, image + report M2 (early)", fm, "early_image+report_M2")]:
    vals = [f3(R(src, model, c, s).auroc) for c in ["all", "x40"] for s in ["patient", "institution"]]
    rows.append([lab] + vals)
table(["Model", "All slides: random folds", "All slides: institutions held out", "40x only: random folds",
       "40x only: institutions held out"], rows, [34, 16.5, 16.5, 16.5, 16.5],
      "Headline AUROCs. Branch rows use each branch's full cohort; fusion rows use the matched image + report cohort.",
      num=(1, 2, 3, 4))
pagebreak()

# ================================================================== 2 DESIGN
h1("2. Research question and study design")
p("**Question.** Does the mechanical behaviour of tumour cells add information for distinguishing glioblastoma from "
  "lower-grade gliomas beyond what pathology images and reports already reveal?")
p("**Hypothesis.** The distribution of mechanical responses across a tumour's cells (not only its average stiffness) "
  "carries diagnostic information that images and reports miss.")
p("**Three inputs.** Whole-slide H&E images (tissue architecture and cytology), pathology reports with the diagnosis "
  "masked (descriptive findings and clinical context), and whole-cell biomechanics (stiffness, deformability and their "
  "spread across cells). Nuclear mechanics is out of scope.")
p("**What a valid combined observation is.** Patient A's slide + Patient A's report + mechanics measured on cells from "
  "Patient A's tumour, ideally from the same surgery. A TCGA slide paired with unrelated cell-line measurements is not a "
  "matched observation, and assigning a \"typical GBM stiffness\" to GBM patients would put the label into the input.")
p("**Design decisions used throughout.**")
bullets([
    "One set of cross-validation splits for every model (data/processed/splits.csv), so image-only, report-only and "
    "fused models are compared on the same held-out patients.",
    "Two validation schemes: 5 x 5 repeated stratified patient-level folds, and 5 x 5 institution-grouped folds in which "
    "every patient of an institution sits in the same fold (whole institutions are held out).",
    "Two cohorts: all slides, and slides scanned at 40x only (every 20x-scanned slide is a GBM).",
    "Report masking at three levels, so the report branch cannot simply read the diagnosis.",
    "Nuisance-only baselines (acquisition metadata; stain colour and sharpness) to measure shortcuts directly.",
])
fig("fig0_pipeline", max_in=5.9)
pagebreak()

# ================================================================== 3 DATA
h1("3. Data")
h2("3.1 Sources")
table(["Input", "Source", "What was used", "Access and licence"], [
    ["Labels + molecular annotations", "TCGA pan-glioma study (Ceccarelli et al., Cell 2016) via cBioPortal study lgggbm_tcga_pub",
     "Patient project (GBM/LGG), age, sex, grade, histology, IDH, 1p/19q, TERT promoter, +7/-10, MGMT", "Public (cBioPortal datahub on GitHub)"],
    ["Pathology reports", "TCGA-Reports (Kefeli et al.), TCGA_Reports.csv", "OCR text of 9,523 TCGA reports; 868 glioma patients with labels",
     "Public (tatonetti-lab/tcga-path-reports on GitHub)"],
    ["Diagnostic slides", "NCI Genomic Data Commons (GDC), TCGA-GBM and TCGA-LGG FFPE DX slides",
     "One slide per patient (primary tumour, lowest DX number): thumbnail + 16 tiles + Aperio header", "Open access; GDC tile service, Data Release 46.0"],
    ["Slide index (cross-check)", "Imaging Data Commons index (idc-index-data 25.0.0)", "1,704 FFPE DX series, 880 patients", "Open"],
    ["Pathology encoder", "HIPT ViT-S/16, DINO-pretrained (Chen et al., CVPR 2022)", "vit256_small_dino.pth + model code", "GitHub LFS (mahmoodlab/HIPT)"],
    ["Whole-cell mechanics", "Mechanomics RT-DC Set 01 - Glioblastoma (Urbanska, Herbig, Guck)",
     "21 RT-DC measurements of lines X01, X04, X08 in serum, EGF or FGFJI", "figshare 10.6084/m9.figshare.28161236, CC BY 4.0"],
], [17, 30, 33, 20], "Data sources.", fs=8)
h2("3.2 Cohorts and linkage")
p("Patients were linked across sources by TCGA barcode (TCGA-XX-XXXX). GDC project labels and cBioPortal study labels "
  "agreed for every slide (0 mismatches).")
rows = [[r.modality, f"{r.patients:,}", r.GBM, r.LGG, r.institutions] for r in inv.itertuples()]
table(["Input", "Patients", "GBM", "LGG", "Institutions"], rows, [40, 15, 15, 15, 15],
      "Data inventory (patients with a label and the input). Analysed numbers are slightly lower: 875 slides had usable "
      "tiles and 686 patients had both a report and a usable slide.", num=(1, 2, 3, 4))
fig("fig1_inventory")
p("Five listed slides could not be used: the GDC tile service returned HTTP 404/400 for TCGA-02-0033, TCGA-DU-5871 and "
  "TCGA-E1-A7YI, and tissue detection found no tiles for TCGA-19-1388 and TCGA-19-1389. Three further patients have only "
  "20x tiles (no 5x tiles); their 5x features were imputed inside each training fold.")
h2("3.3 Labels and WHO 2021 re-annotation")
p("The prediction target is the TCGA project: TCGA-GBM (glioblastoma, WHO 2007 grade IV) versus TCGA-LGG (grade II-III "
  "astrocytoma, oligodendroglioma and oligoastrocytoma). For error analysis each tumour was also given a WHO-2021-like "
  "class from the pan-glioma annotations:")
ct = pd.crosstab(co.who2021_like, co.project)
rows = [[k, v.get("TCGA-GBM", 0), v.get("TCGA-LGG", 0)] for k, v in ct.iterrows()]
desc = {"GBM_IDHwt": "IDH-wild-type and (grade 4 or TERT promoter mutation or +7/-10): glioblastoma, IDH-wild-type",
        "Astro_IDHmut": "IDH-mutant, 1p/19q intact: astrocytoma, IDH-mutant",
        "Oligo_IDHmut_codel": "IDH-mutant, 1p/19q codeleted: oligodendroglioma",
        "IDHwt_other": "IDH-wild-type without the criteria above (or criteria unknown)",
        "unknown": "IDH status missing"}
rows = [[desc.get(r[0], r[0]), r[1], r[2]] for r in rows]
table(["WHO-2021-like class (rule)", "TCGA-GBM", "TCGA-LGG"], rows, [70, 15, 15],
      "WHO-2021-like re-annotation by project. EGFR amplification, a further GBM criterion, was not available here, so "
      "molecular GBM is under-called.", num=(1, 2))
h2("3.4 Institutions")
p(f"The 1,122 labelled patients came from {inv.iloc[0].institutions} institutions (tissue source sites from the GDC case "
  "records). Most sent only one class, because TCGA collected GBM and LGG as separate projects under different site "
  "codes and years. Only eight institutions contributed both classes: Henry Ford Hospital, Duke, Case Western, Thomas "
  "Jefferson University, Mayo Clinic - Rochester, University of Florida, Christiana Healthcare and the International "
  "Genomics Consortium.")
fig("fig13_institutions")
h2("3.5 Slide acquisition metadata")
p("For every slide the Aperio ImageDescription tag was read from the SVS header by HTTP range requests: apparent "
  "magnification, microns per pixel, scanner ID, scan date, JPEG quality and software version. Acquisition differs "
  "sharply by class.")
rows = []
for col, lab in [("appmag", "Scan magnification"), ("jpeg_q", "JPEG quality"), ("aperio_major", "Aperio software (major version)"),
                 ("scan_year", "Scan year")]:
    c = pd.crosstab(sl[col].fillna(-1), sl.project)
    for v, r in c.iterrows():
        vs = "missing" if v == -1 else (f"{int(v)}x" if col == "appmag" else str(int(v)))
        rows.append([lab, vs, r.get("TCGA-GBM", 0), r.get("TCGA-LGG", 0)])
table(["Variable", "Value", "GBM slides", "LGG slides"], rows, [40, 20, 20, 20],
      "Acquisition metadata by class (877 harvested slides).", num=(2, 3))
fig("fig2_acquisition_confounding")
h2("3.6 Cell-mechanics dataset")
p("Mechanomics RT-DC Set 01 contains real-time deformability cytometry of three primary glioblastoma lines (X01, X04, "
  "X08) grown in serum, EGF, or FGF + JAK inhibitor (FGFJI); these conditions give low, medium and high activation of "
  "the STAT3-Ser/Hes3 axis. All measurements used a 30 um channel at 0.16 uL/s. Young's modulus per cell comes from the "
  "exported .tsv (Shape-Out look-up table). After gating (finite modulus, porosity < 1.05, area >= 30 um2), "
  f"{len(ev):,} of 138,414 events remained.")
rows = [[r.date, r.line, r.cond, r.n_cells, f"{10 ** r.logE_median:.3f}", f"{r.logE_iqr:.3f}"]
        for r in pm.sort_values(["date", "line", "cond"]).itertuples()]
table(["Date", "Line", "Condition", "Gated cells", "Median E (kPa)", "IQR of log10 E"], rows, [16, 12, 16, 16, 20, 20],
      "Per-measurement summary of the RT-DC data.", fs=8, num=(3, 4, 5))
pagebreak()

# ================================================================== 4 HOW DATA WERE OBTAINED
h1("4. How the data were obtained")
p("The analysis ran in a cloud Linux workspace (2 CPU cores, 8 GB RAM, no GPU) whose network allowlist reached PyPI and "
  "GitHub but not GDC, figshare or Hugging Face. Data from those hosts were pulled through the user's Chrome browser "
  "with permission and saved to ~/Downloads, then copied into the workspace.")
h2("4.1 Slide tiles from the GDC tile service")
p("Downloading 880 whole SVS files (~0.73 TB) was not feasible, so a browser script (harvest/gdc_tile_harvester.js) "
  "pulled a fixed sample from every slide through GDC's public Deep Zoom tile endpoint (/tile/metadata/<id> and "
  "/tile/<id>?level=&x=&y=). For each slide it:")
numbered([
    "selected one FFPE diagnostic slide per patient (primary tumour code 01, lowest DX number) from the GDC /files API, "
    "and the tissue source site for every case from /cases;",
    "read the Aperio header with HTTP range requests (magnification, MPP, scanner, scan date, JPEG quality);",
    "stitched a thumbnail at ~1/64 of full resolution (<= 1,024 px) and built a tissue mask (saturation > 0.08, value > 0.2, "
    "green/blue ink filtered) plus a haematoxylin optical-density map (Ruifrok-Johnston colour deconvolution);",
    "chose the Deep Zoom level closest to 0.5 um/px (20x; 40x slides are read one level down) and listed full 512-px tiles "
    "with >= 75% tissue (relaxed to 50/25/5% when too few);",
    "sampled 6 tiles from the top 10% most haematoxylin-dense candidates and 6 uniformly at random (seeded by file ID), "
    "plus 2 + 2 tiles at ~2 um/px (5x);",
    "removed the 1-px Deep Zoom overlap, saved JPEG (quality 0.85) with a meta.json of level-0 coordinates, and wrote "
    "11 STORE-mode zip parts.",
])
p("The harvest took 49 minutes for 880 slides with 24 concurrent requests and produced 1.4 GB (877 slides, 13,966 tiles). "
  "Chrome blocks a page's second automatic download, so each part was saved from a fresh tab.")
fig("contact_TCGA-06-0156")
h2("4.2 Other downloads")
bullets([
    "TCGA-Reports (TCGA_Reports.csv.zip, 10.9 MB) and cBioPortal clinical files were fetched directly from GitHub.",
    "The IDC slide index came with the idc-index-data Python package (PyPI) and was used to cross-check slide counts.",
    "HIPT weights (vit256_small_dino.pth, 704 MB, SHA-256 6960cd5a...) came from GitHub LFS.",
    "Mechanomics MechSet01_Glio_tsv.zip (11.9 MB) and MechSet01_Glio_rtdc.zip (93.4 MB) were downloaded through Chrome.",
])
pagebreak()

# ================================================================== 5 METHODS
h1("5. Methods")
h2("5.1 Validation design and metrics")
bullets([
    "**Patient folds.** StratifiedKFold(5, shuffle) repeated 5 times with seeds 20261008-20261012 over all 1,122 labelled "
    "patients; every sub-cohort reuses the same assignment.",
    "**Institution folds.** StratifiedGroupKFold(5) by institution name, repeated 5 times; no institution is split across "
    "folds (verified).",
    "**Out-of-fold predictions.** Each patient's P(GBM) is the mean of its out-of-fold predictions over the repeats. "
    "Attention MIL and early fusion used 2 of the 5 repeats to fit the CPU budget.",
    "**Metrics.** AUROC with a 2,000-sample patient bootstrap 95% CI; macro-F1 at the training-fold Youden threshold "
    "(branch tables) or at 0.5 (fusion tables); Brier score; expected calibration error (10 bins).",
    "**Within-institution AUROC.** AUROC computed inside each institution with >= 2 patients per class, weighted by "
    "n_GBM x n_LGG (the probability a GBM ranks above an LGG from the same institution).",
    "**Paired comparisons.** DeLong test for correlated AUROCs and a paired bootstrap of the AUROC difference on the same "
    "patients.",
])
h2("5.2 Report branch")
p("Reports were lower-cased and whitespace-normalised. Masking (src/report_masking.py) replaces each masked span with the "
  "token xmaskx. Diagnosis nouns are also matched by edit distance (<= 2 for words of 10+ letters, <= 1 for 8-9) because "
  "the text is OCR output; hyphenated words are checked part by part.")
table(["Level", "What is masked"], [
    ["M0", "Nothing (leakage reference)."],
    ["M1", "Tumour-type names (glioblastoma, astrocytoma, oligodendroglioma, oligoastrocytoma, glioma, gliosarcoma, GBM), "
           "WHO grade statements and stray Roman-numeral grades, grade adjectives (anaplastic, high/low-grade, malignant, "
           "well-differentiated), diagnosis modifiers before a masked noun (diffuse, infiltrating, gemistocytic, fibrillary, "
           "small cell, giant cell...), lineage terms (oligodendroglial), ICD-O and SNOMED morphology codes (9440/3, M-94403), "
           "molecular findings (IDH, R132H, 1p/19q, codeletion, LOH, EGFR, vIII, TERT, ATRX, MGMT, CDKN2A, FISH, "
           "chromosome 7/10). The article 'an' is rewritten to 'a' so it cannot reveal the masked noun."],
    ["M2", "M1 plus the grading criteria and their negations: necrosis, pseudopalisading, microvascular/endothelial "
           "proliferation, glomeruloid, mitoses, Ki-67/MIB-1, labelling index, percentages, cellularity, pleomorphism, "
           "atypia; 'no xmaskx' and 'xmaskx is not seen' patterns."],
], [10, 90], "Report masking levels.", fs=8)
p("Example (a GBM report, first lines):")
ex = pd.read_csv(PROC / "reports_masked.csv")
r0 = ex[ex.y_gbm == 1].iloc[3]
code([("M1: " + r0.text_M1[:330]), ("M2: " + r0.text_M2[:330])], size=7.5)
p("Model: TfidfVectorizer (word 1-2-grams, min_df 3, max_df 0.9, sublinear tf, up to 30,000 features) + L2 "
  "LogisticRegressionCV (liblinear, C in {0.3, 1, 3, 10, 30, 100}, 3-fold inner CV on AUROC), fitted inside each training "
  "fold. Baselines: age only (logistic regression), and a keyword rule (\"glioblastoma\" or \"grade IV\" anywhere in the "
  "raw report). Audits: residual diagnosis words after masking, largest coefficients of a full-data fit, and reports "
  "whose text contradicts the project label.")
h2("5.3 Image branch")
bullets([
    "**Tile QC.** A tile passes when tissue fraction >= 0.5 (HSV saturation > 0.07, value > 0.15), blood fraction < 0.3 and "
    "Laplacian variance > 15.",
    "**Interpretable tissue features (per 20x tile).** Haematoxylin channel from skimage rgb2hed, Gaussian-smoothed, Otsu "
    "threshold within tissue, opening + hole filling, objects >= 12 um2 = nuclei. Features: nuclear area fraction, nuclei "
    "per mm2, median / IQR / 90th percentile nuclear area, eccentricity, solidity, GLCM contrast / homogeneity / energy / "
    "correlation (8 grey levels, distances 2 and 4, 4 angles), LBP entropy (P = 8, R = 1). At 5x: pale-eosin low-"
    "haematoxylin fraction (a crude necrosis/oedema proxy). Patient features: mean, SD and max over QC-passing tiles.",
    "**Nuisance image features.** Mean RGB, haematoxylin/eosin optical density, sharpness (Laplacian variance), tissue "
    "and blood fractions.",
    "**Deep features.** Each 512-px tile was split into four 256-px crops and embedded with HIPT ViT-S/16 (DINO teacher "
    "weights, input normalised with mean = std = 0.5): 55,864 crops x 384 features, 2.4 h on 2 CPU cores.",
    "**Mean-pool model.** Mean embedding of QC-passing 20x crops and of 5x crops (768 features) -> imputer -> standardise -> "
    "PCA(64) -> L2 LogisticRegressionCV (C from 1e-3 to 1e2, 5-fold inner CV), all inside each training fold.",
    "**Attention MIL.** Gated attention MIL (Ilse et al., 2018): instances = up to 64 crops per patient with 3 flags "
    "(5x, haematoxylin-dense stratum, failed QC); Linear(387 -> 128) + ReLU + dropout 0.25; gated attention (64 hidden); "
    "AdamW (lr 5e-4, weight decay 1e-2), batches of 32 bags, 25% instance dropout, up to 80 epochs with early stopping "
    "on a 15% validation split (patience 12).",
    "**Nuisance baselines.** Acquisition metadata (scan month, magnification, MPP, JPEG quality, software version, slide "
    "area, tissue area, thumbnail tissue fraction, file size, scanner ID one-hot) and stain colour + sharpness, each with L2 "
    "logistic regression.",
])
h2("5.4 Fusion")
p(f"Fusion used the matched cohort ({n_match} patients with a usable slide and a report). **Late fusion** averages the "
  "branch logits with equal weights; nothing is fitted, so nothing can overfit. **Early fusion** refits, inside each "
  "outer fold, one L2 logistic regression on the M2 TF-IDF matrix plus 64 image principal components (scaled by 0.3 so "
  "the dense block does not dominate). Variants add age as a third logit (late) or column (early).")
h2("5.5 Acquisition-confounding analyses")
bullets([
    "**Scanner recognition.** Among GBM slides only (both values occur there), cross-validated logistic regression "
    "predicts 20x scanning, JPEG quality 70 and scan before 2011 from the embedding or from tissue features.",
    "**Stratified AUROC.** AUROC inside institution x scan-year strata that contain both classes (5 strata, 149 "
    "patients), and inside JPEG-quality x software-version strata.",
    "**University of Florida.** 15 GBM and 15 LGG, all scanned in 2014 at 40x with JPEG quality 30.",
    "**Likelihood-ratio test.** On 40x slides: GBM ~ scan year + JPEG quality 70 + software version, with and without "
    "the out-of-fold model score (logit).",
])
h2("5.6 Mechanics analyses")
bullets([
    "**Per-sample features** (src/06_mechanics.py, sample_features): median, 10/25/75/90th percentiles, IQR, 90-10 range, "
    "MAD, skewness and kurtosis of log10 E; fraction of cells below 0.7 or above 1.3 kPa; median and IQR of deformation "
    "and area.",
    "**Mixed models.** (A) per-measurement log10 median E ~ condition + line + (1 | date); (B) single-cell log10 E ~ "
    "condition + line + date + (1 | measurement), up to 400 cells per measurement; (C) per-measurement IQR ~ condition + "
    "line + date + median (OLS). Variance components: partial R2 of condition, line and date for the medians.",
    "**Heterogeneity test (proxy).** Within each measurement, 40 bags of 50, 100 or 300 cells; features = median only, "
    "the stiffness distribution, or distribution + cell shape; logistic regression trained on two dates and tested on "
    "the third (leave-one-date-out) for Serum vs FGFJI, Serum vs EGF and EGF vs FGFJI.",
])
h2("5.7 Power simulation")
p("Binormal model: base score S ~ N(mu_b y, 1) with AUROC A0, mechanics score M ~ N(mu_m y, 1) independent of S given "
  "the class, combined score S + (mu_m/mu_b) M (the optimal weight), mu_m set so the combined AUROC is A0 + delta. For "
  "A0 in {0.85, 0.90, 0.95}, delta in {0.02, 0.03, 0.05, 0.08} and 25-400 patients per class, 300 simulated cohorts each; "
  "power = fraction with DeLong p < 0.05.")
h2("5.8 Checks on the analysis itself")
bullets([
    "Every AUROC in this report was recomputed independently from the saved out-of-fold prediction files (maximum "
    "absolute difference 0.000).",
    "Each patient appears exactly once per repeat and scheme; no institution is split across institution folds.",
    "The DeLong implementation reproduces scikit-learn AUROCs exactly.",
    "Tile sampling, QC and features use no labels; all scaling, PCA, imputation and hyperparameter search sit inside "
    "the training folds.",
    "The DCC tiling and feature scripts were run end to end on synthetic 20x and 40x pyramidal TIFF slides (same tissue "
    "embedded at both resolutions: median cosine similarity 0.999).",
])
pagebreak()

# ================================================================== 6 RESULTS
h1("6. Results")
p("All figures follow Nature's figure specifications and are also supplied as vector PDFs (editable text) and 450-dpi "
  "PNGs in figures/nature/. Figures S1-S9 plot results that the tables give as numbers; the values are the same.")
h2("6.1 Report branch")
rows = []
for model, lab in [("tfidf_M0", "TF-IDF, M0 unmasked"), ("tfidf_M1", "TF-IDF, M1"), ("tfidf_M2", "TF-IDF, M2"),
                   ("age", "Age only"), ("tfidf_M1+age", "TF-IDF M1 + age")]:
    for c in ["all", "x40"]:
        for s in ["patient", "institution"]:
            r = R(rm, model, c, s)
            rows.append([lab, c, s, int(r.n), ci(r), f3(r.auroc_within_institution), f3(r.macro_f1), f3(r.brier), f3(r.ece)])
table(["Model", "Cohort", "Folds", "n", "AUROC [95% CI]", "Within-inst. AUROC", "Macro-F1", "Brier", "ECE"], rows,
      [19, 7, 11, 6, 21, 10, 9, 8, 9], "Report-branch results.", fs=7.5, num=(3, 5, 6, 7, 8))
fig("figS1_report_roc", supplementary=True)
rr = rm[rm.model == "rule_M0"].iloc[0]
p(f"A keyword rule with no training already reaches AUROC {rr.auroc:.3f} (macro-F1 {rr.macro_f1:.3f}) on the raw "
  "reports, which is why masking is essential.")
fig("fig4_report_masking")
rows = []
for lv, t in [("M1", t1), ("M2", t2)]:
    g = t[t.direction == "GBM"].head(12).index.tolist()
    l_ = t[t.direction == "LGG"].head(12).index.tolist()
    rows.append([lv, ", ".join(g), ", ".join(l_)])
table(["Level", "Terms pushing towards GBM", "Terms pushing towards LGG"], rows, [8, 46, 46],
      "Largest logistic-regression coefficients of a full-data fit on masked reports.", fs=7.5)
p("Under M2 the GBM side is dominated by template phrases (birth date, date tumor, received in, cassette, histological "
  "diagnosis), and the LGG side by oligodendroglial morphology (perinuclear halos, round nuclei) and degree words "
  "(mild, moderate). Three leaks were found and closed during the audit before the final run: the article 'an' "
  "before a masked noun, SNOMED morphology codes such as M-94403, and Roman-numeral grades split by OCR line breaks. "
  f"After masking, {int(leak.reports_with_residual_dx_words.sum())} reports contain a residual diagnosis word.")
ga, la = laud.set_index("project").loc["TCGA-GBM"], laud.set_index("project").loc["TCGA-LGG"]
p(f"**Label audit.** {ga.mentions_gbm * 100:.1f}% of GBM reports name glioblastoma and {la.mentions_gbm * 100:.1f}% "
  f"of LGG reports mention it, mostly as a differential diagnosis. {ga.mentions_grade4 * 100:.1f}% of GBM reports state "
  f"grade IV explicitly; {la.mentions_grade23 * 100:.1f}% of LGG reports state grade II or III. Three LGG reports state "
  "glioblastoma or grade IV in the diagnosis line (TCGA-DU-6542, glioblastoma with oligodendroglial component; "
  "TCGA-DU-A76L, anaplastic oligodendroglioma 'grade IV/IV'; TCGA-TM-A84J). They were kept, as candidate label noise.")

h2("6.2 Image branch")
rows = []
for model, lab in [("img_meanpool", "HIPT mean-pool + LR"), ("img_abmil", "Gated attention MIL"),
                   ("handcrafted", "Tissue features + LR"), ("acq_meta", "Acquisition metadata"),
                   ("stain_color", "Stain colour + sharpness")]:
    for c in ["all", "x40"]:
        for s in ["patient", "institution"]:
            r = R(im, model, c, s)
            rows.append([lab, c, s, int(r.n), ci(r), f3(r.auroc_within_institution), f3(r.macro_f1), f3(r.brier), f3(r.ece)])
table(["Model", "Cohort", "Folds", "n", "AUROC [95% CI]", "Within-inst. AUROC", "Macro-F1", "Brier", "ECE"], rows,
      [19, 7, 11, 6, 21, 10, 9, 8, 9], "Image-branch results.", fs=7.5, num=(3, 5, 6, 7, 8))
fig("figS2_image_roc", supplementary=True)
qc_rate = tf.qc_ok.mean()
p(f"Tile QC: {qc_rate * 100:.1f}% of {len(tf):,} tiles passed ({(~tf.qc_ok & (tf.tissue_frac < 0.5)).sum()} failed on "
  f"tissue, {(~tf.qc_ok & (tf.blood_frac >= 0.3)).sum()} on blood, {(~tf.qc_ok & (tf.blur_lapvar <= 15)).sum()} on blur; "
  "a tile can fail on several). The mean-pool and attention models perform the same within error, so the simpler "
  "mean-pool model is used for fusion.")
fig("fig11_embedding_map")
d = hp.merge(co[["patient", "y_gbm"]], on="patient").merge(sl[["patient", "appmag"]], on="patient")
from sklearn.metrics import roc_auc_score  # noqa: E402
rows = []
for f, lab in [("x20_nuc_area_frac_mean", "Nuclear area fraction"), ("x20_nuc_density_per_mm2_mean", "Nuclei per mm2"),
               ("x20_nuc_area_iqr_um2_mean", "Nuclear area IQR (um2)"), ("x20_nuc_area_med_um2_mean", "Median nuclear area (um2)"),
               ("x20_glcm_contrast_mean", "GLCM contrast"), ("x20_lbp_entropy_mean", "LBP entropy"),
               ("x20_blur_lapvar_mean", "Sharpness (Laplacian variance)"), ("x20_E_mean_mean", "Eosin optical density")]:
    v = d[[f, "y_gbm", "appmag"]].dropna()
    g = v[v.y_gbm == 1]
    rows.append([lab, fnum(v[v.y_gbm == 1][f].median()), fnum(v[v.y_gbm == 0][f].median()),
                 f"{roc_auc_score(v.y_gbm, v[f]):.2f}", f"{roc_auc_score(v[v.appmag == 40].y_gbm, v[v.appmag == 40][f]):.2f}",
                 f"{roc_auc_score((g.appmag == 20).astype(int), g[f]):.2f}"])
table(["Feature (mean over 20x tiles)", "Median GBM", "Median LGG", "AUROC GBM vs LGG", "Same, 40x only",
       "AUROC 20x vs 40x within GBM"], rows, [30, 12, 12, 15, 13, 18],
      "Univariate behaviour of interpretable features. AUROC < 0.5 means higher in LGG; the last column near 0.5 means the "
      "feature does not track the scanner.", fs=7.5, num=(1, 2, 3, 4, 5))
fig("figS3_tissue_feature_aurocs", supplementary=True)
fig("fig12_tissue_features")
fig("fig7_attention_tiles")

h2("6.3 Acquisition confounding")
rows = []
for r in scan.itertuples():
    rows.append([r._1, r.features, r.n, r.n_pos, f"{r.auroc:.3f} [{r.lo:.3f}, {r.hi:.3f}]"])
table(["Target (GBM slides only)", "Features", "n", "Positives", "AUROC [95% CI]"], rows, [24, 36, 8, 12, 20],
      "Can the representation recognise acquisition? Cross-validated, among GBM slides only.", fs=7.5, num=(2, 3))
fig("figS4_scanner_recognition", supplementary=True)
rows = []
for r in strat[strat.branch.isin(["image", "fusion"]) | strat.model.isin(["tfidf_M2", "age"])].itertuples():
    rows.append([r.branch, nm(r.model), r.cohort, r.scheme, f3(r.auroc_overall), f3(r.auroc_within_institution_x_scanyear),
                 f3(r.auroc_within_jpegq_x_software)])
table(["Branch", "Model", "Cohort", "Folds", "AUROC overall", "Within institution x scan year", "Within JPEG x software"],
      rows, [10, 26, 8, 12, 14, 16, 14], "AUROC inside acquisition-homogeneous strata (5 institution x scan-year strata, "
      "149 patients).", fs=7, num=(4, 5, 6))
fig("figS5_stratified_auroc", supplementary=True)
rows = [[r.branch, nm(r.model), r.cohort, r.scheme, f"{r.auroc:.3f} [{r.lo:.3f}, {r.hi:.3f}]"]
        for r in uf[uf.scheme == "institution"].itertuples()]
table(["Branch", "Model", "Cohort", "Folds", "AUROC [95% CI]"], rows, [14, 34, 12, 16, 24],
      "University of Florida only (15 GBM vs 15 LGG, same year and scanner settings), institutions held out.", fs=7.5)
rows = [[r.branch, nm(r.model), r.scheme, int(r.n), f"{r.LR_chi2:.1f}", pfmt(r.p), f"{r.odds_ratio_per_logit:.2f}",
         f"{r.acq_only_auroc_in_sample:.3f}", f"{r.acq_plus_score_auroc_in_sample:.3f}"] for r in lrt.itertuples()]
table(["Branch", "Model", "Folds", "n", "LR chi2 (1 df)", "p", "OR per logit", "AUROC covariates only",
       "AUROC + score"], rows, [9, 21, 11, 6, 10, 8, 10, 12, 13],
      "Likelihood-ratio tests on 40x slides: does the out-of-fold score predict GBM after scan year, JPEG quality and "
      "software version?", fs=7, num=(3, 4, 5, 6, 7, 8))
fig("figS6_uf_lrt", supplementary=True)

h2("6.4 Image + report fusion")
rows = []
for model in ["image_meanpool", "image_abmil", "image_handcrafted", "report_M1", "report_M2", "report_M0 (unmasked)", "age",
              "acq_meta (nuisance)", "stain_color (nuisance)", "late_image+report_M1", "late_image+report_M2",
              "late_image+report_M2+age", "late_image+acq_meta", "early_image+report_M2"]:
    for c in ["all", "x40"]:
        r = R(fm, model, c, "institution")
        rp = R(fm, model, c, "patient")
        if r is None:
            continue
        rows.append([nm(model), c, int(r.n), f3(rp.auroc), ci(r), f3(r.auroc_within_institution), f3(r.macro_f1), f3(r.brier)])
rows.append([nm("three_input_with_mechanics"), "all / x40", 0, "-", "not evaluable", "-", "-", "-"])
table(["Model (matched cohort)", "Cohort", "n", "AUROC, random folds", "AUROC, institutions held out [95% CI]",
       "Within-inst.", "Macro-F1 @0.5", "Brier"], rows, [27, 7, 6, 11, 22, 9, 9, 9],
      "All models on the same matched patients.", fs=7, num=(2, 3, 5, 6, 7))
rows = []
for r in fd.itertuples():
    rows.append([r.cohort, r.scheme, nm(r.baseline), nm(r.model), f"{r.delta_auroc:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]",
                 pfmt(r.p_delong), pfmt(r.p_bootstrap)])
table(["Cohort", "Folds", "Baseline", "Compared model", "AUROC change [95% CI]", "p DeLong", "p bootstrap"], rows,
      [7, 11, 22, 25, 19, 8, 8], "Paired AUROC comparisons on the same patients.", fs=6.8, num=(4, 5, 6))
fig("figS7_fusion_random_rocs_deltas", supplementary=True)
fig("fig3_auroc_overview")
fig("fig9_roc_curves")
fig("fig10_calibration")
p("With institutions held out the fused model under-predicts GBM in the middle range, and macro-F1 at a 0.5 cut-off "
  f"drops from {R(fm, 'late_image+report_M2', 'all', 'patient').macro_f1:.2f} to {fus.macro_f1:.2f} at similar ranking: "
  "probabilities need recalibration at a new site before any threshold is used.")
fig("fig6_who2021_subgroups")
rows = []
sub = fs_[(fs_.cohort == "all") & (fs_.scheme == "institution")].rename(columns={"frac_called_gbm@0.5": "frac"})
for r in sub[sub.label != "grade"].sort_values(["label", "group", "model"]).itertuples():
    rows.append([r.label, r.group, nm(r.model), r.n, f"{r.mean_p_gbm:.3f}", f"{r.frac * 100:.1f}%"])
table(["Project label", "WHO-2021-like class", "Model", "n", "Mean P(GBM)", "Called GBM at 0.5"], rows,
      [12, 22, 26, 8, 14, 18], "Predictions by molecular group (all slides, institutions held out).", fs=7, num=(3, 4, 5))
gr = sub[sub.label == "grade"]
rows = [[r.group, nm(r.model), r.n, f"{r.mean_p_gbm:.3f}", f"{r.frac * 100:.1f}%"] for r in gr.sort_values(["model", "group"]).itertuples()]
table(["Grade", "Model", "n", "Mean P(GBM)", "Called GBM at 0.5"], rows, [12, 34, 10, 20, 24],
      "Predictions by histological grade (all slides, institutions held out).", fs=7.5, num=(2, 3, 4))
fig("figS8_molecular_grade", supplementary=True)

h2("6.5 Whole-cell mechanics")
fig("fig5_mechanics")
G = ev.copy()
pmA = pm.copy(); pmA["cond"] = pd.Categorical(pmA.cond, ["Serum", "EGF", "FGFJI"])
mA = smf.mixedlm("logE_median ~ C(cond) + C(line)", pmA, groups=pmA["date"]).fit(reml=True)
sub_ = pd.concat([g.sample(min(len(g), 400), random_state=20261008) for _, g in G.groupby("measurement")])
sub_["logE"] = np.log10(sub_.emodulus); sub_["cond"] = pd.Categorical(sub_.cond, ["Serum", "EGF", "FGFJI"])
mB = smf.mixedlm("logE ~ C(cond) + C(line) + C(date)", sub_, groups=sub_["measurement"]).fit(reml=True)
rows = []
for name, m in [("A: per-measurement median, (1|date)", mA), ("B: single cells, (1|measurement)", mB)]:
    for term in m.params.index:
        if term in ("Group Var",):
            continue
        ci_ = m.conf_int().loc[term]
        rows.append([name, term.replace("C(cond)[T.", "condition ").replace("C(line)[T.", "line ").replace("C(date)[T.", "date ").replace("]", ""),
                     f"{m.params[term]:+.3f}", f"[{ci_[0]:+.3f}, {ci_[1]:+.3f}]", pfmt(m.pvalues[term]),
                     f"{10 ** m.params[term]:.3f}" if term != "Intercept" else "-"])
table(["Model", "Term", "Coefficient (log10 E)", "95% CI", "p", "Fold change"], rows, [28, 22, 13, 18, 8, 11],
      "Mixed models of Young's modulus. Reference levels: serum, line X01, date 2014-11-26.", fs=7, num=(2, 3, 4, 5))
rows = [[r.term, f"{r.partial_R2:.3f}"] for r in vc.itertuples()]
table(["Term", "Partial R2 (per-measurement median)"], rows, [50, 50], "Variance components of the per-measurement "
      "medians.", fs=8, num=(1,))
p("Spread (IQR of log10 E) depended on measurement date (p = 0.02 and 0.04 for the two later dates) but not on "
  "condition once the median was in the model (p = 0.29). Spread is therefore batch-sensitive.")
rows = [[r.task.replace("_", " "), r.bag_cells, r.features.replace("_", " "), f"{r.bag_auroc:.3f}",
         f"{r.measurement_auroc:.3f}", r.n_measurements] for r in bag.sort_values(["task", "bag_cells", "features"]).itertuples()]
table(["Task", "Cells per bag", "Features", "Bag AUROC", "Measurement AUROC", "Measurements"], rows,
      [22, 12, 26, 13, 15, 12], "Heterogeneity test: classification of culture state from bags of cells, "
      "leave-one-date-out.", fs=7, num=(1, 3, 4, 5))
fig("figS9_mechanics_models", supplementary=True)

h2("6.6 Power for the three-input claim")
fig("fig8_power")
pv = pw.pivot_table(index=["auroc_image_report", "delta_auroc"], columns="n_per_class", values="power")
mech = pw.groupby(["auroc_image_report", "delta_auroc"]).mechanics_alone_auroc.first()
rows = [[f"{a:.2f}", f"+{dl:.2f}", f"{mech.loc[(a, dl)]:.3f}"] + [f"{pv.loc[(a, dl), n]:.2f}" for n in pv.columns]
        for (a, dl) in pv.index]
table(["Image + report AUROC", "Gain", "Mechanics alone needs AUROC"] + [f"n = {n}/class" for n in pv.columns], rows,
      [14, 9, 17, 12, 12, 12, 12, 12], "Simulated power (300 cohorts per cell).", fs=7.5, num=tuple(range(2, 8)))
pagebreak()

# ================================================================== 7 INTERPRETATION
h1("7. Interpretation and next steps")
p("**What the data answer.** Within TCGA, slides and masked reports each carry real information about glioblastoma "
  "versus lower-grade glioma, and they are complementary: fusion adds 0.05 AUROC over the image branch and 0.10 over "
  "the report branch on the same held-out institutions. The image signal survives every acquisition check (40x-only, "
  "within-stratum, the University of Florida subset and the likelihood-ratio test), and its errors follow biology: "
  "grade 2 < grade 3 < grade 4 in predicted probability, and molecular glioblastomas labelled lower-grade sit closer to GBM.")
p("**What they do not answer.** The central MechanoPath claim, that whole-cell mechanics adds information beyond images "
  "and reports, cannot be tested with public data. The mechanics branch was developed on glioblastoma lines only, so it "
  "also says nothing about GBM versus lower-grade cells. On that proxy, the spread of stiffness did not beat the "
  "median, and spread was more sensitive to measurement date than to biology, which argues for batch-balanced designs "
  "and for recording cell size and deformation alongside stiffness.")
p("**Recommended next steps.**")
numbered([
    "Run the DCC pipeline (Section 9.5): full-slide tiling, UNI features (not pretrained on TCGA), BiomedBERT on the M2 "
    "reports, then the same fusion and confounding checks. Report institution-held-out and 40x-only numbers as headlines.",
    "Validate externally: any non-TCGA glioma cohort with slides and reports (for example a Duke or Moffitt series) "
    "is worth more than further tuning on TCGA.",
    "Recalibrate per site, or report AUROC rather than thresholded metrics, until calibration transfers.",
    "Plan the matched mechanics cohort with both classes, about 150 patients per class for a +0.03 gain, one assay, "
    "balanced measurement dates, and cell size/deformation recorded (dcc/mechanics_feature_spec.md).",
    "For the Moffitt hackathon, check whether external data (GDC tiles, Mechanomics) may be used; the image-report "
    "pipeline runs unchanged on organiser-provided data once the cohort table and splits are rebuilt.",
])

# ================================================================== 8 LIMITATIONS
h1("8. Limitations and confidence")
table(["Claim", "Confidence (0-1)", "Basis and main threat"], [
    ["Image + report beats either alone across institutions", "0.85", f"Paired tests on {n_match} patients, repeated on 40x slides; TCGA only"],
    ["The image branch reads tissue, not only the scanner", "0.80", "0.87 on 40x slides with institutions held out vs 0.34 for metadata; residual JPEG and era differences remain"],
    ["Random-fold AUROCs overstate performance at a new site", "0.90", "Drops of 0.07-0.18 when institutions are held out; metadata alone reaches 0.93"],
    ["What survives report masking is mostly site template", "0.75", "Top terms are template words; masking rules were refined on these same reports"],
    ["Stiffness spread adds information beyond the median", "0.30", "Proxy task only: 3 GBM lines, 21 measurements, 3 dates"],
    ["Whole-cell mechanics improves GBM vs LGG classification", "untested", "0 patients with linked mechanics"],
], [36, 14, 50], "Confidence in the main claims.", fs=8)
bullets([
    "Labels are TCGA projects (WHO 2007 histology); 61 LGG-labelled tumours meet WHO 2021 molecular criteria for "
    "glioblastoma and 3 reports contradict their label.",
    "HIPT was self-supervised on TCGA slides (no labels), so the encoder may have seen these images. UNI, used in the "
    "DCC pipeline, excluded TCGA from pretraining.",
    "16 tiles per slide cover about 0.8 mm2 at 20x and 4 mm2 at 5x; the median slide holds over 100 mm2 of tissue.",
    "Masking rules were refined after inspecting coefficients on these reports, so undiscovered leaks may remain.",
    "Attention MIL and early fusion used 2 of 5 repeats; the CPU budget also limited the encoder choice.",
    "The GPU steps of the DCC pipeline have not been run on DCC yet; partition names and paths follow DCC documentation "
    "as of October 2026.",
])
pagebreak()

# ================================================================== 9 HOW TO RUN
h1("9. How to run the code")
h2("9.1 Package layout")
code(["mechanopath/",
      "  README.md                     overview, data sources, rerun summary",
      "  src/                          CPU analysis (every number in this report)",
      "    common.py                   paths, cohort/labels, shared CV splits, metrics, DeLong, bootstrap",
      "    00_inventory.py             linkage of labels, reports, slides -> cohort.csv, splits.csv",
      "    report_masking.py           M0/M1/M2 masking rules",
      "    01_reports.py               report branch + audits",
      "    02_tiles.py                 unpack tiles, QC, interpretable features",
      "    03_embed.py                 HIPT embeddings of every 256-px crop",
      "    04_image_models.py          mean-pool LR, attention MIL, tissue and nuisance baselines",
      "    05_fusion.py                late/early fusion on matched patients, paired tests, subgroups",
      "    06_mechanics.py             RT-DC mechanics branch",
      "    07_confounding.py           scanner recognition, stratified AUROC, UF subset, LRT",
      "    08_power.py                 matched-cohort power simulation",
      "    09_three_input.py           three-input test (needs matched mechanics)",
      "    10_figures.py, 11_report_figures.py   first-version figures (figures/*.png)",
      "    12_nature_figures.py        Nature-style figures and plots of tabled results (figures/nature/)",
      "    figure_legends.py           figure titles and legends (shared with the Word report)",
      "  harvest/gdc_tile_harvester.js browser script used to pull slide tiles from GDC",
      "  dcc/                          Duke Compute Cluster pipeline (SLURM)",
      "  data/processed/               cohort, splits, slide metadata, tile features, embeddings, masked reports",
      "  data/raw/                     cBioPortal files, IDC index extract, GDC manifest, HIPT model code",
      "  results/                      metric tables, out-of-fold predictions, audits, logs",
      "  report/                       Word report: build_report.sh runs build_content.py, render_docx.js, toc_pages.py, fix_bookmarks.py",
      "  figures/                      all figures; figures/nature/ holds the Nature-style PDF and PNG files"], size=7.5)
h2("9.2 Software")
p("Run here with Python 3.13.16, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, scikit-image 0.26.0, "
  "opencv 5.0.0, PyTorch 2.14.1 (CPU), statsmodels 0.15.0, dclab 0.73.1, umap-learn 0.5.12, matplotlib 3.11.2, "
  "Pillow 12.3.0, openslide-python 1.4.6 (OpenSlide 4.0.1), h5py 3.16.0. Install with:")
code(["pip install numpy pandas scikit-learn scipy scikit-image opencv-python torch torchvision statsmodels dclab \\",
      "            umap-learn matplotlib pillow pyarrow h5py openslide-python openslide-bin tifffile"])
h2("9.3 CPU pipeline, step by step")
p("Run from the mechanopath/ folder. Steps 2-3 expect the tile archives mechanopath_gdc_tiles_part01-11.zip in "
  "/mnt/user-data/uploads/Downloads/; edit UPLOADS in src/02_tiles.py to point at your Downloads folder instead. "
  "If you use the shipped embeddings, rebuild data/processed/embeddings_hipt.npz from the two parts and skip step 4.")
table(["Step", "Command", "Main outputs", "Time here (2 CPU cores)"], [
    ["1", "python src/00_inventory.py", "data/processed/cohort.csv, splits.csv; results/inventory.csv", "seconds"],
    ["2", "python src/01_reports.py", "results/report_*.csv, data/processed/reports_masked.csv", "25-50 min"],
    ["3", "python src/02_tiles.py", "slides.csv, tiles_index.csv, tile_features.csv, handcrafted_patient.csv", "~45 min"],
    ["4", "python src/03_embed.py", "data/processed/embeddings_hipt.npz", "2.4 h"],
    ["5", "python src/04_image_models.py", "results/image_oof.csv, image_metrics.csv, abmil_attention_full.csv", "~15 min"],
    ["6", "python src/05_fusion.py", "results/fusion_*.csv", "~20 min"],
    ["7", "python src/06_mechanics.py", "results/mechanics_*.csv, rtdc_events_gated.csv.gz", "<1 min"],
    ["8", "python src/07_confounding.py", "results/confounding_*.csv, acquisition_by_class.csv", "~4 min"],
    ["9", "python src/08_power.py", "results/power_matched_cohort.csv", "<1 min"],
    ["10", "python src/10_figures.py; python src/11_report_figures.py", "figures/*.png", "~1 min"],
    ["11", "python src/12_nature_figures.py", "figures/nature/*.pdf, *.png, figure_legends.md", "~1 min"],
    ["12", "bash report/build_report.sh", "report/MechanoPath_full_report.docx (two passes, so the contents list "
     "carries page numbers)", "~3 min"],
], [6, 33, 43, 18], "Running the CPU analysis.", fs=7.5)
p("The step 1 inventory needs data/raw/tiles/gdc_manifest.json (written by the harvester into every zip part) and the "
  "raw TCGA-Reports and cBioPortal files under data/raw/. Rebuilding the embedding file from the shipped parts:")
code(["import numpy as np",
      "a = np.load('mechanopath_embeddings_hipt_part1.npz'); b = np.load('mechanopath_embeddings_hipt_part2.npz')",
      "np.savez_compressed('data/processed/embeddings_hipt.npz', **{k: np.concatenate([a[k], b[k]]) for k in a.files})"])
h2("9.4 Re-harvesting slide tiles")
numbered([
    "Open https://api.gdc.cancer.gov/status in Chrome (same origin as the API, so no CORS problems).",
    "Allow automatic downloads for api.gdc.cancer.gov (address-bar icon, or Settings > Privacy > Site settings), "
    "otherwise only the first zip part is saved.",
    "Open DevTools > Console, paste harvest/gdc_tile_harvester.js and press Enter. Progress is in window.MP "
    "(done, failed, parts). The run is seeded, so the same tiles come back.",
    "Expect about 50 minutes and 11 parts (~1.4 GB). Any part that did not save can be re-downloaded from a fresh tab "
    "with its blob URL in window.MP.partUrls.",
])
h2("9.5 Duke Compute Cluster (full resolution)")
p("Edit dcc/config.sh (NETID, GROUP and storage paths), copy the whole mechanopath/ folder to /hpc/group/<group>/<netid>/, "
  "and run from there. DCC partitions used: common (CPU), gpu-common (GPU, 2-day limit; scavenger-gpu is preemptible). "
  "Large files go to /cwork/<netid> or /work/<netid>, which are purged after about 75-90 days; keep results in "
  "/hpc/group.")
table(["Step", "Command", "What it does", "Resources"], [
    ["0", "bash dcc/00_setup.sh", "conda env (dcc/environment.yml), gdc-client, huggingface-cli login for UNI", "login node"],
    ["1", "sbatch dcc/01_download.sbatch", "GDC manifest + download of all 1,704 DX slides (~1.4 TB); ONE_PER_PATIENT=1 for 880", "common, 8 CPU, up to 2 days"],
    ["2", "sbatch dcc/02_tile_wsi.sbatch", "tissue mask + 256-px tiles at 0.5 um/px for every slide (100-task array)", "common, 2 CPU each"],
    ["3", "ENCODER=uni sbatch dcc/03_extract_features.sbatch", "UNI (or hipt / resnet50_trunc) features per tile, 8 GPU tasks", "gpu-common, A5000"],
    ["4", "ENCODER=uni sbatch dcc/04_train_mil.sbatch", "mean-pool LR + gated attention MIL on all tiles, both cohorts and schemes", "1 GPU"],
    ["5", "LEVEL=M2 sbatch dcc/05_report_bert.sbatch", "BiomedBERT fine-tuning on masked reports (windows of 512 tokens)", "1 GPU, 3-6 h"],
    ["6", "ENCODER=uni REPORT_MODEL_M2=bert_M2 sbatch dcc/06_fusion_and_checks.sbatch", "merge outputs, rerun fusion, confounding and power", "common"],
], [6, 34, 42, 18], "Running the full-resolution pipeline on DCC. Times are planning estimates, not measurements.", fs=7.5)
p("Optional stain-robustness rerun: save a reference tile as dcc/stain_reference.png (an example tile is shipped as "
  "dcc/stain_reference_example.jpg) and run STAIN_NORM=macenko ENCODER=uni sbatch dcc/03_extract_features.sbatch.")
h2("9.6 Adding matched cell mechanics")
p("Once cells from patients who also have a slide and a report are measured, write data/mechanics/matched_cells.csv with "
  "one row per cell and run:")
code(["python src/09_three_input.py --cells data/mechanics/matched_cells.csv --assay RT-DC"])
table(["Column", "Meaning"], [
    ["patient", "same ID as the slide and report (cohort.csv)"], ["sample_id", "biological preparation the cell came from"],
    ["assay", "RT-DC, AFM-indentation or C-Trap (one assay per analysis)"], ["E_kPa", "apparent Young's modulus under the stated protocol"],
    ["deform, area_um2", "RT-DC shape terms (optional)"], ["relax_tau_s, dissipation_frac, recovery_frac", "time-dependent terms (optional)"],
    ["batch, date", "for batch control"],
], [35, 65], "Input format for the three-input test (full spec: dcc/mechanics_feature_spec.md).", fs=8)
h2("9.7 Output files")
table(["File", "Contents"], [
    ["results/report_oof.csv, image_oof.csv, fusion_oof.csv", "patient, P(GBM), model, cohort (all/x40), scheme (patient/institution)"],
    ["results/report_metrics.csv, image_metrics.csv, fusion_metrics.csv", "AUROC with CI, within-institution AUROC, macro-F1, Brier, ECE"],
    ["results/fusion_deltas.csv", "paired AUROC differences, bootstrap CI, DeLong and bootstrap p"],
    ["results/fusion_subgroups.csv", "mean P(GBM) and % called GBM by WHO-2021-like class and grade"],
    ["results/confounding_*.csv", "scanner recognition, stratified AUROCs, University of Florida subset, likelihood-ratio tests"],
    ["results/report_leak_audit.csv, report_label_audit.csv, report_top_terms_*.csv", "masking and label audits"],
    ["results/mechanics_*.csv, mechanics_lmm.txt", "per-measurement features, mixed models, variance components, bag test"],
    ["results/power_matched_cohort.csv", "simulated power by baseline, gain and cohort size"],
    ["data/processed/cohort.csv", "one row per patient: labels, institution, molecular class, modality flags, slide ID"],
    ["data/processed/splits.csv", "patient, scheme, repeat, fold"],
    ["data/processed/slides.csv", "per slide: size, magnification, MPP, scanner, scan date, JPEG quality, tile selection"],
    ["data/processed/tile_features.csv", "per tile: QC values, colour, nuclear and texture features, qc_ok"],
    ["data/processed/embeddings_hipt.npz", "emb (55,864 x 384) with patient, tile, mag, stratum, crop"],
    ["results/umap_patient_hipt.csv, mechanics_mixed_models_terms.csv", "UMAP coordinates and mixed-model terms behind "
     "Figure 7 and Figure S9"],
    ["figures/nature/*.pdf, *.png, figure_legends.md", "Nature-style figures: vector PDF with embedded fonts, 450- and "
     "300-dpi PNG, and every title and legend"],
], [42, 58], "Output files.", fs=7.5)
pagebreak()

# ================================================================== 10 SOURCES
h1("10. Sources")
bullets([
    "TCGA-Reports (Kefeli et al.): https://github.com/tatonetti-lab/tcga-path-reports",
    "cBioPortal datahub, lgggbm_tcga_pub (Ceccarelli et al., Cell 2016): https://github.com/cBioPortal/datahub/tree/master/public/lgggbm_tcga_pub",
    "NCI Genomic Data Commons API (Data Release 46.0): https://api.gdc.cancer.gov/status",
    "Imaging Data Commons index: https://pypi.org/project/idc-index-data/",
    "HIPT (Chen et al., CVPR 2022): https://github.com/mahmoodlab/HIPT",
    "UNI model card: https://huggingface.co/MahmoodLab/UNI",
    "Mechanomics RT-DC Set 01 - Glioblastoma: https://doi.org/10.6084/m9.figshare.28161236",
    "Ilse, Tomczak and Welling, Attention-based deep multiple instance learning, ICML 2018.",
    "DeLong, DeLong and Clarke-Pearson, Biometrics 1988; Sun and Xu, IEEE Signal Processing Letters 2014.",
    "Duke Compute Cluster documentation: https://dcc.duke.edu/dcc/partitions, https://dcc.duke.edu/dcc/slurm, "
    "https://dcc.duke.edu/dcc/architecture/, https://dcc.duke.edu/storage/",
])
pagebreak()

# ================================================================== APPENDIX
h1("Appendix A. Source code")
p("Complete source of every script used for this report and of the DCC pipeline, in run order.")
files = ["src/common.py", "src/00_inventory.py", "src/report_masking.py", "src/01_reports.py", "src/02_tiles.py",
         "src/03_embed.py", "src/04_image_models.py", "src/05_fusion.py", "src/06_mechanics.py", "src/07_confounding.py",
         "src/08_power.py", "src/09_three_input.py", "src/10_figures.py", "src/11_report_figures.py",
         "src/figure_legends.py", "src/12_nature_figures.py", "report/build_content.py", "report/render_docx.js",
         "report/toc_pages.py", "report/fix_bookmarks.py", "report/build_report.sh",
         "harvest/gdc_tile_harvester.js", "dcc/config.sh", "dcc/environment.yml", "dcc/00_setup.sh",
         "dcc/01_make_gdc_manifest.py", "dcc/01_download.sbatch", "dcc/02_tile_wsi.py", "dcc/02_tile_wsi.sbatch",
         "dcc/03_extract_features.py", "dcc/03_extract_features.sbatch", "dcc/04_train_mil.py", "dcc/04_train_mil.sbatch",
         "dcc/05_report_bert.py", "dcc/05_report_bert.sbatch", "dcc/merge_dcc_outputs.py", "dcc/06_fusion_and_checks.sbatch",
         "dcc/mechanics_feature_spec.md", "dcc/README_DCC.md", "README.md"]
for f in files:
    h3(f)
    code(open(ROOT / f, encoding="utf-8").read().rstrip("\n").split("\n"), size=6.5)

toc = [{"level": 1 if b["t"] == "h1" else 2, "text": b["text"]} for b in B if b["t"] in ("h1", "h2")]
for b in B:
    if b["t"] == "toc":
        b["entries"] = toc
json.dump(B, open(ROOT / "report" / "blocks.json", "w"), indent=0)
print(len(B), "blocks;", fig_no[0], "figures +", sfig_no[0], "supplementary figures;", tab_no[0], "tables")
