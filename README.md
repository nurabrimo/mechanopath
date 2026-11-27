# MechanoPath

Glioblastoma (GBM) versus lower-grade glioma (LGG) from H&E whole-slide images and pathology reports, evaluated at
institutions the model has not seen, and a test of whether single-cell mechanics carries diagnostic information beyond
its average.

**Part 1, routine pathology.** 689 TCGA patients (245 GBM, 444 LGG; 33 tissue-source sites). An attention-based
multiple-instance model on Phikon-v2 features of 7.47 million tissue tiles and a BiomedBERT model on pathology reports
with tumour type, grade, molecular findings and grading criteria masked are combined by late fusion. Every analysis is
run with whole institutions held out and repeated on slides scanned at 40×, where scan magnification no longer separates
the classes, and against acquisition-only baselines, so site and scanner shortcuts are measured rather than assumed away.

**Part 2, single-cell mechanics.** Public deformability-cytometry data from glioblastoma lines in three signalling
states (21,738 cells) and from 39 human tumour–healthy tissue pairs, and a simulation-based design of a matched images,
reports and mechanics study.

## Main results

| Model (institutions held out) | AUROC, all slides (n = 689) | AUROC, 40× slides (n = 557) |
|---|---|---|
| Image + report, late fusion (primary) | 0.958 (0.943–0.973) | 0.913 (0.886–0.941) |
| Image + report, TF-IDF fusion (secondary) | 0.971 (0.960–0.981) | – |
| Image, Phikon-v2 attention MIL | 0.948 (0.933–0.964) | 0.890 (0.859–0.920) |
| Report, TF-IDF (masked) | 0.813 (0.782–0.844) | 0.802 (0.764–0.841) |
| Report, BiomedBERT (masked) | 0.701 (0.662–0.740) | 0.691 (0.634–0.747) |
| Acquisition metadata only | 0.752 (0.706–0.798) | 0.434 (0.364–0.504) |

- Random patient folds inflate every model (fusion 0.995) because GBM and LGG came from different sites and scanners.
- LGG-labelled tumours with molecular features of glioblastoma (WHO CNS5) are called GBM by the fused model in 21% of
  cases, against 5% of IDH-mutant astrocytomas.
- Glioblastoma lines: signalling states shift the median stiffness (0.80- and 0.72-fold) but not the shape of the
  distribution; shape descriptors do not reproduce across days.
- Human biopsies: tumour cells deform less than matched healthy cells (median alone, AUROC 0.72); the full distribution
  adds nothing (ΔAUROC −0.020, 95% CI −0.057 to 0.012), although it identifies organ (0.93) and frozen versus fresh
  tissue (0.999).
- Pipeline v1 (16 sampled tiles per slide, HIPT features; `src/`, `results/fusion_metrics.csv`) is kept as a lighter
  replication; it reached 0.916 with institutions held out. The full-resolution results above supersede it.

## Repository layout

```
src/                 CPU analysis, steps 00–12 (pipeline v1: cohort, reports, tiles, models, fusion, confounding, mechanics)
dcc/                 full-resolution pipeline for the Duke Compute Cluster (Slurm, GPU); see dcc/README_DCC.md
                     ext_validate.py      refit of the image model on all TCGA patients, scoring of DeepHisto and CPTAC-GBM
                     extract_real_images.py   slide tiles and single-cell images shown in the overview figure
phase2b_biopsies/    human tumour versus healthy tissue (RT-FDC, DCOR): analysis, extraction scripts and results
harvest/             browser script that sampled slide tiles from the GDC tile service (pipeline v1)
scripts/             fetch_public_data.sh downloads the public inputs that are not stored here
data/processed/      cohort, splits, slide metadata, tile features, masked reports, gated RT-DC and AFM events
data/raw/            GDC manifest and IDC index extract only; slides and model weights are downloaded, not stored
results/             metric tables, out-of-fold predictions and audits; phase1/ (full resolution), phase2/ (mechanics,
                     design analysis), real_images/ (images in the overview figure, with provenance.json)
figures/manuscript/  figures of the first manuscript draft
paper_figures/       scripts and vector PDFs of the figures of the journal submission
report/              scripts that build the full Word report
licenses/            third-party licences of redistributed derived data
```

## Rerunning

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_public_data.sh                 # labels, reports, HIPT code and weights, RT-DC tables (see DATA.md)
python src/00_inventory.py                        # cohort and shared splits
python src/01_reports.py                          # report masking (M0/M1/M2)
```

Full-resolution image and report models (slide download, tiling, Phikon-v2 and HIPT features, attention MIL,
BiomedBERT, fusion and confounding analyses) run on a Slurm cluster: see `dcc/README_DCC.md`. The Phase 1 results are in
`results/phase1/`.

```bash
python phase2b_biopsies/src/15_mech_biopsy_dcor.py --stages qc,paired,distance,classify,sens   # human biopsies
python paper_figures/scripts/make_figures.py      # journal figures 2–8 from results/
python paper_figures/scripts/make_fig1.py         # overview figure and graphical abstract
```

Random seeds are fixed in the code.

## Data

All data are public: TCGA slides (NCI Genomic Data Commons), clinical and molecular annotations (cBioPortal,
lgggbm_tcga_pub), pathology reports (TCGA-Reports, MIT licence), RT-DC data of glioblastoma lines (figshare,
doi:10.6084/m9.figshare.28161236, CC BY 4.0), AFM data (RODBUK, doi:10.57903/UJ/3LOV6Q, CC0) and RT-FDC data of human
biopsies (DCOR, circle soteriou-kubankova; Soteriou et al., Nat. Biomed. Eng. 2023). What is stored here and what is
downloaded, with licences: [DATA.md](DATA.md).

## Licence and citation

Code: MIT (see `LICENSE`). Data files keep the terms of their sources (see `DATA.md`). How to cite: `CITATION.cff`.
