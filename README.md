# MechanoPath

Glioblastoma (TCGA-GBM) vs lower-grade glioma (TCGA-LGG) from H&E whole-slide images, pathology reports with the
diagnosis masked, and whole-cell biomechanics. The question is whether cell mechanics adds information beyond what
slides and reports already carry. Every model shares the same patient-level splits and is also tested with whole
institutions held out, on 40×-scanned slides only, and against acquisition-only baselines, so scanner and site
shortcuts are measured rather than assumed away.

## Main results

Matched cohort (686 patients with a usable slide and a report), whole institutions held out:

| Model | AUROC [95% CI], all slides | AUROC, 40×-scanned slides only (n = 556) |
|---|---|---|
| Image + masked report (late fusion) | 0.916 [0.895, 0.935] | 0.908 [0.880, 0.932] |
| Image only (HIPT mean-pool) | 0.864 [0.834, 0.893] | 0.856 [0.819, 0.890] |
| Masked report only (M2) | 0.813 [0.780, 0.843] | 0.802 [0.761, 0.838] |
| Acquisition metadata only | 0.752 [0.707, 0.799] | 0.434 [0.366, 0.502] |

- Fusion beats either input alone (paired DeLong p < 0.001).
- Random patient folds inflate every model (fusion 0.986), because GBM and LGG came from different sites,
  years and scanners.
- No public dataset links whole-cell mechanics to these patients, so the three-input test is specified but
  not yet runnable. Detecting a +0.03 AUROC gain over a 0.90 baseline needs about 150 matched patients per
  class.

![ROC curves, institutions held out](figures/nature/fig9_roc_curves_300dpi.png)

The full write-up (methods, every table and figure, how to rerun, source code) is the Word report attached to
the v1.0 release.

## Repository layout

```
src/            CPU analysis, steps 00-12 (every number in the report)
dcc/            full-resolution pipeline for the Duke Compute Cluster (SLURM, GPU)
harvest/        browser script that sampled slide tiles from the GDC tile service
report/         builds the Word report from the result files
scripts/        fetch_public_data.sh downloads the public inputs that are not stored here
data/processed/ cohort, splits, slide metadata, tile features, masked reports, gated RT-DC events
data/raw/       only the GDC manifest and IDC index extract are versioned; everything else is downloaded
results/        metric tables, out-of-fold predictions, audits, logs
figures/        first-version PNGs; figures/nature/ = Nature-style vector PDFs and 300-dpi previews
```

## Rerunning

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_public_data.sh        # labels, reports, HIPT code + weights, RT-DC tables
```

Slide tiles (1.4 GB) are not in git: download the 11 `mechanopath_gdc_tiles_partNN.zip` files from the release,
or regenerate them with `harvest/gdc_tile_harvester.js`, and point `UPLOADS` in `src/02_tiles.py` at their folder.
To skip the 2.4-hour embedding step, download the two `mechanopath_embeddings_hipt_part*.npz` files from the
release and join them as described in `data/processed/EMBEDDINGS_README.txt`.

```bash
python src/00_inventory.py        # cohort + shared splits
python src/01_reports.py          # report branch with masking
python src/02_tiles.py            # tile QC + tissue features
python src/03_embed.py            # HIPT embeddings (or use the release files)
python src/04_image_models.py
python src/05_fusion.py
python src/06_mechanics.py
python src/07_confounding.py
python src/08_power.py
python src/12_nature_figures.py   # figures
bash report/build_report.sh       # Word report (needs node docx, LibreOffice, poppler)
```

Full-resolution rerun on Duke DCC: see `dcc/README_DCC.md`.

## Data

What is stored here, what is in the release and what is downloaded, with sources and licences: [DATA.md](DATA.md).

## Licence and citation

Code: MIT (see `LICENSE`). Data files keep the terms of their sources (see `DATA.md`). How to cite: `CITATION.cff`.
