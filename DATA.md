# Data

Three places hold the data for this project:

- **In git:** processed tables, results and figure previews. Every file is under 10 MB.
- **In the v1.0 GitHub release:** large derived files.
- **Downloaded from the original source:** third-party raw data, fetched by `scripts/fetch_public_data.sh`, so
  their terms stay with their owners.

Everything here comes from public sources: TCGA open-access data and the Mechanomics RT-DC set. Patients appear
only by their public TCGA barcodes.

## In this repository

| Path | Contents | Derived from |
|---|---|---|
| `data/processed/cohort.csv` | one row per patient: GBM/LGG label, WHO-2021-like class, institution, modality flags | TCGA pan-glioma annotations (cBioPortal), GDC case records |
| `data/processed/splits.csv` | shared 5 × 5 patient-level and institution-grouped CV folds | this project |
| `data/processed/slides.csv` | per slide: scanner, magnification, MPP, scan date, JPEG quality, tile selection | GDC slide headers |
| `data/processed/tiles_index.csv`, `tile_features.csv`, `handcrafted_patient.csv` | tile list, QC values, nuclear/texture/stain features | GDC tiles |
| `data/processed/reports_masked.csv` | pathology reports with diagnosis, grade and molecular terms masked (M0/M1/M2) | TCGA-Reports, MIT licence (see below) |
| `data/processed/rtdc_events_gated.csv.gz` | gated single-cell RT-DC events | Mechanomics RT-DC Set 01, CC BY 4.0 |
| `data/raw/tiles/gdc_manifest.json` | GDC file IDs and tissue-source sites of the 880 slides | NCI GDC |
| `data/raw/idc/idc_tcga_glioma_sm_series.csv` | IDC slide index extract (cross-check) | idc-index-data 25.0.0 |
| `results/` | metric tables, out-of-fold predictions, audits, logs | this project |
| `figures/` | figures; `figures/nature/` holds the Nature-style PDFs and 300-dpi previews | this project |

## In the v1.0 release (attach these files when you create it)

| File | Size | Contents |
|---|---|---|
| `MechanoPath_full_report.docx` | 7 MB | full report: data, methods, all results and figures, how to rerun, source code |
| `mechanopath_nature_figures.zip` | 26 MB | every figure as vector PDF and 450-dpi image, with legends |
| `mechanopath_embeddings_hipt_part1.npz`, `part2.npz` | 2 × 21 MB | HIPT embeddings of 55,864 tile crops (join with the snippet in `data/processed/EMBEDDINGS_README.txt`) |
| `mechanopath_gdc_tiles_part01-11.zip` | 1.4 GB in total | 16 tiles + thumbnail + metadata per slide, 877 slides |

GitHub accepts release files up to 2 GiB each. Zenodo (50 GB per record, with a DOI) is the alternative if you
want a citable data archive.

## Downloaded by `scripts/fetch_public_data.sh`

| Source | What | Terms |
|---|---|---|
| cBioPortal datahub: `lgggbm_tcga_pub` (Ceccarelli et al., Cell 2016) | clinical patient and sample tables | public; cite the study and cBioPortal |
| TCGA-Reports (Kefeli & Tatonetti), github.com/tatonetti-lab/tcga-path-reports | `TCGA_Reports.csv` | MIT licence, © 2023 Tatonetti Lab |
| HIPT (Chen et al., CVPR 2022), github.com/mahmoodlab/HIPT | ViT-S/16 code and `vit256_small_dino.pth` weights | Commons Clause licence: non-commercial academic use only, so the weights are not re-hosted here |
| Mechanomics RT-DC Set 01, Glioblastoma (Urbanska, Herbig, Guck), doi:10.6084/m9.figshare.28161236 | per-event `.tsv` exports | CC BY 4.0 |
| NCI GDC open-access TCGA-GBM / TCGA-LGG diagnostic slides | tiles (release files above, or `harvest/gdc_tile_harvester.js`) | open access; cite TCGA and the GDC |

## Do not commit

Patient data from new collections, such as matched mechanics for the three-input test in `data/mechanics/`, are
excluded by `.gitignore`. Those need IRB-approved storage, not GitHub.

## Third-party licence

`data/processed/reports_masked.csv` is derived from TCGA-Reports. The MIT licence of TCGA-Reports, © 2023 Tatonetti
Lab, is reproduced in `licenses/TCGA-Reports-LICENSE.txt`.
