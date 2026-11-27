# Running MechanoPath on the Duke Compute Cluster (DCC)

This is the workflow that produced the manuscript results. Everything runs from `$MECHANOPATH_ROOT`
(`~/mechanopath_full`); large data live on `/cwork/$NETID/mechanopath` (slides, tile grids, features) and the conda
environment and Hugging Face cache on `/work/$NETID`. Paths are set in `dcc/config.sh`.

## One-time setup (already done for nb295)

| What | Command |
|---|---|
| Conda environment | `conda env create -f dcc/environment.yml -p /work/$NETID/conda/mechanopath` |
| Library pins that matter | `pip install "transformers>=4.50,<5" python-docx` (transformers 5.x fails to import with torch 2.6 on CPU nodes) |
| Model weights (cached once, jobs run offline) | `bash dcc/step1_start.sh` downloads Phikon-v2, BiomedBERT and HIPT weights |
| Slide manifest (880 slides, one per patient) | `python dcc/01_make_gdc_manifest.py --one-per-patient --out data/gdc_manifest_dx.txt` |

## Pipeline

| Step | Script | Partition / resources | Notes |
|---|---|---|---|
| 1 | `sbatch dcc/01_download.sbatch` | common | gdc-client, ~0.73 TB, ~3 h |
| 2 | `python dcc/02b_make_wsi_list.py` then `sbatch dcc/02_tile_wsi.sbatch` | common, 50-task array | tiles of 256 px at 0.5 µm/px; skips finished slides; faintly stained slides with no tile are re-thresholded after optical-density normalisation (`mask_rule` attribute); `python dcc/02c_mask_validation.py` checks that the fallback reproduces the primary mask on normal slides |
| 3 | `ENCODER=phikon2 sbatch -J mp_feat_phikon2 dcc/03_extract_features.sbatch` (and `ENCODER=hipt`, `-J mp_feat_hipt`) | gpu-common + scavenger-gpu, 8 tasks | resumable; eager attention on RTX 2080/2080 Ti; a task on a broken GPU node resubmits itself excluding that node |
| 4 | `ENCODER=phikon2 sbatch -J mp_mil_phikon2 dcc/04_train_mil.sbatch` (and `ENCODER=hipt`, `-J mp_mil_hipt`) | gpu-common, 64 GB | mean pooling + attention MIL, both CV schemes, both cohorts; waits for feature tasks of the same encoder that are still queued |
| 5 | `sbatch dcc/05_report_bert.sbatch` | gpu-common + scavenger-gpu, 8 tasks | BiomedBERT, {M2, M1} x {all, x40} x {institution, patient}; finished folds are kept |
| 6 | `sbatch dcc/phase_final.sbatch` | common, 24 GB | Phase 1 statistics, Phase 2 mechanics, plasmode design, figures, manuscript; runs after any feature, image-model or BiomedBERT job still queued, and stops if an image-model task is missing |

`dcc/step1_start.sh` submits a first pass on the slides already downloaded, and `dcc/phase1_after_download.sbatch`
(submitted with a dependency on the download) tiles the rest, extracts features, trains the image models and then
submits step 6 automatically. Nothing needs to be run by hand unless a step fails.

## Checking progress

```bash
cd ~/mechanopath_full && source dcc/config.sh
squeue -u $USER                                  # running and pending jobs
python dcc/02b_make_wsi_list.py | tail -1         # downloaded slides
ls $TILE_DIR/*.h5 | wc -l; ls $FEAT_DIR/phikon2 | wc -l; ls $FEAT_DIR/hipt | wc -l
cat logs/job_ids_chain.txt                        # job IDs submitted by the chain
tail logs/final_*.out                             # final analysis
```

## Outputs

| File | Contents |
|---|---|
| `report/MechanoPath_manuscript.docx` (+ `.md`) | manuscript with all numbers filled from the result tables |
| `report/MechanoPath_results_bundle.zip` | manuscript, figures, every result table |
| `figures/manuscript/*.pdf, *.png` | Nature-format figures (vector PDF and 450-dpi PNG) |
| `results/phase1/*.csv` | metrics, paired comparisons, calibration, decision curves, subgroups, confounding, out-of-fold predictions |
| `results/phase2/*.csv` | mechanics descriptors, variance partition, shift functions, shape distances, ICC, precision, bag classification, plasmode |

Download the bundle from OnDemand (Files -> `mechanopath_full/report`) or from the JupyterLab file browser (right-click -> Download).

## Re-running parts

* Only the statistics, figures and manuscript: `sbatch dcc/phase_final.sbatch`.
* Only one BiomedBERT task (0-7): `sbatch --array=3 dcc/05_report_bert.sbatch`.
* Only failed feature slides: resubmit step 3; finished slides are skipped.

## Optional: UNI encoder (pretraining excluded TCGA)

UNI weights are gated. After your Hugging Face account has been granted access, run once on DCC
`source activate /work/$NETID/conda/mechanopath && hf auth login`, then
`HF_HUB_OFFLINE=0 ENCODER=uni sbatch dcc/03_extract_features.sbatch` and `ENCODER=uni sbatch dcc/04_train_mil.sbatch`.
The login token is yours to enter; the pipeline never stores it.

## Adding matched cell mechanics (three-input model)

Follow `dcc/mechanics_feature_spec.md`, place `data/mechanics/matched_cells.csv` (IRB-approved storage only), then
run `python src/09_three_input.py --cells data/mechanics/matched_cells.csv --assay RT-DC`. The pre-registered test of
distribution beyond average stiffness is the likelihood-ratio test of spread and tail descriptors added to a model
with the image-and-report score, the mean and the median of log10 E (see `src/14_design_plasmode.py`).
