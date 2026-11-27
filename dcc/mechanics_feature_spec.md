# Whole-cell mechanics: what to record so it can enter the classifier

This is the input contract for `src/09_three_input.py`. It applies to RT-DC, AFM indentation and
optical-tweezers (C-Trap) whole-cell measurements. Nuclear mechanics stays out of scope.

## Linkage (the part that decides whether the experiment is valid)

| Field | Rule |
|---|---|
| `patient` | Same identifier as the slide and the report. Cells must come from that patient's tumour, ideally the same surgery. |
| `sample_id` | One value per biological preparation (dissociation or culture). Cells are not independent; preparations are closer to it. |
| `specimen_region`, `surgery_date` | Recorded so slide, report and cells can be shown to be the same specimen. |
| Class balance | GBM and lower-grade gliomas both needed. The public GBM cell-line sets cannot supply the LGG side. |

Never assign a "typical GBM stiffness" to a patient, and never pair unrelated cell-line data with TCGA patients.

## Per-cell table (`data/mechanics/matched_cells.csv`)

| Column | Unit | Notes |
|---|---|---|
| `patient`, `sample_id`, `cell_id` | – | see above |
| `assay` | – | one assay per analysis; do not pool RT-DC moduli with AFM moduli |
| `E_kPa` | kPa | apparent Young's modulus; state the model (Hertz, LUT, power-law) in the metadata file |
| `deform`, `area_um2` | –, µm² | RT-DC shape terms (optional) |
| `relax_tau_s` | s | stress-relaxation or creep time under the stated protocol (optional) |
| `dissipation_frac` | – | hysteresis energy / loading work (optional) |
| `recovery_frac` | – | fraction of deformation recovered after unloading (optional) |
| `batch`, `date`, `operator` | – | batch control; balance classes across batches when you plan measurements |

Also keep a metadata file per assay: instrument, probe/bead or channel size, loading rate or flow rate,
indentation depth, temperature, buffer/medium viscosity, culture conditions and passage number.

## Per-patient features (computed by `sample_features()` in `src/06_mechanics.py`)

Location: median log10 E. Spread: IQR, 90th–10th percentile range, MAD. Shape: skewness, kurtosis,
fraction of very soft and very stiff cells. Optional medians/IQRs of relaxation, dissipation and
recovery. Minimum 50 cells per patient (configurable); report the number of preparations.

## What the public RT-DC data already showed (proxy task, 3 GBM lines)

- Culture state changed median stiffness strongly (EGF 0.80×, FGFJI 0.72× serum).
- Measurement date explained a partial R² of 0.28 of the per-measurement medians, so batch must be designed out.
- Distribution shape of E did **not** separate EGF from FGFJI better than the median; adding cell size and
  deformation did. Plan to record shape descriptors alongside stiffness.

## How many matched patients?

See `results/power_matched_cohort.csv` (from `src/08_power.py`). The required size depends on how good
image + report already is; when that baseline is ~0.90–0.95 AUROC, a +0.03 gain needs on the order of a
hundred or more patients per class even with an ideal combination weight.
