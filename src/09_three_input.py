"""Step 9 - the three-input experiment, ready for a matched cohort (not runnable on public data).

Run this once whole-cell mechanics exist for patients who also have a slide and a report
(for example, fresh glioma resections measured with C-Trap / AFM / RT-DC, linked to their
TCGA-style pathology). Input: data/mechanics/matched_cells.csv, one row per measured cell:

  patient       ID shared with the slide and report (same IDs as data/processed/cohort.csv)
  sample_id     biological preparation (dissociation / culture batch) the cell came from
  assay         e.g. 'RT-DC', 'AFM-indentation', 'C-Trap'  (one assay per analysis!)
  E_kPa         apparent Young's modulus under the stated protocol (required)
  deform, area_um2, relax_tau_s, dissipation_frac, recovery_frac   (optional)
  batch, date   for batch control

What it does (same held-out patients for every model, same splits as everything else):
  mechanics_median     median stiffness only
  mechanics_distr      median + spread + tails (the heterogeneity hypothesis)
  image+report         late fusion of the existing branch predictions
  image+report+mech    late fusion with the mechanics model
It reports paired AUROC differences (DeLong + paired bootstrap) for
  image+report  vs  image+report+mechanics      <- the central claim
  mechanics_median vs mechanics_distr            <- the heterogeneity claim

  python src/09_three_input.py --cells data/mechanics/matched_cells.csv --assay RT-DC
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RES, SEED, delong_test, iter_folds, paired_bootstrap_delta, summarize  # noqa: E402

spec = importlib.util.spec_from_file_location("mech", Path(__file__).parent / "06_mechanics.py")
mech = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mech)                     # only defines functions when imported
sample_features = mech.sample_features

ap = argparse.ArgumentParser()
ap.add_argument("--cells", required=True)
ap.add_argument("--assay", required=True)
ap.add_argument("--min-cells", type=int, default=50)
ap.add_argument("--scheme", default="patient", choices=["patient", "institution"])
ap.add_argument("--fusion-oof", default=str(RES / "fusion_oof.csv"))
ap.add_argument("--fusion-model", default="late_image+report_M2")
ap.add_argument("--out-prefix", default="three_input")
a = ap.parse_args()

cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
splits = pd.read_csv(PROC / "splits.csv")
C = pd.read_csv(a.cells)
C = C[C.assay == a.assay]
assert C.E_kPa.gt(0).all(), "E_kPa must be positive"

# one feature row per patient: pool that patient's preparations, keep #preparations as a covariate
rows = []
for p, g in C.groupby("patient"):
    if len(g) < a.min_cells:
        continue
    f = sample_features(g.E_kPa.values, g.get("deform"), g.get("area_um2"))
    for extra in ["relax_tau_s", "dissipation_frac", "recovery_frac"]:
        if extra in g and g[extra].notna().any():
            v = g[extra].dropna().values
            f[f"{extra}_median"], f[f"{extra}_iqr"] = np.median(v), np.subtract(*np.percentile(v, [75, 25]))
    f["n_preparations"] = g.sample_id.nunique()
    f["patient"] = p
    rows.append(f)
F = pd.DataFrame(rows).set_index("patient")
fus = pd.read_csv(a.fusion_oof)
fus = fus[(fus.model == a.fusion_model) & (fus.scheme == a.scheme) & (fus.cohort == "all")].set_index("patient").p
matched = sorted(set(F.index) & set(fus.index) & set(cohort.index))
y = cohort.loc[matched, "y_gbm"]
print(f"matched three-input patients: {len(matched)} (GBM {int(y.sum())}, LGG {int((1 - y).sum())})")
if y.nunique() < 2 or min(y.sum(), (1 - y).sum()) < 10:
    sys.exit("need at least 10 patients per class with all three inputs")

FEATS = {"mechanics_median": ["logE_median"],
         "mechanics_distr": [c for c in F.columns if c.startswith("logE_") or c.startswith("frac_")] +
                            [c for c in F.columns if c.endswith(("_median", "_iqr")) and not c.startswith("logE")]}


def cv_predict(cols):
    preds = []
    for r, k, tr, te in iter_folds(splits, matched, a.scheme):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegressionCV(Cs=np.logspace(-3, 2, 11), max_iter=5000, scoring="roc_auc",
                                               cv=StratifiedKFold(min(5, int(y.loc[tr].value_counts().min())), shuffle=True,
                                                                  random_state=SEED)))
        m.fit(F.loc[tr, cols], y.loc[tr])
        preds.append(pd.DataFrame({"patient": te, "p": m.predict_proba(F.loc[te, cols])[:, 1]}))
    return pd.concat(preds).groupby("patient").p.mean().reindex(matched)


P = {k: cv_predict(v) for k, v in FEATS.items()}
P["image+report"] = fus.reindex(matched)
clip = lambda s: logit(np.clip(s, 1e-4, 1 - 1e-4))
P["image+report+mechanics"] = expit((clip(P["image+report"]) * 2 + clip(P["mechanics_distr"])) / 3)
M = pd.DataFrame([summarize(y.values, p.values, name=k) for k, p in P.items()])
D = []
for base, new in [("image+report", "image+report+mechanics"), ("mechanics_median", "mechanics_distr")]:
    d0, ci, pb = paired_bootstrap_delta(y.values, P[base].values, P[new].values)
    _, _, _, pdl = delong_test(y.values, P[new].values, P[base].values)
    D.append({"baseline": base, "model": new, "delta_auroc": d0, "lo": ci[0], "hi": ci[1], "p_bootstrap": pb, "p_delong": pdl})
M.to_csv(RES / f"{a.out_prefix}_metrics.csv", index=False)
pd.DataFrame(D).to_csv(RES / f"{a.out_prefix}_deltas.csv", index=False)
print(M.round(3).to_string(index=False)); print(pd.DataFrame(D).round(4).to_string(index=False))
