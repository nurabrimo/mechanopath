"""Step 7 - batch / acquisition confounding checks.

TCGA-GBM and TCGA-LGG were collected and scanned in different eras. Every 20x-scanned
slide is a GBM, older scans are GBM, and each institution sent its GBM and LGG cases
under different site codes at different times. These checks ask how much of each
branch's signal survives once acquisition is held fixed.

  1. acquisition metadata by class (table)
  2. can the image representation recognise the scanner? (predict 20x-vs-40x scanning and
     JPEG quality from embeddings among GBM slides only, where both values occur)
  3. stratified AUROC: inside institution x scan-year strata that contain both classes
  4. University of Florida: 15 GBM vs 15 LGG, same year (2014), same scanner settings
  5. logistic regression on the 40x cohort: label ~ branch score + scan year + JPEG quality
     + software version (likelihood-ratio test for the branch score)

Outputs: results/acquisition_by_class.csv, results/confounding_*.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import logit
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RES, SEED, bootstrap_auc  # noqa: E402

warnings.filterwarnings("ignore")
cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
slides = pd.read_csv(PROC / "slides.csv").set_index("patient")
S = slides.join(cohort[["y_gbm", "institution", "has_report"]])

# 1 ---------------------------------------------------------------------------
tabs = []
for col in ["appmag", "jpeg_q", "aperio_major", "scan_year", "compression"]:
    t = pd.crosstab(S[col].fillna("missing"), S.project)
    t.index = [f"{col}={v}" for v in t.index]
    tabs.append(t)
A = pd.concat(tabs)
A.to_csv(RES / "acquisition_by_class.csv")
print(A)

# 2 ---------------------------------------------------------------------------
Z = np.load(PROC / "embeddings_hipt.npz", allow_pickle=True)  # self-generated file
emb = Z["emb"].astype(np.float32)
idx = pd.DataFrame({c: Z[c] for c in ["patient", "mag"]})
MEAN = pd.DataFrame({p: emb[g.index].mean(0) for p, g in idx[idx.mag == "x20"].groupby("patient")}).T.dropna()
hp = pd.read_csv(PROC / "handcrafted_patient.csv").set_index("patient")
rows = []
gbm = S[(S.y_gbm == 1)].index.intersection(MEAN.index)
for target, fn in [("scanned_at_20x", lambda d: (d.appmag == 20).astype(int)),
                   ("jpeg_q70", lambda d: (d.jpeg_q == 70).astype(int)),
                   ("scan_before_2011", lambda d: (d.scan_year < 2011).astype(int))]:
    yv = fn(S.loc[gbm])
    if yv.nunique() < 2:
        continue
    bio = [c for c in hp.columns if any(k in c for k in ["nuc_", "glcm_", "lbp_", "pale_eosin"]) and "nuc_H_mean" not in c]
    for fname, X in [("HIPT embedding (20x tiles)", MEAN.loc[gbm]),
                     ("hand-crafted tissue features", hp.reindex(gbm)[bio].fillna(hp[bio].median())),
                     ("all hand-crafted incl. colour/sharpness", hp.reindex(gbm).fillna(hp.median(numeric_only=True)))]:
        X = X.select_dtypes(include=[np.number]) if hasattr(X, "select_dtypes") else X
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 2, 8), max_iter=5000,
                                                                    cv=StratifiedKFold(5, shuffle=True, random_state=SEED)))
        p = cross_val_predict(clf, X.values, yv.values, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                              method="predict_proba")[:, 1]
        lo, hi = bootstrap_auc(yv.values, p)
        rows.append({"target (GBM slides only)": target, "features": fname, "n": len(yv), "n_pos": int(yv.sum()),
                     "auroc": roc_auc_score(yv, p), "lo": lo, "hi": hi})
R2 = pd.DataFrame(rows)
R2.to_csv(RES / "confounding_scanner_recognition.csv", index=False)
print(R2.round(3).to_string(index=False))

# 3/4/5 ------------------------------------------------------------------------
img = pd.read_csv(RES / "image_oof.csv")
rep = pd.read_csv(RES / "report_oof.csv")
fus = pd.read_csv(RES / "fusion_oof.csv")


def strat_auc(df, keys):
    num = den = 0.0
    used = []
    for k, g in df.groupby(keys):
        n1, n0 = int(g.y_gbm.sum()), int((1 - g.y_gbm).sum())
        if n1 >= 2 and n0 >= 2:
            a = roc_auc_score(g.y_gbm, g.p)
            num += a * n1 * n0; den += n1 * n0
            used.append((k, n1, n0, round(a, 3)))
    return (num / den if den else np.nan), used


out, uf_rows, lrt = [], [], []
sources = [("image", img, ["img_meanpool", "img_abmil", "handcrafted", "acq_meta", "stain_color"]),
           ("report", rep, ["tfidf_M1", "tfidf_M2", "age"]),
           ("fusion", fus, ["late_image+report_M2", "early_image+report_M2"])]
for src, df, models in sources:
    for (cname, scheme), dd in df.groupby(["cohort", "scheme"]):
        for m in models:
            d = dd[dd.model == m].set_index("patient")[["p"]].join(S[["y_gbm", "institution", "scan_year", "appmag", "jpeg_q",
                                                                       "aperio_major"]], how="inner")
            if d.empty:
                continue
            a_iy, used = strat_auc(d.dropna(subset=["scan_year"]), ["institution", "scan_year"])
            a_q, _ = strat_auc(d, ["jpeg_q", "aperio_major"])
            out.append({"branch": src, "model": m, "cohort": cname, "scheme": scheme, "n": len(d),
                        "auroc_overall": roc_auc_score(d.y_gbm, d.p),
                        "auroc_within_institution_x_scanyear": a_iy,
                        "n_strata": len(used), "patients_in_strata": int(sum(u[1] + u[2] for u in used)),
                        "auroc_within_jpegq_x_software": a_q})
            uf = d[d.institution == "University of Florida"]
            if uf.y_gbm.nunique() == 2:
                lo, hi = bootstrap_auc(uf.y_gbm.values, uf.p.values)
                uf_rows.append({"branch": src, "model": m, "cohort": cname, "scheme": scheme, "n_gbm": int(uf.y_gbm.sum()),
                                "n_lgg": int((1 - uf.y_gbm).sum()), "auroc": roc_auc_score(uf.y_gbm, uf.p), "lo": lo, "hi": hi})
            if cname == "x40":
                e = d.dropna(subset=["scan_year"]).copy()
                e["score"] = logit(np.clip(e.p, 1e-4, 1 - 1e-4))
                # numeric/binary coding avoids complete separation on rare scan-year levels
                cov = pd.DataFrame({"scan_year": e.scan_year - 2013, "jpeg_q70": (e.jpeg_q == 70).astype(float),
                                    "aperio_major": e.aperio_major.fillna(12) - 12}, index=e.index)
                X0 = sm.add_constant(cov)
                X1 = X0.assign(score=e.score)
                try:
                    f0 = sm.Logit(e.y_gbm, X0).fit(disp=0, method="bfgs", maxiter=2000)
                    f1 = sm.Logit(e.y_gbm, X1).fit(disp=0, method="bfgs", maxiter=2000)
                    lr = 2 * (f1.llf - f0.llf)
                    lrt.append({"branch": src, "model": m, "scheme": scheme, "n": len(e), "LR_chi2": lr,
                                "p": stats.chi2.sf(lr, 1), "odds_ratio_per_logit": float(np.exp(f1.params["score"])),
                                "acq_only_auroc_in_sample": roc_auc_score(e.y_gbm, f0.predict(X0)),
                                "acq_plus_score_auroc_in_sample": roc_auc_score(e.y_gbm, f1.predict(X1))})
                except Exception as ex:
                    lrt.append({"branch": src, "model": m, "scheme": scheme, "error": str(ex)[:80]})

O = pd.DataFrame(out); O.to_csv(RES / "confounding_stratified_auroc.csv", index=False)
U = pd.DataFrame(uf_rows); U.to_csv(RES / "confounding_university_of_florida.csv", index=False)
L = pd.DataFrame(lrt); L.to_csv(RES / "confounding_lrt_x40.csv", index=False)
pd.set_option("display.width", 220)
print(O.round(3).to_string(index=False))
print(U.round(3).to_string(index=False))
print(L.round(4).to_string(index=False))
