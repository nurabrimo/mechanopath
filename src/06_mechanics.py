"""Step 6 - whole-cell mechanics branch (public RT-DC data, kept separate from patient models).

Data: Mechanomics RT-DC Set 01 - Glioblastoma (Urbanska, Herbig, Guck; figshare
10.6084/m9.figshare.28161236, CC BY 4.0). Three primary glioblastoma lines (X01, X04,
X08) x three culture conditions (Serum, EGF, FGFJI = low/medium/high STAT3-Ser/Hes3
activation), measured on 3 dates by real-time deformability cytometry (30 um channel,
0.16 uL/s). Young's modulus per cell comes from the exported .tsv (Shape-Out LUT).

What this branch can and cannot test
  - It develops and checks the per-sample mechanical feature set (median, spread,
    tails) and quantifies batch (measurement date) versus biological effects.
  - It tests the *heterogeneity idea* on a proxy task: does the distribution of
    single-cell stiffness separate culture states better than the median alone?
  - It cannot test GBM vs lower-grade glioma: every line here is a glioblastoma
    line and none is linked to a TCGA patient.

Outputs
  data/processed/rtdc_events_gated.csv.gz     gated single-cell events
  results/mechanics_per_measurement.csv       per-measurement feature table
  results/mechanics_lmm.txt                   mixed-model summaries
  results/mechanics_variance_components.csv
  results/mechanics_bag_classification.csv    median-only vs distribution features
"""
import glob
import io
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RAW, RES, SEED  # noqa: E402

warnings.filterwarnings("ignore")


def read_tsv(path):
    lines = open(path, encoding="utf-8-sig").read().splitlines()
    cols = lines[3].lstrip("#").split()
    d = pd.read_csv(io.StringIO("\n".join(lines[5:])), sep="\t", header=None, names=cols)
    m = re.search(r"(\d{8})_XO(\d)_(\w+) - (M\d)", path)
    d["date"], d["line"], d["cond"], d["meas"] = m.group(1), "X0" + m.group(2), m.group(3), m.group(4)
    d["measurement"] = f"{m.group(1)}_X0{m.group(2)}_{m.group(3)}_{m.group(4)}"
    return d


def sample_features(E, deform=None, area=None):
    """Per-sample mechanical summary used throughout MechanoPath.

    E: single-cell apparent Young's moduli (kPa) from ONE sample (tumour / preparation).
    Returns location, spread and tail descriptors on log10(E) plus optional shape terms.
    The same function applies to AFM/optical-tweezers stiffness once units and the
    loading protocol are fixed (see dcc/mechanics_feature_spec.md)."""
    lE = np.log10(np.asarray(E, float))
    q = np.percentile(lE, [10, 25, 50, 75, 90])
    f = {"n_cells": len(lE), "logE_median": q[2], "logE_q10": q[0], "logE_q25": q[1], "logE_q75": q[3],
         "logE_q90": q[4], "logE_iqr": q[3] - q[1], "logE_p90_p10": q[4] - q[0],
         "logE_mad": stats.median_abs_deviation(lE), "logE_skew": stats.skew(lE),
         "logE_kurt": stats.kurtosis(lE), "frac_soft": np.mean(lE < np.log10(0.7)),
         "frac_stiff": np.mean(lE > np.log10(1.3))}
    if deform is not None:
        f["deform_median"] = np.median(deform)
        f["deform_iqr"] = np.subtract(*np.percentile(deform, [75, 25]))
    if area is not None:
        f["area_median_um2"] = np.median(area)
        f["area_iqr_um2"] = np.subtract(*np.percentile(area, [75, 25]))
    return f


if __name__ == "__main__":
    D = pd.concat([read_tsv(f) for f in sorted(glob.glob(str(RAW / "mechanics/tsv/*.tsv")))])
    n0 = len(D)
    # gating: valid LUT region (finite E), smooth contours (porosity < 1.05), no debris
    G = D[np.isfinite(D.emodulus) & (D.area_ratio < 1.05) & (D.area_um >= 30)].copy()
    print(f"events: {n0} raw -> {len(G)} gated")
    G[["measurement", "date", "line", "cond", "meas", "area_um", "deform", "emodulus", "volume", "aspect"]] \
        .to_csv(PROC / "rtdc_events_gated.csv.gz", index=False)

    # ---------------- per-measurement features ----------------
    rows = []
    for m, g in G.groupby("measurement"):
        f = sample_features(g.emodulus.values, g.deform.values, g.area_um.values)
        f.update({"measurement": m, "date": g.date.iloc[0], "line": g.line.iloc[0], "cond": g.cond.iloc[0]})
        rows.append(f)
    PM = pd.DataFrame(rows)
    PM.to_csv(RES / "mechanics_per_measurement.csv", index=False)
    print(PM.groupby("cond")[["logE_median", "logE_iqr", "logE_skew"]].mean().round(3))

    # ---------------- mixed models ----------------
    out = io.StringIO()
    PM["cond"] = pd.Categorical(PM.cond, ["Serum", "EGF", "FGFJI"])
    mA = smf.mixedlm("logE_median ~ C(cond) + C(line)", PM, groups=PM["date"]).fit(reml=True)
    out.write("Model A: per-measurement log10 median E ~ condition + line + (1|date)\n")
    out.write(str(mA.summary()) + "\n\n")
    # event level: random intercept per measurement, date as fixed effect, 400 cells/measurement
    sub = pd.concat([g.sample(min(len(g), 400), random_state=SEED) for _, g in G.groupby("measurement")])
    sub["logE"] = np.log10(sub.emodulus)
    sub["cond"] = pd.Categorical(sub.cond, ["Serum", "EGF", "FGFJI"])
    mB = smf.mixedlm("logE ~ C(cond) + C(line) + C(date)", sub, groups=sub["measurement"]).fit(reml=True)
    out.write("Model B: single-cell log10 E ~ condition + line + date + (1|measurement), <=400 cells/measurement\n")
    out.write(str(mB.summary()) + "\n\n")
    # spread beyond location: does condition change the IQR after adjusting for the median?
    mC = smf.ols("logE_iqr ~ C(cond) + C(line) + C(date) + logE_median", PM).fit()
    out.write("Model C: per-measurement IQR(log10 E) ~ condition + line + date + median\n")
    out.write(str(mC.summary()) + "\n")
    (RES / "mechanics_lmm.txt").write_text(out.getvalue())
    eff = 10 ** mA.params.filter(like="cond")
    print("fold-change in median E vs Serum (Model A):", eff.round(3).to_dict())

    # variance components of the per-measurement median: line, condition, date, residual
    vc = []
    full = smf.ols("logE_median ~ C(cond) + C(line) + C(date)", PM).fit()
    for term in ["C(cond)", "C(line)", "C(date)"]:
        red = smf.ols("logE_median ~ " + " + ".join(t for t in ["C(cond)", "C(line)", "C(date)"] if t != term), PM).fit()
        vc.append({"term": term, "partial_R2": (red.ssr - full.ssr) / red.ssr})
    vc.append({"term": "model_R2", "partial_R2": full.rsquared})
    pd.DataFrame(vc).to_csv(RES / "mechanics_variance_components.csv", index=False)
    print(pd.DataFrame(vc).round(3))

    # ---------------- heterogeneity test on a proxy task ----------------
    # bags of n cells drawn within each measurement; classifier trained on two dates and
    # tested on the third (leave-one-date-out), so no measurement contributes to both sides.
    feats = {
        "median_E_only": ["logE_median"],
        "E_distribution": ["logE_median", "logE_q10", "logE_q90", "logE_iqr", "logE_skew", "logE_kurt"],
        "E_distribution+shape": ["logE_median", "logE_q10", "logE_q90", "logE_iqr", "logE_skew", "logE_kurt",
                                 "deform_median", "deform_iqr", "area_median_um2", "area_iqr_um2"],
    }
    tasks = {"Serum_vs_FGFJI": ("Serum", "FGFJI"), "Serum_vs_EGF": ("Serum", "EGF"), "EGF_vs_FGFJI": ("EGF", "FGFJI")}
    res = []
    for bag_n in [50, 100, 300]:
        bags = []
        for m, g in G.groupby("measurement"):
            if len(g) < bag_n:
                continue
            for b in range(40):
                s = g.sample(bag_n, replace=False, random_state=SEED + b)
                f = sample_features(s.emodulus.values, s.deform.values, s.area_um.values)
                f.update({"measurement": m, "date": g.date.iloc[0], "line": g.line.iloc[0], "cond": g.cond.iloc[0]})
                bags.append(f)
        B = pd.DataFrame(bags)
        for tname, (c0, c1) in tasks.items():
            T = B[B.cond.isin([c0, c1])].copy()
            T["y"] = (T.cond == c1).astype(int)
            for fname, cols in feats.items():
                preds = []
                for d in sorted(T.date.unique()):
                    tr, te = T[T.date != d], T[T.date == d]
                    if tr.y.nunique() < 2 or te.y.nunique() < 2:
                        continue
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(tr[cols], tr.y)
                    preds.append(te.assign(p=clf.predict_proba(te[cols])[:, 1]))
                P = pd.concat(preds)
                bag_auc = roc_auc_score(P.y, P.p)
                Pm = P.groupby("measurement").agg(y=("y", "first"), p=("p", "mean"))
                res.append({"bag_cells": bag_n, "task": tname, "features": fname, "bag_auroc": bag_auc,
                            "measurement_auroc": roc_auc_score(Pm.y, Pm.p), "n_measurements": len(Pm)})
    R = pd.DataFrame(res)
    R.to_csv(RES / "mechanics_bag_classification.csv", index=False)
    print(R.pivot_table(index=["task", "bag_cells"], columns="features", values="bag_auroc").round(3))
