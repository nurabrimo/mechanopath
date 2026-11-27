"""Phase 2 design analysis: how many patients and cells would a matched study need to show that the distribution
of single-cell mechanics adds diagnostic information to images and reports?

No mechanical measurement exists for any TCGA patient, so this is a plasmode simulation, not an analysis of
patient mechanics. Its real components are
  (i)  the out-of-fold GBM probabilities of the Phase 1 image + report model and the true labels of the matched
       TCGA cohort (patients are resampled with replacement within class), and
  (ii) the within-sample shapes of 21 real RT-DC single-cell log10 E distributions (Mechanomics Set 01), each
       standardised to median 0 and IQR 1 and used as a template.
For a simulated tumour i of class y (1 = GBM), with template k and m cells drawn from it:
  location   mu_i = -delta * y + tau_loc * e_i                         (GBM softer by delta log10 units)
  spread     s_i  = 0.13 * r^(1 - y) * exp(tau_scale * h_i)            (LGG spread larger by factor r)
  cells      x_ij = mu_i + s_i * t_kj ; in the tail scenario a fraction pi of LGG cells is shifted by +0.3 log10
  redundant  mu_i = -beta * z_i + tau_loc * e_i, z_i the standardised logit of the image + report score
The between-tumour SDs are anchored on real data (results/phase2/A9_design_anchors.csv): tau_loc 0.03 log10 units
(RT-DC between-sample SD of the median within condition 0.032; IDH-stratified primary cultures 0.025) and
tau_scale 0.09 (RT-DC SD of ln IQR within condition); 0.47 (primary-culture AFM) in scenarios S2b and S3b and in the
grid. delta =
0.072 is the IDH1-R132H vs IDH1-wild-type difference in median Young's modulus (Shmelev et al. 2023).

Per simulated study (n patients per class, m cells per tumour), nested logistic models
  M0 logit(p_image+report) | M1 M0 + mean and median log10 E | M2 M1 + log IQR + upper tail (q90 - q50) + lower tail
  (q50 - q10) | Mmech mechanics only
are compared by likelihood-ratio tests (M1 vs M0: average stiffness beyond images and reports; M2 vs M1: shape of
the distribution beyond its average) and by 5-fold cross-validated AUROC. M1 carries two location estimators because
with few cells the tails of a sample also inform on the error of a single location estimate; with both, the shape
features are close to ancillary for location and M2 vs M1 keeps its nominal size under a pure location effect.

Outputs: results/phase2/plasmode_main.csv, plasmode_grid.csv, plasmode_inputs.json
  python src/14_design_plasmode.py [--oof results/phase1/phase1_oof.csv --model fusion_primary --cohort all
                                    --scheme institution] [--reps 500]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RES, SEED  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--oof", default=None)
ap.add_argument("--model", default=None)
ap.add_argument("--cohort", default="all")
ap.add_argument("--scheme", default="institution")
ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--grid-reps", type=int, default=300)
a = ap.parse_args()
OUT = RES / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED + 14)

# ------------------------------------------------------------------------------------------------ real inputs
cands = [(a.oof, a.model)] if a.oof else [(RES / "phase1" / "phase1_oof.csv", "fusion_primary"),
                                           (RES / "fusion_oof.csv", "late_image+report_M2")]
for path, model in cands:
    if path and Path(path).exists():
        O = pd.read_csv(path)
        O = O[(O.model == model) & (O.cohort == a.cohort) & (O.scheme == a.scheme)]
        if len(O):
            src = f"{Path(path).name}:{model}:{a.cohort}:{a.scheme}"
            break
cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
O = O.merge(cohort[["y_gbm"]], left_on="patient", right_index=True)
L = logit(np.clip(O.p.values, 1e-4, 1 - 1e-4))
Y = O.y_gbm.values
LZ = (L - L.mean()) / L.std()
score_by_class = {c: L[Y == c] for c in (0, 1)}
z_by_class = {c: LZ[Y == c] for c in (0, 1)}


def auc_fast(y, s):
    r = stats.rankdata(s)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


base_auc = auc_fast(Y, L)
print(f"scores: {src} | n={len(Y)} ({Y.sum()} GBM) | AUROC {base_auc:.3f}", flush=True)

ev = pd.read_csv(PROC / "rtdc_events_gated.csv.gz")
ev["logE"] = np.log10(ev.emodulus)
TPL = []
for m, g in ev.groupby("measurement"):
    x = g.logE.values
    TPL.append((x - np.median(x)) / np.subtract(*np.percentile(x, [75, 25])))
anch = pd.read_csv(OUT / "A9_design_anchors.csv", index_col=0)["value"] if (OUT / "A9_design_anchors.csv").exists() \
    else pd.Series(dtype=float)
TAU_LOC, TAU_SCALE, DELTA, S0, TAIL_SHIFT = 0.03, 0.09, 0.072, 0.13, 0.30
MS = [30, 100, 300, 1000]
NS = [25, 50, 100, 200, 400]

# ------------------------------------------------------------------------------------------------ descriptor banks
BANK_N = 3000


def bank(m, pi=0.0):
    """Descriptors of m cells drawn from each standardised template (median 0, IQR 1); a fraction pi of cells
    shifted by TAIL_SHIFT / S0 template units (stiff subpopulation)."""
    out = []
    for t in TPL:
        S = t[rng.integers(0, len(t), (BANK_N, m))]
        if pi > 0:
            S = S + (rng.random((BANK_N, m)) < pi) * (TAIL_SHIFT / S0)
        q10, q25, q50, q75, q90 = np.percentile(S, [10, 25, 50, 75, 90], axis=1)
        out.append(np.column_stack([S.mean(1), q50, q75 - q25, q90 - q50, q50 - q10]))
    return np.stack(out)                                       # templates x BANK_N x 5


t0 = time.time()
BANKS = {(m, pi): bank(m, pi) for m in MS for pi in (0.0, 0.10)}
print(f"banks ready ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ fast logistic fits
def fit_logistic(Xm, y, ridge=1e-3, iters=30):
    """Newton-Raphson logistic regression with an intercept and a tiny ridge on slopes; returns beta, loglik."""
    Xd = np.column_stack([np.ones(len(y)), Xm])
    beta = np.zeros(Xd.shape[1])
    R = np.eye(Xd.shape[1]) * ridge
    R[0, 0] = 0
    for _ in range(iters):
        eta = Xd @ beta
        p = expit(eta)
        W = p * (1 - p) + 1e-9
        g = Xd.T @ (y - p) - R @ beta
        H = (Xd * W[:, None]).T @ Xd + R
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    eta = Xd @ beta
    ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
    return beta, ll


def cv_auc(Xm, y, folds):
    pred = np.empty(len(y))
    for f in range(5):
        te = folds == f
        b, _ = fit_logistic(Xm[~te], y[~te])
        pred[te] = b[0] + Xm[te] @ b[1:]
    return auc_fast(y, pred)


def standardise(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-12)


def simulate(n, m, delta=0.0, r=1.0, pi=0.0, tau_loc=TAU_LOC, tau_scale=TAU_SCALE, redundant_beta=0.0,
             do_cv=True):
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    idx1 = rng.integers(0, len(score_by_class[1]), n)
    idx0 = rng.integers(0, len(score_by_class[0]), n)
    sc = np.r_[score_by_class[1][idx1], score_by_class[0][idx0]]
    zz = np.r_[z_by_class[1][idx1], z_by_class[0][idx0]]
    mu = -delta * y - redundant_beta * zz + tau_loc * rng.standard_normal(2 * n)
    s = S0 * np.where(y == 0, r, 1.0) * np.exp(tau_scale * rng.standard_normal(2 * n))
    k = rng.integers(0, len(TPL), 2 * n)
    j = rng.integers(0, BANK_N, 2 * n)
    d_lgg = BANKS[(m, pi)][k, j] if pi > 0 else BANKS[(m, 0.0)][k, j]
    d = np.where((y == 0)[:, None], d_lgg, BANKS[(m, 0.0)][k, j])
    # location: mean and median of log10 E (two estimators, so that location-invariant shape features cannot
    # improve the location estimate); shape: log IQR, upper tail (q90 - q50), lower tail (q50 - q10)
    feats = np.column_stack([mu + s * d[:, 0], mu + s * d[:, 1], np.log(s * d[:, 2]), s * d[:, 3], s * d[:, 4]])
    F = standardise(feats)
    S = standardise(sc[:, None])
    _, l0 = fit_logistic(S, y)
    _, l1 = fit_logistic(np.column_stack([S, F[:, :2]]), y)
    _, l2 = fit_logistic(np.column_stack([S, F]), y)
    out = {"p_M1_vs_M0": stats.chi2.sf(2 * (l1 - l0), 2), "p_M2_vs_M1": stats.chi2.sf(2 * (l2 - l1), 3),
           "p_M2_vs_M0": stats.chi2.sf(2 * (l2 - l0), 5)}
    if do_cv:
        folds = np.empty(2 * n, int)
        for c in (0, 1):
            i = np.where(y == c)[0]
            folds[i] = rng.permutation(np.arange(len(i)) % 5)
        out.update({"auc_M0": cv_auc(S, y, folds), "auc_M1": cv_auc(np.column_stack([S, F[:, :2]]), y, folds),
                    "auc_M2": cv_auc(np.column_stack([S, F]), y, folds), "auc_mech": cv_auc(F, y, folds),
                    "auc_mech_location": cv_auc(F[:, :2], y, folds)})
    return out


def summarise(res, **kw):
    R = pd.DataFrame(res)
    row = dict(kw, reps=len(R))
    for t in ("M1_vs_M0", "M2_vs_M1", "M2_vs_M0"):
        pw = float((R[f"p_{t}"] < 0.05).mean())
        row[f"power_{t}"] = pw
        row[f"power_{t}_mcse"] = float(np.sqrt(pw * (1 - pw) / len(R)))
    if "auc_M0" in R:
        for c in ("auc_M0", "auc_M1", "auc_M2", "auc_mech", "auc_mech_location"):
            row[c] = float(R[c].mean())
        for b, c in (("auc_M0", "auc_M1"), ("auc_M1", "auc_M2"), ("auc_M0", "auc_M2")):
            dd = R[c] - R[b]
            row[f"delta_{c[4:]}_vs_{b[4:]}"] = float(dd.mean())
            row[f"delta_{c[4:]}_vs_{b[4:]}_lo"], row[f"delta_{c[4:]}_vs_{b[4:]}_hi"] = np.percentile(dd, [2.5, 97.5])
    return row


# redundant scenario: same GBM-LGG difference in mean location as S1, but carried entirely by the score
b_red = DELTA / (LZ[Y == 1].mean() - LZ[Y == 0].mean())        # same class difference in mean location
SCEN = {
    "S0 no association": dict(),
    "S1 location (IDH-anchored)": dict(delta=DELTA),
    "S2 spread (LGG IQR x1.25)": dict(r=1.25),
    "S3 stiff tail (10% of LGG cells, 2-fold)": dict(pi=0.10),
    "S4 location + spread": dict(delta=DELTA, r=1.25),
    "S5 redundant with image + report": dict(redundant_beta=b_red),
    "S2b spread, primary-culture variation": dict(r=1.25, tau_scale=0.47),
    "S3b stiff tail, primary-culture variation": dict(pi=0.10, tau_scale=0.47),
}
rows = []
t0 = time.time()
for sname, kw in SCEN.items():
    for n in NS:
        for m in MS:
            res = [simulate(n, m, **kw) for _ in range(a.reps)]
            params = dict(tau_loc=TAU_LOC, tau_scale=TAU_SCALE)
            params.update(kw)
            rows.append(summarise(res, scenario=sname, n_per_class=n, cells_per_tumour=m, **params))
    pd.DataFrame(rows).to_csv(OUT / "plasmode_main.csv", index=False)
    print(f"{sname} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
MAIN = pd.DataFrame(rows)
print(MAIN[["scenario", "n_per_class", "cells_per_tumour", "power_M1_vs_M0", "power_M2_vs_M1", "auc_M0", "auc_M2",
            "auc_mech"]].round(3).to_string(index=False), flush=True)

rows = []
for tau_scale in (TAU_SCALE, 0.47):
    for delta in (0.0, DELTA / 2, DELTA):
        for r in (1.0, 1.1, 1.25, 1.5):
            for n in (100, 200):
                for m in (100, 1000):
                    res = [simulate(n, m, delta=delta, r=r, tau_scale=tau_scale, do_cv=False)
                           for _ in range(a.grid_reps)]
                    rows.append(summarise(res, n_per_class=n, cells_per_tumour=m, delta=delta, r=r,
                                          tau_loc=TAU_LOC, tau_scale=tau_scale))
GRID = pd.DataFrame(rows)
GRID.to_csv(OUT / "plasmode_grid.csv", index=False)
print(GRID.pivot_table(index=["tau_scale", "delta", "r"], columns=["n_per_class", "cells_per_tumour"],
                       values="power_M2_vs_M1").round(2).to_string(), flush=True)
json.dump({"scores": src, "n_patients": int(len(Y)), "n_gbm": int(Y.sum()), "score_auroc": float(base_auc),
           "templates": len(TPL), "tau_loc": TAU_LOC, "tau_scale": TAU_SCALE, "delta": DELTA, "s0": S0,
           "tail_shift": TAIL_SHIFT, "redundant_beta": float(b_red), "reps_main": a.reps, "reps_grid": a.grid_reps,
           "anchors": {k: float(v) for k, v in anch.items()}},
          open(OUT / "plasmode_inputs.json", "w"), indent=2)
print("PLASMODE_DONE", flush=True)
