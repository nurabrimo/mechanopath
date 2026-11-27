"""Phase 2b - single-cell mechanics of human tumour and matched healthy tissue (RT-FDC):
does the distribution of cell responses carry diagnostic information beyond its average?

Data: DCOR circle soteriou-kubankova (Soteriou, Kubankova et al., Nat Biomed Eng 2023, doi:10.1038/s41551-023-01015-3),
human biopsies measured by real-time fluorescence and deformability cytometry (20 um channel, 0.06 ul/s,
CellCarrier B). 39 patient-tissue units (14 fresh colon, 16 frozen colon, 9 fresh lung), one tumour and one
matched healthy measurement per unit = 78 measurements; 8 units belong to the blind test sets of the original
study. Scalar per-event features only (dcc/06_dcor_extract.py, dcc/17_dcor_compact.py): up to 5000 gated events
per measurement (random subset of all gated events).

This is a test of the premise on a different disease (colorectal and lung carcinoma versus adjacent normal
tissue), not of glioma subtypes: it asks whether, in real human tumour biopsies, the shape of the single-cell
response distribution separates tumour from healthy tissue, and how much of that separation a location summary
(median) already contains.

Gate (original study): area >= 25 um2, area ratio < 1.10 (< 1.05 below 60 um2), aspect ratio <= 2.
Mechanical response = log10 deformation; cell size = log10 area. Young's modulus is not used as the primary
quantity (the look-up table covers only 21-51 % of the events and the original study does not report it); it is
analysed in a sensitivity step.

Steps (stage names for --stages)
  qc       event flow, per-measurement descriptors (per_measurement.csv, gating_flow.csv)
  paired   paired tumour - healthy differences of descriptors, Wilcoxon signed-rank, BH q (paired_effects.csv)
  distance W1 distance decomposition into median shift and median-centred shape, split-half noise floor,
           PERMANOVA with labels swapped inside units (distance_pair_types.csv, permanova.csv)
  classify leave-one-unit-out classification of tumour versus healthy with location, spread and
           distribution descriptors, cluster-bootstrap and within-unit permutation tests of the gains,
           blind-set validation, cell-count curve, positive controls (classification_*.csv)
  sens     sensitivity analyses (permissive gate, large cells, Young's modulus on LUT-covered events,
           temperature, acquisition-time drift)
  figdata  small tables for plotting off the cluster (figdata_*.csv: 24 quantiles of log10 deformation and log10
           area per measurement, paired descriptors and distances per unit, out-of-fold scores of the primary models)

Outputs: results/phase2b/*.csv
"""
import argparse
import os
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW, RES, SEED  # noqa: E402

warnings.filterwarnings("ignore")
OUT = RES / "phase2b"
OUT.mkdir(exist_ok=True)
SRC = Path(os.environ.get("DCOR_COMPACT", RAW / "dcor"))
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QN = ["q05", "q10", "q25", "median", "q75", "q90", "q95"]


# ----------------------------------------------------------------------------------------------
# data and gates
# ----------------------------------------------------------------------------------------------
def load():
    ev = pd.read_csv(SRC / "dcor_events_compact.csv.gz")
    mt = pd.read_csv(SRC / "dcor_measurements_full.csv").rename(columns={"id": "measurement"})
    mt["unit"] = mt["patient"]
    mt["stratum"] = mt["tissue"] + "-" + mt["preservation"]
    mt["y"] = (mt["condition"] == "tumour").astype(int)
    cnt = mt.groupby("unit").y.agg(["size", "sum"])
    assert (cnt["size"] == 2).all() and (cnt["sum"] == 1).all(), "every unit needs one tumour and one healthy measurement"
    return ev, mt


def gate_mask(df, mode):
    ok = np.isfinite(df.area_um) & np.isfinite(df.deform) & (df.deform > 0)
    if mode == "permissive":                                    # what the DCC extraction already applied
        return ok & (df.area_um >= 20) & (df.area_ratio < 1.2)
    ar_lim = np.where(df.area_um < 60, 1.05, 1.10)
    ok &= (df.area_um >= 25) & (df.area_ratio < ar_lim)
    ok &= (df.aspect <= 2.0) & (df.aspect >= 0.5)
    if mode == "large":
        ok &= df.area_um >= 100
    return ok


def prepare(ev, mt, mode="paper"):
    ev = ev.copy()
    m = gate_mask(ev, mode)
    flow = ev.assign(ok=m).groupby("measurement").agg(sampled=("ok", "size"), kept=("ok", "sum")).reset_index()
    ev = ev[m].copy()
    ev["logA"] = np.log10(ev.area_um)
    ev["logD"] = np.log10(ev.deform)
    E = {k: g.reset_index(drop=True) for k, g in ev.groupby("measurement")}
    flow = flow.merge(mt[["measurement", "unit", "stratum", "condition", "n_total", "n_gated"]], on="measurement")
    flow["retained_of_sample"] = flow.kept / flow.sampled
    flow["est_events_after_gate"] = (flow.n_gated * flow.retained_of_sample).round()
    return E, flow


# ----------------------------------------------------------------------------------------------
# descriptors
# ----------------------------------------------------------------------------------------------
def describe(g, prefix_cols=(("logD", "D"), ("logA", "A"))):
    f = {"n_events": len(g)}
    for col, p in prefix_cols:
        v = g[col].to_numpy(float)
        q = np.quantile(v, QS)
        for nm, val in zip(QN, q):
            f[f"{p}_{nm}"] = val
        f[f"{p}_iqr"] = q[4] - q[2]
        f[f"{p}_p90_p10"] = q[5] - q[1]
        f[f"{p}_mean"] = v.mean()
        f[f"{p}_sd"] = v.std(ddof=1)
        f[f"{p}_mad"] = stats.median_abs_deviation(v, scale="normal")
        f[f"{p}_skew"] = stats.skew(v)
        f[f"{p}_kurt"] = stats.kurtosis(v)
    for col in ("bright_avg", "bright_sd"):
        v = g[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) > 5:
            f[f"{col}_median"], f[f"{col}_iqr"] = np.median(v), np.subtract(*np.percentile(v, [75, 25]))
    f["deform_median"] = float(np.median(g.deform))
    f["area_median"] = float(np.median(g.area_um))
    return f


def per_measurement(E, mt):
    rows = []
    for m, g in E.items():
        f = describe(g)
        f["measurement"] = m
        rows.append(f)
    PM = pd.DataFrame(rows).merge(mt[["measurement", "unit", "stratum", "tissue", "preservation", "condition", "y", "role", "date",
                                      "temperature", "temp_used", "frac_emodulus_valid"]], on="measurement")
    return PM


FEATSETS = {
    "median deformation": ["D_median"],
    "mean deformation": ["D_mean"],
    "median deformation + size": ["D_median", "A_median"],
    "median + spread (deformation)": ["D_median", "D_iqr"],
    "median + spread (deformation, size)": ["D_median", "D_iqr", "A_median", "A_iqr"],
    "moments (deformation)": ["D_mean", "D_sd", "D_skew", "D_kurt"],
    "deciles (deformation)": ["D_q05", "D_q10", "D_q25", "D_median", "D_q75", "D_q90", "D_q95"],
    "deciles (deformation, size)": ["D_q05", "D_q10", "D_q25", "D_median", "D_q75", "D_q90", "D_q95",
                                    "A_q05", "A_q10", "A_q25", "A_median", "A_q75", "A_q90", "A_q95"],
    "deciles (deformation, size) + brightness": ["D_q05", "D_q10", "D_q25", "D_median", "D_q75", "D_q90", "D_q95",
                                                  "A_q05", "A_q10", "A_q25", "A_median", "A_q75", "A_q90", "A_q95",
                                                  "bright_avg_median", "bright_avg_iqr", "bright_sd_median", "bright_sd_iqr"],
    "brightness only": ["bright_avg_median", "bright_avg_iqr", "bright_sd_median", "bright_sd_iqr"],
}
PRIMARY = [("deciles (deformation)", "median deformation"), ("deciles (deformation, size)", "median deformation + size")]


# ----------------------------------------------------------------------------------------------
# stage: paired effects
# ----------------------------------------------------------------------------------------------
def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    n = len(p)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def paired_effects(PM):
    cols = [c for c in PM.columns if c.split("_")[0] in ("D", "A", "bright", "deform", "area") and PM[c].dtype.kind == "f"]
    T = PM[PM.y == 1].set_index("unit")
    H = PM[PM.y == 0].set_index("unit").loc[T.index]
    rows = []
    r = np.random.default_rng(SEED)
    for c in cols:
        for lab, idx in [("all", T.index)] + [(s, T.index[T.stratum == s]) for s in sorted(T.stratum.unique())]:
            d = (T.loc[idx, c] - H.loc[idx, c]).dropna()
            if len(d) < 5:
                continue
            bs = np.array([d.values[r.integers(0, len(d), len(d))].mean() for _ in range(2000)])
            try:
                w = stats.wilcoxon(d, zero_method="wilcox")
                p = w.pvalue
            except ValueError:
                p = np.nan
            rows.append({"descriptor": c, "stratum": lab, "n_units": len(d), "mean_diff": d.mean(), "median_diff": d.median(),
                         "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5), "d_z": d.mean() / d.std(ddof=1),
                         "frac_units_tumour_higher": float((d > 0).mean()), "p_wilcoxon": p})
    R = pd.DataFrame(rows)
    R["q_bh"] = np.nan
    for lab, g in R.groupby("stratum"):
        R.loc[g.index, "q_bh"] = bh(g.p_wilcoxon.fillna(1).values)
    R.to_csv(OUT / "paired_effects.csv", index=False)
    return R


# ----------------------------------------------------------------------------------------------
# stage: distance decomposition and PERMANOVA
# ----------------------------------------------------------------------------------------------
def w1(a, b, n=200):
    u = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, u) - np.quantile(b, u))))


def pseudo_f(D2, labels):
    N = len(labels)
    labs, inv = np.unique(labels, return_inverse=True)
    g = len(labs)
    sst = D2[np.triu_indices(N, 1)].sum() / N
    ssw = 0.0
    for k in range(g):
        idx = np.where(inv == k)[0]
        if len(idx) > 1:
            sub = D2[np.ix_(idx, idx)]
            ssw += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
    ssb = sst - ssw
    return (ssb / (g - 1)) / (ssw / (N - g)), ssb / sst


def permanova_swap(D, labels, units, n_perm=20000, seed=SEED):
    """Pseudo-F with the tumour/healthy label swapped inside each unit (exact paired permutation scheme).
    The statistic is computed on D after removing the unit effect is not needed: units enter through the
    restricted permutation only."""
    r = np.random.default_rng(seed)
    D2 = D ** 2
    labels = np.asarray(labels)
    f0, r2 = pseudo_f(D2, labels)
    pairs = [np.where(units == u)[0] for u in np.unique(units)]
    ge = 1
    for _ in range(n_perm):
        pl = labels.copy()
        flip = r.random(len(pairs)) < 0.5
        for (i, j), fl in zip(pairs, flip):
            if fl:
                pl[i], pl[j] = labels[j], labels[i]
        if pseudo_f(D2, pl)[0] >= f0 - 1e-12:
            ge += 1
    return f0, r2, ge / (n_perm + 1)


def distance_stage(E, PM, n_perm):
    out_pairs, out_perm, out_types = [], [], []
    meas = list(PM.measurement)
    r = np.random.default_rng(SEED)
    for col, lab in [("logD", "log10 deformation"), ("logA", "log10 area")]:
        V = {m: E[m][col].to_numpy(float) for m in meas}
        N = len(meas)
        Dfull, Dloc, Dshape = (np.zeros((N, N)) for _ in range(3))
        for i in range(N):
            for j in range(i + 1, N):
                a, b = V[meas[i]], V[meas[j]]
                ma, mb = np.median(a), np.median(b)
                Dfull[i, j] = w1(a, b)
                Dloc[i, j] = abs(ma - mb)
                Dshape[i, j] = w1(a - ma, b - mb)
        for D in (Dfull, Dloc, Dshape):
            D += D.T
        floor = []
        for m in meas:
            v = r.permutation(V[m])
            h = len(v) // 2
            floor.append(w1(v[:h] - np.median(v[:h]), v[h:2 * h] - np.median(v[h:2 * h])))
        iu = np.triu_indices(N, 1)
        pr = pd.DataFrame({"i": np.array(meas)[iu[0]], "j": np.array(meas)[iu[1]], "w1_full": Dfull[iu], "median_shift": Dloc[iu],
                           "shape_w1": Dshape[iu]})
        cm = PM.set_index("measurement")
        for k in ("unit", "stratum", "condition"):
            pr["i_" + k] = cm[k].reindex(pr.i).values
            pr["j_" + k] = cm[k].reindex(pr.j).values
        pr["feature"] = lab
        out_pairs.append(pr)
        same_u = pr.i_unit == pr.j_unit
        same_s = pr.i_stratum == pr.j_stratum
        same_c = pr.i_condition == pr.j_condition
        for tname, mask in [("same unit: tumour vs healthy", same_u),
                            ("same stratum, different units, same condition: tumour-tumour / healthy-healthy", ~same_u & same_s & same_c),
                            ("same stratum, different units, different condition", ~same_u & same_s & ~same_c)]:
            s = pr[mask]
            out_types.append({"feature": lab, "pair type": tname, "n_pairs": len(s), "w1_full": s.w1_full.mean(),
                              "median_shift": s.median_shift.mean(), "shape_w1": s.shape_w1.mean(),
                              "location_share_of_w1": float((s.median_shift / s.w1_full).mean())})
        out_types.append({"feature": lab, "pair type": "split-half noise floor (shape)", "n_pairs": len(floor), "shape_w1": float(np.mean(floor))})
        labels = PM.condition.values
        units = PM.unit.values
        for dn, D in [("full W1", Dfull), ("median shift", Dloc), ("shape (median-centred W1)", Dshape)]:
            f0, r2, p = permanova_swap(D, labels, units, n_perm)
            out_perm.append({"feature": lab, "distance": dn, "pseudo_F": f0, "R2": r2, "p_perm": p, "n_measurements": N, "n_perm": n_perm})
    pd.concat(out_pairs).to_csv(OUT / "distance_decomposition.csv", index=False)
    pd.DataFrame(out_types).to_csv(OUT / "distance_pair_types.csv", index=False)
    PR = pd.DataFrame(out_perm)
    PR.to_csv(OUT / "permanova.csv", index=False)
    return PR


# ----------------------------------------------------------------------------------------------
# stage: classification
# ----------------------------------------------------------------------------------------------
def bags(E, n_cells, n_bags, seed):
    rows = []
    for m, g in E.items():
        k = min(n_cells, len(g))
        r = np.random.default_rng(seed + zlib.crc32(m.encode()))
        for b in range(n_bags):
            f = describe(g.iloc[r.choice(len(g), k, replace=False)])
            f["measurement"] = m
            rows.append(f)
    return pd.DataFrame(rows)


def design(B, MT, cols):
    X = B[cols].to_numpy(float)
    S = pd.get_dummies(MT.set_index("measurement").loc[B.measurement, "stratum"]).to_numpy(float)
    return np.hstack([X, S]) if S.shape[1] > 1 else X


def unit_folds(MT, k, rep):
    """assign whole units to k folds, balanced within stratum (tumour and healthy of one unit never separated)"""
    r = np.random.default_rng(SEED + 101 * rep)
    fold, off = {}, int(r.integers(k))
    for _, g in MT.drop_duplicates("unit").groupby("stratum"):
        for i, u in enumerate(r.permutation(g.unit.values)):
            fold[u] = (i + off) % k
        off = (off + len(g)) % k
    return fold


def cv_scores(B, MT, cols, C=0.1, k=5, repeats=10, y=None):
    """grouped K-fold (units) decision values, averaged over the bags of a measurement; one column per repeat"""
    mt = MT.set_index("measurement")
    yb = (mt.loc[B.measurement, "y"] if y is None else y.loc[B.measurement]).to_numpy()
    ub = mt.loc[B.measurement, "unit"].to_numpy()
    X = design(B, MT, cols)
    out = {}
    for rep in range(repeats):
        fold = unit_folds(MT, k, rep)
        fb = np.array([fold[u] for u in ub])
        s = np.zeros(len(B))
        for f in range(k):
            te = fb == f
            if not te.any() or len(np.unique(yb[~te])) < 2:
                continue
            sc = StandardScaler().fit(X[~te])
            clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[~te]), yb[~te])
            s[te] = clf.decision_function(sc.transform(X[te]))
        out[rep] = pd.Series(s, index=B.measurement.values).groupby(level=0).mean()
    return pd.DataFrame(out)


def paired_acc(sc, MT):
    mt = MT.set_index("measurement").loc[sc.index]
    t = sc[mt.y == 1].groupby(mt.loc[mt.y == 1, "unit"]).mean()
    h = sc[mt.y == 0].groupby(mt.loc[mt.y == 0, "unit"]).mean()
    d = (t - h.reindex(t.index))
    return float(((d > 0) + 0.5 * (d == 0)).mean())


def metrics(S, MT, y=None):
    """mean over repeats of pooled AUROC and of the within-unit (matched) accuracy; S has one column per repeat"""
    yy = MT.set_index("measurement").loc[S.index, "y"] if y is None else y.loc[S.index]
    aucs = [roc_auc_score(yy, S[c].values) for c in S.columns]
    paccs = [paired_acc(S[c], MT) for c in S.columns]
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(paccs))


def cluster_boot_delta(s1, s2, MT, n=2000, seed=SEED):
    """paired cluster bootstrap over units of AUROC(s2) - AUROC(s1) using repeat-averaged out-of-fold scores"""
    mt = MT.set_index("measurement").loc[s1.index]
    units = mt.unit.unique()
    by = {u: np.where(mt.unit.values == u)[0] for u in units}
    y, a, b = mt.y.values, s1.values, s2.values
    r = np.random.default_rng(seed)
    d0 = roc_auc_score(y, b) - roc_auc_score(y, a)
    ds = []
    for _ in range(n):
        idx = np.concatenate([by[u] for u in r.choice(units, len(units))])
        if y[idx].min() == y[idx].max():
            continue
        ds.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    ds = np.array(ds)
    return d0, np.percentile(ds, [2.5, 97.5]), 2 * min((ds <= 0).mean(), (ds >= 0).mean())


def swap_labels(MT, r):
    """tumour/healthy label swapped inside randomly chosen units (restricted permutation)"""
    y = MT.set_index("measurement").y.copy()
    for u, g in MT.groupby("unit"):
        if r.random() < 0.5:
            y.loc[g.measurement] = 1 - y.loc[g.measurement].values
    return y


def delta_stat(B, MT, base_cols, new_cols, C, y=None, repeats=3):
    sb = cv_scores(B, MT, base_cols, C=C, repeats=repeats, y=y)
    sn = cv_scores(B, MT, new_cols, C=C, repeats=repeats, y=y)
    return metrics(sn, MT, y)[0] - metrics(sb, MT, y)[0]


def perm_delta(args):
    seed, B, MT, base_cols, new_cols, C = args
    yp = swap_labels(MT, np.random.default_rng(seed))
    return delta_stat(B, MT, base_cols, new_cols, C, y=yp)


def matched_design(B, MT, cols, C=0.1):
    """within-unit analysis: classifier on descriptor differences (tumour bag minus healthy bag of the same unit),
    leave-one-unit-out; returns the share of units whose tumour measurement scores higher than the healthy one"""
    mt = MT.set_index("measurement")
    Bt = B[mt.loc[B.measurement, "y"].values == 1].copy()
    Bh = B[mt.loc[B.measurement, "y"].values == 0].copy()
    Bt["unit"], Bh["unit"] = mt.loc[Bt.measurement, "unit"].values, mt.loc[Bh.measurement, "unit"].values
    Bt["b"], Bh["b"] = Bt.groupby("measurement").cumcount().values, Bh.groupby("measurement").cumcount().values
    M = Bt.merge(Bh, on=["unit", "b"], suffixes=("_t", "_h"))
    D = np.column_stack([M[f"{c}_t"].to_numpy(float) - M[f"{c}_h"].to_numpy(float) for c in cols])
    units = M.unit.to_numpy()
    sc = D.std(0) + 1e-12
    win = []
    for u in np.unique(units):
        te = units == u
        Xtr = np.vstack([D[~te] / sc, -D[~te] / sc])
        ytr = np.r_[np.ones((~te).sum()), np.zeros((~te).sum())]
        clf = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(Xtr, ytr)
        win.append(float(np.mean(clf.decision_function(D[te] / sc))) > 0)
    return float(np.mean(win)), len(win)


def classify_stage(E, PM, MT, n_cells, n_bags, n_perm, jobs, C=0.1):
    t0 = time.time()
    B = bags(E, n_cells, n_bags, SEED)
    res, oof, S = [], [], {}
    for name, cols in FEATSETS.items():
        sc = cv_scores(B, MT, cols, C=C)
        S[name] = sc.mean(axis=1)
        auc, sd, pacc = metrics(sc, MT)
        mt = MT.set_index("measurement").loc[sc.index]
        ws = {s: float(np.mean([roc_auc_score(mt.y[mt.stratum == s], sc[c][mt.stratum == s].values) for c in sc.columns]))
              for s in sorted(mt.stratum.unique())}
        pm, nu = matched_design(B, MT, cols, C=C)
        res.append({"features": name, "n_features": len(cols), "auroc_pooled": auc, "auroc_sd_over_repeats": sd, "paired_accuracy": pacc,
                    "matched_design_accuracy": pm, **{f"auroc_{k}": v for k, v in ws.items()}})
        oof.append(pd.DataFrame({"measurement": S[name].index, "features": name, "score": S[name].values}))
    R = pd.DataFrame(res)
    R.to_csv(OUT / "classification_cv.csv", index=False)
    pd.concat(oof).to_csv(OUT / "classification_oof.csv", index=False)
    print(R.round(3).to_string(index=False), flush=True)
    gains = []
    for new, base in PRIMARY + [("deciles (deformation, size) + brightness", "median deformation + size"),
                                ("median + spread (deformation)", "median deformation"),
                                ("moments (deformation)", "median deformation"),
                                ("median deformation + size", "median deformation"),
                                ("deciles (deformation, size)", "median + spread (deformation, size)")]:
        d0, ci, pb = cluster_boot_delta(S[base], S[new], MT)
        gains.append({"baseline": base, "model": new, "delta_auroc": d0, "ci_lo": ci[0], "ci_hi": ci[1], "p_boot": pb,
                      "primary": (new, base) in PRIMARY})
    G = pd.DataFrame(gains)
    if n_perm > 0:
        for i, row in G.iterrows():
            if not row.primary:
                continue
            obs = delta_stat(B, MT, FEATSETS[row.baseline], FEATSETS[row.model], C)
            nulls = np.array(Parallel(n_jobs=jobs)(delayed(perm_delta)((SEED + 17 * k, B, MT, FEATSETS[row.baseline], FEATSETS[row.model], C))
                                                    for k in range(n_perm)))
            G.loc[i, "delta_perm_statistic"] = obs
            G.loc[i, "p_perm_one_sided"] = (1 + (nulls >= obs).sum()) / (1 + len(nulls))
            G.loc[i, "null_mean"], G.loc[i, "null_sd"], G.loc[i, "null_q95"] = nulls.mean(), nulls.std(), np.quantile(nulls, 0.95)
            pd.Series(nulls).to_csv(OUT / f"perm_null_{i}.csv", index=False, header=["delta_auroc_null"])
            print(f"permutation {row.model} vs {row.baseline}: obs {obs:.3f} p {G.loc[i, 'p_perm_one_sided']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    G.to_csv(OUT / "classification_gains.csv", index=False)
    print(G.round(4).to_string(index=False), flush=True)

    mt = MT.set_index("measurement")
    bl = []
    for name, cols in FEATSETS.items():
        X = design(B, MT, cols)
        role = mt.loc[B.measurement, "role"].to_numpy()
        y = mt.loc[B.measurement, "y"].to_numpy()
        tr, te = role == "main", role == "blind-set"
        sc_ = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc_.transform(X[tr]), y[tr])
        s = pd.Series(clf.decision_function(sc_.transform(X[te])), index=B.measurement.values[te]).groupby(level=0).mean()
        yy = mt.loc[s.index, "y"]
        n_u = mt.loc[s.index, "unit"].nunique()
        pacc = paired_acc(s, MT)
        bl.append({"features": name, "n_units": n_u, "auroc": roc_auc_score(yy, s.values), "paired_accuracy": pacc,
                   "units_correct": round(pacc * n_u, 1)})
    pd.DataFrame(bl).to_csv(OUT / "blind_validation.csv", index=False)
    print(pd.DataFrame(bl).round(3).to_string(index=False), flush=True)

    cur = []
    for nc in [30, 100, 300, 1000, 3000]:
        Bn = bags(E, nc, max(6, n_bags) if nc < 300 else n_bags, SEED + nc)
        for name in ["median deformation", "median deformation + size", "deciles (deformation)", "deciles (deformation, size)"]:
            auc, sd, pacc = metrics(cv_scores(Bn, MT, FEATSETS[name], C=C, repeats=5), MT)
            cur.append({"cells_per_bag": nc, "features": name, "auroc_pooled": auc, "auroc_sd_over_repeats": sd, "paired_accuracy": pacc})
    pd.DataFrame(cur).to_csv(OUT / "classification_cells.csv", index=False)
    print(pd.DataFrame(cur).pivot(index="cells_per_bag", columns="features", values="auroc_pooled").round(3).to_string(), flush=True)

    ctrl = []
    for target in ["tissue", "preservation"]:
        sub = MT[MT.condition == "healthy"] if target == "tissue" else MT[(MT.tissue == "colon") & (MT.condition == "healthy")]
        pos = "colon" if target == "tissue" else "frozen"
        lab = (sub[target] == pos).astype(int).set_axis(sub.measurement)
        Bs = B[B.measurement.isin(sub.measurement)]
        sub2 = sub.copy()
        sub2["y"] = lab.loc[sub2.measurement].values
        for name in ["median deformation", "median deformation + size", "deciles (deformation, size)"]:
            sc = cv_scores(Bs, sub2.assign(stratum="all"), FEATSETS[name], C=C, repeats=5)
            ctrl.append({"control": f"healthy tissue: {target} ({pos} vs other)", "features": name,
                         "auroc": float(np.mean([roc_auc_score(lab.loc[sc.index], sc[c].values) for c in sc.columns])), "n_measurements": len(sc)})
    pd.DataFrame(ctrl).to_csv(OUT / "positive_controls.csv", index=False)
    print(pd.DataFrame(ctrl).round(3).to_string(index=False), flush=True)
    return S


# ----------------------------------------------------------------------------------------------
# stage: sensitivity
# ----------------------------------------------------------------------------------------------
def sens_stage(ev, mt, n_cells, n_bags, C=0.1):
    rows = []

    def one(analysis, B, MT, name, C_, n_units=None):
        """One cross-validated analysis; a failure is logged and recorded, never fatal."""
        try:
            sc = cv_scores(B, MT, FEATSETS[name] if isinstance(name, str) else name[1], C=C_, repeats=5)
            auc, _, pacc = metrics(sc, MT)
        except Exception as ex:                                   # noqa: BLE001
            print(f"{analysis} / {name}: {type(ex).__name__}: {ex}", flush=True)
            auc = pacc = np.nan
        rows.append({"analysis": analysis, "n_units": MT.unit.nunique() if n_units is None else n_units,
                     "features": name if isinstance(name, str) else name[0], "auroc_pooled": auc, "paired_accuracy": pacc})
    for mode in ["paper", "permissive", "large"]:
        E, flow = prepare(ev, mt, mode)
        keep = [m for m, g in E.items() if len(g) >= 200]
        if len(keep) < len(mt):
            ok_units = mt[mt.measurement.isin(keep)].groupby("unit").size()
            ok_units = ok_units[ok_units == 2].index
            MT = mt[mt.unit.isin(ok_units)]
        else:
            MT = mt
        if MT.unit.nunique() < 8:                                # too few units with enough events under this gate
            print(f"gate {mode}: {MT.unit.nunique()} units with >= 200 events in both measurements; not analysed", flush=True)
            rows.append({"analysis": f"gate: {mode}", "n_units": MT.unit.nunique(), "features": "(fewer than 8 evaluable units)",
                         "auroc_pooled": np.nan, "paired_accuracy": np.nan})
            continue
        Es = {m: E[m] for m in MT.measurement}
        B = bags(Es, n_cells, n_bags, SEED)
        for name in ["median deformation", "median deformation + size", "deciles (deformation)", "deciles (deformation, size)"]:
            one(f"gate: {mode}", B, MT, name, C)
    E, _ = prepare(ev, mt, "paper")
    Bp = bags(E, n_cells, n_bags, SEED)
    for cc in [0.03, 0.3, 1.0]:
        for name in ["median deformation", "median deformation + size", "deciles (deformation)", "deciles (deformation, size)"]:
            one(f"regularisation C={cc}", Bp, mt, name, cc)
    # per-stratum and without blind-set units
    for lab, sel in [(f"stratum {s}", mt.stratum == s) for s in sorted(mt.stratum.unique())] + [("main units only", mt.role == "main")]:
        MT = mt[sel]
        Bs = Bp[Bp.measurement.isin(MT.measurement)]
        for name in ["median deformation", "median deformation + size", "deciles (deformation)", "deciles (deformation, size)"]:
            one(lab, Bs, MT, name, C)
    # Young's modulus on LUT-covered events
    E2 = {}
    for m, g in ev.groupby("measurement"):
        g = g[gate_mask(g, "paper") & np.isfinite(g.emodulus)]
        g = g.assign(logD=np.log10(g.deform), logA=np.log10(g.area_um), logE=np.log10(g.emodulus))
        if len(g) >= 200:
            E2[m] = g.reset_index(drop=True)
    ok_units = mt[mt.measurement.isin(E2)].groupby("unit").size()
    ok_units = ok_units[ok_units == 2].index
    MT2 = mt[mt.unit.isin(ok_units)]
    rowsE = []
    for m in MT2.measurement:
        f = describe(E2[m], (("logD", "D"), ("logA", "A"), ("logE", "E")))
        f["measurement"] = m
        rowsE.append(f)
    PE = pd.DataFrame(rowsE).merge(MT2[["measurement", "unit", "stratum", "y"]], on="measurement")
    FE = {"median log10 E (LUT-covered events)": ["E_median"], "deciles of log10 E": ["E_q05", "E_q10", "E_q25", "E_median", "E_q75", "E_q90", "E_q95"],
          "median log10 E + median size": ["E_median", "A_median"]}
    for name, cols in FE.items():
        one("Young's modulus (LUT-covered events, one fit per measurement)", PE, MT2, (name, cols), C)
    # temperature and acquisition-time drift
    drift = []
    for m, g in E.items():
        g = g.sort_values("time")
        h = len(g) // 2
        drift.append({"measurement": m, "dD_median_second_minus_first_half": g.logD.iloc[h:].median() - g.logD.iloc[:h].median(),
                      "dA_median_second_minus_first_half": g.logA.iloc[h:].median() - g.logA.iloc[:h].median()})
    drift = pd.DataFrame(drift).merge(mt[["measurement", "condition", "stratum"]], on="measurement")
    drift.to_csv(OUT / "drift_halves.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# stage: compact tables for plotting
# ----------------------------------------------------------------------------------------------
def figdata_stage(E, PM, MT):
    P = np.round(np.arange(0.04, 0.9601, 0.04), 2)
    rows = []
    for m, g in E.items():
        for col in ("logD", "logA"):
            q = np.quantile(g[col].to_numpy(float), P)
            rows.append({"measurement": m, "feature": col, **{f"q{int(round(p * 100)):02d}": round(float(v), 4) for p, v in zip(P, q)}})
    Q = pd.DataFrame(rows).merge(MT[["measurement", "unit", "stratum", "condition"]], on="measurement")
    Q = Q[["unit", "stratum", "condition", "feature"] + [c for c in Q.columns if c.startswith("q")]]
    Q.sort_values(["stratum", "unit", "condition", "feature"]).to_csv(OUT / "figdata_quantiles.csv", index=False)

    keep = ["D_median", "D_iqr", "D_skew", "D_kurt", "A_median", "A_iqr", "A_skew"]
    T = PM[PM.y == 1].set_index("unit")[keep + ["stratum", "role", "n_events"]]
    H = PM[PM.y == 0].set_index("unit").loc[T.index][keep + ["n_events"]]
    U = T.join(H, lsuffix="_t", rsuffix="_h")
    for u in U.index:
        a = E[PM[(PM.unit == u) & (PM.y == 1)].measurement.iloc[0]].logD.to_numpy(float)
        b = E[PM[(PM.unit == u) & (PM.y == 0)].measurement.iloc[0]].logD.to_numpy(float)
        ma, mb = np.median(a), np.median(b)
        U.loc[u, "w1_full"], U.loc[u, "median_shift"], U.loc[u, "shape_w1"] = w1(a, b), abs(ma - mb), w1(a - ma, b - mb)
    U.round(5).reset_index().to_csv(OUT / "figdata_pairs.csv", index=False)

    f = OUT / "classification_oof.csv"
    if f.exists():
        O = pd.read_csv(f)
        O = O[O.features.isin([x for pair in PRIMARY for x in pair])]
        O = O.merge(MT[["measurement", "unit", "condition"]], on="measurement")[["unit", "condition", "features", "score"]]
        O["score"] = O.score.round(4)
        O.to_csv(OUT / "figdata_scores.csv", index=False)
    gs = (PM.groupby(["stratum", "condition"]).n_events.agg(["count", "median", "min", "max"]).reset_index())
    gs.to_csv(OUT / "gating_summary.csv", index=False)
    print("figdata written", flush=True)


def variance_stage(PM):
    rows = []
    for c in ["D_median", "D_iqr", "A_median", "A_iqr"]:
        full = smf.ols(f"{c} ~ C(condition) + C(unit)", PM).fit()
        for term in ["C(condition)", "C(unit)"]:
            red = smf.ols(f"{c} ~ " + ("C(unit)" if term == "C(condition)" else "C(condition)"), PM).fit()
            rows.append({"descriptor": c, "component": term, "partial_R2": (red.ssr - full.ssr) / red.ssr})
        rows.append({"descriptor": c, "component": "residual", "partial_R2": 1 - full.rsquared})
    VP = pd.DataFrame(rows)
    VP.to_csv(OUT / "variance_partition.csv", index=False)
    return VP


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="qc,paired,distance,classify,sens,figdata")
    ap.add_argument("--cells", type=int, default=1000)
    ap.add_argument("--bags", type=int, default=12)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--permanova-perm", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    stages = a.stages.split(",")
    ev, mt = load()
    E, flow = prepare(ev, mt, "paper")
    flow.to_csv(OUT / "gating_flow.csv", index=False)
    PM = per_measurement(E, mt)
    PM.to_csv(OUT / "per_measurement.csv", index=False)
    print(f"{len(mt)} measurements, {mt.unit.nunique()} units; events after gate: median {flow.kept.median():.0f}, min {flow.kept.min()}", flush=True)
    if "paired" in stages:
        R = paired_effects(PM)
        variance_stage(PM)
        print(R[R.stratum == "all"].sort_values("p_wilcoxon").head(20).round(4).to_string(index=False), flush=True)
    if "distance" in stages:
        print(distance_stage(E, PM, a.permanova_perm).round(4).to_string(index=False), flush=True)
    if "classify" in stages:
        classify_stage(E, PM, mt, a.cells, a.bags, a.perm, a.jobs)
    if "sens" in stages:
        sens_stage(ev, mt, a.cells, a.bags)
    if "figdata" in stages:
        figdata_stage(E, PM, mt)
