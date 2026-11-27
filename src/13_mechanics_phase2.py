"""Phase 2 - single-cell mechanics: is there information in the distribution beyond its median?

Data (public; no mechanical measurement is linked to any TCGA patient)
  RT-DC  Mechanomics RT-DC Set 01 - Glioblastoma (Urbanska, Herbig, Guck; figshare doi:10.6084/m9.figshare.28161236,
         CC BY 4.0). Primary glioblastoma lines X01, X04 and X08 (Poser et al., FASEB J 2019) in Serum, EGF and
         FGFJI (FGF + JAK inhibitor) culture, i.e. low, medium and high STAT3-Ser/Hes3 activation; 21 measurements
         on three dates, 21,738 gated single cells (data/processed/rtdc_events_gated.csv.gz).
  AFM    LN229 and U87 in 2D culture, 1,160 single-cell Young's moduli (RODBUK doi:10.57903/UJ/3LOV6Q, CC0); no
         replicate or batch labels, so AFM results are descriptive.

The design is crossed: on each date every line was measured in all three culture conditions (7 line x date blocks),
so condition effects are estimated within blocks and permutations are restricted to blocks.

Analyses (all on log10 E)
  A1 descriptors per sample (location, spread, tails, robust skewness and kurtosis) with cell-bootstrap CIs
  A2 variance partition of every descriptor: condition + line + date (type II ANOVA), restricted-permutation p
  A3 shift functions: decile differences between conditions paired within blocks; tail-minus-median and scale
     contrasts (exact sign-flip test over blocks)
  A4 shape distances after removing location (median) or location and scale (median, IQR); within- vs between-group
     distance with restricted permutation
  A5 reproducibility of each descriptor across dates within line x condition (ICC(1), group bootstrap CI)
  A6 sampling precision of each descriptor for n cells and its reliability as a sample-level marker
  A7 bag discrimination (n cells per bag): location (mean, median) vs distribution vs shape-only descriptors;
     conditions with leave-one-date-out, lines with leave-one-condition-out; cluster-bootstrap CIs
  A8 AFM: descriptors and location-aligned shape comparison of LN229 and U87
  A9 literature anchors (Shmelev et al., Int J Mol Sci 2023) converted to log10 scale

Outputs: results/phase2/*.csv, data/processed/afm_events.csv
"""
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RAW, RES, SEED  # noqa: E402

warnings.filterwarnings("ignore")
OUT = RES / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
N_PERM = 5000
N_BOOT = 2000

# ------------------------------------------------------------------------------------------------ data
ev = pd.read_csv(PROC / "rtdc_events_gated.csv.gz")
ev["logE"] = np.log10(ev.emodulus)
ev["date"] = ev.date.astype(str)
COND = ["Serum", "EGF", "FGFJI"]
LINES = ["X01", "X04", "X08"]
afm_path = PROC / "afm_events.csv"
if not afm_path.exists():
    import openpyxl
    wb = openpyxl.load_workbook(RAW / "mechanics/rodbuk/AFM_2D_Youngs_modulus.xlsx", read_only=True, data_only=True)
    rows = [r for r in wb.worksheets[0].iter_rows(values_only=True)][1:]
    afm = pd.DataFrame([(float(r[1]), str(r[2]).replace("LN299", "LN229")) for r in rows if r[1] is not None],
                       columns=["E_kPa", "line"])
    afm["logE"] = np.log10(afm.E_kPa)
    afm.to_csv(afm_path, index=False)
afm = pd.read_csv(afm_path)
meas = ev.drop_duplicates("measurement").set_index("measurement")[["date", "line", "cond"]].sort_index()
meas["n_cells"] = ev.groupby("measurement").size()
meas["block"] = meas.line + "_" + meas.date                       # line x date: one measurement per condition
X = {m: ev.loc[ev.measurement == m, "logE"].values for m in meas.index}
print(f"RT-DC: {len(ev)} cells, {len(meas)} measurements, {meas.block.nunique()} line x date blocks | "
      f"AFM: {dict(afm.line.value_counts())}")

DESC = ["mean", "median", "iqr", "q10", "q90", "upper_tail", "lower_tail", "tail_asym", "bowley", "moors", "mad"]
DESC_LABEL = {"mean": "mean", "median": "median", "iqr": "IQR", "q10": "10th percentile", "q90": "90th percentile",
              "upper_tail": "upper tail (q90 - q50)", "lower_tail": "lower tail (q50 - q10)",
              "tail_asym": "tail asymmetry", "bowley": "quartile skewness", "moors": "octile kurtosis",
              "mad": "MAD"}


def descr_matrix(S):
    """Descriptors of every row of S (k samples x n cells), vectorised."""
    q = np.percentile(S, [10, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 90], axis=-1)
    q10, o1, q25, o3, q50, o5, q75, o7, q90 = q
    iqr = q75 - q25
    with np.errstate(divide="ignore", invalid="ignore"):
        bowley = (q75 + q25 - 2 * q50) / iqr
        moors = ((o7 - o5) + (o3 - o1)) / iqr
    mad = np.median(np.abs(S - q50[..., None]), axis=-1)
    return {"mean": S.mean(-1), "median": q50, "iqr": iqr, "q10": q10, "q90": q90, "upper_tail": q90 - q50, "lower_tail": q50 - q10,
            "tail_asym": (q90 - q50) - (q50 - q10), "bowley": bowley, "moors": moors, "mad": mad}


def descr(x):
    d = descr_matrix(np.asarray(x)[None, :])
    return {k: float(v[0]) for k, v in d.items()}


# ------------------------------------------------------------------------------------------------ A1
rows = []
for m, r in meas.iterrows():
    x = X[m]
    d = descr(x)
    B = descr_matrix(x[rng.integers(0, len(x), (N_BOOT // 4, len(x)))])
    g = ev[ev.measurement == m]
    row = {"assay": "RT-DC", "sample": m, "line": r.line, "condition": r.cond, "date": r.date, "n_cells": len(x), **d}
    for k in ("median", "iqr", "upper_tail", "lower_tail", "bowley"):
        row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(B[k], [2.5, 97.5])
    row["deform_median"] = g.deform.median()
    row["area_um2_median"] = g.area_um.median()
    row["rho_logE_logarea"] = stats.spearmanr(g.logE, np.log10(g.area_um))[0]
    rows.append(row)
for ln, g in afm.groupby("line"):
    x = g.logE.values
    d = descr(x)
    B = descr_matrix(x[rng.integers(0, len(x), (N_BOOT // 4, len(x)))])
    row = {"assay": "AFM", "sample": f"AFM_{ln}", "line": ln, "condition": "2D", "date": "", "n_cells": len(x), **d}
    for k in ("median", "iqr", "upper_tail", "lower_tail", "bowley"):
        row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(B[k], [2.5, 97.5])
    rows.append(row)
PS = pd.DataFrame(rows)
PS.to_csv(OUT / "A1_sample_descriptors.csv", index=False)
D = PS[PS.assay == "RT-DC"].set_index("sample").loc[meas.index]
print("A1 done:", PS.groupby(["assay", "condition"])[["median", "iqr", "bowley"]].mean().round(3).to_dict())


# ------------------------------------------------------------------------------------------------ helpers
def perm_within(labels, blocks, rng_):
    """Permute labels within blocks (restricted permutation that respects the crossed design)."""
    out = labels.copy()
    for b in np.unique(blocks):
        i = np.where(blocks == b)[0]
        out[i] = labels[rng_.permutation(i)]
    return out


def dummies(f):
    lv = np.unique(f)
    return np.column_stack([(f == l_).astype(float) for l_ in lv[1:]])


def rss_multi(Xd, Y):
    beta, *_ = np.linalg.lstsq(Xd, Y, rcond=None)
    return ((Y - Xd @ beta) ** 2).sum(0)


# ------------------------------------------------------------------------------------------------ A2 variance partition
cond, line, date, block = (meas[c].to_numpy(dtype=object) for c in ("cond", "line", "date", "block"))
blk_cond = block                                                         # permute condition within line x date
blk_line = (meas.cond + "_" + meas.date).to_numpy(dtype=object)          # permute line within condition x date
blk_date = (meas.line + "_" + meas.cond).to_numpy(dtype=object)          # permute date within line x condition
A2_DESC = DESC + ["deform_median", "area_um2_median"]
Y = D[A2_DESC].to_numpy(dtype=float)
fac = {"condition": cond, "line": line, "date": date}
blocks = {"condition": blk_cond, "line": blk_line, "date": blk_date}
ones = np.ones((len(Y), 1))
DM = {k: dummies(v) for k, v in fac.items()}
rss_full = rss_multi(np.column_stack([ones] + list(DM.values())), Y)
sst = ((Y - Y.mean(0)) ** 2).sum(0)
res = pd.DataFrame({"descriptor": A2_DESC, "label": [DESC_LABEL.get(k, k) for k in A2_DESC], "ss_total": sst,
                    "ss_residual": rss_full, "share_residual": rss_full / sst})
for name in fac:
    rss_red = rss_multi(np.column_stack([ones] + [DM[j] for j in DM if j != name]), Y)
    ss = rss_red - rss_full
    res[f"share_{name}"] = ss / sst
    res[f"partial_eta2_{name}"] = ss / (ss + rss_full)
    obs = ss / rss_full
    others = [DM[j] for j in DM if j != name]
    exceed = np.zeros(len(A2_DESC))
    for _ in range(N_PERM):
        Dp = dummies(perm_within(fac[name], blocks[name], rng))
        r_full = rss_multi(np.column_stack([ones] + others + [Dp]), Y)
        exceed += (rss_red - r_full) / r_full >= obs - 1e-12
    res[f"p_perm_{name}"] = (exceed + 1) / (N_PERM + 1)
# sequential (type I) sums of squares, order condition -> line -> date, which partition the total exactly
# (condition is balanced within every line x day block, so its share does not depend on the order)
r_c = rss_multi(np.column_stack([ones, DM["condition"]]), Y)
r_cl = rss_multi(np.column_stack([ones, DM["condition"], DM["line"]]), Y)
res["seqshare_condition"] = (sst - r_c) / sst
res["seqshare_line"] = (r_c - r_cl) / sst
res["seqshare_date"] = (r_cl - rss_full) / sst
res["seqshare_residual"] = rss_full / sst
A2 = res
A2.to_csv(OUT / "A2_variance_partition.csv", index=False)
print(A2[["descriptor", "share_condition", "share_line", "share_date", "share_residual", "p_perm_condition",
          "p_perm_line", "p_perm_date"]].round(3).to_string(index=False))

# ------------------------------------------------------------------------------------------------ A3 shift functions
DEC = np.arange(10, 100, 10)


def signflip_p(v):
    """Exact two-sided sign-flip test of mean(v) = 0 over blocks."""
    v = np.asarray(v)
    n = len(v)
    signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
    null = np.abs((signs * v).mean(1))
    return float((null >= abs(v.mean()) - 1e-12).mean())


def tci(v):
    v = np.asarray(v)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean() - h, v.mean() + h


shift_rows, contrast_rows = [], []
for c0, c1 in [("Serum", "EGF"), ("Serum", "FGFJI"), ("EGF", "FGFJI")]:
    per_block = []
    for b, g in meas.groupby("block"):
        m0, m1 = g.index[g.cond == c0][0], g.index[g.cond == c1][0]
        q0, q1 = np.percentile(X[m0], DEC), np.percentile(X[m1], DEC)
        d0, d1 = descr(X[m0]), descr(X[m1])
        per_block.append({"block": b, **{f"d{q}": v for q, v in zip(DEC, q1 - q0)},
                          "upper_minus_median": (q1[8] - q0[8]) - (q1[4] - q0[4]),
                          "lower_minus_median": (q1[0] - q0[0]) - (q1[4] - q0[4]),
                          "log_iqr_ratio": np.log10(d1["iqr"] / d0["iqr"]),
                          "bowley_diff": d1["bowley"] - d0["bowley"], "moors_diff": d1["moors"] - d0["moors"]})
    pb = pd.DataFrame(per_block)
    for q in DEC:
        v = pb[f"d{q}"].values
        lo, hi = tci(v)
        shift_rows.append({"contrast": f"{c1} vs {c0}", "decile": q, "shift_log10E": v.mean(), "lo": lo, "hi": hi,
                           "fold_change": 10 ** v.mean(), "n_blocks": len(v), "p_signflip": signflip_p(v)})
    for k in ("upper_minus_median", "lower_minus_median", "log_iqr_ratio", "bowley_diff", "moors_diff"):
        v = pb[k].values
        lo, hi = tci(v)
        contrast_rows.append({"contrast": f"{c1} vs {c0}", "quantity": k, "mean": v.mean(), "lo": lo, "hi": hi,
                              "n_blocks": len(v), "p_signflip": signflip_p(v), "p_t": stats.ttest_1samp(v, 0).pvalue})
# lines: paired within condition x date (dates with all three lines)
for l0, l1 in combinations(LINES, 2):
    per_block = []
    for (c, d_), g in meas.groupby(["cond", "date"]):
        if not {l0, l1} <= set(g.line):
            continue
        m0, m1 = g.index[g.line == l0][0], g.index[g.line == l1][0]
        q0, q1 = np.percentile(X[m0], DEC), np.percentile(X[m1], DEC)
        d0, d1 = descr(X[m0]), descr(X[m1])
        per_block.append({**{f"d{q}": v for q, v in zip(DEC, q1 - q0)},
                          "upper_minus_median": (q1[8] - q0[8]) - (q1[4] - q0[4]),
                          "lower_minus_median": (q1[0] - q0[0]) - (q1[4] - q0[4]),
                          "log_iqr_ratio": np.log10(d1["iqr"] / d0["iqr"]),
                          "bowley_diff": d1["bowley"] - d0["bowley"], "moors_diff": d1["moors"] - d0["moors"]})
    pb = pd.DataFrame(per_block)
    for q in DEC:
        v = pb[f"d{q}"].values
        lo, hi = tci(v)
        shift_rows.append({"contrast": f"{l1} vs {l0}", "decile": q, "shift_log10E": v.mean(), "lo": lo, "hi": hi,
                           "fold_change": 10 ** v.mean(), "n_blocks": len(v), "p_signflip": signflip_p(v)})
    for k in ("upper_minus_median", "lower_minus_median", "log_iqr_ratio", "bowley_diff", "moors_diff"):
        v = pb[k].values
        lo, hi = tci(v)
        contrast_rows.append({"contrast": f"{l1} vs {l0}", "quantity": k, "mean": v.mean(), "lo": lo, "hi": hi,
                              "n_blocks": len(v), "p_signflip": signflip_p(v), "p_t": stats.ttest_1samp(v, 0).pvalue})
A3s, A3c = pd.DataFrame(shift_rows), pd.DataFrame(contrast_rows)
A3s.to_csv(OUT / "A3_shift_functions.csv", index=False)
A3c.to_csv(OUT / "A3_shape_contrasts.csv", index=False)
print(A3c.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------ A4 shape distances
def qgrid(x, k=199):
    return np.percentile(x, np.linspace(0.5, 99.5, k))


Q = {m: qgrid(X[m]) for m in meas.index}


def dist_matrix(mode):
    """1-Wasserstein distance between quantile functions after removing location (or location and scale)."""
    Z = []
    for m in meas.index:
        z = Q[m] - np.median(X[m])
        if mode == "location_scale":
            z = z / np.subtract(*np.percentile(X[m], [75, 25]))
        Z.append(z)
    Z = np.array(Z)
    return np.abs(Z[:, None, :] - Z[None, :, :]).mean(-1)


iu = np.triu_indices(len(meas), 1)
rows, mats = [], {}
for mode in ("location", "location_scale", "raw"):
    if mode == "raw":
        Z = np.array([Q[m] for m in meas.index])
        Dm = np.abs(Z[:, None, :] - Z[None, :, :]).mean(-1)
    else:
        Dm = dist_matrix(mode)
    mats[mode] = Dm
    for name, lab, blk in (("condition", cond, blk_cond), ("line", line, blk_line), ("date", date, blk_date)):
        def stat(l_):
            same = (l_[:, None] == l_[None, :])[iu]
            return Dm[iu][~same].mean() / Dm[iu][same].mean()
        obs = stat(lab)
        null = np.array([stat(perm_within(lab, blk, rng)) for _ in range(N_PERM)])
        same = (lab[:, None] == lab[None, :])[iu]
        rows.append({"distance": mode, "factor": name, "within_mean": Dm[iu][same].mean(),
                     "between_mean": Dm[iu][~same].mean(), "between_over_within": obs,
                     "p_perm": (np.sum(null >= obs - 1e-12) + 1) / (N_PERM + 1)})
A4 = pd.DataFrame(rows)
A4.to_csv(OUT / "A4_shape_distance_tests.csv", index=False)
for mode, Dm in mats.items():
    pd.DataFrame(Dm, index=meas.index, columns=meas.index).to_csv(OUT / f"A4_distance_matrix_{mode}.csv")
print(A4.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------ A5 reproducibility
def icc1(values, groups):
    df = pd.DataFrame({"v": values, "g": groups})
    k = df.groupby("g").size()
    n, a = len(df), len(k)
    gm = df.groupby("g").v.mean()
    msb = (k * (gm - df.v.mean()) ** 2).sum() / (a - 1)
    msw = ((df.v - df.g.map(gm)) ** 2).sum() / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


grp = (meas.line + "_" + meas.cond).to_numpy(dtype=object)
ug = np.unique(grp)
rows = []
for k in DESC:
    v = D[k].values
    est = icc1(v, grp)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(ug, len(ug), replace=True)
        vv, gg = [], []
        for j, g_ in enumerate(pick):
            i = np.where(grp == g_)[0]
            vv += list(v[i]); gg += [f"{g_}_{j}"] * len(i)
        if len(set(gg)) > 1:
            boots.append(icc1(np.array(vv), np.array(gg)))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    rows.append({"descriptor": k, "label": DESC_LABEL[k], "icc1": est, "lo": lo, "hi": hi,
                 "n_groups": len(ug), "n_measurements": len(v)})
A5 = pd.DataFrame(rows)
A5.to_csv(OUT / "A5_reproducibility_icc.csv", index=False)
print(A5.round(3).to_string(index=False))

# ------------------------------------------------------------------------------------------------ A6 precision vs cells
NS = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
resid = {}
for k in DESC:                                               # between-sample variance within condition, full samples
    y = D[k].values.astype(float)
    r = y - pd.Series(y).groupby(cond).transform("mean").values
    resid[k] = r
samp_var_full = {k: [] for k in DESC}
rows = []
for n in NS:
    sd = {k: [] for k in DESC}
    for m in meas.index:
        x = X[m]
        S = x[rng.integers(0, len(x), (400, n))]
        Dn = descr_matrix(S)
        for k in DESC:
            sd[k].append(np.nanstd(Dn[k], ddof=1))
    for k in DESC:
        rows.append({"assay": "RT-DC", "n_cells": n, "descriptor": k, "sampling_sd": float(np.mean(sd[k])),
                     "sampling_sd_min": float(np.min(sd[k])), "sampling_sd_max": float(np.max(sd[k]))})
for m in meas.index:                                          # sampling variance of the full-sample estimates
    x = X[m]
    Dn = descr_matrix(x[rng.integers(0, len(x), (300, len(x)))])
    for k in DESC:
        samp_var_full[k].append(np.nanvar(Dn[k], ddof=1))
A6 = pd.DataFrame(rows)
rel = []
for k in DESC:
    df_res = len(meas) - 3
    var_between = max(float((resid[k] ** 2).sum() / df_res - np.mean(samp_var_full[k])), 0.0)
    for n in NS:
        s2 = float(A6.loc[(A6.descriptor == k) & (A6.n_cells == n), "sampling_sd"].iloc[0]) ** 2
        rel.append({"descriptor": k, "n_cells": n, "between_sample_sd": np.sqrt(var_between),
                    "reliability": var_between / (var_between + s2) if var_between > 0 else 0.0})
A6 = A6.merge(pd.DataFrame(rel), on=["descriptor", "n_cells"], how="left")
for ln, g in afm.groupby("line"):
    x = g.logE.values
    for n in [n for n in NS if n <= 500]:
        Dn = descr_matrix(x[rng.integers(0, len(x), (400, n))])
        for k in DESC:
            A6 = pd.concat([A6, pd.DataFrame([{"assay": f"AFM_{ln}", "n_cells": n, "descriptor": k,
                                               "sampling_sd": float(np.nanstd(Dn[k], ddof=1))}])])
A6.to_csv(OUT / "A6_precision_vs_cells.csv", index=False)
need = []
for k in DESC:
    t = A6[(A6.assay == "RT-DC") & (A6.descriptor == k)].sort_values("n_cells")
    ok = t[t.reliability >= 0.8]
    need.append({"descriptor": k, "between_sample_sd": t.between_sample_sd.iloc[0],
                 "cells_for_reliability_0.8": int(ok.n_cells.iloc[0]) if len(ok) else np.nan,
                 "reliability_at_100": float(t.loc[t.n_cells == 100, "reliability"].iloc[0]),
                 "reliability_at_1000": float(t.loc[t.n_cells == 1000, "reliability"].iloc[0])})
A6n = pd.DataFrame(need)
A6n.to_csv(OUT / "A6_cells_needed.csv", index=False)
print(A6n.round(3).to_string(index=False))

# ------------------------------------------------------------------------------------------------ A7 bag discrimination
FEATS = {"location": ["mean", "median"],
         "distribution": ["mean", "median", "iqr", "upper_tail", "lower_tail", "bowley"],
         "shape_only": ["iqr", "upper_tail", "lower_tail", "bowley"]}
N_BAGS = 50


def make_bags(n_cells):
    out = []
    for m, r in meas.iterrows():
        x = X[m]
        S = x[rng.integers(0, len(x), (N_BAGS, n_cells))]
        d = pd.DataFrame(descr_matrix(S))
        d["measurement"], d["cond"], d["line"], d["date"] = m, r.cond, r.line, r.date
        out.append(d)
    return pd.concat(out, ignore_index=True)


def cv_predict(B, y, cols, fold_col):
    p = pd.Series(np.nan, index=B.index)
    for f in B[fold_col].unique():
        tr, te = B[fold_col] != f, B[fold_col] == f
        if y[tr].nunique() < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(B.loc[tr, cols], y[tr])
        p[te] = clf.predict_proba(B.loc[te, cols])[:, 1]
    return p


def cluster_boot(B, y, P, n_boot=1000):
    """AUROC of each feature set and differences, CI by resampling measurements within class."""
    meas_by_class = {c: B.loc[y == c, "measurement"].unique() for c in (0, 1)}
    idx_by_meas = {m: np.where(B.measurement.values == m)[0] for m in B.measurement.unique()}
    names = list(P)
    stats_ = {n: [] for n in names}
    diffs = {("location", "distribution"): [], ("location", "shape_only"): []}
    for _ in range(n_boot):
        ii = np.concatenate([idx_by_meas[m] for c in (0, 1)
                             for m in rng.choice(meas_by_class[c], len(meas_by_class[c]), replace=True)])
        a = {n: roc_auc_score(y.values[ii], P[n].values[ii]) for n in names}
        for n in names:
            stats_[n].append(a[n])
        for (b0, b1) in diffs:
            diffs[(b0, b1)].append(a[b1] - a[b0])
    return stats_, diffs


rows = []
for n_cells in (30, 100, 300, 1000):
    B = make_bags(n_cells)
    tasks = [(f"{c1} vs {c0}", "cond", c0, c1, "date") for c0, c1 in [("Serum", "EGF"), ("Serum", "FGFJI"),
                                                                       ("EGF", "FGFJI")]]
    tasks += [(f"{l1} vs {l0}", "line", l0, l1, "cond") for l0, l1 in combinations(LINES, 2)]
    for name, col, v0, v1, fold_col in tasks:
        T = B[B[col].isin([v0, v1])].reset_index(drop=True)
        if col == "line":                                     # dates on which both lines were measured
            ok = T.groupby("date")[col].nunique()
            T = T[T.date.isin(ok.index[ok == 2])].reset_index(drop=True)
        y = (T[col] == v1).astype(int)
        P = {fs: cv_predict(T, y, cols, fold_col) for fs, cols in FEATS.items()}
        keep = pd.concat(P, axis=1).notna().all(1)
        T, y, P = T[keep].reset_index(drop=True), y[keep].reset_index(drop=True), {k: v[keep].reset_index(drop=True)
                                                                                   for k, v in P.items()}
        bs, bd = cluster_boot(T, y, P)
        row = {"task": name, "factor": col, "cv": f"leave-one-{fold_col}-out", "cells_per_bag": n_cells,
               "n_bags": len(T), "n_measurements": T.measurement.nunique()}
        for fs in FEATS:
            row[f"auroc_{fs}"] = roc_auc_score(y, P[fs])
            row[f"auroc_{fs}_lo"], row[f"auroc_{fs}_hi"] = np.percentile(bs[fs], [2.5, 97.5])
            # measurement-level AUROC (mean bag prediction per measurement)
            pm = pd.DataFrame({"m": T.measurement, "y": y, "p": P[fs]}).groupby("m").mean()
            row[f"auroc_meas_{fs}"] = roc_auc_score(pm.y, pm.p) if pm.y.nunique() == 2 else np.nan
        for (b0, b1), v in bd.items():
            row[f"delta_{b1}_vs_{b0}"] = row[f"auroc_{b1}"] - row[f"auroc_{b0}"]
            row[f"delta_{b1}_vs_{b0}_lo"], row[f"delta_{b1}_vs_{b0}_hi"] = np.percentile(v, [2.5, 97.5])
        rows.append(row)
A7 = pd.DataFrame(rows)
A7.to_csv(OUT / "A7_bag_discrimination.csv", index=False)
print(A7[["task", "cells_per_bag", "auroc_location", "auroc_distribution", "auroc_shape_only",
          "delta_distribution_vs_location", "delta_distribution_vs_location_lo",
          "delta_distribution_vs_location_hi", "auroc_meas_location", "auroc_meas_distribution"]].round(3).to_string(index=False))

# ------------------------------------------------------------------------------------------------ A8 AFM
a0, a1 = afm.loc[afm.line == "LN229", "logE"].values, afm.loc[afm.line == "U87", "logE"].values
z0, z1 = a0 - np.median(a0), a1 - np.median(a1)
s0, s1 = z0 / np.subtract(*np.percentile(a0, [75, 25])), z1 / np.subtract(*np.percentile(a1, [75, 25]))


def boot_diff(f, n_boot=N_BOOT):
    v = [f(a0[rng.integers(0, len(a0), len(a0))], a1[rng.integers(0, len(a1), len(a1))]) for _ in range(n_boot)]
    return np.percentile(v, [2.5, 97.5])


rows = []
for k, f in [("median_diff", lambda u, v: np.median(v) - np.median(u)),
             ("log_iqr_ratio", lambda u, v: np.log10(np.subtract(*np.percentile(v, [75, 25])) /
                                                     np.subtract(*np.percentile(u, [75, 25])))),
             ("bowley_diff", lambda u, v: descr(v)["bowley"] - descr(u)["bowley"]),
             ("upper_minus_median_shift", lambda u, v: (np.percentile(v, 90) - np.percentile(u, 90)) -
              (np.median(v) - np.median(u))),
             ("lower_minus_median_shift", lambda u, v: (np.percentile(v, 10) - np.percentile(u, 10)) -
              (np.median(v) - np.median(u)))]:
    lo, hi = boot_diff(f)
    rows.append({"quantity": f"U87 vs LN229: {k}", "estimate": f(a0, a1), "lo": lo, "hi": hi})
rows.append({"quantity": "KS on median-aligned log10 E (p)", "estimate": stats.ks_2samp(z0, z1).pvalue})
rows.append({"quantity": "KS on median/IQR-standardised log10 E (p)", "estimate": stats.ks_2samp(s0, s1).pvalue})
rows.append({"quantity": "Fligner-Killeen scale test (p)", "estimate": stats.fligner(a0, a1).pvalue})
rows.append({"quantity": "Mann-Whitney location test (p)", "estimate": stats.mannwhitneyu(a0, a1).pvalue})
A8 = pd.DataFrame(rows)
A8.to_csv(OUT / "A8_afm_comparison.csv", index=False)
print(A8.round(4).to_string(index=False))

# ------------------------------------------------------------------------------------------------ A9 literature anchors
# Shmelev et al., Int. J. Mol. Sci. 2023, 24, 4056 (doi:10.3390/ijms24044056): AFM (PeakForce QNM, DMT model) of
# early-passage glioma cultures; medians and IQRs of Young's modulus in kPa as reported in the paper.
lit = pd.DataFrame([
    ("IDH1 wild-type (pooled)", "IDH1-wt", 28.59, 8.287, 102),
    ("IDH1 R132H (pooled)", "IDH1-R132H", 33.718, 10.881, 150),
    ("BT1", "IDH1-wt", 26.31, 9.842, np.nan), ("BT40", "IDH1-wt", 28.67, 7.344, np.nan),
    ("BT32", "IDH1-R132H", 31.27, 15.974, np.nan), ("BT39", "IDH1-R132H", 33.71, 7.367, np.nan)],
    columns=["sample", "idh1", "median_kPa", "iqr_kPa", "n_cells"])
lit["log10_median"] = np.log10(lit.median_kPa)
lit["relative_iqr"] = lit.iqr_kPa / lit.median_kPa
# log10 IQR, assuming quartiles symmetric about the median on the linear scale (only median and IQR are reported)
lit["log10_iqr_approx"] = np.log10((lit.median_kPa + lit.iqr_kPa / 2) / (lit.median_kPa - lit.iqr_kPa / 2))
cult = lit.iloc[2:]
anch = {
    "delta_log10_median_R132H_vs_wt": lit.log10_median[1] - lit.log10_median[0],
    "fold_median_R132H_vs_wt": lit.median_kPa[1] / lit.median_kPa[0],
    "ratio_log10_iqr_R132H_vs_wt": lit.log10_iqr_approx[1] / lit.log10_iqr_approx[0],
    "sd_log10_median_between_cultures_within_idh": float(np.sqrt(np.mean(
        [np.var(g.log10_median, ddof=1) for _, g in cult.groupby("idh1")]))),
    "sd_ln_log10_iqr_between_cultures_within_idh": float(np.sqrt(np.mean(
        [np.var(np.log(g.log10_iqr_approx), ddof=1) for _, g in cult.groupby("idh1")]))),
    "rtdc_sd_median_between_samples_within_condition": float(np.std(resid["median"], ddof=3)),
    "rtdc_sd_ln_iqr_between_samples_within_condition": float(np.std(
        np.log(D.iqr.values) - pd.Series(np.log(D.iqr.values)).groupby(cond).transform("mean").values, ddof=3)),
}
lit.to_csv(OUT / "A9_literature_values.csv", index=False)
pd.DataFrame([anch]).T.rename(columns={0: "value"}).to_csv(OUT / "A9_design_anchors.csv")
print(pd.Series(anch).round(4).to_string())
print("PHASE2_DONE")
