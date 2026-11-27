"""Shared helpers for the MechanoPath analyses: paths, cohort tables, CV splits, metrics.

Every patient-level model in this project is evaluated with the same splits
(created once by make_splits and saved to data/processed/splits.csv), so that
image-only, report-only and fused models are compared on identical held-out
patients.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (brier_score_loss, f1_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(os.environ.get("MECHANOPATH_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
for p in (PROC, RES, FIG):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20261008
N_REPEATS = 5
N_FOLDS = 5


# ---------------------------------------------------------------------------
# Cohort / labels
# ---------------------------------------------------------------------------
def load_cbio_glioma() -> pd.DataFrame:
    """Patient-level labels and molecular annotations from the TCGA pan-glioma
    study (Ceccarelli et al., Cell 2016) as distributed by cBioPortal
    (study id lgggbm_tcga_pub). One row per patient."""
    d = RAW / "cbioportal" / "lgggbm_tcga_pub"
    pt = pd.read_csv(d / "data_clinical_patient.txt", sep="\t", comment="#")
    sa = pd.read_csv(d / "data_clinical_sample.txt", sep="\t", comment="#")
    sa = sa.sort_values("SAMPLE_ID").drop_duplicates("PATIENT_ID")
    df = pt.merge(sa, on="PATIENT_ID", how="left")
    df["y_gbm"] = (df["STUDY"] == "Glioblastoma multiforme").astype(int)
    df["project"] = np.where(df["y_gbm"] == 1, "TCGA-GBM", "TCGA-LGG")
    df["tss_code"] = df["PATIENT_ID"].str.split("-").str[1]
    return df.rename(columns={"PATIENT_ID": "patient"})


def who2021_like(df: pd.DataFrame) -> pd.Series:
    """Approximate WHO CNS5 (2021) integrated class from pan-glioma annotations.

    'GBM_IDHwt'  : IDH wild-type AND (histological grade 4 OR TERT promoter
                   mutation OR chr7 gain/chr10 loss). EGFR amplification is a
                   further criterion that is not available in this table, so
                   some molecular GBMs are missed (conservative).
    'Astro_IDHmut': IDH mutant, non-codeleted.
    'Oligo_IDHmut_codel': IDH mutant, 1p/19q codeleted.
    'IDHwt_other': IDH wild-type without the above criteria (or criteria unknown).
    'unknown'    : IDH status missing.
    """
    idh = df["IDH_STATUS"].astype(str)
    sub = df["IDH_CODEL_SUBTYPE"].astype(str)
    tert = df["TERT_PROMOTER_STATUS"].astype(str).str.lower()
    c710 = df["CHR_7_GAIN_CHR_10_LOSS"].astype(str).str.lower()
    grade4 = df["GRADE"].astype(str).eq("G4")
    mol = tert.str.contains("mutant") | c710.str.contains(r"gain chr 7 & loss chr 10|^gain.*loss", regex=True)
    out = pd.Series("unknown", index=df.index)
    out[idh.eq("WT") & (grade4 | mol)] = "GBM_IDHwt"
    out[idh.eq("WT") & ~(grade4 | mol)] = "IDHwt_other"
    out[sub.eq("IDHmut-non-codel")] = "Astro_IDHmut"
    out[sub.eq("IDHmut-codel")] = "Oligo_IDHmut_codel"
    return out


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def make_splits(patients: pd.DataFrame, group_col: str = "institution") -> pd.DataFrame:
    """Create (a) repeated stratified patient-level K-fold splits and
    (b) institution-grouped stratified K-fold splits.

    Returns a long table: patient, scheme, repeat, fold.
    """
    rows = []
    y = patients["y_gbm"].values
    pids = patients["patient"].values
    for r in range(N_REPEATS):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + r)
        for k, (_, te) in enumerate(skf.split(pids, y)):
            rows += [(pids[i], "patient", r, k) for i in te]
    groups = patients[group_col].fillna("unknown").values
    for r in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED + r)
        for k, (_, te) in enumerate(sgkf.split(pids, y, groups)):
            rows += [(pids[i], "institution", r, k) for i in te]
    return pd.DataFrame(rows, columns=["patient", "scheme", "repeat", "fold"])


def iter_folds(splits: pd.DataFrame, cohort: list[str], scheme: str):
    """Yield (repeat, fold, train_ids, test_ids) restricted to a cohort."""
    s = splits[(splits.scheme == scheme) & splits.patient.isin(cohort)]
    for (r, k), g in s.groupby(["repeat", "fold"]):
        test = g.patient.tolist()
        train = s[(s.repeat == r) & (s.fold != k)].patient.tolist()
        yield r, k, train, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ece(y, p, n_bins=10):
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmax(tpr - fpr)]


def bootstrap_auc(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def paired_bootstrap_delta(y, p1, p2, n=2000, seed=SEED):
    """AUROC(p2) - AUROC(p1) with a paired bootstrap 95% CI and two-sided p."""
    rng = np.random.default_rng(seed)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d0 = roc_auc_score(y, p2) - roc_auc_score(y, p1)
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ds.append(roc_auc_score(y[i], p2[i]) - roc_auc_score(y[i], p1[i]))
    ds = np.array(ds)
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return d0, np.percentile(ds, [2.5, 97.5]), max(p, 1 / len(ds))


def delong_test(y, p1, p2):
    """DeLong et al. (1988) test for two correlated AUROCs (fast version,
    Sun & Xu 2014). Returns (auc1, auc2, z, p)."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")
    y = y[order]
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    m = int(y.sum())
    n = len(y) - m

    def midrank(x):
        return stats.rankdata(x, method="average")     # mid-ranks for ties

    k = preds.shape[0]
    tx = np.array([midrank(preds[r, :m]) for r in range(k)])
    ty = np.array([midrank(preds[r, m:]) for r in range(k)])
    tz = np.array([midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    l = np.array([1, -1])
    var = l @ s @ l
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return aucs[0], aucs[1], z, p


def summarize(y, p, thr=0.5, name=""):
    y, p = np.asarray(y), np.asarray(p)
    lo, hi = bootstrap_auc(y, p)
    yhat = (p >= thr).astype(int)
    return {
        "model": name, "n": len(y), "n_gbm": int(y.sum()),
        "auroc": roc_auc_score(y, p), "auroc_lo": lo, "auroc_hi": hi,
        "macro_f1": f1_score(y, yhat, average="macro"),
        "brier": brier_score_loss(y, p), "ece": ece(y, p),
    }


def within_group_auc(df: pd.DataFrame, pcol: str, group_col: str = "institution", min_per_class: int = 2):
    """AUROC computed inside each institution that has both classes, plus the
    patient-weighted mean. Removes between-institution differences from the
    evaluation (each comparison is GBM vs LGG from the same institution)."""
    rows = []
    for g, d in df.groupby(group_col):
        n1, n0 = int(d.y_gbm.sum()), int((1 - d.y_gbm).sum())
        if n1 >= min_per_class and n0 >= min_per_class:
            rows.append({"institution": g, "n_gbm": n1, "n_lgg": n0,
                         "auroc": roc_auc_score(d.y_gbm, d[pcol])})
    t = pd.DataFrame(rows)
    if len(t):
        w = t.n_gbm * t.n_lgg
        pooled = float((t.auroc * w).sum() / w.sum())
    else:
        pooled = np.nan
    return t, pooled


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
