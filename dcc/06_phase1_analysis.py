"""Phase 1 analysis of the full-resolution DCC outputs: discrimination, paired comparisons, calibration, clinical
utility, WHO 2021 subgroups and confounding checks for GBM vs LGG classification from slides and reports.

  python dcc/06_phase1_analysis.py            (CPU, ~10 min; run after 04_train_mil and 05_report_bert)

Branch predictions are patient-level out-of-fold probabilities produced with the shared splits
(data/processed/splits.csv), so every model is evaluated on identical held-out patients:
  image   Phikon-v2 or HIPT tile embeddings -> gated attention MIL (abmil) or mean pooling + logistic regression
  report  BiomedBERT fine-tuned on masked reports (M1: tumour-type names, WHO grade statements, morphology codes and
          molecular findings masked; M2: M1 plus the histological grading criteria - necrosis, microvascular
          proliferation, mitoses, proliferation indices, cellularity, atypia); TF-IDF logistic regression on the same
          texts; age alone
  nuisance acquisition metadata (scanner, magnification, scan date, JPEG quality) and stain colour, which carry no
          biology and measure shortcut capacity
Late fusion = mean of branch logits with equal weights (no fitted parameters, so no extra optimism).

Pre-specified primary analysis: late fusion of Phikon-v2 attention MIL and BiomedBERT-M2 versus each branch alone, in
the matched cohort (diagnostic slide and report available), with whole tissue-source sites held out
(institution-grouped CV). Key secondary: the same contrasts in patients whose slides were scanned at 40x (every
20x-scanned TCGA-GBM/LGG slide is a glioblastoma, so magnification is a shortcut). Holm correction within each
family of four contrasts; DeLong tests for correlated AUROCs with paired-bootstrap CIs.

Outputs: results/phase1/*.csv and phase1_summary.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import PROC, RES, SEED, delong_test, ece, within_group_auc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = RES / "phase1"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)
cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
slides = pd.read_csv(PROC / "slides.csv").set_index("patient")


# ------------------------------------------------------------------------------------------------ load branches
def load(path, model, name, encoder=None):
    path = Path(path)
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d[d.model == model]
    if encoder is not None and "encoder" in d:
        d = d[d.encoder == encoder]
    return d.assign(name=name)[["patient", "cohort", "scheme", "p", "name"]] if len(d) else None


def load_bert(level):
    parts = [load(f, f"bert_{level}", f"report_bert_{level}") for f in sorted((RES / "bert").glob(
        f"report_oof_bert_{level}_*.csv"))]
    parts = [p for p in parts if p is not None]
    return pd.concat(parts) if parts else None


SRC = {
    "image_phikon2_abmil": load(RES / "image_oof_phikon2.csv", "img_abmil", "image_phikon2_abmil"),
    "image_phikon2_meanpool": load(RES / "image_oof_phikon2.csv", "img_meanpool", "image_phikon2_meanpool"),
    "image_hipt_abmil": load(RES / "image_oof_hipt.csv", "img_abmil", "image_hipt_abmil"),
    "image_hipt_meanpool": load(RES / "image_oof_hipt.csv", "img_meanpool", "image_hipt_meanpool"),
    "report_bert_M2": load_bert("M2"),
    "report_bert_M1": load_bert("M1"),
    "report_tfidf_M2": load(RES / "report_oof.csv", "tfidf_M2", "report_tfidf_M2"),
    "report_tfidf_M1": load(RES / "report_oof.csv", "tfidf_M1", "report_tfidf_M1"),
    "report_unmasked_M0": load(RES / "report_oof.csv", "tfidf_M0", "report_unmasked_M0"),
    "age": load(RES / "report_oof.csv", "age", "age"),
    "acq_meta": load(RES / "image_oof.csv", "acq_meta", "acq_meta"),
    "stain_color": load(RES / "image_oof.csv", "stain_color", "stain_color"),
    "tissue_handcrafted": load(RES / "image_oof.csv", "handcrafted", "tissue_handcrafted"),
}
available = {k: v is not None for k, v in SRC.items()}
print("available:", {k: v for k, v in available.items()})
IMG = "image_phikon2_abmil" if available["image_phikon2_abmil"] else "image_hipt_abmil"
REP = "report_bert_M2" if available["report_bert_M2"] else "report_tfidf_M2"
primary_def = f"late fusion of {IMG} and {REP}"
LONG = pd.concat([v for v in SRC.values() if v is not None], ignore_index=True)
CORE = [k for k in ("image_phikon2_abmil", "image_phikon2_meanpool", "image_hipt_abmil", "image_hipt_meanpool",
                    "report_bert_M2", "report_bert_M1", "report_tfidf_M2", "age") if available[k]]

FUSIONS = {
    "fusion_primary": [IMG, REP],
    "fusion_image_reportM1": [IMG, "report_bert_M1" if available["report_bert_M1"] else "report_tfidf_M1"],
    "fusion_hipt_reportM2": ["image_hipt_abmil", REP],
    "fusion_image_tfidfM2": [IMG, "report_tfidf_M2"],
    "fusion_primary_age": [IMG, REP, "age"],
    "fusion_image_acq": [IMG, "acq_meta"],
}


def clipl(p):
    return logit(np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4))


# ------------------------------------------------------------------------------------------------ metrics
def delong_ci(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    allr = stats.rankdata(np.r_[pos, neg])
    rp, rn = stats.rankdata(pos), stats.rankdata(neg)
    auc = (allr[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (allr[:m] - rp) / n
    v01 = 1 - (allr[m:] - rn) / m
    se = np.sqrt(np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n)
    return auc, max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se), se


def calib(y, p):
    lp = clipl(p)
    try:
        sl = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y)
        slope = float(sl.coef_[0, 0])
        import statsmodels.api as sm
        cil = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        slope, cil = np.nan, np.nan
    return slope, float(cil)


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    z, ph = 1.96, k / n
    den = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def metrics(y, p, inst):
    auc, lo, hi, se = delong_ci(y, p)
    yhat = (p >= 0.5).astype(int)
    slope, cil = calib(y, p)
    tmp = pd.DataFrame({"p": p, "y_gbm": y, "institution": inst})
    _, wi = within_group_auc(tmp, "p")
    tp, tn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
    return {"n": len(y), "n_gbm": int(y.sum()), "auroc": auc, "auroc_lo": lo, "auroc_hi": hi, "auroc_se": se,
            "auprc": average_precision_score(y, p), "prevalence": y.mean(),
            "sensitivity@0.5": tp / max(1, y.sum()), "specificity@0.5": tn / max(1, (1 - y).sum()),
            "balanced_accuracy@0.5": balanced_accuracy_score(y, yhat), "macro_f1@0.5": f1_score(y, yhat, average="macro"),
            "brier": brier_score_loss(y, p), "ece": ece(y, p), "calibration_slope": slope,
            "calibration_intercept": cil, "auroc_within_institution": wi}


def fast_auc(y, s):
    r = stats.rankdata(s)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def paired_boot(y, p1, p2, n=2000):
    y, p1, p2 = (np.asarray(v, float) for v in (y, p1, p2))
    y = y.astype(int)
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ds.append(fast_auc(y[i], p2[i]) - fast_auc(y[i], p1[i]))
    return np.percentile(ds, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


# ------------------------------------------------------------------------------------------------ assemble per cohort
oof_rows, met_rows, cmp_rows, sizes = [], [], [], {}
COMPARE = {
    "primary": [("fusion_primary", IMG), ("fusion_primary", REP)],
    "secondary": [("fusion_image_reportM1", IMG), ("fusion_image_tfidfM2", IMG), ("fusion_image_tfidfM2", "fusion_primary"),
                  ("fusion_primary_age", "fusion_primary"), (IMG, "acq_meta"), ("fusion_primary", "fusion_image_acq")],
    "branch": [(IMG, "image_hipt_abmil"), (IMG, "image_phikon2_meanpool"), ("image_hipt_abmil", "image_hipt_meanpool"),
               (REP, "report_tfidf_M2"), (REP, "report_bert_M1"), (REP, "age"), ("report_unmasked_M0", REP)],
}
for cname in ("all", "x40"):
    for scheme in ("institution", "patient"):
        L_ = LONG[(LONG.cohort == cname) & (LONG.scheme == scheme)]
        P = L_.pivot_table(index="patient", columns="name", values="p")
        matched = P.index[P[CORE].notna().all(1)]
        matched = [p for p in matched if p in cohort.index]
        P = P.loc[matched]
        for f, parts in FUSIONS.items():
            if all(c in P for c in parts):
                P[f] = expit(np.mean([clipl(P[c]) for c in parts], axis=0))
        y = cohort.loc[matched, "y_gbm"].values
        inst = cohort.loc[matched, "institution"].values
        sizes[f"{cname}_{scheme}"] = {"n": len(matched), "n_gbm": int(y.sum()), "n_institutions": int(len(set(inst)))}
        for m in P.columns:
            ok = P[m].notna().values
            if ok.sum() < 20:
                continue
            r = metrics(y[ok], P[m].values[ok], inst[ok])
            met_rows.append({"model": m, "cohort": cname, "scheme": scheme, **r})
            oof_rows.append(pd.DataFrame({"patient": np.array(matched)[ok], "model": m, "cohort": cname,
                                          "scheme": scheme, "p": P[m].values[ok], "y": y[ok]}))
        for fam, pairs in COMPARE.items():
            rows = []
            for a_, b_ in pairs:
                if a_ not in P or b_ not in P:
                    continue
                ok = P[a_].notna().values & P[b_].notna().values
                auc_a, auc_b, z, pdl = delong_test(y[ok], P[a_].values[ok], P[b_].values[ok])
                lo, hi = paired_boot(y[ok], P[b_].values[ok], P[a_].values[ok])
                rows.append({"family": fam, "cohort": cname, "scheme": scheme, "model": a_, "comparator": b_,
                             "n": int(ok.sum()), "auroc_model": auc_a, "auroc_comparator": auc_b,
                             "delta_auroc": auc_a - auc_b, "lo": lo, "hi": hi, "z": z, "p_delong": pdl})
            cmp_rows += rows
OOF = pd.concat(oof_rows, ignore_index=True)
MET = pd.DataFrame(met_rows)
CMP = pd.DataFrame(cmp_rows)
# Holm within the pre-specified families: primary = 2 contrasts x {all, x40} (institution scheme) = 4 tests
fam_mask = (CMP.family == "primary") & (CMP.scheme == "institution")
CMP["p_holm"] = np.nan
CMP.loc[fam_mask, "p_holm"] = holm(CMP.loc[fam_mask, "p_delong"])
for (fam, c, s), g in CMP[~fam_mask].groupby(["family", "cohort", "scheme"]):
    CMP.loc[g.index, "p_holm"] = holm(g.p_delong)
OOF.to_csv(OUT / "phase1_oof.csv", index=False)
MET.to_csv(OUT / "phase1_metrics.csv", index=False)
CMP.to_csv(OUT / "phase1_comparisons.csv", index=False)
pd.set_option("display.width", 250)
print(MET[MET.scheme == "institution"][["cohort", "model", "n", "auroc", "auroc_lo", "auroc_hi", "auroc_within_institution",
                                        "auprc", "brier", "calibration_slope"]].round(3).to_string(index=False))
print(CMP[CMP.scheme == "institution"][["family", "cohort", "model", "comparator", "delta_auroc", "lo", "hi", "p_delong",
                                        "p_holm"]].round(4).to_string(index=False))

# ------------------------------------------------------------------------------------------------ repeat variability
rep_rows = []
for enc in ("phikon2", "hipt"):
    f = RES / f"image_fold_log_{enc}.csv"
    if f.exists():
        fl = pd.read_csv(f)
        for (m, c, s), g in fl.groupby(["model", "cohort", "scheme"]):
            rep_rows.append({"source": f"{enc}_{m}", "cohort": c, "scheme": s, "folds": len(g),
                             "fold_auroc_mean": g.test_auroc.mean(), "fold_auroc_sd": g.test_auroc.std(),
                             "fold_auroc_min": g.test_auroc.min(), "fold_auroc_max": g.test_auroc.max()})
for f in sorted((RES / "bert").glob("report_fold_log_bert_*.csv")):
    g = pd.read_csv(f)
    lv, c, s = f.stem.replace("report_fold_log_bert_", "").split("_")
    rep_rows.append({"source": f"bert_{lv}", "cohort": c, "scheme": s, "folds": len(g),
                     "fold_auroc_mean": g.test_auroc.mean(), "fold_auroc_sd": g.test_auroc.std(),
                     "fold_auroc_min": g.test_auroc.min(), "fold_auroc_max": g.test_auroc.max(),
                     "best_epoch_median": g.best_epoch.median()})
pd.DataFrame(rep_rows).to_csv(OUT / "phase1_fold_variability.csv", index=False)

# ------------------------------------------------------------------------------------------------ calibration, decision curves
KEY = [IMG, REP, "fusion_primary", "image_hipt_abmil", "report_tfidf_M2", "acq_meta"]
cal_rows, dca_rows = [], []
for (c, s, m), g in OOF[OOF.model.isin(KEY)].groupby(["cohort", "scheme", "model"]):
    q = pd.qcut(g.p.rank(method="first"), 10, labels=False)
    for b, gg in g.groupby(q):
        lo, hi = wilson(int(gg.y.sum()), len(gg))
        cal_rows.append({"cohort": c, "scheme": s, "model": m, "bin": int(b), "n": len(gg), "mean_p": gg.p.mean(),
                         "observed": gg.y.mean(), "lo": lo, "hi": hi})
    n, prev = len(g), g.y.mean()
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        tp = ((g.p >= t) & (g.y == 1)).sum() / n
        fp = ((g.p >= t) & (g.y == 0)).sum() / n
        dca_rows.append({"cohort": c, "scheme": s, "model": m, "threshold": t, "net_benefit": tp - fp * t / (1 - t),
                         "treat_all": prev - (1 - prev) * t / (1 - t)})
pd.DataFrame(cal_rows).to_csv(OUT / "phase1_calibration_bins.csv", index=False)
pd.DataFrame(dca_rows).to_csv(OUT / "phase1_decision_curve.csv", index=False)

# ------------------------------------------------------------------------------------------------ WHO 2021 subgroups
sub_rows = []
lab = cohort[["y_gbm", "who2021_like", "GRADE", "IDH_STATUS"]]
for (c, s, m), g in OOF[OOF.model.isin([IMG, REP, "fusion_primary"])].groupby(["cohort", "scheme", "model"]):
    g = g.merge(lab, left_on="patient", right_index=True)
    for (yy, w), gg in g.groupby(["y_gbm", "who2021_like"]):
        k = int((gg.p >= 0.5).sum())
        lo, hi = wilson(k, len(gg))
        sub_rows.append({"cohort": c, "scheme": s, "model": m, "tcga_label": "GBM" if yy else "LGG", "group": w,
                         "n": len(gg), "called_gbm": k, "frac_called_gbm": k / len(gg), "lo": lo, "hi": hi,
                         "mean_p": gg.p.mean()})
    for gr, gg in g.groupby("GRADE"):
        k = int((gg.p >= 0.5).sum())
        lo, hi = wilson(k, len(gg))
        sub_rows.append({"cohort": c, "scheme": s, "model": m, "tcga_label": "grade", "group": gr, "n": len(gg),
                         "called_gbm": k, "frac_called_gbm": k / len(gg), "lo": lo, "hi": hi, "mean_p": gg.p.mean()})
SUB = pd.DataFrame(sub_rows)
# LGG-labelled molecular glioblastomas vs IDH-mutant astrocytomas: Fisher exact test of the call rate
tests = []
for (c, s, m), g in SUB[SUB.tcga_label == "LGG"].groupby(["cohort", "scheme", "model"]):
    a_ = g[g.group == "GBM_IDHwt"]
    b_ = g[g.group == "Astro_IDHmut"]
    if len(a_) and len(b_):
        tab = [[int(a_.called_gbm.iloc[0]), int(a_.n.iloc[0] - a_.called_gbm.iloc[0])],
               [int(b_.called_gbm.iloc[0]), int(b_.n.iloc[0] - b_.called_gbm.iloc[0])]]
        orr, pf = stats.fisher_exact(tab)
        tests.append({"cohort": c, "scheme": s, "model": m, "n_molecular_gbm_in_lgg": int(a_.n.iloc[0]),
                      "frac_called_molecular_gbm": float(a_.frac_called_gbm.iloc[0]),
                      "n_astro_idhmut_in_lgg": int(b_.n.iloc[0]),
                      "frac_called_astro_idhmut": float(b_.frac_called_gbm.iloc[0]), "odds_ratio": orr,
                      "p_fisher": pf})
SUB.to_csv(OUT / "phase1_subgroups.csv", index=False)
pd.DataFrame(tests).to_csv(OUT / "phase1_subgroup_tests.csv", index=False)
print(pd.DataFrame(tests).round(3).to_string(index=False))

# ------------------------------------------------------------------------------------------------ by institution
inst_rows = []
for (c, s, m), g in OOF[OOF.model.isin([IMG, REP, "fusion_primary"])].groupby(["cohort", "scheme", "model"]):
    g = g.merge(cohort[["institution"]], left_on="patient", right_index=True)
    for i, gg in g.groupby("institution"):
        n1, n0 = int(gg.y.sum()), int((1 - gg.y).sum())
        if n1 >= 5 and n0 >= 5:
            auc, lo, hi, _ = delong_ci(gg.y.values, gg.p.values)
            inst_rows.append({"cohort": c, "scheme": s, "model": m, "institution": i, "n_gbm": n1, "n_lgg": n0,
                              "auroc": auc, "lo": lo, "hi": hi})
pd.DataFrame(inst_rows).to_csv(OUT / "phase1_by_institution.csv", index=False)

# ------------------------------------------------------------------------------------------------ confounding checks
conf = []
matched_all = sorted(OOF[(OOF.cohort == "all") & (OOF.model == "fusion_primary")].patient.unique())
for enc in ("phikon2", "hipt"):
    f = PROC / f"patient_mean_{enc}.npz"
    if not f.exists():
        continue
    Z = np.load(f, allow_pickle=True)
    E = pd.DataFrame(Z["X"], index=Z["patient"])
    E = E.loc[[p for p in E.index if p in cohort.index]]
    meta = cohort.loc[E.index]
    pipe = make_pipeline(StandardScaler(), PCA(64, random_state=SEED), LogisticRegression(C=0.1, max_iter=5000))
    # (a) tissue-source site, within each diagnostic class (removes the label-site association)
    for cls, cl_name in ((1, "GBM"), (0, "LGG")):
        mm = meta[meta.y_gbm == cls]
        vc = mm.institution.value_counts()
        keep = vc.index[vc >= 15]
        idx = mm.index[mm.institution.isin(keep)]
        if len(keep) < 2:
            continue
        yy = mm.loc[idx, "institution"].to_numpy(dtype=object)
        pr = cross_val_predict(pipe, E.loc[idx].values, yy, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        conf.append({"encoder": enc, "test": f"site identification within {cl_name}", "n": len(idx),
                     "classes": len(keep), "balanced_accuracy": balanced_accuracy_score(yy, pr),
                     "chance": 1 / len(keep)})
    # (b) scan magnification within GBM (20x vs 40x)
    gm = [p for p in meta.index[meta.y_gbm == 1] if p in slides.index]
    yy = (slides.loc[gm, "appmag"] == 20).astype(int).values
    pp = cross_val_predict(pipe, E.loc[gm].values, yy, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                           method="predict_proba")[:, 1]
    auc, lo, hi, _ = delong_ci(yy, pp)
    conf.append({"encoder": enc, "test": "20x vs 40x scanning within GBM (AUROC)", "n": len(gm), "classes": 2,
                 "auroc": auc, "lo": lo, "hi": hi})
# (c) tissue quantity
for enc in ("phikon2", "hipt"):
    f = RES / f"slide_tiles_{enc}.csv"
    if not f.exists():
        continue
    st = pd.read_csv(f).groupby("patient").n_tiles.sum()
    st = st[[p for p in st.index if p in matched_all]]
    yy = cohort.loc[st.index, "y_gbm"].values
    auc, lo, hi, _ = delong_ci(yy, st.values)
    conf.append({"encoder": enc, "test": "tiles per patient alone (AUROC)", "n": len(st), "classes": 2, "auroc": auc,
                 "lo": lo, "hi": hi, "median_tiles_gbm": float(np.median(st.values[yy == 1])),
                 "median_tiles_lgg": float(np.median(st.values[yy == 0])),
                 "p_mannwhitney": stats.mannwhitneyu(st.values[yy == 1], st.values[yy == 0]).pvalue})
    g = OOF[(OOF.cohort == "all") & (OOF.scheme == "institution") & (OOF.model == "fusion_primary")].set_index("patient")
    g = g.loc[[p for p in g.index if p in st.index]]
    terc = pd.qcut(st.loc[g.index], 3, labels=["low", "mid", "high"])
    for t_, gg in g.groupby(terc):
        if gg.y.nunique() == 2:
            auc, lo, hi, _ = delong_ci(gg.y.values, gg.p.values)
            conf.append({"encoder": enc, "test": f"fusion AUROC, tissue tertile {t_}", "n": len(gg), "classes": 2,
                         "auroc": auc, "lo": lo, "hi": hi})
CONF = pd.DataFrame(conf)
CONF.to_csv(OUT / "phase1_confounding.csv", index=False)
print(CONF.round(3).to_string(index=False))

# ------------------------------------------------------------------------------------------------ summary
def pick(model, c, s):
    r = MET[(MET.model == model) & (MET.cohort == c) & (MET.scheme == s)]
    return r.iloc[0].to_dict() if len(r) else None


summary = {"primary_definition": primary_def, "image_branch": IMG, "report_branch": REP, "available": available,
           "cohort_sizes": sizes,
           "primary": {f"{c}_{s}": {m: pick(m, c, s) for m in ("fusion_primary", IMG, REP)}
                       for c in ("all", "x40") for s in ("institution", "patient")},
           "primary_tests": CMP[CMP.family == "primary"].to_dict("records")}
json.dump(summary, open(OUT / "phase1_summary.json", "w"), indent=1, default=float)
print("PHASE1_ANALYSIS_DONE")
