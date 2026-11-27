"""Step 1 - report branch: GBM vs LGG from TCGA pathology reports.

Models (all evaluated on the shared patient-level and institution-grouped splits):
  rule_M0        keyword rule on raw text (glioblastoma / grade IV mentioned) - leakage demo
  tfidf_M0/M1/M2 TF-IDF (word 1-2 grams) + L2 logistic regression, C tuned by inner CV
  age            age at diagnosis only (structured clinical data)
  tfidf_M1+age   masked report + age

Outputs
  results/report_oof.csv            out-of-fold P(GBM), averaged over repeats, per patient/model/scheme
  results/report_metrics.csv        AUROC (bootstrap CI), macro-F1, Brier, ECE
  results/report_top_terms_*.csv    largest positive / negative coefficients (full-data fit)
  results/report_leak_audit.csv     residual diagnosis words after masking
  results/report_label_audit.csv    report statements that disagree with the project label
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import (PROC, RAW, RES, SEED, iter_folds, summarize,  # noqa: E402
                    within_group_auc, youden_threshold)
from report_masking import mask, normalize, residual_dx_terms  # noqa: E402

warnings.filterwarnings("ignore")

cohort = pd.read_csv(PROC / "cohort.csv")
splits = pd.read_csv(PROC / "splits.csv")
rep = pd.read_csv(RAW / "tcga_reports/extracted/TCGA_Reports.csv")
rep["patient"] = rep.patient_filename.str.split(".").str[0]
d = cohort[cohort.has_report].merge(rep[["patient", "text"]], on="patient")
print("report cohort:", len(d), "GBM", d.y_gbm.sum())

for lv in ["M0", "M1", "M2"]:
    d[f"text_{lv}"] = [mask(t, lv) for t in d.text]
d["masked_frac_M1"] = d.text_M1.str.count("xmaskx") / d.text_M0.str.split().str.len()
d[["patient", "y_gbm", "text_M0", "text_M1", "text_M2"]].to_csv(PROC / "reports_masked.csv", index=False)

# ---- leak audit -------------------------------------------------------------
audit = []
for lv in ["M1", "M2"]:
    res = d[f"text_{lv}"].map(residual_dx_terms)
    audit.append({"level": lv, "reports_with_residual_dx_words": int((res.str.len() > 0).sum()),
                  "examples": "; ".join(sorted({w for r in res for w in r})[:20]),
                  "mean_tokens_masked": float(d[f"text_{lv}"].str.count("xmaskx").mean())})
pd.DataFrame(audit).to_csv(RES / "report_leak_audit.csv", index=False)
print(pd.DataFrame(audit))

# ---- label audit: what does the raw report say? ------------------------------
t0 = d.text_M0
d["mentions_gbm"] = t0.str.contains(r"glioblastoma|\bgbm\b", regex=True)
d["mentions_grade4"] = t0.str.contains(r"grade\s*[:\-]?\s*(iv|4)\b", regex=True)
d["mentions_grade23"] = t0.str.contains(r"grade\s*[:\-]?\s*(ii|iii|2|3)\b", regex=True)
d["rule_M0"] = (d.mentions_gbm | d.mentions_grade4).astype(float)
la = d.groupby("project")[["mentions_gbm", "mentions_grade4", "mentions_grade23"]].mean().round(3)
la["n"] = d.groupby("project").size()
la.to_csv(RES / "report_label_audit.csv")
print(la)
# LGG reports whose diagnosis line states glioblastoma/grade IV (candidate label noise)
dx_line = t0.str.extract(r"diagnos[ie]s\s*:?(.{0,300})", expand=False).fillna("")
d["dxline_gbm"] = dx_line.str.contains(r"glioblastoma|grade\s*(iv|4)\b", regex=True)
d.loc[d.y_gbm == 0, ["patient", "institution", "GRADE", "HISTOLOGICAL_DIAGNOSIS", "who2021_like", "dxline_gbm"]] \
    .query("dxline_gbm").to_csv(RES / "report_lgg_cases_with_gbm_in_dx_line.csv", index=False)
print("LGG patients whose report diagnosis line says glioblastoma/grade IV:",
      int(d[(d.y_gbm == 0)].dxline_gbm.sum()))


# ---- models -----------------------------------------------------------------
def tfidf_lr(train_txt, ytr, test_txt, extra_tr=None, extra_te=None):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z\-]+\b", max_features=30000)
    Xtr = vec.fit_transform(train_txt)
    Xte = vec.transform(test_txt)
    if extra_tr is not None:
        sc = StandardScaler().fit(extra_tr)
        Xtr = sparse.hstack([Xtr, sparse.csr_matrix(sc.transform(extra_tr))]).tocsr()
        Xte = sparse.hstack([Xte, sparse.csr_matrix(sc.transform(extra_te))]).tocsr()
    clf = LogisticRegressionCV(Cs=[0.3, 1, 3, 10, 30, 100], cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                               scoring="roc_auc", max_iter=5000, solver="liblinear")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1], clf.predict_proba(Xtr)[:, 1]


def age_lr(Atr, ytr, Ate):
    m = LogisticRegression(max_iter=1000).fit(Atr, ytr)
    return m.predict_proba(Ate)[:, 1], m.predict_proba(Atr)[:, 1]


d = d.set_index("patient")
age = d[["AGE"]].fillna(d.AGE.median())
models = {
    "tfidf_M0": lambda tr, te: tfidf_lr(d.loc[tr, "text_M0"], d.loc[tr, "y_gbm"], d.loc[te, "text_M0"]),
    "tfidf_M1": lambda tr, te: tfidf_lr(d.loc[tr, "text_M1"], d.loc[tr, "y_gbm"], d.loc[te, "text_M1"]),
    "tfidf_M2": lambda tr, te: tfidf_lr(d.loc[tr, "text_M2"], d.loc[tr, "y_gbm"], d.loc[te, "text_M2"]),
    "age": lambda tr, te: age_lr(age.loc[tr].values, d.loc[tr, "y_gbm"], age.loc[te].values),
    "tfidf_M1+age": lambda tr, te: tfidf_lr(d.loc[tr, "text_M1"], d.loc[tr, "y_gbm"], d.loc[te, "text_M1"],
                                            age.loc[tr].values, age.loc[te].values),
}

rows, metrics = [], []
from sklearn.metrics import f1_score, roc_auc_score
# cohorts: every patient with a report, and the acquisition-restricted subset whose
# diagnostic slide was scanned at 40x (removes the 20x-scanner = GBM shortcut used in step 4)
slides_meta = pd.read_csv(PROC / "slides.csv") if (PROC / "slides.csv").exists() else None
cohorts = {"all": d.index.tolist()}
if slides_meta is not None:
    x40 = set(slides_meta.loc[slides_meta.appmag == 40, "patient"])
    cohorts["x40"] = [p for p in d.index if p in x40]
for cname, cohort_ids in cohorts.items():
    for scheme in ["patient", "institution"]:
        for name, fn in models.items():
            preds = []
            for r, k, tr, te in iter_folds(splits, cohort_ids, scheme):
                p_te, p_tr = fn(tr, te)
                thr = youden_threshold(d.loc[tr, "y_gbm"], p_tr)
                preds.append(pd.DataFrame({"patient": te, "repeat": r, "p": p_te, "thr": thr}))
            P = pd.concat(preds)
            # threshold-based metrics use the training-fold Youden threshold of each fold
            P["yhat"] = (P.p >= P.thr).astype(int)
            agg = P.groupby("patient").agg(p=("p", "mean"), yhat=("yhat", lambda s: int(s.mean() >= 0.5)))
            agg["model"], agg["scheme"], agg["cohort"] = name, scheme, cname
            rows.append(agg.reset_index())
            y = d.loc[agg.index, "y_gbm"].values
            s = summarize(y, agg.p.values, name=name)
            s["macro_f1"] = f1_score(y, agg.yhat.values, average="macro")
            rep_auc = [roc_auc_score(d.loc[g.patient, "y_gbm"], g.p) for _, g in P.groupby("repeat")]
            s["auroc_repeat_sd"] = float(np.std(rep_auc))
            tmp = agg.join(d[["y_gbm", "institution"]])
            _, s["auroc_within_institution"] = within_group_auc(tmp, "p")
            s["scheme"], s["cohort"] = scheme, cname
            metrics.append(s)
            print(f"{cname:4s} {scheme:12s} {name:14s} n={s['n']} AUROC {s['auroc']:.3f} [{s['auroc_lo']:.3f},{s['auroc_hi']:.3f}] "
                  f"within-inst {s['auroc_within_institution']:.3f} F1 {s['macro_f1']:.3f}", flush=True)

# rule-based (no training)
s = summarize(d.y_gbm.values, d.rule_M0.values, name="rule_M0"); s["scheme"] = "none"; s["cohort"] = "all"
metrics.append(s)
print("rule_M0 AUROC", round(s["auroc"], 3), "F1", round(s["macro_f1"], 3))

oof = pd.concat(rows)
oof.to_csv(RES / "report_oof.csv", index=False)
pd.DataFrame(metrics).to_csv(RES / "report_metrics.csv", index=False)

# ---- interpretability: coefficients from a fit on all patients ---------------
for lv in ["M1", "M2"]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z\-]+\b", max_features=30000)
    X = vec.fit_transform(d[f"text_{lv}"])
    clf = LogisticRegression(C=10, max_iter=5000, solver="liblinear").fit(X, d.y_gbm)
    coef = pd.Series(clf.coef_[0], index=vec.get_feature_names_out()).sort_values()
    top = pd.concat([coef.tail(30)[::-1].rename("coef").to_frame().assign(direction="GBM"),
                     coef.head(30).rename("coef").to_frame().assign(direction="LGG")])
    top.to_csv(RES / f"report_top_terms_{lv}.csv")
    print(lv, "GBM terms:", ", ".join(coef.tail(15)[::-1].index))
    print(lv, "LGG terms:", ", ".join(coef.head(15).index))
