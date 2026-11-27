"""Step 5 - fusion of image and report branches on the matched cohort.

Matched cohort = patients with a label, a pathology report AND a diagnostic slide.
All models are compared on exactly these held-out patients, under both CV schemes,
for two cohorts:
  all  every matched patient
  x40  matched patients whose slide was scanned at 40x (removes the scanner shortcut;
       every 20x-scanned slide in TCGA-GBM/LGG is a GBM)
Because each patient's CV fold is fixed in splits.csv, branch out-of-fold predictions
are held-out for the same patients.

Fusion variants
  late_*      average of branch logits (no fitted weights)
  early_*     one L2 logistic regression on [PCA(64) of image embedding, TF-IDF(M2) text],
              refit inside every outer fold
Whole-cell mechanics is NOT fused: no patient in any public source has linked cell
mechanics, so the three-input comparison is recorded as not evaluable.

Outputs: results/fusion_metrics.csv, fusion_deltas.csv, fusion_oof.csv, fusion_subgroups.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import (PROC, RES, SEED, delong_test, iter_folds, paired_bootstrap_delta,  # noqa: E402
                    summarize, within_group_auc)

warnings.filterwarnings("ignore")
cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
splits = pd.read_csv(PROC / "splits.csv")
rep_oof = pd.read_csv(RES / "report_oof.csv")
img_oof = pd.read_csv(RES / "image_oof.csv")
texts = pd.read_csv(PROC / "reports_masked.csv").set_index("patient")

# mean-pooled image embeddings for early fusion
Z = np.load(PROC / "embeddings_hipt.npz", allow_pickle=True)  # self-generated file
emb = Z["emb"].astype(np.float32)
idx = pd.DataFrame({c: Z[c] for c in ["patient", "tile", "mag"]})
tf = pd.read_csv(PROC / "tile_features.csv").set_index("tile")
idx["qc_ok"] = idx.tile.map(tf.qc_ok).fillna(False).astype(bool)
MEAN = {}
for p, g in idx.groupby("patient"):
    v = []
    for mag in ["x20", "x5"]:
        m = g[g.mag == mag]
        m = m[m.qc_ok] if m.qc_ok.sum() >= 4 else m
        v.append(emb[m.index].mean(0) if len(m) else np.full(emb.shape[1], np.nan, np.float32))
    MEAN[p] = np.concatenate(v)
MEAN = pd.DataFrame(MEAN).T
AGE = cohort[["AGE"]].fillna(cohort.AGE.median())


# branch model names (override to plug in DCC outputs, e.g. REPORT_MODEL_M2=bert_M2)
import os  # noqa: E402
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "img_meanpool")
REPORT_M1 = os.environ.get("REPORT_MODEL_M1", "tfidf_M1")
REPORT_M2 = os.environ.get("REPORT_MODEL_M2", "tfidf_M2")


def clip(p):
    return np.clip(p, 1e-4, 1 - 1e-4)


def early_fusion(tr, te, y, with_age=False, level="text_M2"):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z\-]+\b", max_features=30000)
    Ttr, Tte = vec.fit_transform(texts.loc[tr, level]), vec.transform(texts.loc[te, level])
    imp = SimpleImputer(strategy="mean").fit(MEAN.loc[tr])
    Mtr, Mte = imp.transform(MEAN.loc[tr]), imp.transform(MEAN.loc[te])
    sc = StandardScaler().fit(Mtr)
    pca = PCA(64, random_state=SEED).fit(sc.transform(Mtr))
    Itr, Ite = pca.transform(sc.transform(Mtr)), pca.transform(sc.transform(Mte))
    s2 = StandardScaler().fit(Itr)
    # image block down-weighted so its 64 dense columns do not swamp the sparse tf-idf block under one C
    Itr, Ite = s2.transform(Itr) * 0.3, s2.transform(Ite) * 0.3
    btr, bte = [Ttr, sparse.csr_matrix(Itr)], [Tte, sparse.csr_matrix(Ite)]
    if with_age:
        sa = StandardScaler().fit(AGE.loc[tr])
        btr.append(sparse.csr_matrix(sa.transform(AGE.loc[tr]) * 0.3))
        bte.append(sparse.csr_matrix(sa.transform(AGE.loc[te]) * 0.3))
    clf = LogisticRegressionCV(Cs=[0.3, 1, 3, 10, 30, 100], cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                               scoring="roc_auc", max_iter=5000, solver="liblinear")
    clf.fit(sparse.hstack(btr).tocsr(), y.loc[tr])
    return clf.predict_proba(sparse.hstack(bte).tocsr())[:, 1]


COMPARISONS = [("image_meanpool", "late_image+report_M2"), ("report_M2", "late_image+report_M2"),
               ("image_meanpool", "late_image+report_M1"), ("report_M1", "late_image+report_M1"),
               ("report_M2", "early_image+report_M2"), ("image_meanpool", "early_image+report_M2"),
               ("image_meanpool", "image_abmil"), ("image_handcrafted", "image_meanpool"),
               ("acq_meta (nuisance)", "late_image+acq_meta"), ("image_meanpool", "late_image+acq_meta"),
               ("acq_meta (nuisance)", "image_meanpool"), ("age", "late_image+report_M2+age")]

rows, deltas, oof_all = [], [], []
for cname in ["all", "x40"]:
    ro = rep_oof[rep_oof.cohort == cname]
    io_ = img_oof[img_oof.cohort == cname]
    matched = sorted(set(ro.patient) & set(io_.patient))
    y = cohort.loc[matched, "y_gbm"]
    print(f"[{cname}] matched cohort: {len(matched)}  GBM {int(y.sum())}")

    def get(df, model, scheme):
        return df[(df.model == model) & (df.scheme == scheme)].set_index("patient").p.reindex(matched)

    for scheme in ["patient", "institution"]:
        P = {"image_meanpool": get(io_, IMAGE_MODEL, scheme), "image_abmil": get(io_, "img_abmil", scheme),
             "image_handcrafted": get(io_, "handcrafted", scheme), "report_M1": get(ro, REPORT_M1, scheme),
             "report_M2": get(ro, REPORT_M2, scheme), "age": get(ro, "age", scheme),
             "acq_meta (nuisance)": get(io_, "acq_meta", scheme), "stain_color (nuisance)": get(io_, "stain_color", scheme),
             "report_M0 (unmasked)": get(ro, "tfidf_M0", scheme)}
        P = {k: v for k, v in P.items() if v.notna().all()}          # skip branches not run for this cohort
        L = {k: logit(clip(v)) for k, v in P.items()}
        P["late_image+report_M1"] = expit((L["image_meanpool"] + L["report_M1"]) / 2)
        P["late_image+report_M2"] = expit((L["image_meanpool"] + L["report_M2"]) / 2)
        P["late_image+report_M2+age"] = expit((L["image_meanpool"] + L["report_M2"] + L["age"]) / 3)
        if "acq_meta (nuisance)" in L:
            P["late_image+acq_meta"] = expit((L["image_meanpool"] + L["acq_meta (nuisance)"]) / 2)
        # early fusion refits TF-IDF + LR per fold: 2 of the 5 repeats (CPU budget)
        for nm, kw in [("early_image+report_M2", {})]:
            preds = [pd.DataFrame({"patient": te, "p": early_fusion(tr, te, y, **kw)})
                     for r, k, tr, te in iter_folds(splits, matched, scheme) if r < 2]
            P[nm] = pd.concat(preds).groupby("patient").p.mean().reindex(matched)
        P["three_input_with_mechanics"] = None
        for name, p in P.items():
            if p is None:
                rows.append({"model": name, "cohort": cname, "scheme": scheme, "n": 0,
                             "note": "not evaluable: 0 patients with linked whole-cell mechanics"})
                continue
            s = summarize(y.values, p.values, name=name)
            s["macro_f1"] = f1_score(y.values, (p.values >= 0.5).astype(int), average="macro")
            tmp = pd.DataFrame({"p": p, "y_gbm": y, "institution": cohort.loc[matched, "institution"]})
            _, s["auroc_within_institution"] = within_group_auc(tmp, "p")
            s["cohort"], s["scheme"] = cname, scheme
            rows.append(s)
            oof_all.append(pd.DataFrame({"patient": matched, "model": name, "cohort": cname, "scheme": scheme, "p": p.values}))
        for a, b in COMPARISONS:
            if P.get(a) is None or P.get(b) is None:
                continue
            d0, ci, pb = paired_bootstrap_delta(y.values, P[a].values, P[b].values)
            _, _, z, pdl = delong_test(y.values, P[b].values, P[a].values)
            deltas.append({"cohort": cname, "scheme": scheme, "baseline": a, "model": b, "delta_auroc": d0,
                           "lo": ci[0], "hi": ci[1], "p_bootstrap": pb, "p_delong": pdl})

M = pd.DataFrame(rows); M.to_csv(RES / "fusion_metrics.csv", index=False)
D = pd.DataFrame(deltas); D.to_csv(RES / "fusion_deltas.csv", index=False)
O = pd.concat(oof_all); O.to_csv(RES / "fusion_oof.csv", index=False)
pd.set_option("display.width", 220)
print(M[["cohort", "scheme", "model", "n", "auroc", "auroc_lo", "auroc_hi", "auroc_within_institution", "macro_f1",
         "brier", "ece"]].round(3).to_string(index=False))
print(D.round(4).to_string(index=False))

# ---- subgroups: fused scores across WHO-2021-like classes and grades ----
sub = []
keep = ["image_meanpool", "report_M2", "late_image+report_M2"]
for (cname, scheme), og in O[O.model.isin(keep)].groupby(["cohort", "scheme"]):
    o = og.merge(cohort[["y_gbm", "who2021_like", "GRADE"]], left_on="patient", right_index=True)
    for (m, lab, cls), g in o.groupby(["model", "y_gbm", "who2021_like"]):
        sub.append({"cohort": cname, "scheme": scheme, "model": m, "label": "GBM" if lab else "LGG", "group": cls,
                    "n": len(g), "mean_p_gbm": g.p.mean(), "frac_called_gbm@0.5": (g.p >= 0.5).mean()})
    for (m, grade), g in o.groupby(["model", "GRADE"]):
        sub.append({"cohort": cname, "scheme": scheme, "model": m, "label": "grade", "group": grade, "n": len(g),
                    "mean_p_gbm": g.p.mean(), "frac_called_gbm@0.5": (g.p >= 0.5).mean()})
S = pd.DataFrame(sub); S.to_csv(RES / "fusion_subgroups.csv", index=False)
print(S[(S.scheme == "institution")].round(3).to_string(index=False))
