"""Step 4 - image branch: patient-level GBM vs LGG from diagnostic-slide tiles.

Models (same shared splits as the report branch):
  img_meanpool    mean HIPT embedding over QC-passing tiles (20x and 5x pooled separately,
                  concatenated, 768-d) + L2 logistic regression (C by inner CV)
  img_abmil       gated-attention MIL (Ilse et al. 2018) over all 20x+5x crops
  handcrafted     interpretable tile features (nuclear density/size, H&E intensities,
                  texture) aggregated per patient + L2 logistic regression
  acq_meta        NUISANCE ONLY: scan date, scanner, magnification/MPP, JPEG quality,
                  software version, slide and tissue size - no image content
  stain_color     NUISANCE-LIKE: mean RGB and H/E optical-density statistics of tissue

Outputs: results/image_oof.csv, results/image_metrics.csv, results/abmil_attention_full.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RES, SEED, iter_folds, summarize, within_group_auc, youden_threshold  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
splits = pd.read_csv(PROC / "splits.csv")
slides = pd.read_csv(PROC / "slides.csv").set_index("patient")
tf = pd.read_csv(PROC / "tile_features.csv")
hp = pd.read_csv(PROC / "handcrafted_patient.csv").set_index("patient")
Z = np.load(PROC / "embeddings_hipt.npz", allow_pickle=True)  # self-generated file
emb = Z["emb"].astype(np.float32)
idx = pd.DataFrame({c: Z[c] for c in ["patient", "tile", "mag", "stratum", "crop"]})
qc = tf.set_index("tile").qc_ok
idx["qc_ok"] = idx.tile.map(qc).fillna(False).astype(bool)

patients = sorted(set(idx.patient) & set(cohort.index) & set(slides.index))
print("image cohort:", len(patients))
y_all = cohort.loc[patients, "y_gbm"]

# ---------------- patient-level feature matrices ----------------
def pooled(p_idx):
    out = []
    for mag in ["x20", "x5"]:
        m = p_idx[p_idx.mag == mag]
        mq = m[m.qc_ok] if m.qc_ok.sum() >= 4 else m
        out.append(emb[mq.index].mean(0) if len(mq) else np.full(emb.shape[1], np.nan, np.float32))
    return np.concatenate(out)


g = idx.groupby("patient")
MEAN = pd.DataFrame(np.stack([pooled(g.get_group(p)) for p in patients]), index=patients)

# interpretable tissue features: nuclear morphology/density, texture, low-power pale-eosin fraction
bio_cols = [c for c in hp.columns if any(k in c for k in ["nuc_", "glcm_", "lbp_", "pale_eosin"])
            and "nuc_H_mean" not in c]
# nuisance-like image statistics: stain colour/intensity, sharpness, tissue and blood fractions
nuis_cols = [c for c in hp.columns if any(k in c for k in ["mean_r", "mean_g", "mean_b", "H_mean", "H_sd", "E_mean",
                                                          "E_sd", "blur_lapvar", "tissue_frac", "blood_frac"])]
HC = hp.reindex(patients)[bio_cols]
COLOR = hp.reindex(patients)[nuis_cols]
print("hand-crafted tissue features:", len(bio_cols), "| stain/sharpness features:", len(nuis_cols))
acq_num = ["scan_month_index", "appmag", "mpp", "jpeg_q", "aperio_major", "slide_area_mpx", "tissue_area_mm2",
           "thumb_tissue_frac", "file_size"]
ACQ = slides.reindex(patients)[acq_num + ["scanner"]].copy()
ACQ["scanner"] = ACQ.scanner.fillna("unknown").astype(str)


def lr_cv():
    return LogisticRegressionCV(Cs=np.logspace(-3, 2, 11), cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                scoring="roc_auc", max_iter=5000)


def fit_predict_tab(X, tr, te, kind):
    if kind == "acq":
        pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), acq_num),
                                 ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["scanner"])])
        model = make_pipeline(pre, lr_cv())
    elif kind == "emb":
        model = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), PCA(64, random_state=SEED), lr_cv())
    else:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), lr_cv())
    model.fit(X.loc[tr], y_all.loc[tr])
    return model.predict_proba(X.loc[te])[:, 1], model.predict_proba(X.loc[tr])[:, 1]


# ---------------- attention MIL ----------------
MAXI = 64
ABMIL_REPEATS = 2
def bag_tensor(p):
    m = g.get_group(p)
    x = emb[m.index]
    extra = np.stack([(m.mag == "x5").values, (m.stratum == "hdense").values, ~m.qc_ok.values], 1).astype(np.float32)
    x = np.concatenate([x, extra], 1)[:MAXI]
    pad = np.zeros((MAXI, x.shape[1]), np.float32); pad[:len(x)] = x
    mask = np.zeros(MAXI, bool); mask[:len(x)] = True
    return pad, mask, m.tile.values[:MAXI]


BAGS = {p: bag_tensor(p) for p in patients}


class GatedABMIL(nn.Module):
    def __init__(self, d_in, d=128, d_att=64, drop=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(drop))
        self.V = nn.Linear(d, d_att); self.U = nn.Linear(d, d_att); self.w = nn.Linear(d_att, 1)
        self.cls = nn.Linear(d, 1)

    def forward(self, x, mask):
        h = self.enc(x)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = a.masked_fill(~mask, -1e9).softmax(-1)
        z = (a.unsqueeze(-1) * h).sum(1)
        return self.cls(z).squeeze(-1), a


def train_abmil(tr, te, seed=0, epochs=80, patience=12):
    torch.manual_seed(seed); np.random.seed(seed)
    ytr = y_all.loc[tr].values
    tr_a, va_a = train_test_split(np.array(tr), test_size=0.15, stratify=ytr, random_state=seed)
    X = lambda ids: torch.from_numpy(np.stack([BAGS[p][0] for p in ids]))
    M = lambda ids: torch.from_numpy(np.stack([BAGS[p][1] for p in ids]))
    Xtr, Mtr = X(tr_a), M(tr_a)
    mu = Xtr[Mtr].mean(0); sd = Xtr[Mtr].std(0) + 1e-6
    mu[-3:] = 0; sd[-3:] = 1
    norm = lambda x: (x - mu) / sd
    Xtr = norm(Xtr); Ytr = torch.tensor(y_all.loc[tr_a].values, dtype=torch.float32)
    Xva, Mva = norm(X(va_a)), M(va_a); yva = y_all.loc[va_a].values
    net = GatedABMIL(Xtr.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            m = Mtr[b] & (torch.rand(Mtr[b].shape) > 0.25)          # instance dropout
            m[m.sum(1) == 0] = Mtr[b][m.sum(1) == 0]
            logit, _ = net(Xtr[b], m)
            loss = lossf(logit, Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva, Mva)[0]).numpy()
        auc = roc_auc_score(yva, pv)
        if auc > best + 1e-4:
            best, best_state, bad = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pte, ate = net(norm(X(te)), M(te))
        ptr, _ = net(norm(X(tr)), M(tr))
    return torch.sigmoid(pte).numpy(), torch.sigmoid(ptr).numpy(), ate.numpy(), net, norm


MODELS = {
    "img_meanpool": lambda tr, te: fit_predict_tab(MEAN, tr, te, "emb"),
    "handcrafted": lambda tr, te: fit_predict_tab(HC, tr, te, "tab"),
    "stain_color": lambda tr, te: fit_predict_tab(COLOR, tr, te, "tab"),
    "acq_meta": lambda tr, te: fit_predict_tab(ACQ, tr, te, "acq"),
    "img_abmil": lambda tr, te: train_abmil(tr, te)[:2],
}

if __name__ == "__main__":
    rows, metrics = [], []
    # cohort "x40": slides scanned at 40x only. Every 20x-scanned slide in TCGA-GBM/LGG is a GBM,
    # so on "all" an image model can partly read scanner/resolution instead of tissue.
    cohorts = {"all": patients, "x40": [p for p in patients if slides.loc[p, "appmag"] == 40]}
    for cname, cids in cohorts.items():
        for scheme in ["patient", "institution"]:
            for name, fn in MODELS.items():
                preds = []
                for r, k, tr, te in iter_folds(splits, cids, scheme):
                    if name == "img_abmil" and r >= ABMIL_REPEATS:   # attention MIL: 2 of the 5 repeats (CPU budget)
                        continue
                    p_te, p_tr = fn(tr, te)
                    thr = youden_threshold(y_all.loc[tr], p_tr)
                    preds.append(pd.DataFrame({"patient": te, "repeat": r, "p": p_te, "thr": thr}))
                P = pd.concat(preds)
                P["yhat"] = (P.p >= P.thr).astype(int)
                agg = P.groupby("patient").agg(p=("p", "mean"), yhat=("yhat", lambda s: int(s.mean() >= 0.5)))
                agg["model"], agg["scheme"], agg["cohort"] = name, scheme, cname
                rows.append(agg.reset_index())
                yy = y_all.loc[agg.index].values
                s = summarize(yy, agg.p.values, name=name)
                s["macro_f1"] = f1_score(yy, agg.yhat.values, average="macro")
                s["auroc_repeat_sd"] = float(np.std([roc_auc_score(y_all.loc[q.patient], q.p) for _, q in P.groupby("repeat")]))
                tmp = agg.join(cohort[["y_gbm", "institution"]])
                _, s["auroc_within_institution"] = within_group_auc(tmp, "p")
                s["scheme"], s["cohort"] = scheme, cname
                metrics.append(s)
                print(f"{cname:4s} {scheme:12s} {name:14s} n={s['n']} AUROC {s['auroc']:.3f} [{s['auroc_lo']:.3f},{s['auroc_hi']:.3f}] "
                      f"within-inst {s['auroc_within_institution']:.3f} F1 {s['macro_f1']:.3f}", flush=True)
                pd.concat(rows).to_csv(RES / "image_oof.csv", index=False)
                pd.DataFrame(metrics).to_csv(RES / "image_metrics.csv", index=False)

    # full-data ABMIL fit for attention inspection (not used for evaluation)
    p_all, _, att, net, norm = train_abmil(patients, patients, seed=1)
    rows = []
    for i, p in enumerate(patients):
        tiles = BAGS[p][2]
        for j, t in enumerate(tiles):
            rows.append({"patient": p, "instance": j, "tile": t, "attention": att[i, j]})
    pd.DataFrame(rows).to_csv(RES / "abmil_attention_full.csv", index=False)
    print("saved attention")
