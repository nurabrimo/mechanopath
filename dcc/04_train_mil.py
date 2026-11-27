"""Patient-level image models on full-resolution tile embeddings (DCC pipeline).

  python dcc/04_train_mil.py --encoder phikon2 [--max-instances 4096 --abmil-repeats 5]

Bag = every tissue tile (256 px at 0.5 um/px) of the patient's diagnostic slide. The patient splits,
cohort table and 40x cohort are the ones used by every other model in the project, so out-of-fold
predictions can be compared and fused patient by patient.

Models
  img_meanpool  mean tile embedding -> standardise -> PCA(128) -> L2 logistic regression (C by inner CV)
  img_abmil     gated attention MIL (Ilse et al., ICML 2018). Training bags are random subsets of up to
                --max-instances tiles, redrawn every epoch; inference uses every tile. AdamW, early stopping
                on a 15% validation split of the training fold.
Cohorts: 'all' and 'x40' (slides scanned at 40x; every 20x-scanned TCGA-GBM/LGG slide is a GBM).
Schemes: 'patient' (5 x 5 stratified folds) and 'institution' (5 x 5 folds grouped by tissue source site).

Outputs (with --part, per-task files go to results/mil_parts and dcc/04b_merge_mil.py combines them)
  results/image_oof_<enc>.csv, results/image_metrics_<enc>.csv, results/image_fold_log_<enc>.csv
  results/slide_tiles_<enc>.csv                       tiles per slide and acquisition attributes
  data/processed/patient_mean_<enc>.npz               patient-level mean embeddings (for confounding checks)
  results/attention_<enc>.npz                         attention of a full-data ABMIL fit for every tile
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import PROC, RES, SEED, iter_folds, summarize, within_group_auc, youden_threshold  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", required=True)
ap.add_argument("--feat-root", default=None, help="defaults to $FEAT_DIR")
ap.add_argument("--max-instances", type=int, default=4096)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--abmil-repeats", type=int, default=5)
ap.add_argument("--models", default="img_meanpool,img_abmil")
ap.add_argument("--combos", default="all:patient,all:institution,x40:patient,x40:institution",
                help="cohort:scheme pairs to run (array tasks run one each)")
ap.add_argument("--part", default=None, help="array-task label; outputs then go to results/mil_parts")
ap.add_argument("--attention", type=int, default=1, help="1 = fit the full-data model for attention maps")
a = ap.parse_args()
COMBOS = [tuple(c.split(":")) for c in a.combos.split(",")]
import os  # noqa: E402
feat_dir = Path(a.feat_root or os.environ["FEAT_DIR"]) / a.encoder
tag = a.encoder
OUTD = RES if a.part is None else RES / "mil_parts"
OUTD.mkdir(parents=True, exist_ok=True)
otag = tag if a.part is None else f"{tag}_part{a.part}"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_start = time.time()

cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
splits = pd.read_csv(PROC / "splits.csv")
slides_meta = pd.read_csv(PROC / "slides.csv").set_index("patient")

# ---------------------------------------------------------------- load bags
bags, coords, stems, rows = {}, {}, {}, []
for f in sorted(glob.glob(str(feat_dir / "*.h5"))):
    stem = Path(f).stem
    p = stem[:12]
    with h5py.File(f, "r") as h:
        F = h["features"][:]
        C = h["coords"][:]
        at = {k: h.attrs[k] for k in h.attrs}
    if len(F) == 0:
        continue
    bags.setdefault(p, []).append(torch.from_numpy(F.astype(np.float16)))
    coords.setdefault(p, []).append(C.astype(np.int64))
    stems.setdefault(p, []).extend([stem] * len(F))
    rows.append({"patient": p, "slide": stem, "n_tiles": len(F), "dim": F.shape[1],
                 **{k: (v.decode() if isinstance(v, bytes) else v) for k, v in at.items()
                    if k in ("appmag", "mpp0", "jpeg_q", "scanner", "scan_date", "read_level", "read_size",
                             "size0", "width0", "height0", "tissue_frac_thumb", "mask_rule", "od_gain")}})
for p in list(bags):
    bags[p] = torch.cat(bags[p])
    coords[p] = np.concatenate(coords[p])
    stems[p] = np.array(stems[p])
ST = pd.DataFrame(rows)
if a.part in (None, "1"):
    ST.to_csv(RES / f"slide_tiles_{tag}.csv", index=False)
patients = sorted(set(bags) & set(cohort.index))
y = cohort.loc[patients, "y_gbm"]
x40 = set(slides_meta.index[slides_meta.appmag == 40])
MEAN = pd.DataFrame(np.stack([bags[p].float().mean(0).numpy() for p in patients]), index=patients)
D = MEAN.shape[1]
if a.part in (None, "1"):
    np.savez_compressed(PROC / f"patient_mean_{tag}.npz", patient=np.array(patients, dtype="U"),
                        X=MEAN.values.astype(np.float32))
print(f"[{tag}] {len(patients)} patients ({int(y.sum())} GBM) | dim {D} | tiles/slide median "
      f"{int(ST.n_tiles.median())} (IQR {int(ST.n_tiles.quantile(.25))}-{int(ST.n_tiles.quantile(.75))}) | "
      f"total tiles {int(ST.n_tiles.sum())} | device {dev}", flush=True)


# ---------------------------------------------------------------- models
class GatedABMIL(nn.Module):
    def __init__(self, d_in, d=256, d_att=128, drop=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(drop))
        self.V, self.U, self.w = nn.Linear(d, d_att), nn.Linear(d, d_att), nn.Linear(d_att, 1)
        self.cls = nn.Linear(d, 1)

    def forward(self, x):                                # x: (N, d_in), one bag
        h = self.enc(x)
        att = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1).softmax(0)
        return self.cls((att.unsqueeze(-1) * h).sum(0)).squeeze(-1), att


def _bag(p, mu, sd, n=None, gen=None):
    x = bags[p]
    if n is not None and len(x) > n:
        x = x[torch.randperm(len(x), generator=gen)[:n]]
    return (x.to(dev, non_blocking=True).float() - mu) / sd


def train_abmil(tr, te, seed=0, return_net=False):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr_a, va_a = train_test_split(np.array(tr), test_size=0.15, stratify=y.loc[tr], random_state=seed)
    mu = torch.tensor(MEAN.loc[tr_a].values.mean(0), device=dev, dtype=torch.float32)
    sd = torch.tensor(MEAN.loc[tr_a].values.std(0) + 1e-6, device=dev, dtype=torch.float32)
    net = GatedABMIL(D).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    best, state, bad, ep_best = -1.0, None, 0, 0
    yy = {p: torch.tensor(float(y[p]), device=dev) for p in tr}
    for ep in range(a.epochs):
        net.train()
        for p in np.random.default_rng(seed + ep).permutation(tr_a):
            n = min(len(bags[p]), a.max_instances)
            n = max(1, int(n * 0.75))                            # 25% instance dropout on top of subsampling
            logit, _ = net(_bag(p, mu, sd, n, gen))
            loss = lossf(logit, yy[p])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = [torch.sigmoid(net(_bag(q, mu, sd))[0]).item() for q in va_a]
        auc = roc_auc_score(y.loc[va_a], pv)
        if auc > best + 1e-4:
            best, state, bad, ep_best = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        pte = np.array([torch.sigmoid(net(_bag(q, mu, sd))[0]).item() for q in te])
        ptr = np.array([torch.sigmoid(net(_bag(q, mu, sd))[0]).item() for q in tr])
    info = {"val_auroc": best, "best_epoch": ep_best, "epochs_run": ep + 1}
    if return_net:
        return pte, ptr, info, net, mu, sd
    return pte, ptr, info


def meanpool_lr(tr, te, seed=0):
    m = make_pipeline(StandardScaler(), PCA(min(128, D), random_state=SEED),
                      LogisticRegressionCV(Cs=np.logspace(-3, 2, 11), scoring="roc_auc", max_iter=5000,
                                           cv=StratifiedKFold(5, shuffle=True, random_state=SEED)))
    m.fit(MEAN.loc[tr], y.loc[tr])
    C = float(m[-1].C_[0])
    return m.predict_proba(MEAN.loc[te])[:, 1], m.predict_proba(MEAN.loc[tr])[:, 1], {"C": C}


MODELS = {"img_meanpool": meanpool_lr, "img_abmil": train_abmil}
use = [m for m in a.models.split(",") if m in MODELS]
rows, metrics, log = [], [], []
COH = {"all": patients, "x40": [p for p in patients if p in x40]}
for cname, scheme in COMBOS:
    cids = COH[cname]
    if True:
        for name in use:
            fn = MODELS[name]
            preds = []
            for r, k, tr, te in iter_folds(splits, cids, scheme):
                if name == "img_abmil" and r >= a.abmil_repeats:
                    continue
                t0 = time.time()
                p_te, p_tr, info = fn(tr, te, seed=SEED + 100 * r + k)
                preds.append(pd.DataFrame({"patient": te, "repeat": r, "fold": k, "p": p_te,
                                           "thr": youden_threshold(y.loc[tr], p_tr)}))
                log.append({"encoder": tag, "model": name, "cohort": cname, "scheme": scheme, "repeat": r,
                            "fold": k, "n_train": len(tr), "n_test": len(te), "test_auroc": roc_auc_score(y.loc[te], p_te)
                            if y.loc[te].nunique() == 2 else np.nan, "seconds": time.time() - t0, **info})
            P = pd.concat(preds)
            P["yhat"] = (P.p >= P.thr).astype(int)
            agg = P.groupby("patient").agg(p=("p", "mean"), yhat=("yhat", lambda s: int(s.mean() >= 0.5)),
                                           n_repeats=("p", "size"))
            agg["model"], agg["scheme"], agg["cohort"], agg["encoder"] = name, scheme, cname, tag
            rows.append(agg.reset_index())
            s = summarize(y.loc[agg.index].values, agg.p.values, name=name)
            s["macro_f1"] = f1_score(y.loc[agg.index], agg.yhat, average="macro")
            _, s["auroc_within_institution"] = within_group_auc(agg.join(cohort[["y_gbm", "institution"]]), "p")
            s["auroc_repeat_sd"] = float(np.std([roc_auc_score(y.loc[g.patient], g.p) for _, g in P.groupby("repeat")]))
            s.update({"scheme": scheme, "cohort": cname, "encoder": tag})
            metrics.append(s)
            print(f"[{tag}] {cname:4s} {scheme:12s} {name:13s} n={s['n']} AUROC {s['auroc']:.3f} "
                  f"[{s['auroc_lo']:.3f},{s['auroc_hi']:.3f}] within-inst {s['auroc_within_institution']:.3f} "
                  f"F1 {s['macro_f1']:.3f} ({(time.time() - t_start) / 60:.0f} min)", flush=True)
            pd.concat(rows).to_csv(OUTD / f"image_oof_{otag}.csv", index=False)
            pd.DataFrame(metrics).to_csv(OUTD / f"image_metrics_{otag}.csv", index=False)
            pd.DataFrame(log).to_csv(OUTD / f"image_fold_log_{otag}.csv", index=False)

# ---------------------------------------------------------------- attention of a full-data fit (for figures only)
if "img_abmil" in use and a.attention:
    _, _, info, net, mu, sd = train_abmil(patients, patients, seed=SEED, return_net=True)
    P_, S_, X_, Y_, A_, R_ = [], [], [], [], [], []
    with torch.no_grad():
        for p in patients:
            logit, att = net(_bag(p, mu, sd))
            att = att.float().cpu().numpy()
            P_.append(np.full(len(att), p, dtype="U12"))
            S_.append(stems[p])
            X_.append(coords[p][:, 0]); Y_.append(coords[p][:, 1])
            A_.append(att.astype(np.float32))
            R_.append(np.full(len(att), torch.sigmoid(logit).item(), dtype=np.float32))
    np.savez_compressed(RES / f"attention_{tag}.npz", patient=np.concatenate(P_), slide=np.concatenate(S_).astype("U40"),
                        x=np.concatenate(X_), y=np.concatenate(Y_), attention=np.concatenate(A_),
                        p_full_fit=np.concatenate(R_))
    print(f"[{tag}] attention saved (full-data fit, val AUROC {info['val_auroc']:.3f})", flush=True)
print(f"[{tag}] MIL_DONE in {(time.time() - t_start) / 60:.0f} min", flush=True)
