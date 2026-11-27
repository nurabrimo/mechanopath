"""External validation of the Phase 1 image model (Phikon-v2 gated attention MIL).

The model is refit on every TCGA patient with a diagnostic-slide bag (same architecture, optimiser, instance sampling
and early stopping as dcc/04_train_mil.py; 5 seeds averaged), frozen, and applied without any recalibration to
  * DeepHisto (University Hospital Bern glioma cohort, Philips scanner): GBM vs IDH-mutant astrocytoma/oligodendroglioma
  * CPTAC-GBM (all glioblastoma): share called GBM (sensitivity) and its IDH-wild-type subset.
A mean-pool logistic regression (standardise -> PCA 128 -> L2 LR, C by inner CV) is the simple comparator.

python ext_validate.py   (on a GPU node; writes ~/mechanopath_full/results/external/)
"""
import glob, json, os, sys
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "mechanopath_full"; PROC = ROOT / "data" / "processed"; OUT = ROOT / "results" / "external"; OUT.mkdir(parents=True, exist_ok=True)
FEAT = Path("/cwork/nb295/mechanopath/features")
MAXI, EPOCHS, PAT, SEEDS = 4096, 60, 10, [0, 1, 2, 3, 4]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(dirpath, key=lambda s: s[:12]):
    bags = {}
    for f in sorted(glob.glob(str(dirpath / "*.h5"))):
        with h5py.File(f, "r") as h:
            F = h["features"][:]
        if len(F):
            bags.setdefault(key(Path(f).stem), []).append(torch.from_numpy(F.astype(np.float16)))
    return {p: torch.cat(v) for p, v in bags.items()}


cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
tcga = load(FEAT / "phikon2")
P = sorted(set(tcga) & set(cohort.index)); y = cohort.loc[P, "y_gbm"].astype(int)
print(f"TCGA training patients {len(P)} ({y.sum()} GBM), device {dev}", flush=True)

# --- external cohorts
dh_lab = pd.read_csv(Path.home() / "recast/results/ext/labels_deephisto.csv")
dh_lab = dh_lab[dh_lab.in_scope.astype(str) == "True"].set_index("patient")
dh = load(FEAT / "ext_deephisto" / "phikon2", key=lambda s: s.split("_")[0][:5])
dh = {p: v for p, v in dh.items() if p in dh_lab.index}
idx = pd.read_csv("/work/nb295/recast/ext/cptac/idc_cptac_gbm_sm.csv")
ser2pat = dict(zip(idx.SeriesInstanceUID, idx.PatientID))
cp_raw = load(FEAT / "ext_cptac" / "phikon2", key=lambda s: s)
cp = {}
for s, v in cp_raw.items():
    p = ser2pat.get(s)
    if p:
        cp.setdefault(p, []).append(v)
cp = {p: torch.cat(v) for p, v in cp.items()}
# IDH status of CPTAC-GBM patients from the cBioPortal mutation table, if present
idh_mut = set()
try:
    M = json.load(open("/work/nb295/recast/glioprior/raw/gbm_cptac_2021_muts.json"))
    for m in M:
        g = (m.get("gene", {}) or {}).get("hugoGeneSymbol") or m.get("hugoGeneSymbol")
        if g in ("IDH1", "IDH2"):
            idh_mut.add(str(m.get("patientId", m.get("sampleId", ""))[:9]))
except Exception as e:
    print("no CPTAC IDH table:", e)
print(f"DeepHisto patients {len(dh)} | CPTAC patients {len(cp)} (IDH-mutant {len(idh_mut & set(cp))})", flush=True)

MEAN = pd.DataFrame(np.stack([tcga[p].float().mean(0).numpy() for p in P]), index=P); D = MEAN.shape[1]


class GatedABMIL(nn.Module):
    def __init__(self, d_in, d=256, d_att=128, drop=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(drop))
        self.V, self.U, self.w = nn.Linear(d, d_att), nn.Linear(d, d_att), nn.Linear(d_att, 1)
        self.cls = nn.Linear(d, 1)

    def forward(self, x):
        h = self.enc(x)
        att = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1).softmax(0)
        return self.cls((att.unsqueeze(-1) * h).sum(0)).squeeze(-1), att


def bag(x, mu, sd, n=None, gen=None):
    if n is not None and len(x) > n:
        x = x[torch.randperm(len(x), generator=gen)[:n]]
    return (x.to(dev).float() - mu) / sd


def fit(seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    tr, va = train_test_split(np.array(P), test_size=0.15, stratify=y.loc[P], random_state=seed)
    mu = torch.tensor(MEAN.loc[tr].values.mean(0), device=dev, dtype=torch.float32)
    sd = torch.tensor(MEAN.loc[tr].values.std(0) + 1e-6, device=dev, dtype=torch.float32)
    net = GatedABMIL(D).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2); lf = nn.BCEWithLogitsLoss()
    best, state, bad = -1, None, 0
    for ep in range(EPOCHS):
        net.train()
        for p in np.random.default_rng(seed + ep).permutation(tr):
            n = max(1, int(min(len(tcga[p]), MAXI) * 0.75))
            lo, _ = net(bag(tcga[p], mu, sd, n, gen)); loss = lf(lo, torch.tensor(float(y[p]), device=dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = [torch.sigmoid(net(bag(tcga[q], mu, sd))[0]).item() for q in va]
        auc = roc_auc_score(y.loc[va], pv)
        if auc > best + 1e-4:
            best, state, bad = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PAT:
                break
    net.load_state_dict(state); net.eval()
    print(f"seed {seed}: val AUROC {best:.3f}, epochs {ep + 1}", flush=True)
    return net, mu, sd


def score(net, mu, sd, bags):
    with torch.no_grad():
        return {p: torch.sigmoid(net(bag(x, mu, sd))[0]).item() for p, x in bags.items()}


S = {"deephisto": [], "cptac": []}
for s in SEEDS:
    net, mu, sd = fit(s)
    S["deephisto"].append(score(net, mu, sd, dh)); S["cptac"].append(score(net, mu, sd, cp))
# mean-pool comparator
lr = make_pipeline(StandardScaler(), PCA(128, random_state=0), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, scoring="roc_auc"))
lr.fit(MEAN.values, y.values)


def mp(bags):
    X = np.stack([bags[p].float().mean(0).numpy() for p in bags]); return dict(zip(bags, lr.predict_proba(X)[:, 1]))


rows = []
for name, bags in [("deephisto", dh), ("cptac", cp)]:
    ab = pd.DataFrame(S[name]).mean(0); mpp = pd.Series(mp(bags))
    for p in bags:
        r = {"cohort": name, "patient": p, "p_abmil": ab[p], "p_meanpool": mpp[p], "n_tiles": len(bags[p])}
        if name == "deephisto":
            r.update(y_gbm=int(dh_lab.loc[p, "y_gbm"]), cns5=dh_lab.loc[p, "cns5"])
        else:
            r.update(y_gbm=1, idh_mutant=p in idh_mut, cns5="Glioblastoma (CPTAC-GBM)")
        rows.append(r)
SC = pd.DataFrame(rows); SC.to_csv(OUT / "external_scores.csv", index=False)

rng = np.random.default_rng(0); met = []


def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n); return c - h, c + h


for model in ["p_abmil", "p_meanpool"]:
    d = SC[SC.cohort == "deephisto"]
    a = roc_auc_score(d.y_gbm, d[model]); bs = []
    for _ in range(4000):
        i = rng.integers(0, len(d), len(d)); dd = d.iloc[i]
        if dd.y_gbm.nunique() == 2:
            bs.append(roc_auc_score(dd.y_gbm, dd[model]))
    sens = ((d[model] >= 0.5) & (d.y_gbm == 1)).sum() / (d.y_gbm == 1).sum(); spec = ((d[model] < 0.5) & (d.y_gbm == 0)).sum() / (d.y_gbm == 0).sum()
    met.append(dict(cohort="DeepHisto", model=model, n=len(d), n_gbm=int(d.y_gbm.sum()), auroc=a, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                    sensitivity=sens, specificity=spec, brier=brier_score_loss(d.y_gbm, d[model])))
    for sub, c in [("CPTAC-GBM, all", SC[SC.cohort == "cptac"]), ("CPTAC-GBM, IDH-wild-type", SC[(SC.cohort == "cptac") & (SC.idh_mutant != True)])]:
        k = int((c[model] >= 0.5).sum()); lo, hi = wilson(k, len(c))
        met.append(dict(cohort=sub, model=model, n=len(c), n_gbm=len(c), sensitivity=k / len(c), lo=lo, hi=hi, median_prob=c[model].median()))
pd.DataFrame(met).to_csv(OUT / "external_metrics.csv", index=False)
print(pd.DataFrame(met).to_string()); print("EXT_DONE", flush=True)
