"""Phase 2b figures (human tumour vs matched healthy biopsies, RT-FDC; Soteriou et al. 2023 data).
Reads only the result tables written by src/15_mech_biopsy_dcor.py on DCC (copied to dcc_copy/p2b/results/phase2b).
Usage: python make_p2b_figures.py <phase2b dir> <out dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#3B2F2A", "#7A6E66", "#E7E0D8"
TERRA, BLUE, OCHRE, SAGE, PLUM, SAND = "#B8462F", "#3A6B8C", "#E0A93B", "#6FA287", "#5E3350", "#D8D0C6"
STRAT = {"colon-fresh": TERRA, "colon-frozen": BLUE, "lung-fresh": OCHRE}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
                     "axes.labelsize": 8.5, "axes.titlesize": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 7, "savefig.dpi": 300})
D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4


def panel(ax, s):
    ax.text(-0.22, 1.13, s, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def save(f, name):
    f.savefig(OUT / f"{name}.png", bbox_inches="tight"); f.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(f)


# ---------------------------------------------------------------- Fig P2b-1: paired location shift and shift functions
P = pd.read_csv(D / "figdata_pairs.csv")
Q = pd.read_csv(D / "figdata_quantiles.csv")
qcols = [c for c in Q.columns if c.startswith("q")]; qx = np.array([int(c[1:]) for c in qcols]) / 100
f, axs = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM), gridspec_kw=dict(width_ratios=[0.8, 1, 1], wspace=0.42))
ax = axs[0]
for s, g in P.groupby("stratum"):
    for _, r in g.iterrows():
        ls = "--" if r.role != "main" else "-"
        ax.plot([0, 1], [r.D_median_h, r.D_median_t], color=STRAT[s], lw=0.8, alpha=0.75, ls=ls)
    ax.plot([], [], color=STRAT[s], label=f"{s} (n={len(g)})")
ax.plot([], [], color=INK2, ls="--", label="blind-set unit")
ax.set_xticks([0, 1], ["healthy", "tumour"]); ax.set_xlim(-0.2, 1.2)
ax.set_ylabel("median log$_{10}$ deformation"); panel(ax, "a")
ax.set_title(f"tumour lower in {np.mean(P.D_median_t < P.D_median_h):.0%} of units", fontsize=8)
for ax, feat, lab, letter in [(axs[1], "logD", "log$_{10}$ deformation", "b"), (axs[2], "logA", "log$_{10}$ area", "c")]:
    for s in STRAT:
        t = Q[(Q.feature == feat) & (Q.stratum == s) & (Q.condition == "tumour")].set_index("unit")[qcols]
        h = Q[(Q.feature == feat) & (Q.stratum == s) & (Q.condition == "healthy")].set_index("unit")[qcols]
        u = t.index.intersection(h.index); d = (t.loc[u].values - h.loc[u].values)
        m = d.mean(0); se = d.std(0, ddof=1) / np.sqrt(len(u))
        ax.plot(qx, m, color=STRAT[s], lw=1.4, label=s); ax.fill_between(qx, m - 1.96 * se, m + 1.96 * se, color=STRAT[s], alpha=0.15, lw=0)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlabel("quantile of the cell distribution"); ax.set_ylabel(f"tumour − healthy, {lab}"); panel(ax, letter)
    ax.set_title("shift function, mean ± 95% CI", fontsize=8)
h, l = axs[0].get_legend_handles_labels(); f.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=7)
save(f, "FigP2b_1_location_and_shift")

# ---------------------------------------------------------------- Fig P2b-2: distance decomposition and variance partition
T = pd.read_csv(D / "distance_pair_types.csv"); V = pd.read_csv(D / "variance_partition.csv"); PM = pd.read_csv(D / "permanova.csv")
f, axs = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM), gridspec_kw=dict(width_ratios=[1.25, 0.9, 0.9], wspace=0.5))
ax = axs[0]
lab = {"same unit: tumour vs healthy": "same patient,\ntumour vs healthy",
       "same stratum, different units, same condition: tumour-tumour / healthy-healthy": "different patients,\nsame condition",
       "same stratum, different units, different condition": "different patients,\ndifferent condition"}
t = T[(T.feature == "log10 deformation") & T["pair type"].isin(lab)]
y = np.arange(len(t))
ax.barh(y + 0.18, t.median_shift, 0.34, color=TERRA, label="median shift")
ax.barh(y - 0.18, t.shape_w1, 0.34, color=BLUE, label="shape (median-centred W1)")
nf = T[(T.feature == "log10 deformation") & T["pair type"].str.startswith("split-half")].shape_w1.iloc[0]
ax.axvline(nf, color=INK2, ls="--", lw=0.8); ax.text(nf, len(t) - 0.35, " split-half\n noise floor", fontsize=6.3, color=INK2, va="top")
ax.set_yticks(y, [lab[s] for s in t["pair type"]]); ax.set_xlabel("mean distance, log$_{10}$ deformation"); ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), fontsize=6.3, ncol=2)
panel(ax, "a")
ax = axs[1]
pm = PM[PM.feature == "log10 deformation"]
ax.bar(range(len(pm)), pm.R2, color=[INK2, TERRA, BLUE])
for i, r in enumerate(pm.itertuples()):
    ax.text(i, r.R2 + 0.004, f"p={r.p_perm:.4f}", ha="center", fontsize=6.3)
ax.set_xticks(range(len(pm)), ["full W1", "median\nshift", "shape"]); ax.set_ylabel("PERMANOVA R² (tumour vs healthy)")
ax.set_title("deformation", fontsize=8); panel(ax, "b")
ax = axs[2]
descs = ["D_median", "D_iqr", "A_median", "A_iqr"]; names = ["med.\nD", "IQR\nD", "med.\nA", "IQR\nA"]
for j, (comp, col, nm) in enumerate([("C(condition)", TERRA, "tumour vs healthy"), ("C(unit)", BLUE, "patient unit"), ("residual", SAND, "residual")]):
    v = np.array([V[(V.descriptor == d) & (V.component == comp)].partial_R2.iloc[0] for d in descs])
    ax.bar(np.arange(len(descs)) + (j - 1) * 0.27, v, 0.27, color=col, label=nm)
ax.set_xticks(range(len(descs)), names, fontsize=6.5); ax.set_ylabel("partial R²"); ax.set_ylim(0, 0.8)
ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
panel(ax, "c")
save(f, "FigP2b_2_distances_variance")

# ---------------------------------------------------------------- Fig P2b-3: classification
C = pd.read_csv(D / "classification_cv.csv"); B = pd.read_csv(D / "blind_validation.csv")
K = pd.read_csv(D / "classification_cells.csv"); PC = pd.read_csv(D / "positive_controls.csv")
f, axs = plt.subplots(1, 3, figsize=(183 * MM, 66 * MM), gridspec_kw=dict(width_ratios=[1.3, 0.9, 1.0], wspace=0.55))
ax = axs[0]
c = C.iloc[::-1].reset_index(drop=True); y = np.arange(len(c))
ax.errorbar(c.auroc_pooled, y, xerr=c.auroc_sd_over_repeats, fmt="o", color=TERRA, ms=4, label="5-fold CV by patient unit, 39 units (±1 s.d., 10 repeats)")
b = B.set_index("features").reindex(c.features)
ax.plot(b.auroc.values, y, "D", color=BLUE, ms=3.5, mfc="white", label="trained on 31 units, tested on 8 blind-set units")
ax.axvline(0.5, color=INK2, lw=0.7, ls=":")
ax.set_yticks(y, c.features); ax.set_xlabel("AUROC, tumour vs healthy"); ax.set_xlim(0.45, 0.9); ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.16), fontsize=6.3)
panel(ax, "a")
ax = axs[1]
for feat, col in [("median deformation", TERRA), ("deciles (deformation)", BLUE), ("deciles (deformation, size)", OCHRE), ("median deformation + size", SAGE)]:
    k = K[K.features == feat]
    ax.errorbar(k.cells_per_bag, k.auroc_pooled, yerr=k.auroc_sd_over_repeats, marker="o", ms=3, color=col, lw=1.1, label=feat)
ax.set_xscale("log"); ax.set_xlabel("cells per measurement (sub-sampled)"); ax.set_ylabel("AUROC"); ax.set_ylim(0.6, 0.8)
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25)); panel(ax, "b")
ax = axs[2]
pc = PC.copy(); pc["ctl"] = pc.control.str.replace("healthy tissue: ", "").str.replace(r" \(.*\)", "", regex=True)
ctls = pc.ctl.unique(); feats = pc.features.unique(); w = 0.26
for j, (fe, col) in enumerate(zip(feats, [TERRA, SAGE, BLUE])):
    v = [pc[(pc.ctl == cc) & (pc.features == fe)].auroc.iloc[0] for cc in ctls]
    ax.bar(np.arange(len(ctls)) + (j - 1) * w, v, w, color=col, label=fe)
ax.axhline(0.5, color=INK2, lw=0.7, ls=":")
ax.set_xticks(range(len(ctls)), ["colon vs\nlung", "frozen vs\nfresh"]); ax.set_ylabel("AUROC (healthy tissue only)")
ax.set_ylim(0, 1.05); ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25)); panel(ax, "c")
save(f, "FigP2b_3_classification")

# ---------------------------------------------------------------- Fig P2b-4: sensitivity analyses
S = pd.read_csv(D / "sensitivity.csv").dropna(subset=["auroc_pooled"])
S["features"] = S.features.replace({"median log10 E (LUT-covered events)": "median deformation", "deciles of log10 E": "deciles (deformation)",
                                     "median log10 E + median size": "median deformation + size"})
S["analysis"] = S.analysis.replace({"Young's modulus (LUT-covered events, one fit per measurement)": "Young's modulus instead of deformation (26 units)"})
f, ax = plt.subplots(figsize=(120 * MM, 78 * MM))
an = list(dict.fromkeys(S.analysis)); feats = ["median deformation", "median deformation + size", "deciles (deformation)", "deciles (deformation, size)"]
cols = [TERRA, SAGE, BLUE, OCHRE]
for j, (fe, col) in enumerate(zip(feats, cols)):
    s = S[S.features == fe].set_index("analysis").reindex(an)
    ax.plot(s.auroc_pooled.values, np.arange(len(an)) + (j - 1.5) * 0.15, "o", color=col, ms=3.5, label=fe)
ax.set_yticks(range(len(an)), an); ax.invert_yaxis(); ax.axvline(0.5, color=INK2, lw=0.7, ls=":")
ax.set_xlabel("AUROC, tumour vs healthy"); ax.legend(fontsize=6.3, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
save(f, "FigP2b_4_sensitivity")
print("saved", sorted(p.name for p in OUT.glob("FigP2b_*.png")))
