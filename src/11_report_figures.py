"""Step 11 - supplementary figures for the written report (pipeline diagram, ROC, calibration,
embedding map, tissue features, institution composition)."""
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import FIG, PROC, RES, SEED  # noqa: E402

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                          "#e87ba4", "#008300", "#4a3aa7", "#e34948")
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "legend.frameon": False})


def save(fig, name):
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


co = pd.read_csv(PROC / "cohort.csv").set_index("patient")
sl = pd.read_csv(PROC / "slides.csv").set_index("patient")

# ------------------------------------------------------------- A pipeline diagram
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 110); ax.set_ylim(0, 62); ax.axis("off")


def box(x, y, w, h, title, lines, edge, fill="#ffffff", dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=1.6,
                       edgecolor=edge, facecolor=fill, linestyle="--" if dashed else "-")
    ax.add_patch(p)
    ax.text(x + 1.2, y + h - 1.6, title, fontsize=8.6, fontweight="bold", color=INK, va="top")
    for i, l in enumerate(lines):
        ax.text(x + 1.2, y + h - 4.6 - i * 2.55, l, fontsize=7.3, color=INK2, va="top")


def arrow(x0, y0, x1, y1, color=MUTED, dashed=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, linewidth=1.4,
                                 color=color, linestyle="--" if dashed else "-"))


for x, t in [(1, "Data sources"), (29, "Processing and features"), (63, "Branch models"), (88, "Fusion and evaluation")]:
    ax.text(x, 61, t, fontsize=9, color=MUTED, fontweight="bold")
box(1, 47, 24, 12, "Labels", ["TCGA pan-glioma (cBioPortal)", "1,122 patients, GBM vs LGG", "IDH, 1p/19q, TERT, +7/-10"], ORANGE)
box(1, 32, 24, 12, "Pathology reports", ["TCGA-Reports (OCR text)", "868 patients"], VIOLET)
box(1, 17, 24, 12, "Diagnostic H&E slides", ["GDC tile service via browser", "880 slides: 16 tiles + thumb", "scanner info from SVS header"], AQUA)
box(1, 2, 24, 12, "Whole-cell mechanics", ["Mechanomics RT-DC Set 01", "3 GBM lines x 3 conditions", "not linked to any patient"], GREEN, dashed=True)
box(29, 47, 30, 12, "Cohort linkage + shared splits", ["TCGA barcode linkage, institutions", "5x5 stratified patient CV", "5x5 institution-grouped CV"], ORANGE)
box(29, 32, 30, 12, "Masking + TF-IDF", ["M0 raw; M1 masks diagnosis, grade", "and molecular results; M2 also", "masks grading criteria"], VIOLET)
box(29, 17, 30, 12, "Tiles -> QC -> features", ["13,966 tiles, 95% pass QC", "HIPT ViT-S/16: 55,864 crops x 384", "nuclei, texture, stain, sharpness"], AQUA)
box(29, 2, 30, 12, "Gating -> per-sample features", ["21,738 gated cells, 21 measurements", "median, spread, tails of log E", "deformation and cell size"], GREEN, dashed=True)
box(63, 32, 21, 12, "Report models", ["TF-IDF + L2 logistic", "regression; age baseline"], VIOLET)
box(63, 17, 21, 12, "Image models", ["mean-pool + L2 LR", "gated attention MIL", "metadata / stain baselines"], AQUA)
box(63, 2, 21, 12, "Mechanics models", ["mixed models (date)", "bag test, held-out date"], GREEN, dashed=True)
box(88, 47, 21, 12, "Fusion (686 matched)", ["late: mean of logits", "early: one LR on both"], BLUE)
box(88, 24, 21, 19, "Validation", ["random patient folds", "institutions held out", "40x-scanned slides only", "institution x scan-year", "UF 15 v 15; LRT", "DeLong + bootstrap"], INK2)
box(88, 2, 21, 18, "Three-input test", ["needs slide, report AND", "cells from the same", "tumour (0 today)", "~150 per class for", "+0.03 AUROC (power)"], RED, dashed=True)
for y in (53, 38, 23):
    arrow(25.6, y, 28.6, y)
arrow(25.6, 8, 28.6, 8, GREEN, True)
arrow(59.6, 38, 62.6, 38); arrow(59.6, 23, 62.6, 23); arrow(59.6, 8, 62.6, 8, GREEN, True)
arrow(44, 46.6, 44, 44.6, ORANGE)
arrow(84.6, 40, 87.6, 50, VIOLET); arrow(84.6, 26, 87.6, 48.5, AQUA)
arrow(98.5, 46.6, 98.5, 43.6, BLUE)
arrow(84.6, 8, 87.6, 8, GREEN, True)
save(fig, "fig0_pipeline.png")

# ------------------------------------------------------------- B ROC curves
fo = pd.read_csv(RES / "fusion_oof.csv")
series = [("late_image+report_M2", "Image + report M2 (late fusion)", BLUE),
          ("image_meanpool", "Image (HIPT mean-pool)", AQUA),
          ("report_M2", "Report, masked M2", VIOLET),
          ("acq_meta (nuisance)", "Acquisition metadata only", MUTED)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, cohort, title in [(axes[0], "all", "All matched patients (n = 686)"), (axes[1], "x40", "40x-scanned slides only (n = 556)")]:
    for m, lab, c in series:
        d = fo[(fo.model == m) & (fo.cohort == cohort) & (fo.scheme == "institution")].join(co.y_gbm, on="patient")
        fpr, tpr, _ = roc_curve(d.y_gbm, d.p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{lab}  {roc_auc_score(d.y_gbm, d.p):.3f}")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    ax.set_xlabel("False-positive rate (LGG called GBM)")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=7.6, title="AUROC, institutions held out", title_fontsize=7.6)
axes[0].set_ylabel("True-positive rate (GBM called GBM)")
fig.tight_layout()
save(fig, "fig9_roc_curves.png")

# ------------------------------------------------------------- C calibration
fig, ax = plt.subplots(figsize=(5.2, 4.2))
for scheme, c, mk, lab in [("patient", VIOLET, "o", "Random patient folds"), ("institution", ORANGE, "D", "Institutions held out")]:
    d = fo[(fo.model == "late_image+report_M2") & (fo.cohort == "all") & (fo.scheme == scheme)].join(co.y_gbm, on="patient")
    q = pd.qcut(d.p, 10, duplicates="drop")
    g = d.groupby(q, observed=True).agg(p=("p", "mean"), y=("y_gbm", "mean"), n=("p", "size"))
    ax.plot(g.p, g.y, marker=mk, ms=6, lw=2, color=c, label=lab)
ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
ax.set_xlabel("Predicted P(GBM), decile mean")
ax.set_ylabel("Observed fraction GBM")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper left")
ax.set_title("Calibration of the fused model")
save(fig, "fig10_calibration.png")

# ------------------------------------------------------------- D embedding map
import umap  # noqa: E402
Z = np.load(PROC / "embeddings_hipt.npz", allow_pickle=True)
emb = Z["emb"].astype(np.float32)
idx = pd.DataFrame({c: Z[c] for c in ["patient", "mag"]})
M = pd.DataFrame({p: emb[g.index].mean(0) for p, g in idx[idx.mag == "x20"].groupby("patient")}).T
M = M.loc[M.index.intersection(co.index)]
X = (M - M.mean()) / (M.std() + 1e-6)
U = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=SEED).fit_transform(X.values)
U = pd.DataFrame(U, index=M.index, columns=["u1", "u2"]).join(co.y_gbm).join(sl[["appmag", "scan_year"]])
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for lab, c, v in [("LGG", BLUE, 0), ("GBM", ORANGE, 1)]:
    d = U[U.y_gbm == v]
    axes[0].scatter(d.u1, d.u2, s=12, color=c, alpha=0.75, edgecolors=SURF, linewidths=0.4, label=f"{lab} (n={len(d)})")
axes[0].set_title("Coloured by diagnosis")
for lab, c, sel in [("40x scan, LGG", BLUE, (U.appmag == 40) & (U.y_gbm == 0)),
                    ("40x scan, GBM", AQUA, (U.appmag == 40) & (U.y_gbm == 1)),
                    ("20x scan (all GBM)", ORANGE, U.appmag == 20)]:
    d = U[sel]
    axes[1].scatter(d.u1, d.u2, s=12, color=c, alpha=0.75, edgecolors=SURF, linewidths=0.4, label=f"{lab} (n={len(d)})")
axes[1].set_title("Coloured by scan magnification")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=7.6, markerscale=1.5)
fig.suptitle("Patient-level HIPT embeddings (mean of 20x crops): scanner groups are as visible as diagnosis", x=0.01,
             ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "fig11_embedding_map.png")

# ------------------------------------------------------------- E tissue features by class and scanner
hp = pd.read_csv(PROC / "handcrafted_patient.csv").set_index("patient").join(co.y_gbm).join(sl.appmag)
groups = [("GBM, 20x scan", (hp.y_gbm == 1) & (hp.appmag == 20), ORANGE),
          ("GBM, 40x scan", (hp.y_gbm == 1) & (hp.appmag == 40), AQUA),
          ("LGG, 40x scan", (hp.y_gbm == 0) & (hp.appmag == 40), BLUE)]
feats = [("x20_nuc_area_frac_mean", "Nuclear area fraction"), ("x20_nuc_density_per_mm2_mean", "Nuclei per mm²"),
         ("x20_nuc_area_iqr_um2_mean", "Nuclear size IQR (µm²)"), ("x20_blur_lapvar_mean", "Sharpness (Laplacian variance)")]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
rng = np.random.default_rng(0)
for ax, (f, title) in zip(axes, feats):
    for i, (lab, m, c) in enumerate(groups):
        v = hp.loc[m, f].dropna().values
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=c, alpha=0.45, edgecolors="none")
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2)
        ax.plot([i, i], [q1, q3], color=INK, lw=1.2)
    ax.set_xticks(range(3), [g[0].replace(", ", "\n") for g in groups], fontsize=7.6)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", visible=False)
    if f == "x20_blur_lapvar_mean":
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.suptitle("Tissue features separate GBM from LGG at either scan magnification; sharpness tracks the scanner",
             x=0.01, ha="left", fontweight="bold")
fig.text(0.01, -0.03, "Bar = median, line = interquartile range; one point per patient (mean over QC-passing 20x tiles).",
         color=MUTED, fontsize=8)
fig.tight_layout()
save(fig, "fig12_tissue_features.png")

# ------------------------------------------------------------- F institutions
it = co.groupby("institution").y_gbm.agg(n="size", gbm="sum")
it["lgg"] = it.n - it.gbm
it = it[it.n >= 10].sort_values("n")
fig, ax = plt.subplots(figsize=(8, 6.2))
yy = np.arange(len(it))
ax.barh(yy, it.gbm, height=0.6, color=ORANGE, label="GBM")
ax.barh(yy, it.lgg, left=it.gbm + 1, height=0.6, color=BLUE, label="LGG")
for y, (name, r) in zip(yy, it.iterrows()):
    tag = "  both classes" if (r.gbm > 0 and r.lgg > 0) else ""
    ax.text(r.n + 4, y, f"{int(r.n)}{tag}", va="center", fontsize=7.6, color=INK2)
ax.set_yticks(yy, [n[:42] for n in it.index], fontsize=7.6)
ax.set_xlabel("Patients with pan-glioma labels")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
ax.set_title("Institutions with 10 or more patients: most sent only one class")
save(fig, "fig13_institutions.png")
