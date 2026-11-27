"""Fig 1 (study overview) in Nature-overview style with real images and real data in every panel.
Real inputs: TCGA slide thumbnail and tiles, an M2-masked TCGA pathology report, RT-DC stiffness distributions of
glioblastoma lines, RT-FDC images of tumour and healthy cells (DCOR, Soteriou et al. 2023), Phase 1 AUROCs and plasmode power.
Run after make_figures.py (imports its style and helpers)."""
import re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.stats import gaussian_kde
from PIL import Image

HERE = Path(__file__).resolve().parent
exec(open(HERE / "make_figures.py").read().split("# ================================================================================ Fig 1")[0])  # style, paths, save(), met()
RI = REPO / "results/real_images"

f = plt.figure(figsize=(W2, 150 * MM))
bg = f.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")


def rbox(x, y, w, h, fc, ec="none", lw=0.9):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=lw, transform=bg.transAxes))


def arr(x0, y0, x1, y1, rad=0.0):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=1.3, color=NAVY,
                                 connectionstyle=f"arc3,rad={rad}", transform=bg.transAxes))


def label(x, y, s, size=7.5, bold=False, color=INK, ha="center", va="center"):
    bg.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal", color=color, ha=ha, va=va, transform=bg.transAxes, linespacing=1.2)


def img_ax(x, y, w, h, path, border=None):
    ax = f.add_axes([x, y, w, h]); ax.imshow(Image.open(path).convert("RGB")); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(border is not None); s_.set_color(border or "none"); s_.set_linewidth(0.6)
    ax.grid(False); return ax


def letter(x, y, s):
    bg.text(x, y, s, fontsize=11, fontweight="bold", color=INK, ha="left", va="top", transform=bg.transAxes)


# ------------------------------------------------------------------ panel a: routine pathology
letter(0.005, 0.995, "a")
label(0.03, 0.975, "Glioblastoma or lower-grade glioma from routine pathology", 8, True, NAVY, ha="left")
# slide
rbox(0.02, 0.69, 0.2, 0.26, LBLUE)
img_ax(0.035, 0.735, 0.17, 0.2, RI / "slide_GBM.png")
label(0.12, 0.712, "H&E whole-slide image", 7.2)
# report
rbox(0.02, 0.52, 0.2, 0.15, LPINK)
rep = ("final diagnosis: left temporal parietal tumor: ▇▇ ▇▇, ▇▇ … the tumor consists of a moderately ▇▇ and highly "
       "infiltrative ▇▇ of astrocytes. there are rare ▇▇ …")
words = rep.split(" "); lines, cur = [], ""
for w_ in words:
    if len(cur) + len(w_) > 27:
        lines.append(cur); cur = w_
    else:
        cur = (cur + " " + w_).strip()
lines.append(cur)
for i, ln in enumerate(lines[:6]):
    bg.text(0.028, 0.66 - i * 0.019, ln, fontsize=5.6, family="DejaVu Sans Mono", color=INK, ha="left", va="top", transform=bg.transAxes)
label(0.12, 0.532, "masked pathology report", 6.8, color=INK2)
# tiles: real GBM and LGG tiles
rbox(0.27, 0.69, 0.2, 0.26, LBLUE)
for k, (cls, j) in enumerate([("GBM", 0), ("GBM", 1), ("LGG", 0), ("LGG", 2)]):
    xx = 0.282 + (k % 2) * 0.09; yy = 0.835 - (k // 2) * 0.105
    img_ax(xx, yy, 0.085, 0.1, RI / f"tile_{cls}_{j}.png", border="white")
bg.text(0.276, 0.885, "GBM", fontsize=6.5, color=TERRA, rotation=90, ha="center", va="center", transform=bg.transAxes, fontweight="bold")
bg.text(0.276, 0.78, "LGG", fontsize=6.5, color=BLUE, rotation=90, ha="center", va="center", transform=bg.transAxes, fontweight="bold")
label(0.37, 0.712, "7.5 million 128-µm tiles → attention MIL", 7.0)
# text model
rbox(0.27, 0.52, 0.2, 0.15, LPINK)
label(0.37, 0.61, "language model\n(BiomedBERT) and\nword-frequency model", 7.0)
label(0.37, 0.535, "trained per fold", 6.5, color=INK2)
arr(0.22, 0.82, 0.27, 0.82); arr(0.22, 0.595, 0.27, 0.595)
# fusion
rbox(0.52, 0.62, 0.13, 0.2, LSAND)
label(0.585, 0.745, "late fusion", 7.5, True)
# probability bar (illustrative scale)
bg.add_patch(Rectangle((0.535, 0.665), 0.10, 0.018, fc="white", ec=INK2, lw=0.5, transform=bg.transAxes))
bg.add_patch(Rectangle((0.535, 0.665), 0.083, 0.018, fc=TERRA, ec="none", transform=bg.transAxes))
label(0.585, 0.645, "P(glioblastoma)", 6.5, color=INK2)
arr(0.47, 0.82, 0.52, 0.76, -0.1); arr(0.47, 0.595, 0.52, 0.68, 0.1)
# evaluation: real AUROCs, all slides vs 40x, institutions held out
rbox(0.70, 0.52, 0.29, 0.43, "white", ec=NAVY)
label(0.845, 0.925, "unseen institutions", 7.5, True, NAVY)
arr(0.65, 0.72, 0.70, 0.72)
ax = f.add_axes([0.80, 0.585, 0.17, 0.29])
ms = [("fusion_primary", "image + report"), ("image_phikon2_abmil", "image"), ("report_bert_M2", "report"), ("acq_meta", "scanner\nmetadata")]
yy = np.arange(len(ms))[::-1]
for y_, (m, n) in zip(yy, ms):
    a_all, a_40 = met(m, "all"), met(m, "x40")
    ax.plot([a_40, a_all], [y_, y_], color=SAND, lw=2, zorder=1)
    ax.plot(a_all, y_, "o", color=MODEL_COL[m], ms=4.5, zorder=2); ax.plot(a_40, y_, "o", color="white", mec=MODEL_COL[m], mew=1.1, ms=4.5, zorder=2)
ax.set_yticks(yy, [n for _, n in ms], fontsize=6.5); ax.set_xlim(0.4, 1.0); ax.set_xticks([0.5, 0.75, 1.0]); ax.tick_params(labelsize=6.5)
ax.set_xlabel("AUROC", fontsize=6.8, labelpad=1); ax.axvline(0.5, color=INK2, lw=0.5, ls=":")
ax.plot([], [], "o", color=INK2, ms=4, label="all slides"); ax.plot([], [], "o", color="white", mec=INK2, ms=4, label="40× slides")
ax.legend(fontsize=6.2, loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.6)

# ------------------------------------------------------------------ panels b–d: single-cell mechanics
bg.plot([0.02, 0.98], [0.485, 0.485], color=GRID, lw=0.8, transform=bg.transAxes)
label(0.03, 0.465, "Does the spread of single-cell stiffness add to its average?", 8, True, NAVY, ha="left")
# b: glioblastoma lines, real RT-DC distributions
letter(0.005, 0.445, "b"); rbox(0.02, 0.03, 0.29, 0.40, LBLUE)
EV = pd.read_csv(DATA / "rtdc_events_gated.csv.gz"); EV = EV[EV.emodulus > 0]; EV["logE"] = np.log10(EV.emodulus)
ax = f.add_axes([0.06, 0.13, 0.23, 0.22]); xs = np.linspace(-0.6, 0.5, 200)
for c, col, nm in [("Serum", SAGE, "serum"), ("EGF", OCHRE, "EGF"), ("FGFJI", PLUM, "FGF + JAKi")]:
    ax.plot(xs, gaussian_kde(EV[EV.cond == c].logE)(xs), color=col, lw=1.4, label=nm)
ax.set_yticks([]); ax.set_xlabel("log$_{10}$ stiffness (kPa)", fontsize=6.8, labelpad=1); ax.tick_params(labelsize=6.5)
ax.spines["left"].set_visible(False); ax.set_facecolor("none"); ax.grid(False)
ax.legend(fontsize=6.2, loc="upper left", handlelength=1, frameon=False)
label(0.165, 0.395, "glioblastoma cells, 3 signalling states", 7.0)
label(0.165, 0.065, "21,738 cells · deformability cytometry", 6.5, color=INK2)
# c: human biopsies, real RT-FDC cell images
letter(0.335, 0.445, "c"); rbox(0.35, 0.03, 0.30, 0.40, LPINK)
label(0.50, 0.395, "human tissue: tumour and matched healthy", 7.0)
heal = [1, 4, 6, 7]; tum = [0, 1, 3, 5]
for k, j in enumerate(heal):
    img_ax(0.385 + k * 0.064, 0.225, 0.06, 0.12, RI / f"cell_healthy_{j}.png")
for k, j in enumerate(tum):
    img_ax(0.385 + k * 0.064, 0.085, 0.06, 0.12, RI / f"cell_tumour_{j}.png")
bg.text(0.372, 0.285, "healthy", fontsize=6.5, rotation=90, ha="center", va="center", transform=bg.transAxes, color=SAGE, fontweight="bold")
bg.text(0.372, 0.145, "tumour", fontsize=6.5, rotation=90, ha="center", va="center", transform=bg.transAxes, color=TERRA, fontweight="bold")
label(0.50, 0.055, "39 patients · colon and lung", 6.5, color=INK2)
# d: design analysis, real power grid
letter(0.675, 0.445, "d"); rbox(0.69, 0.03, 0.29, 0.40, "white", ec=NAVY)
label(0.835, 0.395, "matched study design", 7.0, True, NAVY)
PM = pd.read_csv(R / "phase2/plasmode_main.csv"); sc = "S2 spread (LGG IQR x1.25)"; d_ = PM[PM.scenario == sc]
n_ = sorted(d_.n_per_class.unique()); c_ = sorted(d_.cells_per_tumour.unique())
Hp = np.array([[d_[(d_.n_per_class == a) & (d_.cells_per_tumour == b)].power_M2_vs_M1.iloc[0] for b in c_] for a in n_])
ax = f.add_axes([0.76, 0.11, 0.18, 0.24]); ax.imshow(Hp, cmap=SEQ, vmin=0, vmax=1, origin="lower", aspect="auto")
ax.set_xticks(range(len(c_)), c_, fontsize=6.2); ax.set_yticks(range(len(n_)), n_, fontsize=6.2)
ax.set_xlabel("cells per tumour", fontsize=6.6, labelpad=1); ax.set_ylabel("patients per class", fontsize=6.6, labelpad=1); ax.grid(False)
[s_.set_visible(False) for s_ in ax.spines.values()]
label(0.835, 0.055, "power to detect a wider spread (dark = high)", 6.5, color=INK2)
arr(0.31, 0.24, 0.35, 0.24); arr(0.65, 0.24, 0.69, 0.24)
save(f, "Fig1")

# ------------------------------------------------------------------ graphical abstract (real images; Elsevier: readable at 13 x 5 cm)
f = plt.figure(figsize=(160 * MM, 62 * MM))
bg = f.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
rbox(0.01, 0.52, 0.2, 0.46, LBLUE); img_ax(0.025, 0.62, 0.17, 0.33, RI / "slide_GBM.png"); label(0.11, 0.565, "H&E slide", 7.5)
rbox(0.01, 0.03, 0.2, 0.45, LPINK)
for i, ln in enumerate(["final diagnosis: left", "temporal parietal tumor:", "▇▇ ▇▇, ▇▇ … infiltrative", "▇▇ of astrocytes …"]):
    bg.text(0.02, 0.4 - i * 0.07, ln, fontsize=6.2, family="DejaVu Sans Mono", color=INK, ha="left", va="top", transform=bg.transAxes)
label(0.11, 0.075, "masked report", 7.5)
rbox(0.26, 0.3, 0.17, 0.42, LSAND); label(0.345, 0.51, "fusion model\nGBM vs LGG", 8, True)
arr(0.21, 0.75, 0.26, 0.6, -0.1); arr(0.21, 0.25, 0.26, 0.42, 0.1)
rbox(0.48, 0.3, 0.2, 0.42, "white", ec=NAVY)
label(0.58, 0.51, f"AUROC {met('fusion_primary'):.2f}\nat unseen hospitals\n{met('fusion_primary', 'x40'):.2f} on 40× slides", 7.8)
arr(0.43, 0.51, 0.48, 0.51)
rbox(0.73, 0.03, 0.26, 0.95, LPINK)
label(0.86, 0.93, "single cells: tumour vs healthy", 7.2, True)
for k, j in enumerate([1, 4, 6]):
    img_ax(0.745 + k * 0.08, 0.6, 0.075, 0.26, RI / f"cell_healthy_{j}.png")
for k, j in enumerate([0, 1, 3]):
    img_ax(0.745 + k * 0.08, 0.3, 0.075, 0.26, RI / f"cell_tumour_{j}.png")
label(0.86, 0.15, "stiffness signal in the median,\nnot in the spread", 7.2)
bg.text(0.737, 0.73, "healthy", fontsize=6.5, rotation=90, ha="center", va="center", transform=bg.transAxes, color=SAGE, fontweight="bold")
bg.text(0.737, 0.43, "tumour", fontsize=6.5, rotation=90, ha="center", va="center", transform=bg.transAxes, color=TERRA, fontweight="bold")
save(f, "Graphical_abstract")
