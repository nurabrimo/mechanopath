"""Figures for the Computers in Biology and Medicine submission (MechanoPath).

All panels are drawn from result files written by the analysis pipeline on DCC (copied to ../../MechanoPath_Report/dcc_copy)
and the processed data in ../../MechanoPath_PLOS/work/data. Output: ../figures/FigN.tif (300 dpi, LZW, RGB) + FigN.pdf.
Elsevier sizes: double column 190 mm; text >= 7 pt Arial.
"""
from pathlib import Path
import io
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
from sklearn.metrics import roc_curve
from scipy.stats import gaussian_kde
from PIL import Image

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]          # repository root
DC = REPO; R = REPO / "results"; P2B = REPO / "phase2b_biopsies/results/phase2b"
DATA = REPO / "data/processed"
OUT = REPO / "paper_figures/figures"; OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4; W2 = 190 * MM

INK, INK2, GRID = "#3B2F2A", "#7A6E66", "#E7E0D8"
TERRA, BLUE, OCHRE, SAGE, PLUM, SAND, SAND_D = "#B8462F", "#3A6B8C", "#E0A93B", "#6FA287", "#5E3350", "#D8D0C6", "#A89F95"
NAVY, LBLUE, LPINK, LSAND = "#24485F", "#DCEAF4", "#F7E1DC", "#F4EEE6"
SEQ = LinearSegmentedColormap.from_list("cozy_seq", ["#FBF4E6", "#F3DDB0", "#E9B872", "#D98A4E", "#B8462F", "#6B2A1C"])
SEQB = LinearSegmentedColormap.from_list("cozy_seqb", ["#F4F1EC", "#C9D8E2", "#8FB0C4", "#3A6B8C", "#1F3F56"])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
                     "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "legend.frameon": False, "axes.titleweight": "bold", "pdf.fonttype": 42, "svg.fonttype": "none"})
MODEL_COL = {"fusion_primary": TERRA, "image_phikon2_abmil": BLUE, "report_bert_M2": OCHRE, "report_tfidf_M2": SAGE, "acq_meta": SAND_D,
             "fusion_image_tfidfM2": PLUM}
MODEL_NAME = {"fusion_primary": "Image + report (fusion)", "fusion_image_tfidfM2": "Image + report (TF-IDF fusion)",
              "image_phikon2_abmil": "Image, attention MIL", "image_phikon2_meanpool": "Image, mean pooling",
              "image_hipt_abmil": "Image, HIPT encoder", "report_bert_M2": "Report, BiomedBERT", "report_tfidf_M2": "Report, TF-IDF",
              "report_unmasked_M0": "Report, unmasked (reference)", "age": "Age", "acq_meta": "Acquisition metadata",
              "stain_color": "Stain colour", "tissue_handcrafted": "Hand-crafted tissue features"}


def panel(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def save(f, name, max_mm=190):
    # shrink the canvas (not the text) until the trimmed figure is at most max_mm wide
    for _ in range(8):
        f.canvas.draw(); bb = f.get_tightbbox(f.canvas.get_renderer())
        if bb.width / MM <= max_mm + 0.5:
            break
        wi, hi = f.get_size_inches(); f.set_size_inches(wi * max_mm / (bb.width / MM) * 0.995, hi)
    f.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    buf = io.BytesIO(); f.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white"); plt.close(f)
    im = Image.open(buf).convert("RGB"); im.save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    print(name, im.size, f"{im.size[0] / 300 * 25.4:.0f} x {im.size[1] / 300 * 25.4:.0f} mm")


M1 = pd.read_csv(R / "phase1/phase1_metrics.csv"); OOF = pd.read_csv(R / "phase1/phase1_oof.csv")
cohort = pd.read_csv(DATA / "cohort.csv"); slides = pd.read_csv(DATA / "slides.csv")


def met(model, cohort_="all", scheme="institution", col="auroc"):
    r = M1[(M1.model == model) & (M1.cohort == cohort_) & (M1.scheme == scheme)]
    return r[col].iloc[0] if len(r) else np.nan


# ================================================================================ Fig 1: study design and cohort
def box(ax, x, y, w, h, fc, text, fs=7.2, bold=False, ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018", fc=fc, ec=ec or "none", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=INK, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=1.1, color=NAVY, shrinkA=0, shrinkB=0))


def asp(ax):
    bb = ax.get_position(); W, H = ax.figure.get_size_inches(); return (bb.height * H) / (bb.width * W)


def icon_slide(ax, x, y, s=0.05):
    a = asp(ax)
    ax.add_patch(Rectangle((x, y), s * 2.6 * a, s * 1.0, fc="white", ec=NAVY, lw=0.7))
    ax.add_patch(Ellipse((x + s * 1.3 * a, y + s * 0.5), s * 1.9 * a, s * 0.7, fc="#E8A9B8", ec="#B76E86", lw=0.5))
    ax.add_patch(Ellipse((x + s * 1.0 * a, y + s * 0.55), s * 0.7 * a, s * 0.3, fc="#9C5C9E", ec="none", alpha=0.7))
    ax.add_patch(Ellipse((x + s * 1.6 * a, y + s * 0.42), s * 0.5 * a, s * 0.22, fc="#9C5C9E", ec="none", alpha=0.5))


def icon_doc(ax, x, y, s=0.05):
    a = asp(ax)
    ax.add_patch(Rectangle((x, y), s * 0.85 * a, s * 1.1, fc="white", ec=NAVY, lw=0.7))
    for k in range(5):
        ax.plot([x + s * 0.12 * a, x + s * (0.72 if k % 2 == 0 else 0.55) * a], [y + s * (0.92 - 0.17 * k)] * 2, color=SAND_D, lw=0.8)
    ax.add_patch(Rectangle((x + s * 0.12 * a, y + s * 0.12), s * 0.4 * a, s * 0.09, fc=INK, ec="none"))


def icon_cells(ax, x, y, s=0.05):
    a = asp(ax); rng = np.random.default_rng(3)
    for k in range(8):
        r = s * rng.uniform(0.16, 0.24); cx, cy = x + s * rng.uniform(0.2, 2.2) * a, y + s * rng.uniform(0.2, 0.9)
        ax.add_patch(Ellipse((cx, cy), r * 2 * a, r * 2 * rng.uniform(0.8, 1.0), fc="#F2C7A5" if k % 3 else "#D98A4E", ec="#9C5A33", lw=0.5))
        ax.add_patch(Ellipse((cx, cy), r * 0.8 * a, r * 0.7, fc="#9C5A33", ec="none", alpha=0.55))


f = plt.figure(figsize=(W2, 150 * MM))
ax = f.add_axes([0, 0.43, 1, 0.57]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
# Phase 1 lane
ax.text(0.03, 0.93, "Question 1 · Can routine pathology tell glioblastoma from lower-grade glioma at hospitals the model has not seen?",
        fontsize=7.5, fontweight="bold", color=NAVY, va="top")
box(ax, 0.03, 0.55, 0.17, 0.3, LBLUE, "\n\n\nH&E whole-slide image\n689 TCGA patients\n33 contributing sites")
icon_slide(ax, 0.075, 0.74, 0.075)
box(ax, 0.03, 0.17, 0.17, 0.3, LPINK, "\n\n\nPathology report\ntumour type, grade and\nmolecular terms masked")
icon_doc(ax, 0.1, 0.33, 0.085)
box(ax, 0.26, 0.55, 0.17, 0.3, LBLUE, "7.47 million tissue tiles\nfoundation-model features\n(Phikon-v2)\n→ attention-based\nmultiple-instance model")
box(ax, 0.26, 0.17, 0.17, 0.3, LPINK, "biomedical language\nmodel (BiomedBERT)\nand bag-of-words\nmodel (TF-IDF)")
arrow(ax, 0.20, 0.70, 0.26, 0.70); arrow(ax, 0.20, 0.32, 0.26, 0.32)
box(ax, 0.49, 0.36, 0.15, 0.3, LSAND, "late fusion\nP(glioblastoma)", bold=True)
arrow(ax, 0.43, 0.70, 0.49, 0.56); arrow(ax, 0.43, 0.32, 0.49, 0.46)
box(ax, 0.70, 0.36, 0.27, 0.49, "white",
    "Evaluation designed against shortcuts\n\n• whole institutions held out\n• 40× scans only (magnification\n   no longer marks the class)\n"
    "• acquisition-only models\n• WHO CNS5 molecular groups", ec=NAVY, fs=7)
arrow(ax, 0.64, 0.51, 0.70, 0.56)
# Phase 2 lane
ax2 = f.add_axes([0, 0.0, 1, 0.43]); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
ax2.text(0.03, 0.98, "Question 2 · Does the spread of single-cell stiffness carry diagnostic information beyond its average?",
         fontsize=7.5, fontweight="bold", color=NAVY, va="top")
box(ax2, 0.03, 0.47, 0.25, 0.4, LBLUE, "\n\n\nGlioblastoma cell lines (RT-DC)\n3 lines × 3 STAT3/Hes3 states\n× 3 days · 21,738 cells")
icon_cells(ax2, 0.1, 0.67, 0.14)
box(ax2, 0.03, 0.02, 0.25, 0.4, LPINK, "\n\n\nHuman tumour biopsies (RT-FDC)\n39 tumour/healthy tissue pairs\ncolon and lung")
icon_cells(ax2, 0.1, 0.22, 0.14)
box(ax2, 0.34, 0.25, 0.3, 0.4, LSAND, "location (median) versus shape\n(spread, tails, skewness) of the\nsingle-cell distribution\n\nreproducibility · cells needed ·\nclassification beyond the median")
arrow(ax2, 0.28, 0.67, 0.34, 0.52); arrow(ax2, 0.28, 0.22, 0.34, 0.38)
box(ax2, 0.70, 0.25, 0.27, 0.4, "white", "Design analysis\nreal image + report scores\ncombined with real cell\ndistributions → patients and\ncells needed for a matched study",
    ec=NAVY, fs=7)
arrow(ax2, 0.64, 0.45, 0.70, 0.45)
save(f, "Fig1")

inst = pd.read_csv(R / "institutions.csv").sort_values("n", ascending=False)
# ================================================================================ Fig 2: discrimination at unseen institutions
f = plt.figure(figsize=(W2, 92 * MM))
roc_models = ["fusion_primary", "image_phikon2_abmil", "report_tfidf_M2", "report_bert_M2", "acq_meta"]
SHORT = {"fusion_primary": "Image + report", "image_phikon2_abmil": "Image", "report_tfidf_M2": "Report, TF-IDF", "report_bert_M2": "Report, BiomedBERT",
         "acq_meta": "Acquisition metadata"}
axA = f.add_axes([0.065, 0.36, 0.25, 0.55]); axB = f.add_axes([0.375, 0.36, 0.25, 0.55]); axC = f.add_axes([0.80, 0.36, 0.185, 0.55])
for ax, coh, title in [(axA, "all", "All slides"), (axB, "x40", "40× slides only")]:
    for m in roc_models[::-1]:
        d = OOF[(OOF.model == m) & (OOF.cohort == coh) & (OOF.scheme == "institution")]
        fpr, tpr, _ = roc_curve(d.y, d.p)
        ax.plot(fpr, tpr, color=MODEL_COL[m], lw=1.7 if m == "fusion_primary" else 1.1, zorder=3 if m == "fusion_primary" else 2)
    ax.plot([0, 1], [0, 1], ls=":", color=INK2, lw=0.7); ax.set_aspect("equal"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity", labelpad=2)
    ax.set_title(f"{title}\ninstitutions held out", fontsize=8, pad=4)
# shared key with AUROC values (all / 40x), below panels a and b
from matplotlib.lines import Line2D
hs = [Line2D([], [], color=MODEL_COL[m], lw=2) for m in roc_models]
ls_ = [f"{SHORT[m]}   {met(m, 'all'):.3f} / {met(m, 'x40'):.3f}" for m in roc_models]
f.legend(hs, ls_, loc="upper center", bbox_to_anchor=(0.345, 0.2), ncol=2, fontsize=6.8, columnspacing=1.6, handlelength=1.6,
         title="AUROC, all slides / 40× slides", title_fontsize=6.8)
# panel c: random folds versus institutions held out
y = np.arange(len(roc_models))[::-1]
for yy, m in zip(y, roc_models):
    a_, b_ = met(m, "all", "patient"), met(m, "all", "institution")
    axC.plot([b_, a_], [yy, yy], color=SAND, lw=2.6, solid_capstyle="round", zorder=1)
    axC.plot(a_, yy, "o", color="white", mec=MODEL_COL[m], mew=1.3, ms=5.5, zorder=2); axC.plot(b_, yy, "o", color=MODEL_COL[m], ms=5.5, zorder=2)
labs = {"fusion_primary": "Image +\nreport", "image_phikon2_abmil": "Image", "report_tfidf_M2": "Report\nTF-IDF", "report_bert_M2": "Report\nBiomedBERT",
        "acq_meta": "Acquisition\nmetadata"}
axC.set_yticks(y, [labs[m] for m in roc_models], fontsize=6.8); axC.set_xlim(0.66, 1.02); axC.set_xticks([0.7, 0.8, 0.9, 1.0]); axC.set_xlabel("AUROC")
axC.set_ylim(-0.6, len(roc_models) - 0.4); axC.set_title("Effect of holding out\ninstitutions (all slides)", fontsize=8, pad=4)
axC.plot([], [], "o", color="white", mec=INK2, mew=1.2, label="random patient folds"); axC.plot([], [], "o", color=INK2, label="institutions held out")
axC.legend(loc="upper center", bbox_to_anchor=(0.3, -0.25), fontsize=6.8, ncol=1, handletextpad=0.3)
for ax, s_, dx in [(axA, "a", 0.055), (axB, "b", 0.055), (axC, "c", 0.115)]:
    bb = ax.get_position(); f.text(bb.x0 - dx, 0.985, s_, fontsize=11, fontweight="bold", va="top")
save(f, "Fig2")

# ================================================================================ Fig 3: model profiles (heat map + radar)
f = plt.figure(figsize=(W2, 92 * MM))
ax = f.add_axes([0.25, 0.1, 0.33, 0.72])
rows = ["fusion_primary", "fusion_image_tfidfM2", "image_phikon2_abmil", "image_phikon2_meanpool", "image_hipt_abmil", "report_tfidf_M2",
        "report_bert_M2", "report_unmasked_M0", "tissue_handcrafted", "stain_color", "age", "acq_meta"]
cols = [("all", "patient"), ("all", "institution"), ("x40", "patient"), ("x40", "institution")]
H = np.array([[met(m, c, s) for c, s in cols] for m in rows])
im = ax.imshow(H, cmap=SEQ, vmin=0.4, vmax=1.0, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=6.6, color="white" if H[i, j] > 0.86 else INK)
ax.set_yticks(range(len(rows)), [MODEL_NAME[m] for m in rows], fontsize=7)
ax.set_xticks(range(4), ["random\nfolds", "sites\nheld out", "random\nfolds", "sites\nheld out"], fontsize=6.8)
ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
ax.text(0.5, -2.1, "all slides", ha="center", fontsize=7.2, fontweight="bold"); ax.text(2.5, -2.1, "40× slides", ha="center", fontsize=7.2, fontweight="bold")
ax.plot([1.5, 1.5], [-0.5, len(rows) - 0.5], color="white", lw=2)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
cax = f.add_axes([0.6, 0.1, 0.012, 0.72]); cb = f.colorbar(im, cax=cax); cb.outline.set_visible(False)
cax.set_title("AUROC", fontsize=7, pad=4)
ax.text(-0.95, 1.13, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")
ax = f.add_axes([0.76, 0.24, 0.21, 0.48], projection="polar")
axes_ = [("AUROC", lambda m: met(m)), ("AUROC\n40×", lambda m: met(m, "x40")), ("AUPRC", lambda m: met(m, col="auprc")),
         ("AUROC\nwithin site", lambda m: met(m, col="auroc_within_institution")), ("balanced\naccuracy", lambda m: met(m, col="balanced_accuracy@0.5")),
         ("1 − ECE", lambda m: 1 - met(m, col="ece"))]
ang = np.linspace(0, 2 * np.pi, len(axes_), endpoint=False).tolist(); ang += ang[:1]
for m in ["acq_meta", "report_bert_M2", "report_tfidf_M2", "image_phikon2_abmil", "fusion_primary"]:
    v = [fn(m) for _, fn in axes_]; v += v[:1]
    ax.plot(ang, v, color=MODEL_COL[m], lw=1.5 if m == "fusion_primary" else 1.0, label=MODEL_NAME[m])
    ax.fill(ang, v, color=MODEL_COL[m], alpha=0.10 if m == "fusion_primary" else 0.03)
ax.set_xticks(ang[:-1], [a for a, _ in axes_], fontsize=6.8); ax.tick_params(axis="x", pad=6)
ax.set_ylim(0.3, 1.0); ax.set_yticks([0.5, 0.75, 1.0], ["0.5", "0.75", "1.0"], fontsize=6.2); ax.set_rlabel_position(60)
ax.grid(color=GRID, lw=0.6); ax.spines["polar"].set_color(GRID)
h, l = ax.get_legend_handles_labels()
f.legend(h[::-1], l[::-1], loc="lower center", bbox_to_anchor=(0.835, -0.07), fontsize=6.6, ncol=2, columnspacing=1.0, handlelength=1.4)
f.text(0.70, 0.9, "b", fontsize=11, fontweight="bold")
ax.set_title("institutions held out", fontsize=7.5, pad=16, fontweight="normal", color=INK2)
save(f, "Fig3")

# ================================================================================ Fig 3: calibration, utility, molecular groups, shortcuts
CB = pd.read_csv(R / "phase1/phase1_calibration_bins.csv"); DCV = pd.read_csv(R / "phase1/phase1_decision_curve.csv")
SG = pd.read_csv(R / "phase1/phase1_subgroups.csv"); CF = pd.read_csv(R / "phase1/phase1_confounding.csv")
f, axs = plt.subplots(2, 2, figsize=(W2, 145 * MM), gridspec_kw=dict(hspace=0.62, wspace=0.62, left=0.08, right=0.98, top=0.94, bottom=0.12))
ax = axs[0, 0]
for m in ["fusion_primary", "image_phikon2_abmil", "report_bert_M2"]:
    d = CB[(CB.model == m) & (CB.cohort == "all") & (CB.scheme == "institution")]
    ax.errorbar(d.mean_p, d.observed, yerr=[d.observed - d.lo, d.hi - d.observed], fmt="o-", ms=3, lw=1, color=MODEL_COL[m], capsize=0,
                label=f"{MODEL_NAME[m]} (slope {met(m, col='calibration_slope'):.2f})")
ax.plot([0, 1], [0, 1], ls=":", color=INK2, lw=0.7); ax.set_xlabel("predicted probability of glioblastoma"); ax.set_ylabel("observed fraction glioblastoma")
ax.legend(fontsize=6.3, loc="upper left"); ax.set_title("Calibration (institutions held out)", fontsize=8); panel(ax, "a", -0.2)
ax = axs[0, 1]
d0 = DCV[(DCV.model == "fusion_primary") & (DCV.cohort == "all") & (DCV.scheme == "institution")]
ax.plot(d0.threshold, d0.treat_all, color=SAND_D, ls="--", lw=1, label="treat all as glioblastoma"); ax.axhline(0, color=INK2, lw=0.7, label="treat none")
for m in ["fusion_primary", "image_phikon2_abmil", "report_bert_M2"]:
    d = DCV[(DCV.model == m) & (DCV.cohort == "all") & (DCV.scheme == "institution")]
    ax.plot(d.threshold, d.net_benefit, color=MODEL_COL[m], lw=1.3, label=MODEL_NAME[m])
ax.set_ylim(-0.05, 0.38); ax.set_xlabel("threshold probability"); ax.set_ylabel("net benefit"); ax.legend(fontsize=6.3)
ax.set_title("Decision curve", fontsize=8); panel(ax, "b", -0.2)
ax = axs[1, 0]
groups = [("LGG", "Astro_IDHmut", "LGG label\nIDH-mut.\nastrocytoma"), ("LGG", "Oligo_IDHmut_codel", "LGG label\noligodendro-\nglioma"),
          ("LGG", "GBM_IDHwt", "LGG label\nmolecular\nGBM (CNS5)"), ("GBM", "GBM_IDHwt", "GBM label\nIDH-\nwild-type"),
          ("GBM", "Astro_IDHmut", "GBM label\nIDH-\nmutant")]
x = np.arange(len(groups)); w = 0.26
for j, m in enumerate(["fusion_primary", "image_phikon2_abmil", "report_bert_M2"]):
    v, lo, hi, nn = [], [], [], []
    for lab, g, _ in groups:
        r = SG[(SG.model == m) & (SG.cohort == "all") & (SG.scheme == "institution") & (SG.tcga_label == lab) & (SG.group == g)].iloc[0]
        v.append(r.frac_called_gbm); lo.append(r.frac_called_gbm - r.lo); hi.append(r.hi - r.frac_called_gbm); nn.append(r.n)
    ax.bar(x + (j - 1) * w, v, w, color=MODEL_COL[m], yerr=[lo, hi], error_kw=dict(lw=0.6, ecolor=INK2), label=MODEL_NAME[m])
for i, (_, _, lab) in enumerate(groups):
    r = SG[(SG.model == "fusion_primary") & (SG.cohort == "all") & (SG.scheme == "institution") & (SG.tcga_label == groups[i][0]) & (SG.group == groups[i][1])].iloc[0]
    ax.text(i, 1.04, f"n={r.n}", ha="center", fontsize=6.3, color=INK2)
ax.set_xticks(x, [g[2] for g in groups], fontsize=6.3); ax.set_ylim(0, 1.12); ax.set_ylabel("fraction called glioblastoma")
ax.legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0, 0.93)); ax.set_title("Calls by WHO CNS5 molecular group", fontsize=8); panel(ax, "c", -0.2)
ax = axs[1, 1]
tests = [("phikon2", "site identification within GBM", "site within glioblastoma"), ("phikon2", "site identification within LGG", "site within LGG"),
         ("hipt", "site identification within GBM", "site within glioblastoma"), ("hipt", "site identification within LGG", "site within LGG")]
lab, val, ch, col = [], [], [], []
for enc, t, nm in tests:
    r = CF[(CF.encoder == enc) & (CF.test == t)].iloc[0]
    lab.append(f"{'Phikon-v2' if enc == 'phikon2' else 'HIPT'} · {nm.replace('site within glioblastoma', 'site, GBM').replace('site within LGG', 'site, LGG')}"); val.append(r.balanced_accuracy); ch.append(r.chance); col.append(BLUE if enc == "phikon2" else SAGE)
for enc, nm in [("phikon2", "Phikon-v2"), ("hipt", "HIPT")]:
    r = CF[(CF.encoder == enc) & (CF.test.str.startswith("20x vs 40x"))].iloc[0]
    lab.append(f"{nm} · 20× vs 40× scan"); val.append(r.auroc); ch.append(0.5); col.append(BLUE if enc == "phikon2" else SAGE)
y = np.arange(len(lab))[::-1]
ax.barh(y, val, color=col, height=0.6); ax.scatter(ch, y, marker="|", s=120, color=INK, zorder=3)
for yy, v in zip(y, val):
    ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=6.5)
ax.set_yticks(y, lab, fontsize=6.6); ax.set_xlim(0, 1.12); ax.set_xlabel("balanced accuracy (site) or AUROC (scan)")
ax.set_title("What slide features also encode (| = chance)", fontsize=8); panel(ax, "d", -0.62)
save(f, "Fig4")

# ================================================================================ Fig 4: attention (existing tile figure, re-saved to spec)
src = DC / "figures/manuscript/Fig4_attention.png"
im = Image.open(src).convert("RGB"); wpx = int(190 / 25.4 * 300)
im = im.resize((wpx, int(im.size[1] * wpx / im.size[0])), Image.LANCZOS); im.save(OUT / "Fig5.tif", compression="tiff_lzw", dpi=(300, 300))
im.save(OUT / "Fig5.pdf", resolution=300); print("Fig5", im.size)

# ================================================================================ Fig 5: cell-line mechanics (Phase 2)
EV = pd.read_csv(DATA / "rtdc_events_gated.csv.gz"); EV = EV[EV.emodulus > 0]; EV["logE"] = np.log10(EV.emodulus)
A3 = pd.read_csv(R / "phase2/A3_shift_functions.csv"); A5 = pd.read_csv(R / "phase2/A5_reproducibility_icc.csv")
A6 = pd.read_csv(R / "phase2/A6_precision_vs_cells.csv"); A7 = pd.read_csv(R / "phase2/A7_bag_discrimination.csv")
SC = {"Serum": SAGE, "EGF": OCHRE, "FGFJI": PLUM}
f = plt.figure(figsize=(W2, 150 * MM))
g1 = f.add_gridspec(1, 3, wspace=0.75, left=0.07, right=0.98, top=0.93, bottom=0.6)
g2 = f.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.85, left=0.14, right=0.93, top=0.42, bottom=0.08)
ax = f.add_subplot(g1[0, 0]); xs = np.linspace(-0.8, 0.6, 300)
for c in ["Serum", "EGF", "FGFJI"]:
    for meas, g in EV[EV.cond == c].groupby("measurement"):
        if len(g) > 50:
            ax.plot(xs, gaussian_kde(g.logE)(xs), color=SC[c], lw=0.6, alpha=0.55)
    ax.plot([], [], color=SC[c], lw=1.5, label={"Serum": "serum", "EGF": "EGF", "FGFJI": "FGF + JAKi"}[c])
ax.set_xlabel("log$_{10}$ apparent Young's modulus (kPa)"); ax.set_ylabel("density"); ax.legend(fontsize=6.3, loc="upper left", title="culture state", title_fontsize=6.3, handlelength=1.2)
ax.set_title("Stiffness distributions", fontsize=8); panel(ax, "a", -0.3, 1.06)
ax = f.add_subplot(g1[0, 1])
for con, col in [("EGF vs Serum", OCHRE), ("FGFJI vs Serum", PLUM)]:
    d = A3[A3.contrast == con]
    ax.fill_between(d.decile, d.lo, d.hi, color=col, alpha=0.18, lw=0); ax.plot(d.decile, d.shift_log10E, "o-", color=col, ms=3, lw=1.2, label=con.replace("FGFJI", "FGF + JAKi").replace("Serum", "serum"))
ax.axhline(0, color=INK2, lw=0.7); ax.set_xlabel("decile of the cell distribution"); ax.set_ylabel("shift in log$_{10}$ E")
ax.legend(fontsize=6.3, loc="lower left"); ax.set_title("Shift functions", fontsize=8); panel(ax, "b", -0.36, 1.06)
ax = f.add_subplot(g1[0, 2])
a5 = A5.sort_values("icc1"); y = np.arange(len(a5))
ax.hlines(y, a5.lo, a5.hi, color=SAND_D, lw=1.6)
ax.scatter(a5.icc1, y, c=[TERRA if d in ("mean", "median", "q10", "q90") else BLUE for d in a5.descriptor], s=18, zorder=3)
ax.axvline(0, color=INK2, lw=0.7); ax.set_yticks(y, a5.label, fontsize=6.5); ax.set_xlabel("intraclass correlation across days")
ax.set_title("Reproducibility across days", fontsize=8); panel(ax, "c", -0.85, 1.06)
ax = f.add_subplot(g2[0, 0])
a6 = A6[A6.assay == "RT-DC"]; order = ["median", "mean", "q10", "q90", "iqr", "mad", "upper_tail", "lower_tail", "tail_asym", "bowley", "moors"]
lab = dict(zip(A5.descriptor, A5.label)); cells = sorted(a6.n_cells.unique())
Hm = np.array([[a6[(a6.descriptor == d) & (a6.n_cells == n)].reliability.iloc[0] for n in cells] for d in order])
im = ax.imshow(np.clip(Hm, 0, 1), cmap=SEQB, vmin=0, vmax=1, aspect="auto")
for i in range(Hm.shape[0]):
    for j in range(Hm.shape[1]):
        ax.text(j, i, f"{Hm[i, j]:.2f}" if Hm[i, j] > 0 else "<0", ha="center", va="center", fontsize=5.8, color="white" if Hm[i, j] > 0.6 else INK)
ax.set_xticks(range(len(cells)), cells); ax.set_yticks(range(len(order)), [lab[d] for d in order], fontsize=6.4)
ax.set_xlabel("cells measured per sample"); ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
cb = f.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("reliability", fontsize=7); cb.outline.set_visible(False)
ax.set_title("Reliability by number of cells", fontsize=8); panel(ax, "d", -0.4, 1.06)
ax = f.add_subplot(g2[0, 1])
a7 = A7[A7.cells_per_bag == 100].copy()
tasks = ["EGF vs Serum", "FGFJI vs Serum", "FGFJI vs EGF", "X04 vs X01", "X08 vs X01", "X08 vs X04"]
Hb = np.array([[a7[a7.task == t][c].iloc[0] for c in ["auroc_location", "auroc_distribution", "auroc_shape_only"]] for t in tasks])
im = ax.imshow(Hb, cmap=SEQ, vmin=0.3, vmax=1.0, aspect="auto")
for i in range(Hb.shape[0]):
    for j in range(Hb.shape[1]):
        ax.text(j, i, f"{Hb[i, j]:.2f}", ha="center", va="center", fontsize=6.6, color="white" if Hb[i, j] > 0.86 else INK)
ax.set_xticks(range(3), ["location", "location\n+ shape", "shape\nonly"], fontsize=6.8)
ax.set_yticks(range(len(tasks)), [t.replace("FGFJI", "FGF+JAKi").replace("Serum", "serum") + (" (state)" if i < 3 else " (line)") for i, t in enumerate(tasks)], fontsize=6.6)
ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
cb = f.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("AUROC", fontsize=7); cb.outline.set_visible(False)
ax.set_title("Classifying bags of 100 cells", fontsize=8); panel(ax, "e", -0.75, 1.06)
save(f, "Fig6")

# ================================================================================ Fig 6: human biopsies (Phase 2b)
P = pd.read_csv(P2B / "figdata_pairs.csv"); Q = pd.read_csv(P2B / "figdata_quantiles.csv"); T = pd.read_csv(P2B / "distance_pair_types.csv")
C = pd.read_csv(P2B / "classification_cv.csv"); B = pd.read_csv(P2B / "blind_validation.csv"); PC = pd.read_csv(P2B / "positive_controls.csv")
STR = {"colon-fresh": TERRA, "colon-frozen": BLUE, "lung-fresh": OCHRE}
f = plt.figure(figsize=(W2, 150 * MM))
gs = f.add_gridspec(2, 3, hspace=0.68, wspace=0.75, left=0.08, right=0.98, top=0.93, bottom=0.1)
ax = f.add_subplot(gs[0, 0])
for s, g in P.groupby("stratum"):
    for _, r in g.iterrows():
        ax.plot([0, 1], [r.D_median_h, r.D_median_t], color=STR[s], lw=0.8, alpha=0.75, ls="--" if r.role != "main" else "-")
    ax.plot([], [], color=STR[s], label=f"{s.replace('-', ', ')} (n = {len(g)})")
ax.set_xticks([0, 1], ["healthy", "tumour"]); ax.set_xlim(-0.25, 1.25); ax.set_ylabel("median log$_{10}$ deformation")
ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=1); ax.set_title(f"Tumour lower in {np.mean(P.D_median_t < P.D_median_h):.0%} of pairs", fontsize=8); panel(ax, "a", -0.38, 1.08)
ax = f.add_subplot(gs[0, 1]); qc = [c for c in Q.columns if c.startswith("q")]; qx = np.array([int(c[1:]) for c in qc]) / 100
for s in STR:
    t = Q[(Q.feature == "logD") & (Q.stratum == s) & (Q.condition == "tumour")].set_index("unit")[qc]
    h = Q[(Q.feature == "logD") & (Q.stratum == s) & (Q.condition == "healthy")].set_index("unit")[qc]
    u = t.index.intersection(h.index); dd = t.loc[u].values - h.loc[u].values; m = dd.mean(0); se = dd.std(0, ddof=1) / np.sqrt(len(u))
    ax.plot(qx, m, color=STR[s], lw=1.3); ax.fill_between(qx, m - 1.96 * se, m + 1.96 * se, color=STR[s], alpha=0.15, lw=0)
ax.axhline(0, color=INK2, lw=0.7); ax.set_xlabel("quantile of the cell distribution"); ax.set_ylabel("tumour − healthy, log$_{10}$ deformation")
ax.set_title("Shift function", fontsize=8); panel(ax, "b", -0.42, 1.08)
ax = f.add_subplot(gs[0, 2])
lab = {"same unit: tumour vs healthy": "same patient,\ntumour vs healthy",
       "same stratum, different units, same condition: tumour-tumour / healthy-healthy": "two patients,\nsame condition",
       "same stratum, different units, different condition": "two patients,\ndifferent condition"}
t = T[(T.feature == "log10 deformation") & T["pair type"].isin(lab)]; yy = np.arange(len(t))
ax.barh(yy + 0.18, t.median_shift, 0.34, color=TERRA, label="median shift"); ax.barh(yy - 0.18, t.shape_w1, 0.34, color=BLUE, label="shape")
nf = T[(T.feature == "log10 deformation") & T["pair type"].str.startswith("split-half")].shape_w1.iloc[0]
ax.axvline(nf, color=INK2, ls="--", lw=0.8); ax.set_ylim(-0.6, 3.0); ax.text(nf + 0.003, 2.62, "noise floor", fontsize=6.2, color=INK2, va="center")
ax.set_yticks(yy, [lab[s] for s in t["pair type"]], fontsize=6.6); ax.set_xlabel("Wasserstein distance")
ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2); ax.set_title("Distribution distances", fontsize=8); panel(ax, "c", -0.8, 1.08)
ax = f.add_subplot(gs[1, 0:2])
c = C.iloc[::-1].reset_index(drop=True); yy = np.arange(len(c)); b = B.set_index("features").reindex(c.features)
ax.errorbar(c.auroc_pooled, yy, xerr=c.auroc_sd_over_repeats, fmt="o", color=TERRA, ms=4, label="cross-validated by patient (39 pairs)")
ax.plot(b.auroc.values, yy, "D", color=BLUE, ms=3.5, mfc="white", label="held-out blind set (8 pairs)")
ax.axvline(0.5, color=INK2, lw=0.7, ls=":"); ax.set_yticks(yy, c.features, fontsize=6.6); ax.set_xlim(0.45, 0.9); ax.set_xlabel("AUROC, tumour vs healthy")
ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); ax.set_title("Tumour vs healthy: shape adds nothing to the median", fontsize=8); panel(ax, "d", -0.42, 1.06)
ax = f.add_subplot(gs[1, 2])
pc = PC.copy(); pc["ctl"] = np.where(pc.control.str.contains("preservation"), "frozen vs fresh", "colon vs lung")
fe = ["median deformation", "median deformation + size", "deciles (deformation, size)"]
for j, (ff, col) in enumerate(zip(fe, [TERRA, SAGE, BLUE])):
    v = [pc[(pc.ctl == cc) & (pc.features == ff)].auroc.iloc[0] for cc in ["colon vs lung", "frozen vs fresh"]]
    ax.bar(np.arange(2) + (j - 1) * 0.26, v, 0.26, color=col, label=ff)
ax.axhline(0.5, color=INK2, lw=0.7, ls=":"); ax.set_xticks([0, 1], ["organ\ncolon vs\nlung", "handling\nfrozen vs\nfresh"], fontsize=6.6)
ax.set_ylim(0, 1.05); ax.set_ylabel("AUROC, healthy tissue only"); ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.38))
ax.set_title("Shape encodes tissue origin", fontsize=8); panel(ax, "e", -0.4, 1.06)
save(f, "Fig7")

# ================================================================================ Fig 7: design analysis heat maps
PM = pd.read_csv(R / "phase2/plasmode_main.csv")
f, axs = plt.subplots(1, 3, figsize=(W2, 72 * MM), gridspec_kw=dict(wspace=0.35, left=0.07, right=0.93, top=0.82, bottom=0.18))
spec = [("S1 location (IDH-anchored)", "power_M1_vs_M0", "Average stiffness beyond\nimage + report (location effect)"),
        ("S2 spread (LGG IQR x1.25)", "power_M2_vs_M1", "Spread beyond average\n(cell-line-like variation)"),
        ([s for s in PM.scenario.unique() if s.startswith("S2b")][0], "power_M2_vs_M1", "Spread beyond average\n(primary-culture-like variation)")]
for k, (sc, col, title) in enumerate(spec):
    ax = axs[k]; d = PM[PM.scenario == sc]
    n = sorted(d.n_per_class.unique()); c = sorted(d.cells_per_tumour.unique())
    Hp = np.array([[d[(d.n_per_class == a) & (d.cells_per_tumour == b)][col].iloc[0] for b in c] for a in n])
    im = ax.imshow(Hp, cmap=SEQ, vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(len(n)):
        for j in range(len(c)):
            ax.text(j, i, f"{Hp[i, j]:.2f}", ha="center", va="center", fontsize=6.8, color="white" if Hp[i, j] > 0.62 else INK,
                    fontweight="bold" if Hp[i, j] >= 0.8 else "normal")
    ax.set_xticks(range(len(c)), c); ax.set_yticks(range(len(n)), n); ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_xlabel("cells per tumour"); ax.set_ylabel("patients per class" if k == 0 else ""); ax.set_title(title, fontsize=7.8)
    panel(ax, "abc"[k], -0.25, 1.12)
cax = f.add_axes([0.945, 0.18, 0.012, 0.64]); cb = f.colorbar(im, cax=cax); cb.set_label("power (α = 0.05)", fontsize=7); cb.outline.set_visible(False)
save(f, "Fig8")

# ================================================================================ Graphical abstract (13 x 5 cm readable; saved larger)
f = plt.figure(figsize=(160 * MM, 62 * MM)); ax = f.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.02, 0.56, 0.2, 0.38, LBLUE, "\n\n\nH&E slide", fs=8); icon_slide(ax, 0.07, 0.76, 0.1)
box(ax, 0.02, 0.08, 0.2, 0.38, LPINK, "\n\n\nmasked report", fs=8); icon_doc(ax, 0.1, 0.26, 0.11)
box(ax, 0.29, 0.3, 0.2, 0.4, LSAND, "fusion model\nglioblastoma vs\nlower-grade glioma", fs=8, bold=True)
arrow(ax, 0.22, 0.75, 0.29, 0.6); arrow(ax, 0.22, 0.27, 0.29, 0.4)
box(ax, 0.56, 0.3, 0.19, 0.4, "white", f"AUROC {met('fusion_primary'):.2f}\nat unseen hospitals\n{met('fusion_primary', 'x40'):.2f} on 40× slides", fs=8, ec=NAVY)
arrow(ax, 0.49, 0.5, 0.56, 0.5)
box(ax, 0.79, 0.56, 0.19, 0.38, LBLUE, "\n\n\ncell lines and\nhuman biopsies", fs=7.5); icon_cells(ax, 0.82, 0.74, 0.13)
box(ax, 0.79, 0.08, 0.19, 0.38, "white", "single-cell stiffness:\nsignal in the median,\nnot in the spread", fs=7.5, ec=NAVY)
arrow(ax, 0.885, 0.56, 0.885, 0.46)
save(f, "Graphical_abstract")
