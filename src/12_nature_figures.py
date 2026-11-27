"""Step 12 - Nature-style figures.

One Nature-style version of every report figure, drawn from the same data with the same panels, labels and
numbers as src/10_figures.py, src/11_report_figures.py and the QC contact sheet, plus supplementary plots
(figS1-figS9) of results that the report otherwise gives only as tables.

Style, following Nature's figure guidelines: final widths of 89 mm (single column), 120-136 mm or 183 mm
(double column); sans-serif type metric-identical to Arial (Liberation Sans); 5.5-7 pt text and 8 pt bold
lowercase panel letters; no gridlines or background shading; colour-blind-safe Wong palette; vector PDF with
embedded TrueType fonts (editable text) plus 450-dpi and 300-dpi PNG; scale bars on micrographs. Figure
titles and footnotes go in the legends (src/figure_legends.py -> figures/nature/figure_legends.md).

Usage:  python src/12_nature_figures.py              # every figure
        python src/12_nature_figures.py fig3 figS4   # selected figures (prefix match)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import FIG, PROC, RAW, RES, SEED  # noqa: E402
from figure_legends import LEGENDS, MODEL_NAMES  # noqa: E402

OUT = FIG / "nature"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4

# Wong (2011) colour-blind-safe palette
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMIL, PURPLE, GREY = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999")
DGREY, LGREY = "#555555", "#BBBBBB"
C_GBM, C_LGG = VERMIL, BLUE                                   # classes
C_REP, C_IMG, C_TIS, C_FUS, C_NUIS, C_AGE = PURPLE, GREEN, SKY, BLUE, GREY, ORANGE   # model families
C_PAT, C_INST = PURPLE, ORANGE                                # validation schemes
REF = dict(color=LGREY, lw=0.5, ls=(0, (2.5, 2)), zorder=0)   # reference lines (chance, 80% power)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],   # Arial where installed
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7, "axes.titleweight": "normal",
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "legend.title_fontsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5, "xtick.direction": "out", "ytick.direction": "out", "lines.linewidth": 1.0,
    "lines.markersize": 3, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.35,
    "legend.handletextpad": 0.5, "legend.columnspacing": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.labelpad": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.titlepad": 3,
    "mathtext.default": "regular", "axes.unicode_minus": True,
    "figure.constrained_layout.w_pad": 1.5 / 72, "figure.constrained_layout.h_pad": 1.5 / 72,
    "figure.constrained_layout.wspace": 0.04, "figure.constrained_layout.hspace": 0.05})


def nm(x):
    return MODEL_NAMES.get(x, x)


def figure(w_mm, h_mm, layout="constrained"):
    return plt.figure(figsize=(w_mm * MM, h_mm * MM), layout=layout)


def letters(fig, pairs):
    """Bold lowercase panel letters at the top-left corner of each panel (axes plus its labels)."""
    pairs = list(pairs)
    for ax, _ in pairs:                       # keep an 8-pt line above untitled panels free for the letter
        if not ax.get_title():
            ax.set_title(" ", loc="left", fontsize=8)
    fig.canvas.draw()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, l in pairs:
        bb = ax.get_tightbbox(r)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0, y1, l, fontsize=8, fontweight="bold", ha="left", va="top")
    fig.set_layout_engine("none")             # freeze the layout so the letters stay aligned


def save(fig, name, photo=False):
    fig.savefig(OUT / f"{name}.pdf", dpi=600)
    fig.savefig(OUT / f"{name}.png", dpi=450)
    fig.savefig(OUT / f"{name}_300dpi.png", dpi=300)
    plt.close(fig)
    if photo:                                 # micrographs: high-quality JPEGs keep the Word file and archive small
        Image.open(OUT / f"{name}_300dpi.png").convert("RGB").save(OUT / f"{name}_word.jpg", quality=92)
        Image.open(OUT / f"{name}.png").convert("RGB").save(OUT / f"{name}.jpg", quality=95, dpi=(450, 450))
    print("saved", name)


def log_ticks(ax, ticks, axis="x"):
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks, [f"{t:g}" for t in ticks])
    a.set_minor_formatter(NullFormatter())


def outline(lw=1.2, colour="black"):
    return [pe.withStroke(linewidth=lw, foreground=colour, alpha=0.55)]


def scale_bar(ax, img, um, um_per_px, label_size=6, inset=20):
    """Scale bar in the lower-right corner; black on pale backgrounds, white otherwise."""
    n = img.size[0]
    L = um / um_per_px
    x0, x1, yb = n - inset - L, n - inset, n - inset - 2
    patch = np.asarray(img.convert("L"))[int(yb - 40):int(yb + 4), int(x0):int(x1)]
    fg, bg = ("black", "white") if patch.mean() > 175 else ("white", "black")
    ax.plot([x0, x1], [yb, yb], color=fg, lw=1.5, solid_capstyle="butt", path_effects=outline(2.4, bg))
    ax.text((x0 + x1) / 2, yb - 12, f"{um} µm", color=fg, fontsize=label_size, ha="center", va="bottom",
            path_effects=outline(1.2, bg))


# ---------------------------------------------------------------- data (read once)
co = pd.read_csv(PROC / "cohort.csv").set_index("patient")
sl = pd.read_csv(PROC / "slides.csv").set_index("patient")
slr = pd.read_csv(PROC / "slides.csv")
inv = pd.read_csv(RES / "inventory.csv")
rm = pd.read_csv(RES / "report_metrics.csv")
im = pd.read_csv(RES / "image_metrics.csv")
fm = pd.read_csv(RES / "fusion_metrics.csv")
fd = pd.read_csv(RES / "fusion_deltas.csv")
fo = pd.read_csv(RES / "fusion_oof.csv")
ro = pd.read_csv(RES / "report_oof.csv")
io = pd.read_csv(RES / "image_oof.csv")
pm = pd.read_csv(RES / "mechanics_per_measurement.csv")
bag = pd.read_csv(RES / "mechanics_bag_classification.csv")
pw = pd.read_csv(RES / "power_matched_cohort.csv")
scan = pd.read_csv(RES / "confounding_scanner_recognition.csv")
strat = pd.read_csv(RES / "confounding_stratified_auroc.csv")
uf = pd.read_csv(RES / "confounding_university_of_florida.csv")
lrt = pd.read_csv(RES / "confounding_lrt_x40.csv")
vc = pd.read_csv(RES / "mechanics_variance_components.csv")
fsg = pd.read_csv(RES / "fusion_subgroups.csv")
ev = pd.read_csv(PROC / "rtdc_events_gated.csv.gz")


# =============================================================== original report figures, restyled
def fig0_pipeline():
    fig = figure(183, 105, layout=None)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 110); ax.set_ylim(0, 63); ax.axis("off")
    C_LAB, C_MECH, C_VAL, C_3 = ORANGE, SKY, BLACK, VERMIL

    def box(x, y, w, h, title, lines, edge, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", linewidth=0.8,
                                    edgecolor=edge, facecolor="white", linestyle=(0, (3, 2)) if dashed else "-"))
        ax.text(x + 1.2, y + h - 1.6, title, fontsize=7, fontweight="bold", va="top")
        for i, l in enumerate(lines):
            ax.text(x + 1.2, y + h - 4.6 - i * 2.55, l, fontsize=6, va="top")

    def arrow(x0, y0, x1, y1, color=DGREY, dashed=False):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
                                     color=color, linestyle=(0, (3, 2)) if dashed else "-"))

    for x, t in [(1, "Data sources"), (29, "Processing and features"), (63, "Branch models"), (88, "Fusion and evaluation")]:
        ax.text(x, 61, t, fontsize=7, fontweight="bold")
    box(1, 47, 24, 12, "Labels", ["TCGA pan-glioma (cBioPortal)", "1,122 patients, GBM vs LGG", "IDH, 1p/19q, TERT, +7/−10"], C_LAB)
    box(1, 32, 24, 12, "Pathology reports", ["TCGA-Reports (OCR text)", "868 patients"], C_REP)
    box(1, 17, 24, 12, "Diagnostic H&E slides", ["GDC tile service via browser", "880 slides: 16 tiles + thumb", "scanner info from SVS header"], C_IMG)
    box(1, 2, 24, 12, "Whole-cell mechanics", ["Mechanomics RT-DC Set 01", "3 GBM lines × 3 conditions", "not linked to any patient"], C_MECH, dashed=True)
    box(29, 47, 30, 12, "Cohort linkage + shared splits", ["TCGA barcode linkage, institutions", "5×5 stratified patient CV", "5×5 institution-grouped CV"], C_LAB)
    box(29, 32, 30, 12, "Masking + TF-IDF", ["M0 raw; M1 masks diagnosis, grade", "and molecular results; M2 also", "masks grading criteria"], C_REP)
    box(29, 17, 30, 12, "Tiles → QC → features", ["13,966 tiles, 95% pass QC", "HIPT ViT-S/16: 55,864 crops × 384", "nuclei, texture, stain, sharpness"], C_IMG)
    box(29, 2, 30, 12, "Gating → per-sample features", ["21,738 gated cells, 21 measurements", "median, spread, tails of log E", "deformation and cell size"], C_MECH, dashed=True)
    box(63, 32, 21, 12, "Report models", ["TF-IDF + L2 logistic", "regression; age baseline"], C_REP)
    box(63, 17, 21, 12, "Image models", ["mean-pool + L2 LR", "gated attention MIL", "metadata / stain baselines"], C_IMG)
    box(63, 2, 21, 12, "Mechanics models", ["mixed models (date)", "bag test, held-out date"], C_MECH, dashed=True)
    box(88, 47, 21, 12, "Fusion (686 matched)", ["late: mean of logits", "early: one LR on both"], C_FUS)
    box(88, 24, 21, 19, "Validation", ["random patient folds", "institutions held out", "40×-scanned slides only", "institution × scan-year", "UF 15 v 15; LRT", "DeLong + bootstrap"], C_VAL)
    box(88, 2, 21, 18, "Three-input test", ["needs slide, report AND", "cells from the same", "tumour (0 today)", "~150 per class for", "+0.03 AUROC (power)"], C_3, dashed=True)
    for y in (53, 38, 23):
        arrow(25.6, y, 28.6, y)
    arrow(25.6, 8, 28.6, 8, C_MECH, True)
    arrow(59.6, 38, 62.6, 38); arrow(59.6, 23, 62.6, 23); arrow(59.6, 8, 62.6, 8, C_MECH, True)
    arrow(44, 46.6, 44, 44.6, C_LAB)
    arrow(84.6, 40, 87.6, 50, C_REP); arrow(84.6, 26, 87.6, 48.5, C_IMG)
    arrow(98.5, 46.6, 98.5, 43.6, C_FUS)
    arrow(84.6, 8, 87.6, 8, C_MECH, True)
    save(fig, "fig0_pipeline")


def fig1_inventory():
    fig = figure(120, 50)
    ax = fig.add_subplot()
    yy = np.arange(len(inv))[::-1]
    ax.barh(yy, inv.GBM, height=0.5, color=C_GBM, label="GBM", lw=0)
    ax.barh(yy, inv.LGG, left=inv.GBM, height=0.5, color=C_LGG, label="Lower-grade glioma", lw=0)
    for y, (_, r) in zip(yy, inv.iterrows()):
        ax.text(r.GBM + r.LGG + 14, y, f"{r.patients:,}" if r.patients else "0  (no public source)", va="center")
    ax.set_yticks(yy, inv.modality)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Patients")
    ax.set_xlim(0, 1350)
    ax.legend(loc="lower right", ncol=2)
    save(fig, "fig1_inventory")


def fig2_acquisition_confounding():
    fig = figure(183, 52)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 1, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ct = pd.crosstab(slr.scan_year.fillna(0).astype(int), slr.project)
    ct = ct[ct.index > 0]
    x = np.arange(len(ct))
    w = 0.36
    axes[0].bar(x - w / 2 - 0.01, ct["TCGA-GBM"], w, color=C_GBM, label="GBM", lw=0)
    axes[0].bar(x + w / 2 + 0.01, ct["TCGA-LGG"], w, color=C_LGG, label="LGG", lw=0)
    axes[0].set_xticks(x, ct.index)
    axes[0].set_title("Scan year")
    axes[0].legend(loc="upper left")
    for ax, col, title in [(axes[1], "appmag", "Scan magnification"), (axes[2], "jpeg_q", "JPEG quality")]:
        c = pd.crosstab(slr[col], slr.project)
        xx = np.arange(len(c))
        ax.bar(xx - w / 2 - 0.01, c["TCGA-GBM"], w, color=C_GBM, lw=0)
        ax.bar(xx + w / 2 + 0.01, c["TCGA-LGG"], w, color=C_LGG, lw=0)
        ax.set_xticks(xx, [f"{int(v)}×" if col == "appmag" else f"Q={int(v)}" for v in c.index])
        ax.set_title(title)
    for ax in axes:
        ax.set_ylabel("Slides")
    letters(fig, zip(axes, "abc"))
    save(fig, "fig2_acquisition_confounding")


def fig3_auroc_overview():
    fm_ = fm[fm.n > 0]
    lab = {"tfidf_M0": "Report, unmasked (M0)", "tfidf_M1": "Report, masked M1", "tfidf_M2": "Report, masked M2",
           "age": "Age only", "img_meanpool": "Image, HIPT mean-pool", "img_abmil": "Image, attention MIL",
           "handcrafted": "Image, hand-crafted", "acq_meta": "Acquisition metadata only",
           "stain_color": "Stain colour & sharpness only"}
    rows = []
    for _, r in pd.concat([rm[rm.model.isin(lab)], im[im.model.isin(lab)]]).iterrows():
        rows.append({"label": lab[r.model], "cohort": r.cohort, "scheme": r.scheme, "auroc": r.auroc, "lo": r.auroc_lo,
                     "hi": r.auroc_hi, "src": "report" if r.model.startswith(("tfidf", "age")) else
                     ("nuis" if r.model in ("acq_meta", "stain_color") else "image")})
    for _, r in fm_[fm_.model.isin(["late_image+report_M2", "early_image+report_M2"])].iterrows():
        rows.append({"label": {"late_image+report_M2": "Fusion: image + report M2 (late)",
                               "early_image+report_M2": "Fusion: image + report M2 (early)"}[r.model],
                     "cohort": r.cohort, "scheme": r.scheme, "auroc": r.auroc, "lo": r.auroc_lo, "hi": r.auroc_hi,
                     "src": "fusion"})
    T = pd.DataFrame(rows)
    order = ["Report, unmasked (M0)", "Report, masked M1", "Report, masked M2", "Age only", "Image, HIPT mean-pool",
             "Image, attention MIL", "Image, hand-crafted", "Fusion: image + report M2 (late)",
             "Fusion: image + report M2 (early)", "Acquisition metadata only", "Stain colour & sharpness only"]
    colors = {"report": C_REP, "image": C_IMG, "fusion": C_FUS, "nuis": C_NUIS}
    fig = figure(183, 92)
    gs = fig.add_gridspec(1, 2)
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[0, 1], sharey=a0)
    for ax, cohort, title in [(a0, "all", "All slides"), (a1, "x40", "40×-scanned slides only")]:
        for i, name in enumerate(order[::-1]):
            for scheme, dy, mk in [("patient", 0.17, "o"), ("institution", -0.17, "D")]:
                r = T[(T.label == name) & (T.cohort == cohort) & (T.scheme == scheme)]
                if r.empty:
                    continue
                r = r.iloc[0]
                c = colors[r.src]
                ax.plot([r.lo, r.hi], [i + dy] * 2, color=c, lw=1.0, solid_capstyle="butt")
                ax.plot(r.auroc, i + dy, mk, ms=3.6 if mk == "o" else 3.1, color=c,
                        mfc=c if scheme == "patient" else "white", mec=c, mew=0.8)
        ax.axvline(0.5, **REF)
        ax.set_xlim(0.25, 1.005)
        ax.set_xticks(np.arange(0.3, 1.01, 0.1), [f"{v:.1f}" for v in np.arange(0.3, 1.01, 0.1)])
        ax.set_title(title)
        ax.set_xlabel("AUROC (95% bootstrap CI)")
    a0.set_yticks(range(len(order)), order[::-1])
    a0.tick_params(axis="y", length=0)
    a1.tick_params(axis="y", length=0, labelleft=False)
    h1 = Line2D([], [], marker="o", ls="", color=BLACK, ms=3.6, label="Random patient folds")
    h2 = Line2D([], [], marker="D", ls="", color=BLACK, mfc="white", mew=0.8, ms=3.1, label="Whole institutions held out")
    fig.legend(handles=[h1, h2], loc="outside lower center", ncol=2)
    letters(fig, [(a0, "a"), (a1, "b")])
    save(fig, "fig3_auroc_overview")


def fig4_report_masking():
    fig = figure(89, 74)
    ax = fig.add_subplot()
    lv = ["tfidf_M0", "tfidf_M1", "tfidf_M2"]
    for scheme, c, mk in [("patient", C_PAT, "o"), ("institution", C_INST, "D")]:
        r = rm[(rm.cohort == "all") & (rm.scheme == scheme)].set_index("model").loc[lv]
        ax.plot(range(3), r.auroc, color=c, lw=1.0, marker=mk, ms=3.4 if mk == "o" else 3.0,
                label={"patient": "Random patient folds", "institution": "Institutions held out"}[scheme])
        ax.fill_between(range(3), r.auroc_lo, r.auroc_hi, color=c, alpha=0.15, lw=0)
        ax.text(2.08, r.auroc.iloc[-1], f"{r.auroc.iloc[-1]:.3f}", va="center")
    age = rm[(rm.cohort == "all") & (rm.scheme == "patient") & (rm.model == "age")].auroc.iloc[0]
    ax.axhline(age, color=GREY, lw=0.8)
    ax.text(2.08, age - 0.009, f"age only {age:.3f}", va="center")
    ax.set_xticks(range(3), ["M0 unmasked", "M1 no diagnosis /\ngrade / molecular", "M2 also no grading\ncriteria"])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.75, 1.0)
    ax.set_xlim(-0.2, 2.55)
    fig.legend(loc="outside lower center", ncol=2)
    save(fig, "fig4_report_masking")


def fig5_mechanics():
    fig = figure(183, 60)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.1, 1.2])
    a0, a1, a2 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cc = {"Serum": BLUE, "EGF": ORANGE, "FGFJI": GREEN}
    for cond, c in cc.items():
        v = np.sort(np.log10(ev[ev.cond == cond].emodulus.values))
        a0.plot(10 ** v, np.linspace(0, 1, len(v)), color=c, lw=1.0, label=cond)
    log_ticks(a0, [0.4, 0.6, 1.0, 1.5, 2.5])
    a0.set_xticklabels(["0.4", "0.6", "1.0", "1.5", "2.5"])
    a0.set_xlim(0.35, 3)
    a0.set_xlabel("Young's modulus (kPa), RT-DC")
    a0.set_ylabel("Cumulative fraction of cells")
    a0.legend(loc="lower right")
    a0.set_title("Single cells, 3 GBM lines pooled")
    lines = ["X01", "X04", "X08"]
    for j, cond in enumerate(cc):
        d = pm[pm.cond == cond]
        xs = np.array([lines.index(l) for l in d.line]) + (j - 1) * 0.22
        a1.plot(xs, 10 ** d.logE_median, "o", ms=3.6, color=cc[cond], mec="white", mew=0.4, label=cond)
    a1.set_xticks(range(3), lines)
    a1.set_xlabel("Glioblastoma line")
    a1.set_ylabel("Median modulus per measurement (kPa)")
    a1.set_title("Per measurement (3 dates; colours as left)")
    tk = bag[(bag.task == "EGF_vs_FGFJI")]
    fl = {"median_E_only": ("Median E", GREY), "E_distribution": ("E distribution", PURPLE),
          "E_distribution+shape": ("E distribution + cell shape", BLACK)}
    for f, (lbl, c) in fl.items():
        r = tk[tk.features == f].sort_values("bag_cells")
        a2.plot(r.bag_cells, r.bag_auroc, marker="o", color=c, lw=1.0, ms=3.2, label=lbl)
    log_ticks(a2, [50, 100, 300])
    a2.set_xlabel("Cells per sample")
    a2.set_ylabel("AUROC, held-out date")
    a2.set_ylim(0.5, 1.0)
    a2.legend(loc="lower left")
    a2.set_title("EGF vs FGFJI from samples of cells")
    letters(fig, zip([a0, a1, a2], "abc"))
    save(fig, "fig5_mechanics")


def fig6_who2021_subgroups():
    d = fo[(fo.model == "late_image+report_M2") & (fo.cohort == "all") & (fo.scheme == "institution")].join(co, on="patient")
    groups = [("GBM", "GBM_IDHwt", "GBM, IDH-wt"), ("GBM", "Astro_IDHmut", "GBM label, IDH-mutant"),
              ("LGG", "GBM_IDHwt", "LGG label, molecular GBM*"), ("LGG", "IDHwt_other", "LGG, IDH-wt other"),
              ("LGG", "Astro_IDHmut", "LGG, astro IDH-mut"), ("LGG", "Oligo_IDHmut_codel", "LGG, oligo 1p/19q")]
    fig = figure(120, 60)
    ax = fig.add_subplot()
    rng = np.random.default_rng(0)
    for i, (labl, cls, name) in enumerate(groups):
        g = d[(d.y_gbm == (labl == "GBM")) & (d.who2021_like == cls)]
        c = C_GBM if labl == "GBM" else C_LGG
        ax.plot(g.p, i + rng.uniform(-0.18, 0.18, len(g)), "o", ms=2.0, color=c, alpha=0.55, mec="none")
        if len(g):
            ax.plot([g.p.median()] * 2, [i - 0.3, i + 0.3], color=BLACK, lw=1.2)
            ax.text(1.02, i, f"n={len(g)}", va="center")
    ax.set_yticks(range(len(groups)), [g[2] for g in groups])
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.axvline(0.5, **REF)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_xlabel("Fused P(GBM), institutions held out")
    save(fig, "fig6_who2021_subgroups")


def fig7_attention_tiles():
    att = pd.read_csv(RES / "abmil_attention_full.csv")
    att = att[att.tile.str.contains("x20")]
    mpp_eff = pd.read_csv(PROC / "tiles_index.csv").set_index("tile").mpp_eff
    pick = []
    for lab_v in [1, 0]:                                       # same selection as src/10_figures.py
        pats = co[(co.y_gbm == lab_v)].index.intersection(att.patient.unique())
        sub = fo[(fo.model == "image_meanpool") & (fo.cohort == "all") & (fo.scheme == "patient")].set_index("patient").p
        pats = sorted(pats, key=lambda p: -sub.get(p, 0) if lab_v == 1 else sub.get(p, 1))[:3]
        for p in pats:
            t = att[att.patient == p].sort_values("attention", ascending=False).tile.iloc[0]
            pick.append((lab_v, p, t))
    W, side, gap, top, title_h, bottom = 183.0, 1.0, 3.0, 1.0, 5.0, 1.0
    t_mm = (W - 2 * side - 2 * gap) / 3
    H = top + 2 * (title_h + t_mm) + 2.0 + bottom
    fig = figure(W, H, layout=None)
    for k, (lab_v, p, t) in enumerate(pick):
        r_, c_ = divmod(k, 3)
        x = side + c_ * (t_mm + gap)
        y = top + r_ * (title_h + t_mm + 2.0) + title_h
        ax = fig.add_axes([x / W, 1 - (y + t_mm) / H, t_mm / W, t_mm / H])
        img = Image.open(RAW / "tiles" / t)
        ax.imshow(img, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        scale_bar(ax, img, 50, mpp_eff[t])
        fig.text((x + t_mm / 2) / W, 1 - (y - 1.2) / H, f"{'GBM' if lab_v else 'LGG'}  {p}", fontsize=6.5,
                 ha="center", va="bottom")
        fig.text(x / W, 1 - (y - 1.2) / H, "abcdef"[k], fontsize=8, fontweight="bold", ha="left", va="bottom")
    save(fig, "fig7_attention_tiles", photo=True)


def fig8_power():
    fig = figure(183, 56)
    gs = fig.add_gridspec(1, 3)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2)]
    dc = {0.02: GREY, 0.03: PURPLE, 0.05: BLUE, 0.08: GREEN}
    for ax, a0 in zip(axes, [0.85, 0.90, 0.95]):
        for dl, c in dc.items():
            r = pw[(pw.auroc_image_report == a0) & (pw.delta_auroc == dl)]
            if len(r):
                ax.plot(r.n_per_class, r.power, marker="o", ms=3.0, lw=1.0, color=c, label=f"+{dl:.2f}")
        ax.axhline(0.8, **REF)
        log_ticks(ax, [25, 50, 100, 200, 400])
        ax.set_title(f"Image + report AUROC {a0:.2f}")
        ax.set_xlabel("Matched patients per class")
    axes[0].set_ylabel("Power (DeLong, α = 0.05)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    h, l = axes[0].get_legend_handles_labels()
    axes[2].legend(h, l, title="AUROC gain from mechanics", loc="lower right")
    letters(fig, zip(axes, "abc"))
    save(fig, "fig8_power")


def roc_series(ax, df, series, cohort, scheme, lw=1.0, fmt="{lab}  {a:.3f}"):
    n = 0
    for s in series:
        model, lab, c = s[:3]
        ls = s[3] if len(s) > 3 else "-"
        d = df[(df.model == model) & (df.cohort == cohort) & (df.scheme == scheme)].join(co.y_gbm, on="patient")
        fpr, tpr, _ = roc_curve(d.y_gbm, d.p)
        ax.plot(fpr, tpr, color=c, ls=ls, lw=lw, label=fmt.format(lab=lab, a=roc_auc_score(d.y_gbm, d.p)))
        n = len(d)
    ax.plot([0, 1], [0, 1], **REF)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_aspect("equal")
    return n


FUSION_SERIES = [("late_image+report_M2", "Image + report M2 (late fusion)", C_FUS),
                 ("image_meanpool", "Image (HIPT mean-pool)", C_IMG),
                 ("report_M2", "Report, masked M2", C_REP),
                 ("acq_meta (nuisance)", "Acquisition metadata only", C_NUIS)]


def fig9_roc_curves():
    fig = figure(183, 96)
    gs = fig.add_gridspec(1, 2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    for ax, cohort, title in [(axes[0], "all", "All matched patients (n = 686)"),
                              (axes[1], "x40", "40×-scanned slides only (n = 556)")]:
        roc_series(ax, fo, FUSION_SERIES, cohort, "institution")
        ax.set_xlabel("False-positive rate (LGG called GBM)")
        ax.set_title(title)
        ax.legend(loc="lower right", title="AUROC, institutions held out")
    axes[0].set_ylabel("True-positive rate (GBM called GBM)")
    letters(fig, zip(axes, "ab"))
    save(fig, "fig9_roc_curves")


def fig10_calibration():
    fig = figure(89, 82)
    ax = fig.add_subplot()
    for scheme, c, mk, lab in [("patient", C_PAT, "o", "Random patient folds"), ("institution", C_INST, "D", "Institutions held out")]:
        d = fo[(fo.model == "late_image+report_M2") & (fo.cohort == "all") & (fo.scheme == scheme)].join(co.y_gbm, on="patient")
        q = pd.qcut(d.p, 10, duplicates="drop")
        g = d.groupby(q, observed=True).agg(p=("p", "mean"), y=("y_gbm", "mean"), n=("p", "size"))
        ax.plot(g.p, g.y, marker=mk, ms=3.4 if mk == "o" else 3.0, lw=1.0, color=c, label=lab, clip_on=False, zorder=3)
    ax.plot([0, 1], [0, 1], **REF)
    ax.set_xlabel("Predicted P(GBM), decile mean")
    ax.set_ylabel("Observed fraction GBM")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    save(fig, "fig10_calibration")


def fig11_embedding_map():
    import umap
    Z = np.load(PROC / "embeddings_hipt.npz", allow_pickle=True)   # same inputs and settings as 11_report_figures
    emb = Z["emb"].astype(np.float32)
    idx = pd.DataFrame({c: Z[c] for c in ["patient", "mag"]})
    M = pd.DataFrame({p: emb[g.index].mean(0) for p, g in idx[idx.mag == "x20"].groupby("patient")}).T
    M = M.loc[M.index.intersection(co.index)]
    X = (M - M.mean()) / (M.std() + 1e-6)
    U = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=SEED).fit_transform(X.values)
    U = pd.DataFrame(U, index=M.index, columns=["u1", "u2"]).join(co.y_gbm).join(sl[["appmag", "scan_year"]])
    U.to_csv(RES / "umap_patient_hipt.csv")
    fig = figure(183, 86)
    gs = fig.add_gridspec(1, 2)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    kw = dict(s=4, alpha=0.75, edgecolors="white", linewidths=0.2)
    for lab, c, v in [("LGG", C_LGG, 0), ("GBM", C_GBM, 1)]:
        d = U[U.y_gbm == v]
        axes[0].scatter(d.u1, d.u2, color=c, label=f"{lab} (n={len(d)})", **kw)
    axes[0].set_title("Coloured by diagnosis")
    for lab, c, sel in [("40× scan, LGG", C_LGG, (U.appmag == 40) & (U.y_gbm == 0)),
                        ("40× scan, GBM", GREEN, (U.appmag == 40) & (U.y_gbm == 1)),
                        ("20× scan (all GBM)", C_GBM, U.appmag == 20)]:
        d = U[sel]
        axes[1].scatter(d.u1, d.u2, color=c, label=f"{lab} (n={len(d)})", **kw)
    axes[1].set_title("Coloured by scan magnification")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(loc="best", markerscale=1.5, handletextpad=0.2)
    letters(fig, zip(axes, "ab"))
    save(fig, "fig11_embedding_map")


def fig12_tissue_features():
    hp = pd.read_csv(PROC / "handcrafted_patient.csv").set_index("patient").join(co.y_gbm).join(sl.appmag)
    groups = [("GBM, 20× scan", (hp.y_gbm == 1) & (hp.appmag == 20), C_GBM),
              ("GBM, 40× scan", (hp.y_gbm == 1) & (hp.appmag == 40), GREEN),
              ("LGG, 40× scan", (hp.y_gbm == 0) & (hp.appmag == 40), C_LGG)]
    feats = [("x20_nuc_area_frac_mean", "Nuclear area fraction"), ("x20_nuc_density_per_mm2_mean", "Nuclei per mm²"),
             ("x20_nuc_area_iqr_um2_mean", "Nuclear size IQR (µm²)"), ("x20_blur_lapvar_mean", "Sharpness (Laplacian variance)")]
    fig = figure(183, 56)
    gs = fig.add_gridspec(1, 4)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    rng = np.random.default_rng(0)
    for ax, (f, title) in zip(axes, feats):
        for i, (lab, m, c) in enumerate(groups):
            v = hp.loc[m, f].dropna().values
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=1.6, color=c, alpha=0.45, edgecolors="none")
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i - 0.28, i + 0.28], [med, med], color=BLACK, lw=1.2)
            ax.plot([i, i], [q1, q3], color=BLACK, lw=0.8)
        ax.set_xticks(range(3), [g[0].replace(", ", "\n") for g in groups])
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylabel(title)
        if f == "x20_blur_lapvar_mean":
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
    letters(fig, zip(axes, "abcd"))
    save(fig, "fig12_tissue_features")


def fig13_institutions():
    it = co.groupby("institution").y_gbm.agg(n="size", gbm="sum")
    it["lgg"] = it.n - it.gbm
    it = it[it.n >= 10].sort_values("n")
    fig = figure(136, 118)
    ax = fig.add_subplot()
    yy = np.arange(len(it))
    ax.barh(yy, it.gbm, height=0.6, color=C_GBM, label="GBM", lw=0)
    ax.barh(yy, it.lgg, left=it.gbm, height=0.6, color=C_LGG, label="LGG", lw=0)
    for y, (name, r) in zip(yy, it.iterrows()):
        tag = "  both classes" if (r.gbm > 0 and r.lgg > 0) else ""
        ax.text(r.n + 4, y, f"{int(r.n)}{tag}", va="center")
    ax.set_yticks(yy, [n[:42] for n in it.index])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(it) - 0.4)
    ax.set_xlabel("Patients with pan-glioma labels")
    ax.legend(loc="lower right")
    save(fig, "fig13_institutions")


def contact_sheet(patient="TCGA-06-0156"):
    meta = json.load(open(RAW / "tiles" / patient / "meta.json"))
    thumb = Image.open(RAW / "tiles" / patient / "thumb.jpg")
    ds, mpp = meta["thumb_downsample"], meta["mpp"]
    tw, th = thumb.size
    W, side, gap_ab, g, top, bottom = 183.0, 1.0, 5.0, 1.0, 4.5, 1.0
    grid = (W - 2 * side - gap_ab) / (1 + tw / th)        # grid side = thumbnail height
    t = (grid - 3 * g) / 4
    H = top + grid + bottom
    fig = figure(W, H, layout=None)

    def axes_mm(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    a = axes_mm(side, top, grid * tw / th, grid)
    a.imshow(thumb, interpolation="none")
    a.set_axis_off()
    for tl in meta["tiles"]:
        s = tl["size0"] / ds
        hd = tl["stratum"] == "hdense"
        a.add_patch(Rectangle((tl["x0"] / ds, tl["y0"] / ds), s, s, fill=False, lw=0.7,
                              ec=VERMIL if hd else GREEN, ls="-" if hd else (0, (2, 1.2))))
    L = 2000 / (mpp * ds)
    a.plot([10, 10 + L], [th - 14, th - 14], color=BLACK, lw=1.4, solid_capstyle="butt")
    a.text(10 + L / 2, th - 19, "2 mm", ha="center", va="bottom", fontsize=6)
    fig.text(side / W, 1 - 0.4 / H, "a", fontsize=8, fontweight="bold", ha="left", va="top")
    x0 = side + grid * tw / th + gap_ab
    fig.text(x0 / W, 1 - 0.4 / H, "b", fontsize=8, fontweight="bold", ha="left", va="top")
    for k, tl in enumerate(meta["tiles"]):
        r_, c_ = divmod(k, 4)
        ax = axes_mm(x0 + c_ * (t + g), top + r_ * (t + g), t, t)
        img = Image.open(RAW / "tiles" / tl["name"])
        ax.imshow(img, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(True)
            s_.set_color(VERMIL if tl["stratum"] == "hdense" else GREEN)
            s_.set_linewidth(1.2)
        scale_bar(ax, img, 50 if tl["mag"] == "x20" else 200, mpp * tl["downsample"], label_size=5.5, inset=24)
    save(fig, f"contact_{patient}", photo=True)


# =============================================================== supplementary: results that were tables only
def roc_grid(name, df, series):
    fig = figure(136, 136)
    gs = fig.add_gridspec(2, 2)
    pairs = []
    for i, (cohort, ctitle) in enumerate([("all", "All slides"), ("x40", "40×-scanned slides only")]):
        for j, (scheme, stitle) in enumerate([("patient", "random patient folds"), ("institution", "institutions held out")]):
            ax = fig.add_subplot(gs[i, j])
            n = roc_series(ax, df, series, cohort, scheme, lw=0.9, fmt="{lab}  {a:.3f}")
            ax.set_title(f"{ctitle} (n = {n})\n{stitle}")
            ax.set_xlabel("False-positive rate")
            ax.set_ylabel("True-positive rate")
            ax.legend(loc="lower right", title="AUROC", fontsize=5.5, title_fontsize=5.5, handlelength=1.6)
            pairs.append((ax, "abcd"[2 * i + j]))
    letters(fig, pairs)
    save(fig, name)


def figS1_report_roc():
    roc_grid("figS1_report_roc", ro, [("tfidf_M0", "Report, unmasked M0", BLACK, ":"),
                                      ("tfidf_M1", "Report, masked M1", C_REP, "--"),
                                      ("tfidf_M2", "Report, masked M2", C_REP, "-"),
                                      ("tfidf_M1+age", "TF-IDF M1 + age", BLUE, "-"),
                                      ("age", "Age only", C_AGE, "-")])


def figS2_image_roc():
    roc_grid("figS2_image_roc", io, [("img_meanpool", "HIPT mean-pool + LR", C_IMG, "-"),
                                     ("img_abmil", "Gated attention MIL", C_IMG, "--"),
                                     ("handcrafted", "Tissue features + LR", C_TIS, "-"),
                                     ("stain_color", "Stain colour + sharpness", BLACK, ":"),
                                     ("acq_meta", "Acquisition metadata", C_NUIS, "-")])


def figS3_tissue_feature_aurocs():
    hpr = pd.read_csv(PROC / "handcrafted_patient.csv")
    d = hpr.merge(co.reset_index()[["patient", "y_gbm"]], on="patient").merge(slr[["patient", "appmag"]], on="patient")
    feats = [("x20_nuc_area_frac_mean", "Nuclear area fraction"), ("x20_nuc_density_per_mm2_mean", "Nuclei per mm²"),
             ("x20_nuc_area_iqr_um2_mean", "Nuclear area IQR (µm²)"), ("x20_nuc_area_med_um2_mean", "Median nuclear area (µm²)"),
             ("x20_glcm_contrast_mean", "GLCM contrast"), ("x20_lbp_entropy_mean", "LBP entropy"),
             ("x20_blur_lapvar_mean", "Sharpness (Laplacian variance)"), ("x20_E_mean_mean", "Eosin optical density")]
    rows = []
    for f, lab in feats:                       # same computation as the table in report/build_content.py
        v = d[[f, "y_gbm", "appmag"]].dropna()
        g = v[v.y_gbm == 1]
        rows.append((lab, roc_auc_score(v.y_gbm, v[f]), roc_auc_score(v[v.appmag == 40].y_gbm, v[v.appmag == 40][f]),
                     roc_auc_score((g.appmag == 20).astype(int), g[f])))
    fig = figure(120, 62)
    ax = fig.add_subplot()
    yy = np.arange(len(rows))[::-1]
    for y, (lab, a1, a2, a3) in zip(yy, rows):
        ax.plot([min(a1, a2, a3), max(a1, a2, a3)], [y, y], color=LGREY, lw=0.6, zorder=1)
        ax.plot(a1, y, "o", color=BLACK, ms=3.4, zorder=3)
        ax.plot(a2, y, "o", color=BLACK, mfc="white", mew=0.8, ms=3.4, zorder=3)
        ax.plot(a3, y, "s", color=VERMIL, ms=3.0, zorder=2)
    ax.axvline(0.5, **REF)
    ax.set_yticks(yy, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUROC")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLACK, ms=3.4, label="AUROC GBM vs LGG"),
                        Line2D([], [], marker="o", ls="", color=BLACK, mfc="white", mew=0.8, ms=3.4, label="Same, 40× only"),
                        Line2D([], [], marker="s", ls="", color=VERMIL, ms=3.0, label="AUROC 20× vs 40× within GBM")],
               loc="outside lower center", ncol=3)
    save(fig, "figS3_tissue_feature_aurocs")


def figS4_scanner_recognition():
    S = scan.set_index(["target (GBM slides only)", "features"])
    targets = [("scanned_at_20x", "Scanned at 20×"), ("jpeg_q70", "JPEG quality 70"), ("scan_before_2011", "Scan before 2011")]
    feats = [("HIPT embedding (20x tiles)", "HIPT embedding (20× tiles)", C_IMG, "o"),
             ("hand-crafted tissue features", "Hand-crafted tissue features", C_TIS, "s"),
             ("all hand-crafted incl. colour/sharpness", "All hand-crafted incl. colour/sharpness", C_NUIS, "D")]
    fig = figure(89, 72)
    ax = fig.add_subplot()
    for j, (key, lab, c, mk) in enumerate(feats):
        for i, (tg, _) in enumerate(targets):
            r = S.loc[(tg, key)]
            x = i + (j - 1) * 0.24
            ax.plot([x, x], [r.lo, r.hi], color=c, lw=0.9)
            ax.plot(x, r.auroc, mk, color=c, ms=3.4 if mk == "o" else 3.0, label=lab if i == 0 else None)
    ax.axhline(0.5, **REF)
    ax.set_xticks(range(3), [f"{t}\n({int(S.loc[(k, feats[0][0])].n_pos)} of {int(S.loc[(k, feats[0][0])].n)})"
                             for k, t in targets])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("AUROC (GBM slides only)")
    fig.legend(loc="outside lower center", ncol=1)
    save(fig, "figS4_scanner_recognition")


FAMILY = {"img_meanpool": C_IMG, "img_abmil": C_IMG, "handcrafted": C_TIS, "acq_meta": C_NUIS, "stain_color": C_NUIS,
          "tfidf_M1": C_REP, "tfidf_M2": C_REP, "age": C_AGE, "late_image+report_M2": C_FUS, "early_image+report_M2": C_FUS}


def figS5_stratified_auroc():
    S = strat[strat.branch.isin(["image", "fusion"]) | strat.model.isin(["tfidf_M2", "age"])]
    models = list(dict.fromkeys(S.model))
    rows = [(m, c) for m in models for c in ["all", "x40"]]
    lo = min(S[["auroc_overall", "auroc_within_institution_x_scanyear", "auroc_within_jpegq_x_software"]].min())
    fig = figure(183, 100)
    gs = fig.add_gridspec(1, 2)
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[0, 1], sharey=a0)
    yy = np.arange(len(rows))[::-1]
    for ax, scheme, title in [(a0, "patient", "Random patient folds"), (a1, "institution", "Institutions held out")]:
        for y, (m, c) in zip(yy, rows):
            r = S[(S.model == m) & (S.cohort == c) & (S.scheme == scheme)]
            if r.empty:
                continue
            r = r.iloc[0]
            vals = [r.auroc_overall, r.auroc_within_institution_x_scanyear, r.auroc_within_jpegq_x_software]
            ax.plot([min(vals), max(vals)], [y, y], color=LGREY, lw=0.6, zorder=1)
            ax.plot(vals[0], y, "o", color=BLACK, ms=3.2, zorder=3)
            ax.plot(vals[1], y, "o", color=VERMIL, mfc="white", mew=0.8, ms=3.2, zorder=3)
            ax.plot(vals[2], y, "^", color=BLUE, mfc="white", mew=0.8, ms=3.4, zorder=2)
        ax.axvline(0.5, **REF)
        ax.set_xlim(np.floor(lo * 10) / 10, 1.0)
        ax.set_xlabel("AUROC")
        ax.set_title(title)
    for y in yy[1::2][:-1]:
        for ax in (a0, a1):
            ax.axhline(y - 0.5, color="#E6E6E6", lw=0.4, zorder=0)
    a0.set_yticks(yy, [f"{nm(m)} ({'all slides' if c == 'all' else '40× only'})" for m, c in rows])
    a0.tick_params(axis="y", length=0)
    a1.tick_params(axis="y", length=0, labelleft=False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLACK, ms=3.2, label="AUROC overall"),
                        Line2D([], [], marker="o", ls="", color=VERMIL, mfc="white", mew=0.8, ms=3.2,
                               label="Within institution × scan year"),
                        Line2D([], [], marker="^", ls="", color=BLUE, mfc="white", mew=0.8, ms=3.4,
                               label="Within JPEG × software")], loc="outside lower center", ncol=3)
    letters(fig, [(a0, "a"), (a1, "b")])
    save(fig, "figS5_stratified_auroc")


def figS6_uf_lrt():
    U = uf[uf.scheme == "institution"]
    fig = figure(183, 112)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05])
    a = fig.add_subplot(gs[:, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 1], sharex=b)
    yy = np.arange(len(U))[::-1]
    for y, r in zip(yy, U.itertuples()):
        col = FAMILY.get(r.model, BLACK)
        a.plot([r.lo, r.hi], [y, y], color=col, lw=0.9, solid_capstyle="butt", clip_on=False)
        a.plot(r.auroc, y, "o", color=col, ms=3.2, clip_on=False, zorder=3)
    a.axvline(0.5, **REF)
    a.set_yticks(yy, [f"{nm(r.model)} ({'all slides' if r.cohort == 'all' else '40× only'})" for r in U.itertuples()])
    a.tick_params(axis="y", length=0)
    a.set_xlim(0.2, 1.0)
    a.set_ylim(-0.6, len(U) - 0.4)
    a.set_xlabel("AUROC (95% CI)")
    a.set_title("University of Florida, 15 GBM vs 15 LGG\n(institutions held out)")
    for ax, scheme, title in [(b, "patient", "Random patient folds"), (c, "institution", "Institutions held out")]:
        L = lrt[lrt.scheme == scheme]
        y2 = np.arange(len(L))[::-1]
        for y, r in zip(y2, L.itertuples()):
            col = FAMILY.get(r.model, BLACK)
            ax.plot([r.acq_only_auroc_in_sample, r.acq_plus_score_auroc_in_sample], [y, y], color=col, lw=0.8)
            ax.plot(r.acq_only_auroc_in_sample, y, "o", color=col, mfc="white", mew=0.8, ms=3.2)
            ax.plot(r.acq_plus_score_auroc_in_sample, y, "o", color=col, ms=3.2, clip_on=False, zorder=3)
            ax.text(1.02, y, f"{r.LR_chi2:.1f}", va="center", ha="left", fontsize=5.5, transform=ax.get_yaxis_transform())
        ax.set_ylim(-0.6, len(L) - 0.4)
        ax.text(1.02, len(L) - 0.4, "LR χ²", va="bottom", ha="left", fontsize=5.5, transform=ax.get_yaxis_transform())
        ax.set_yticks(y2, [nm(m) for m in L.model])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0.6, 1.0)
        ax.set_title(title)
    c.set_xlabel("In-sample AUROC on 40×-scanned slides")
    b.tick_params(labelbottom=False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=DGREY, mfc="white", mew=0.8, ms=3.2,
                               label="b, c: scan year + JPEG quality + software version"),
                        Line2D([], [], marker="o", ls="", color=DGREY, ms=3.2, label="b, c: same + out-of-fold model score")],
               loc="outside lower right", ncol=2)
    letters(fig, [(a, "a"), (b, "b"), (c, "c")])
    save(fig, "figS6_uf_lrt")


def figS7_fusion_random_rocs_deltas():
    fig = figure(183, 200)
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 1.3])
    a, b = top.subplots(1, 2)
    c = bot.add_subplot(1, 2, 1)
    d = bot.add_subplot(1, 2, 2, sharey=c)
    for ax, cohort, title in [(a, "all", "All matched patients (n = 686)"), (b, "x40", "40×-scanned slides only (n = 556)")]:
        roc_series(ax, fo, FUSION_SERIES, cohort, "patient")
        ax.set_xlabel("False-positive rate (LGG called GBM)")
        ax.set_ylabel("True-positive rate (GBM called GBM)")
        ax.set_title(title)
        ax.legend(loc="lower right", title="AUROC, random patient folds")
    comps = list(dict.fromkeys(zip(fd.baseline, fd.model)))
    yy = np.arange(len(comps))[::-1]
    for ax, cohort, title in [(c, "all", "All slides"), (d, "x40", "40×-scanned slides only")]:
        for y, (bl, md) in zip(yy, comps):
            for scheme, dy, mk, mfc in [("patient", 0.17, "o", BLACK), ("institution", -0.17, "s", "white")]:
                r = fd[(fd.baseline == bl) & (fd.model == md) & (fd.cohort == cohort) & (fd.scheme == scheme)].iloc[0]
                ax.plot([r.lo, r.hi], [y + dy] * 2, color=BLACK, lw=0.8, solid_capstyle="butt")
                ax.plot(r.delta_auroc, y + dy, mk, color=BLACK, mfc=mfc, mew=0.8, ms=3.2 if mk == "o" else 2.8)
        ax.axvline(0, **REF)
        ax.set_xlabel("AUROC change (95% CI)")
        ax.set_title(title)
        for y in yy[:-1]:
            ax.axhline(y - 0.5, color="#E6E6E6", lw=0.4, zorder=0)
        ax.set_ylim(-0.6, len(comps) - 0.4)
    c.set_yticks(yy, [f"{nm(md)}\nvs {nm(bl)}" for bl, md in comps], fontsize=5.5)
    c.tick_params(axis="y", length=0)
    d.tick_params(axis="y", length=0, labelleft=False)
    d.legend(handles=[Line2D([], [], marker="o", ls="", color=BLACK, ms=3.2, label="Random patient folds"),
                      Line2D([], [], marker="s", ls="", color=BLACK, mfc="white", mew=0.8, ms=2.8, label="Institutions held out")],
             loc="upper right")
    letters(fig, zip([a, b, c, d], "abcd"))
    save(fig, "figS7_fusion_random_rocs_deltas")


def figS8_molecular_grade():
    sub = fsg[(fsg.cohort == "all") & (fsg.scheme == "institution")].rename(columns={"frac_called_gbm@0.5": "frac"})
    grp = sub[sub.label != "grade"]
    gr = sub[sub.label == "grade"]
    keys = list(grp[["label", "group"]].drop_duplicates().sort_values(["label", "group"]).itertuples(index=False, name=None))
    pretty = {"GBM_IDHwt": "glioblastoma, IDH-wild-type", "Astro_IDHmut": "astrocytoma, IDH-mutant",
              "Oligo_IDHmut_codel": "oligodendroglioma", "IDHwt_other": "IDH-wild-type, other", "unknown": "IDH status missing"}
    models = [("image_meanpool", "Image, HIPT mean-pool", C_IMG, "o"), ("report_M2", "Report, masked M2", C_REP, "s"),
              ("late_image+report_M2", "Late fusion: image + report M2", C_FUS, "D")]
    fig = figure(183, 132)
    top, bot = fig.subfigures(2, 1, height_ratios=[1.45, 1])
    a = top.add_subplot(1, 2, 1)
    b = top.add_subplot(1, 2, 2, sharey=a)
    c, d = bot.subplots(1, 2)
    yy = np.arange(len(keys))[::-1]
    for ax, col, xl in [(a, "mean_p_gbm", "Mean P(GBM)"), (b, "frac", "Called GBM at P = 0.5 (%)")]:
        top_ = 100 if col == "frac" else 1
        for j, (m, lab, cc, mk) in enumerate(models):
            vals = [grp[(grp.label == L) & (grp.group == G) & (grp.model == m)][col].iloc[0] for L, G in keys]
            vals = np.array(vals) * (100 if col == "frac" else 1)
            ax.plot(vals, yy + (1 - j) * 0.22, mk, ls="", color=cc, ms=3.2 if mk != "D" else 2.8, label=lab,
                    clip_on=False, zorder=3)
        ax.axvline(top_ / 2, **REF)
        ax.set_xlim(-0.03 * top_, 1.03 * top_)
        ax.spines["bottom"].set_bounds(0, top_)
        ax.set_xlabel(xl)
        ax.axhline(4.5, color="#E6E6E6", lw=0.5, zorder=0)
        ax.set_ylim(-0.6, len(keys) - 0.4)
    ns = [int(grp[(grp.label == L) & (grp.group == G)].n.iloc[0]) for L, G in keys]
    for y, n in zip(yy, ns):
        b.text(1.05, y, f"n = {n}", va="center", fontsize=5.5, transform=b.get_yaxis_transform())
    a.set_yticks(yy, [f"{L} label: {pretty.get(G, G)}" for L, G in keys])
    a.tick_params(axis="y", length=0)
    b.tick_params(axis="y", length=0, labelleft=False)
    grades = ["G2", "G3", "G4"]
    gn = [int(gr[gr.group == g_].n.iloc[0]) for g_ in grades]
    for ax, col, yl in [(c, "mean_p_gbm", "Mean P(GBM)"), (d, "frac", "Called GBM at P = 0.5 (%)")]:
        for m, lab, cc, mk in models:
            r = gr[gr.model == m].set_index("group").loc[grades]
            ax.plot(range(3), r[col] * (100 if col == "frac" else 1), marker=mk, ms=3.2 if mk != "D" else 2.8, color=cc,
                    lw=1.0, label=lab)
        ax.set_xticks(range(3), [f"Grade {g_[1]}\n(n = {n})" for g_, n in zip(grades, gn)])
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(0, 100 if col == "frac" else 1)
        ax.set_ylabel(yl)
    c.legend(loc="upper left")
    letters(fig, zip([a, b, c, d], "abcd"))
    save(fig, "figS8_molecular_grade")


def figS9_mechanics_models():
    import statsmodels.formula.api as smf
    pmA = pm.copy()                                           # same fits as the table in report/build_content.py
    pmA["cond"] = pd.Categorical(pmA.cond, ["Serum", "EGF", "FGFJI"])
    mA = smf.mixedlm("logE_median ~ C(cond) + C(line)", pmA, groups=pmA["date"]).fit(reml=True)
    G = ev.copy()
    sub_ = pd.concat([g.sample(min(len(g), 400), random_state=20261008) for _, g in G.groupby("measurement")])
    sub_["logE"] = np.log10(sub_.emodulus)
    sub_["cond"] = pd.Categorical(sub_.cond, ["Serum", "EGF", "FGFJI"])
    mB = smf.mixedlm("logE ~ C(cond) + C(line) + C(date)", sub_, groups=sub_["measurement"]).fit(reml=True)
    rows = []
    for mname, m in [("A", mA), ("B", mB)]:
        ci = m.conf_int()
        for term in m.params.index:
            if term in ("Group Var", "Intercept"):
                continue
            t = term.replace("C(cond)[T.", "condition ").replace("C(line)[T.", "line ").replace("C(date)[T.", "date ").replace("]", "")
            rows.append((mname, t, m.params[term], ci.loc[term, 0], ci.loc[term, 1], m.pvalues[term]))
    R = pd.DataFrame(rows, columns=["model", "term", "coef_log10E", "lo", "hi", "p"])
    R.to_csv(RES / "mechanics_mixed_models_terms.csv", index=False)
    fig = figure(183, 122)
    top, bot = fig.subfigures(2, 1, height_ratios=[1.2, 1])
    gs = top.add_gridspec(1, 3)
    a = top.add_subplot(gs[0, 0:2])
    b = top.add_subplot(gs[0, 2])
    tasks = list(bot.subplots(1, 3))
    terms = list(dict.fromkeys(R.term))
    yy = np.arange(len(terms))[::-1]
    for mname, dy, mk, mfc, lab in [("A", 0.15, "o", BLACK, "Model A: per-measurement median, (1 | date)"),
                                    ("B", -0.15, "s", "white", "Model B: single cells, (1 | measurement)")]:
        for y, t in zip(yy, terms):
            r = R[(R.model == mname) & (R.term == t)]
            if r.empty:
                continue
            r = r.iloc[0]
            a.plot([10 ** r.lo, 10 ** r.hi], [y + dy] * 2, color=BLACK, lw=0.8, solid_capstyle="butt")
            a.plot(10 ** r.coef_log10E, y + dy, mk, color=BLACK, mfc=mfc, mew=0.8, ms=3.2 if mk == "o" else 2.8,
                   label=lab if t == terms[0] else None)
    a.axvline(1, **REF)
    log_ticks(a, [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    a.set_yticks(yy, terms)
    a.tick_params(axis="y", length=0)
    a.set_ylim(-0.6, len(terms) - 0.4)
    a.set_xlabel("Fold change in Young's modulus (95% CI)")
    a.legend(loc="upper right")
    vlab = {"C(cond)": "Condition", "C(line)": "Line", "C(date)": "Date", "model_R2": "Whole model (R²)"}
    yv = np.arange(len(vc))[::-1]
    b.barh(yv, vc.partial_R2, height=0.55, color=[DGREY if t != "model_R2" else LGREY for t in vc.term], lw=0)
    for y, v in zip(yv, vc.partial_R2):
        b.text(v + 0.02, y, f"{v:.3f}", va="center", fontsize=5.5)
    b.set_yticks(yv, [vlab.get(t, t) for t in vc.term])
    b.tick_params(axis="y", length=0)
    b.set_xlim(0, 1.05)
    b.set_xlabel("Partial R² (per-measurement medians)")
    fl = {"median_E_only": ("Median E", GREY, "o"), "E_distribution": ("E distribution", PURPLE, "s"),
          "E_distribution+shape": ("E distribution + cell shape", BLACK, "D")}
    for ax, task in zip(tasks, ["Serum_vs_FGFJI", "Serum_vs_EGF", "EGF_vs_FGFJI"]):
        tk = bag[bag.task == task]
        for f, (lbl, c, mk) in fl.items():
            r = tk[tk.features == f].sort_values("bag_cells")
            ax.plot(r.bag_cells, r.bag_auroc, marker=mk, color=c, lw=1.0, ms=3.0 if mk != "D" else 2.6, label=lbl,
                    clip_on=False, zorder=3)
            ax.plot(r.bag_cells, r.measurement_auroc, marker=mk, color=c, lw=0.7, ls=(0, (2.5, 1.5)),
                    ms=3.0 if mk != "D" else 2.6, mfc="white", mew=0.7, clip_on=False, zorder=3)
        log_ticks(ax, [50, 100, 300])
        ax.set_ylim(0.5, 1.0)
        ax.set_xlabel("Cells per bag")
        ax.set_ylabel("AUROC, held-out date")
        ax.set_title(task.replace("_vs_", " vs "), pad=7)
    tasks[0].legend(loc="lower left", fontsize=5.5)
    tasks[1].legend(handles=[Line2D([], [], marker="o", color=DGREY, lw=1.0, ms=3.0, label="Bag AUROC"),
                             Line2D([], [], marker="o", color=DGREY, lw=0.7, ls=(0, (2.5, 1.5)), ms=3.0, mfc="white",
                                    mew=0.7, label="Measurement AUROC")], loc="lower left", fontsize=5.5)
    letters(fig, zip([a, b] + tasks, "abcde"))
    save(fig, "figS9_mechanics_models")


FIGURES = [fig0_pipeline, fig1_inventory, fig2_acquisition_confounding, fig3_auroc_overview, fig4_report_masking,
           fig5_mechanics, fig6_who2021_subgroups, fig7_attention_tiles, fig8_power, fig9_roc_curves, fig10_calibration,
           fig11_embedding_map, fig12_tissue_features, fig13_institutions, contact_sheet, figS1_report_roc,
           figS2_image_roc, figS3_tissue_feature_aurocs, figS4_scanner_recognition, figS5_stratified_auroc, figS6_uf_lrt,
           figS7_fusion_random_rocs_deltas, figS8_molecular_grade, figS9_mechanics_models]


def write_legends():
    lines = ["# MechanoPath figures: legends", "",
             "Nature-style figures: PDF = vector artwork with embedded, editable fonts; PNG at 450 dpi (micrographs "
             "fig7 and the contact sheet as 450-dpi JPEG). Titles and footnotes that the earlier figures carried on "
             "the image are given here.", ""]
    for name, (title, legend) in LEGENDS.items():
        lines += [f"**{name}** — {title} {legend}".rstrip(), ""]
    (OUT / "figure_legends.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sel = sys.argv[1:]
    for f in FIGURES:
        name = f.__name__ if f is not contact_sheet else "contact_TCGA-06-0156"
        if not sel or any(name.startswith(s) or f.__name__.startswith(s) for s in sel):
            f()
    write_legends()
