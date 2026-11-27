"""Manuscript figures (Nature format) for both phases, drawn from the result tables.

  python dcc/08_figures.py [fig1 fig5 ...]        (default: every figure)

Inputs: results/phase1/*.csv (dcc/06_phase1_analysis.py), results/phase2/*.csv (src/13_mechanics_phase2.py,
src/14_design_plasmode.py), cohort tables in data/processed, and for Fig. 4 the attention weights
(results/attention_phikon2.npz) with the slides and tile grids on /cwork.
Style: 89 mm or 183 mm wide, sans-serif 5-7 pt, bold lowercase 8-pt panel letters, no gridlines, colour-blind-safe
Wong palette; vector PDF (editable text) plus 450-dpi PNG. Outputs: figures/manuscript/.
"""
import json
import os
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"          # jobs submitted from Jupyter inherit an inline backend that is invalid here
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import FIG, PROC, RES  # noqa: E402

from matplotlib import font_manager  # noqa: E402
for _f in (ROOT / "dcc" / "fonts").glob("*.ttf"):            # Liberation Sans (Arial metrics, SIL OFL) when Arial is absent
    font_manager.fontManager.addfont(str(_f))
OUT = FIG / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)
P1, P2 = RES / "phase1", RES / "phase2"
MM = 1 / 25.4
BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERMIL, PURPLE, GREY = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999")
DGREY, LGREY = "#555555", "#C8C8C8"
C_GBM, C_LGG = VERMIL, BLUE
C_IMG, C_REP, C_FUS, C_NUIS, C_AGE = GREEN, PURPLE, BLUE, GREY, ORANGE
C_PAT, C_INST = PURPLE, ORANGE
C_COND = {"Serum": "#4D4D4D", "EGF": ORANGE, "FGFJI": SKY}
REF = dict(color=LGREY, lw=0.5, ls=(0, (2.5, 2)), zorder=0)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 6.5, "axes.titlesize": 6.5, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "lines.linewidth": 0.9, "lines.markersize": 3, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.labelspacing": 0.3, "legend.handletextpad": 0.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.labelpad": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.titlepad": 3, "figure.constrained_layout.w_pad": 1.2 / 72, "figure.constrained_layout.h_pad": 1.2 / 72,
    "figure.constrained_layout.wspace": 0.03, "figure.constrained_layout.hspace": 0.04, "mathtext.default": "regular"})

NAMES = {
    "fusion_primary": "Fusion (image + report)", "image_phikon2_abmil": "Image: Phikon-v2, attention MIL",
    "image_phikon2_meanpool": "Image: Phikon-v2, mean pooling", "image_hipt_abmil": "Image: HIPT, attention MIL",
    "image_hipt_meanpool": "Image: HIPT, mean pooling", "report_bert_M2": "Report: BiomedBERT, M2 masking",
    "report_bert_M1": "Report: BiomedBERT, M1 masking", "report_tfidf_M2": "Report: TF-IDF, M2 masking",
    "report_tfidf_M1": "Report: TF-IDF, M1 masking", "report_unmasked_M0": "Report: unmasked (M0)",
    "age": "Age only", "acq_meta": "Acquisition metadata", "stain_color": "Stain colour",
    "tissue_handcrafted": "Hand-crafted tissue features", "fusion_image_reportM1": "Fusion (image + report M1)",
    "fusion_hipt_reportM2": "Fusion (HIPT image + report)", "fusion_image_tfidfM2": "Fusion (image + TF-IDF report)",
    "fusion_primary_age": "Fusion + age", "fusion_image_acq": "Image + acquisition metadata"}
DESC_NAMES = {"mean": "Mean", "median": "Median", "q10": "10th percentile", "q90": "90th percentile",
              "iqr": "IQR", "upper_tail": "Upper tail (q90-q50)", "lower_tail": "Lower tail (q50-q10)",
              "tail_asym": "Tail asymmetry", "bowley": "Quartile skewness", "moors": "Octile kurtosis", "mad": "MAD"}


def nm(m):
    return NAMES.get(m, m)


def read(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 5:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def figure(w, h):
    return plt.figure(figsize=(w * MM, h * MM), layout="constrained")


def letters(fig, pairs):
    for ax, _ in pairs:
        if not (ax.get_title() or ax.get_title(loc="left") or ax.get_title(loc="right")):
            ax.set_title(" ", loc="left", fontsize=8)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, lab in pairs:
        bb = ax.get_tightbbox(r)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(max(x0, 0.002), min(y1, 0.998), lab, fontsize=8, fontweight="bold", ha="left", va="top")
    fig.set_layout_engine("none")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=450)
    plt.close(fig)
    print("saved", name, flush=True)


def logx(ax, ticks):
    from matplotlib.ticker import NullFormatter
    ax.set_xscale("log")
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def note(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=6, color=DGREY, wrap=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
slides = pd.read_csv(PROC / "slides.csv").set_index("patient")
OOF = read(P1 / "phase1_oof.csv")
MET = read(P1 / "phase1_metrics.csv")
CMP = read(P1 / "phase1_comparisons.csv")
SUMM = json.load(open(P1 / "phase1_summary.json")) if (P1 / "phase1_summary.json").exists() else {}
IMG = SUMM.get("image_branch", "image_phikon2_abmil")
REP = SUMM.get("report_branch", "report_bert_M2")


def met(model, c="all", s="institution"):
    r = MET[(MET.model == model) & (MET.cohort == c) & (MET.scheme == s)] if MET is not None else []
    return r.iloc[0] if len(r) else None


# ================================================================================================ Fig. 1
def box(ax, x, y, w, h, text, fc="#F2F2F2", ec=DGREY, fs=5.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None,
            linespacing=1.15)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5, lw=0.6, color=DGREY))


def fig1():
    fig = figure(183, 128)
    top, bot = fig.subfigures(2, 1, height_ratios=[1.2, 1], hspace=0.02)
    ax = top.add_subplot(111)
    ax.set_xlim(-0.005, 1.005); ax.set_ylim(0, 1); ax.axis("off")
    sz = SUMM.get("cohort_sizes", {})
    n_all = sz.get("all_institution", {}).get("n", "-")
    n40 = sz.get("x40_institution", {}).get("n", "-")
    ax.text(0.0, 0.99, "Phase 1  |  GBM vs LGG from slides and reports (TCGA)", fontsize=6.5, fontweight="bold",
            va="top")
    box(ax, 0.00, 0.62, 0.15, 0.22, f"TCGA-GBM / LGG\nmatched cohort\nn = {n_all}\n(40x subset n = {n40})", "#FFFFFF")
    box(ax, 0.20, 0.73, 0.18, 0.12, "Diagnostic H&E slide\n256-px tiles at 0.5 µm/px", "#E8F4EE")
    box(ax, 0.42, 0.73, 0.17, 0.12, "Phikon-v2 ViT-L/16\n-> gated attention MIL", "#E8F4EE")
    box(ax, 0.20, 0.58, 0.18, 0.12, "Pathology report\ndiagnosis/grade masked", "#F6ECF3")
    box(ax, 0.42, 0.58, 0.17, 0.12, "BiomedBERT\nfine-tuned per fold", "#F6ECF3")
    box(ax, 0.64, 0.64, 0.12, 0.15, "Late fusion\nmean of logits", "#E5EEF7")
    box(ax, 0.81, 0.64, 0.19, 0.15, "Evaluation: 5x5 CV with\nwhole institutions held out;\n40x-only subset; DeLong, Holm",
        "#FFFFFF", fs=5)
    for (x0, y0, x1, y1) in [(0.15, 0.76, 0.20, 0.79), (0.15, 0.70, 0.20, 0.64), (0.38, 0.79, 0.42, 0.79),
                             (0.38, 0.64, 0.42, 0.64), (0.59, 0.79, 0.64, 0.74), (0.59, 0.64, 0.64, 0.69),
                             (0.76, 0.715, 0.81, 0.715)]:
        arrow(ax, x0, y0, x1, y1)
    ax.text(0.0, 0.47, "Phase 2  |  Does the single-cell stiffness distribution carry information beyond its "
                       "average?", fontsize=6.5, fontweight="bold", va="top")
    box(ax, 0.00, 0.06, 0.22, 0.32, "RT-DC (Mechanomics Set 01)\n3 primary GBM lines x 3 culture\nstates x 3 days"
                                   "\n21 measurements, 21,738 cells\n\nAFM: LN229, U87 (1,160 cells)", "#FFF6E5", fs=5)
    box(ax, 0.27, 0.16, 0.17, 0.14, "Per-sample descriptors\nlocation | spread | tails", "#FFF6E5")
    box(ax, 0.49, 0.24, 0.20, 0.13, "Location vs shape effects;\nreproducibility; cells needed", "#FFFFFF", fs=5)
    box(ax, 0.49, 0.06, 0.20, 0.13, "Bag classification:\naverage vs distribution", "#FFFFFF", fs=5)
    box(ax, 0.74, 0.06, 0.26, 0.32, "Design analysis (plasmode)\nreal Phase 1 scores + real RT-DC\nshapes; power "
                                   "of distribution\nbeyond average stiffness beyond\nimages and reports", "#E5EEF7", fs=5)
    for (x0, y0, x1, y1) in [(0.22, 0.23, 0.27, 0.23), (0.44, 0.25, 0.49, 0.30), (0.44, 0.20, 0.49, 0.13),
                             (0.69, 0.30, 0.74, 0.26), (0.69, 0.13, 0.74, 0.17)]:
        arrow(ax, x0, y0, x1, y1)
    ax.plot([0.70, 0.70], [0.62, 0.42], color=DGREY, lw=0.5, ls=(0, (2, 2)))
    ax.text(0.705, 0.53, "Phase 1 out-of-fold\nscores", fontsize=5, color=DGREY, va="center")
    arrow(ax, 0.70, 0.42, 0.79, 0.385)
    gs = bot.add_gridspec(1, 3, width_ratios=[1.15, 0.75, 1.2])
    # b: cohort flow
    axb = bot.add_subplot(gs[0, 0])
    lab = cohort
    has_slide = lab.index.isin(slides.index)
    rep = pd.read_csv(PROC / "reports_masked.csv").set_index("patient")
    has_rep = lab.index.isin(rep.index)
    x40 = lab.index.isin(slides.index[slides.appmag == 40])
    matched = OOF[(OOF.cohort == "all") & (OOF.scheme == "institution") & (OOF.model == "fusion_primary")].patient \
        if OOF is not None else lab.index[has_slide & has_rep]
    m40 = OOF[(OOF.cohort == "x40") & (OOF.scheme == "institution") & (OOF.model == "fusion_primary")].patient \
        if OOF is not None else lab.index[has_slide & has_rep & x40]
    rows = [("Labelled (pan-glioma)", lab.index), ("Diagnostic slide", lab.index[has_slide]),
            ("Pathology report", lab.index[has_rep]), ("Matched, analysed", pd.Index(matched)),
            ("Matched, 40x scan", pd.Index(m40))]
    for i, (name, idx) in enumerate(rows):
        g = int(lab.loc[idx, "y_gbm"].sum()); l_ = len(idx) - g
        axb.barh(i, g, color=C_GBM, height=0.6)
        axb.barh(i, l_, left=g, color=C_LGG, height=0.6)
        axb.text(len(idx) + 15, i, f"{len(idx)}", va="center", fontsize=5.5)
    axb.set_yticks(range(len(rows)), [r[0] for r in rows])
    axb.invert_yaxis()
    axb.set_xlabel("Patients")
    axb.set_xlim(0, 1300)
    axb.legend(handles=[Patch(color=C_GBM, label="GBM"), Patch(color=C_LGG, label="LGG")], loc="lower right")
    # c: magnification by class
    axc = bot.add_subplot(gs[0, 1])
    s = slides.join(cohort[["y_gbm"]], how="inner")
    for j, (cls, col) in enumerate(((1, C_GBM), (0, C_LGG))):
        n20 = int(((s.y_gbm == cls) & (s.appmag == 20)).sum()); n40_ = int(((s.y_gbm == cls) & (s.appmag == 40)).sum())
        axc.bar(j, n40_, color=col, width=0.6)
        axc.bar(j, n20, bottom=n40_, color=col, width=0.6, alpha=0.4, hatch="////", lw=0,
                label="20x" if j == 0 else None)
        axc.text(j, n40_ + n20 + 10, f"20x: {n20}\n40x: {n40_}", ha="center", va="bottom", fontsize=5)
    axc.set_xticks([0, 1], ["GBM", "LGG"])
    axc.set_ylabel("Slides")
    axc.set_ylim(0, max(len(s[s.y_gbm == 1]), len(s[s.y_gbm == 0])) * 1.3)
    axc.set_title("Scan magnification (hatched: 20x)", fontsize=6)
    # d: institutions
    axd = bot.add_subplot(gs[0, 2])
    mm = cohort.loc[[p for p in matched if p in cohort.index]]
    t = mm.groupby("institution").y_gbm.agg(["sum", "size"]).sort_values("size", ascending=False).head(14)
    t["lgg"] = t["size"] - t["sum"]
    y_ = np.arange(len(t))
    axd.barh(y_, t["sum"], color=C_GBM, height=0.65)
    axd.barh(y_, t["lgg"], left=t["sum"], color=C_LGG, height=0.65)
    axd.set_yticks(y_, [i[:28] for i in t.index], fontsize=5)
    axd.invert_yaxis()
    axd.set_xlabel("Matched patients")
    axd.set_title(f"Largest 14 of {mm.institution.nunique()} tissue-source sites", fontsize=6)
    letters(fig, [(ax, "a"), (axb, "b"), (axc, "c"), (axd, "d")])
    save(fig, "Fig1_design_cohort")


# ================================================================================================ Fig. 2
def roc_panel(ax, c, s, models, styles, title):
    if OOF is None:
        return note(ax, "Phase 1 results not available")
    short = {"fusion_primary": "Fusion", IMG: "Image", REP: "Report", "acq_meta": "Acquisition metadata"}
    for m in models:
        g = OOF[(OOF.cohort == c) & (OOF.scheme == s) & (OOF.model == m)]
        if not len(g):
            continue
        fpr, tpr, _ = roc_curve(g.y, g.p)
        r = met(m, c, s)
        col, ls = styles[m]
        ax.plot(fpr, tpr, color=col, ls=ls, lw=1.0 if m == "fusion_primary" else 0.8,
                label=f"{short.get(m, nm(m))}  {r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]")
    ax.plot([0, 1], [0, 1], **REF)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    n = SUMM.get("cohort_sizes", {}).get(f"{c}_{s}", {})
    ax.set_title(f"{title}\n(n = {n.get('n', '-')}, {n.get('n_gbm', '-')} GBM)", fontsize=6)
    ax.legend(loc="lower right", fontsize=4.8, handlelength=1.2, borderaxespad=0.2)


def dot_panel(ax, c, models, title, show_labels=True):
    if MET is None:
        return note(ax, "Phase 1 results not available")
    y = np.arange(len(models))
    for k, (s, col, off) in enumerate((("patient", C_PAT, -0.17), ("institution", C_INST, 0.17))):
        for i, m in enumerate(models):
            r = met(m, c, s)
            if r is None:
                continue
            ax.errorbar(r.auroc, i + off, xerr=[[r.auroc - r.auroc_lo], [r.auroc_hi - r.auroc]], fmt="o", ms=2.6,
                        color=col, mfc=col if s == "institution" else "white", mew=0.7, elinewidth=0.6, capsize=0)
    ax.axvline(0.5, **REF)
    ax.set_yticks(y, [nm(m) for m in models] if show_labels else [""] * len(models), fontsize=5.2)
    ax.set_ylim(len(models) - 0.4, -0.6)
    ax.set_xlim(0.25, 1.0)
    ax.set_xlabel("AUROC (95% CI)")
    ax.set_title(title, fontsize=6)


CMP_LABEL = {("fusion_primary", "IMG"): "Fusion vs image", ("fusion_primary", "REP"): "Fusion vs report",
             ("fusion_image_reportM1", "IMG"): "Fusion (report M1) vs image",
             ("fusion_image_tfidfM2", "IMG"): "Fusion (TF-IDF report) vs image",
             ("fusion_image_tfidfM2", "fusion_primary"): "Fusion (TF-IDF) vs fusion (BERT)",
             ("fusion_primary_age", "fusion_primary"): "Fusion + age vs fusion"}


def cmp_label(r):
    key = (r.model, "IMG" if r.comparator == IMG else "REP" if r.comparator == REP else r.comparator)
    return CMP_LABEL.get(key)


def fig2():
    fig = figure(183, 160)
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 1.25], hspace=0.02)
    gt = top.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    styles = {"fusion_primary": (C_FUS, "-"), IMG: (C_IMG, "-"), REP: (C_REP, "-"), "acq_meta": (C_NUIS, "--")}
    models = ["fusion_primary", IMG, REP, "acq_meta"]
    a = top.add_subplot(gt[0, 0]); roc_panel(a, "all", "institution", models, styles, "Matched cohort, sites held out")
    b = top.add_subplot(gt[0, 1]); roc_panel(b, "x40", "institution", models, styles, "40x slides, sites held out")
    e = top.add_subplot(gt[0, 2])
    if CMP is not None:
        pr = CMP[(CMP.scheme == "institution") & CMP.family.isin(["primary", "secondary"])].copy()
        pr["lab"] = [cmp_label(r) for r in pr.itertuples()]
        pr = pr[pr.lab.notna()]
        pr["order"] = pr.cohort.map({"all": 0, "x40": 1})
        pr = pr.sort_values(["order", "family"], ascending=[True, True], kind="stable")
        ys, labs, yy = [], [], 0
        for c_, g in pr.groupby("order", sort=True):
            labs.append(("Matched cohort" if c_ == 0 else "40x-scanned slides", yy, True)); yy += 1
            for r in g.itertuples():
                ys.append(yy); labs.append((r.lab, yy, False)); yy += 1
        cols = [C_FUS if f == "primary" else DGREY for f in pr.family]
        e.errorbar(pr.delta_auroc, ys, xerr=[pr.delta_auroc - pr.lo, pr.hi - pr.delta_auroc], fmt="none",
                   ecolor=cols, elinewidth=0.7)
        e.scatter(pr.delta_auroc, ys, c=cols, s=9, zorder=3)
        for y_, r in zip(ys, pr.itertuples()):
            ptxt = "P < 0.001" if r.p_holm < 0.001 else f"P = {r.p_holm:.3f}"
            e.text(1.0, y_, ptxt, transform=e.get_yaxis_transform(), fontsize=4.8, va="center", ha="left")
        e.axvline(0, **REF)
        e.set_yticks([l_[1] for l_ in labs], [l_[0] for l_ in labs], fontsize=5)
        for t, l_ in zip(e.get_yticklabels(), labs):
            if l_[2]:
                t.set_fontweight("bold")
        e.set_ylim(yy - 0.4, -0.6)
        e.set_xlabel("Difference in AUROC (95% CI)")
        e.set_title("Paired contrasts, sites held out\n(blue: pre-specified primary; Holm P)", fontsize=6)
    gb = bot.add_gridspec(1, 2, width_ratios=[1.55, 1])
    models_all = [m for m in ["fusion_primary", "fusion_primary_age", "fusion_image_reportM1", "fusion_hipt_reportM2",
                              "fusion_image_tfidfM2", IMG, "image_phikon2_meanpool", "image_hipt_abmil",
                              "image_hipt_meanpool", REP, "report_bert_M1", "report_tfidf_M2", "report_unmasked_M0",
                              "age", "tissue_handcrafted", "stain_color", "acq_meta"]
                  if MET is not None and (MET.model == m).any()]
    c = bot.add_subplot(gb[0, 0]); dot_panel(c, "all", models_all, "Matched cohort")
    d = bot.add_subplot(gb[0, 1]); dot_panel(d, "x40", models_all, "40x-scanned slides", show_labels=False)
    c.legend(handles=[Line2D([], [], marker="o", ls="", color=C_PAT, mfc="white", ms=3, label="Random patient folds"),
                      Line2D([], [], marker="o", ls="", color=C_INST, ms=3, label="Institutions held out")],
             loc="lower left", fontsize=5)
    letters(fig, [(a, "a"), (b, "b"), (e, "c"), (c, "d"), (d, "e")])
    save(fig, "Fig2_discrimination")


# ================================================================================================ Fig. 3
def fig3():
    fig = figure(183, 125)
    gs = fig.add_gridspec(2, 3)
    cal = read(P1 / "phase1_calibration_bins.csv")
    dca = read(P1 / "phase1_decision_curve.csv")
    a = fig.add_subplot(gs[0, 0])
    if cal is not None:
        for m, col in (("fusion_primary", C_FUS), (IMG, C_IMG), (REP, C_REP)):
            g = cal[(cal.cohort == "all") & (cal.scheme == "institution") & (cal.model == m)]
            r = met(m)
            lab = nm(m).split(":")[0].replace(" (image + report)", "")
            if r is not None:
                lab += f": slope {r.calibration_slope:.2f}, ECE {r.ece:.3f}"
            a.errorbar(g.mean_p, g.observed, yerr=[g.observed - g.lo, g.hi - g.observed], fmt="o-", ms=2, lw=0.7,
                       elinewidth=0.5, color=col, label=lab)
        a.plot([0, 1], [0, 1], **REF)
        a.set_xlabel("Predicted probability of GBM"); a.set_ylabel("Observed fraction GBM")
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")
        a.legend(fontsize=4.6, loc="upper left", handlelength=1.2)
        a.set_title("Calibration (sites held out)", fontsize=6)
    b = fig.add_subplot(gs[0, 1])
    if dca is not None:
        g0 = dca[(dca.cohort == "all") & (dca.scheme == "institution")]
        for m, col in (("fusion_primary", C_FUS), (IMG, C_IMG), (REP, C_REP)):
            g = g0[g0.model == m]
            b.plot(g.threshold, g.net_benefit, color=col, label=nm(m).split(":")[0])
        gg = g0[g0.model == "fusion_primary"]
        b.plot(gg.threshold, gg.treat_all, color=GREY, lw=0.6, ls="--", label="Treat all as GBM")
        b.axhline(0, color=BLACK, lw=0.5, label="Treat none")
        b.set_ylim(-0.05, max(0.05, gg.treat_all.max()) * 1.15)
        b.set_xlabel("Threshold probability"); b.set_ylabel("Net benefit")
        b.legend(fontsize=4.8)
        b.set_title("Decision curves", fontsize=6)
    c = fig.add_subplot(gs[0, 2])
    bi = read(P1 / "phase1_by_institution.csv")
    if bi is not None and len(bi):
        g = bi[(bi.cohort == "all") & (bi.scheme == "institution") & (bi.model == "fusion_primary")]
        g = g.sort_values("auroc")
        y = np.arange(len(g))
        c.errorbar(g.auroc, y, xerr=[g.auroc - g.lo, g.hi - g.auroc], fmt="o", ms=2.4, color=C_FUS, elinewidth=0.6)
        c.set_yticks(y, [f"{i[:22]} ({a_}/{b_})" for i, a_, b_ in zip(g.institution, g.n_gbm, g.n_lgg)], fontsize=4.6)
        r = met("fusion_primary")
        if r is not None:
            c.axvline(r.auroc_within_institution, color=C_FUS, lw=0.5, ls=":")
        c.axvline(0.5, **REF)
        c.set_xlim(0.3, 1.02)
        c.set_xlabel("AUROC within institution")
        c.set_title("Sites with >= 5 GBM and >= 5 LGG", fontsize=6)
    d = fig.add_subplot(gs[1, :2])
    sub = read(P1 / "phase1_subgroups.csv")
    if sub is not None:
        groups = [("LGG", "Oligo_IDHmut_codel", "LGG: IDH-mut, 1p/19q-codeleted"),
                  ("LGG", "Astro_IDHmut", "LGG: IDH-mut astrocytoma"),
                  ("LGG", "IDHwt_other", "LGG: IDH-wt, other"),
                  ("LGG", "GBM_IDHwt", "LGG: IDH-wt, molecular GBM features"),
                  ("GBM", "Astro_IDHmut", "GBM: IDH-mut"), ("GBM", "GBM_IDHwt", "GBM: IDH-wt")]
        x = np.arange(len(groups))
        ns = []
        for lab_, grp, _ in groups:
            r = sub[(sub.cohort == "all") & (sub.scheme == "institution") & (sub.model == "fusion_primary") &
                    (sub.tcga_label == lab_) & (sub.group == grp)]
            ns.append(int(r.n.iloc[0]) if len(r) else 0)
        for k, (m, col) in enumerate(((IMG, C_IMG), (REP, C_REP), ("fusion_primary", C_FUS))):
            vals = []
            for lab_, grp, _ in groups:
                r = sub[(sub.cohort == "all") & (sub.scheme == "institution") & (sub.model == m) &
                        (sub.tcga_label == lab_) & (sub.group == grp)]
                vals.append(r.iloc[0] if len(r) else None)
            for i, r in enumerate(vals):
                if r is None:
                    continue
                d.bar(i + (k - 1) * 0.26, r.frac_called_gbm, width=0.24, color=col, alpha=0.85,
                      label=nm(m).split(":")[0] if i == 0 else None)
                d.errorbar(i + (k - 1) * 0.26, r.frac_called_gbm, yerr=[[r.frac_called_gbm - r.lo],
                                                                         [r.hi - r.frac_called_gbm]],
                           fmt="none", ecolor=BLACK, elinewidth=0.5)
        d.set_xticks(x, [g[2].replace(": ", ":\n", 1).replace(", molecular", ",\nmolecular") + f"\n(n = {n_})"
                         for g, n_ in zip(groups, ns)], fontsize=4.8)
        d.set_ylim(0, 1.0)
        d.set_ylabel("Fraction called GBM (p >= 0.5)")
        d.legend(fontsize=5, ncol=3, loc="upper left")
        d.set_title("WHO CNS5-like molecular groups (sites held out)", fontsize=6)
    e = fig.add_subplot(gs[1, 2])
    conf = read(P1 / "phase1_confounding.csv")
    if conf is not None and len(conf):
        rows = []
        for r in conf.itertuples():
            if str(r.test).startswith("site identification"):
                rows.append((f"{r.encoder}: site ID within {r.test.split()[-1]}", r.balanced_accuracy, r.chance,
                             "balanced accuracy"))
            elif "20x" in str(r.test):
                rows.append((f"{r.encoder}: 20x vs 40x within GBM", r.auroc, 0.5, "AUROC"))
            elif "tiles per patient" in str(r.test):
                rows.append((f"{r.encoder}: tissue amount -> GBM", r.auroc, 0.5, "AUROC"))
        y = np.arange(len(rows))
        e.barh(y, [r[1] for r in rows], color=[C_NUIS if "AUROC" == r[3] else SKY for r in rows], height=0.6)
        e.scatter([r[2] for r in rows], y, marker="|", color=BLACK, s=40, lw=0.8, zorder=3, label="Chance")
        e.set_yticks(y, [r[0] for r in rows], fontsize=4.8)
        e.invert_yaxis()
        e.set_xlim(0, 1)
        e.set_xlabel("Balanced accuracy (blue) or AUROC (grey)")
        e.legend(fontsize=5, loc="lower right")
        e.set_title("Shortcut capacity of slide embeddings", fontsize=6)
    letters(fig, [(a, "a"), (b, "b"), (c, "c"), (d, "d"), (e, "e")])
    save(fig, "Fig3_calibration_subgroups_confounding")


# ================================================================================================ Fig. 4
def fig4():
    att_f = RES / "attention_phikon2.npz"
    if not att_f.exists() or OOF is None:
        print("Fig. 4 skipped: attention weights not found")
        return
    import h5py
    import openslide
    A = np.load(att_f, allow_pickle=True)
    att = pd.DataFrame({"patient": A["patient"], "slide": A["slide"], "x": A["x"], "y": A["y"], "a": A["attention"]})
    g = OOF[(OOF.cohort == "x40") & (OOF.scheme == "institution") & (OOF.model == "fusion_primary")]
    tile_dir = Path(os.environ.get("TILE_DIR", ""))
    wsi_list = [l.strip() for l in open(ROOT / "data" / "wsi_list.txt")] if (ROOT / "data" / "wsi_list.txt").exists() else []
    wsi_by_stem = {Path(w).name.split(".")[0]: w for w in wsi_list}
    picks = []
    for cls in (1, 0):
        cand = g[(g.y == cls)].sort_values("p", ascending=(cls == 0))
        for p in cand.patient:
            st = att.loc[att.patient == p, "slide"]
            if len(st) and st.iloc[0] in wsi_by_stem and (tile_dir / f"{st.iloc[0]}.h5").exists():
                picks.append((p, cls))
            if sum(1 for _, c in picks if c == cls) == 2:
                break
    if len(picks) < 2:
        print("Fig. 4 skipped: slides not found")
        return
    fig = figure(183, 46 * len(picks))
    gs = fig.add_gridspec(len(picks), 5, width_ratios=[1.6, 1, 1, 1, 1])
    cmap = LinearSegmentedColormap.from_list("att", ["#FFFFFF", "#FDD49E", "#E34A33", "#7F0000"])
    pairs = []
    for i, (p, cls) in enumerate(picks):
        sub = att[att.patient == p]
        stem = sub.slide.iloc[0]
        sl = openslide.OpenSlide(wsi_by_stem[stem])
        with h5py.File(tile_dir / f"{stem}.h5", "r") as h:
            rs, lvl, tile = int(h.attrs["read_size"]), int(h.attrs["read_level"]), int(h.attrs["tile"])
        ds = sl.level_downsamples[lvl]
        span0 = rs * ds                                            # tile side in level-0 pixels
        th = sl.get_thumbnail((900, 900)).convert("RGB")
        sx = th.size[0] / sl.dimensions[0]
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(th)
        ov = np.zeros((th.size[1], th.size[0], 4))                # attention overlay as one raster layer
        side = max(1, int(round(span0 * sx)))
        for (x, y, v) in zip(sub.x.values, sub.y.values, sub.a.rank(pct=True).values):
            if v > 0.5:                                            # upper half of tiles by attention rank
                x0, y0 = int(x * sx), int(y * sx)
                ov[y0:y0 + side, x0:x0 + side] = (*cmap(v)[:3], 0.6)
        ax.imshow(ov)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        pr = float(g.loc[g.patient == p, "p"].iloc[0])
        ax.set_xlabel(f"{p} | TCGA {'GBM' if cls else 'LGG'} | fused P(GBM) = {pr:.2f}", fontsize=5.5)
        mpp0 = float(sl.properties.get("openslide.mpp-x", 0.25))
        bar = 2000 * sx / mpp0                                   # 2 mm in thumbnail pixels
        ax.plot([th.size[0] - 20 - bar, th.size[0] - 20], [th.size[1] - 20] * 2, color=BLACK, lw=1.2)
        ax.text(th.size[0] - 20 - bar / 2, th.size[1] - 35, "2 mm", ha="center", fontsize=5)
        pairs.append((ax, "abcd"[i]))
        top = sub.sort_values("a", ascending=False).head(4)
        for j, (x, y) in enumerate(zip(top.x.values, top.y.values)):
            axt = fig.add_subplot(gs[i, j + 1])
            im = sl.read_region((int(x), int(y)), lvl, (rs, rs)).convert("RGB")
            if rs != tile:
                im = im.resize((tile, tile))
            axt.imshow(im)
            axt.plot([tile - 12 - 100 / 0.5, tile - 12], [tile - 10] * 2, color=BLACK, lw=1.2)  # 100 um
            axt.set_xticks([]); axt.set_yticks([])
            for s_ in axt.spines.values():
                s_.set_visible(True); s_.set_color(cmap(1.0)); s_.set_linewidth(0.8)
            if i == 0:
                axt.set_title(f"Top-{j + 1} attention tile", fontsize=5.5)
    letters(fig, pairs)
    save(fig, "Fig4_attention")


# ================================================================================================ Fig. 5
def fig5():
    ev = pd.read_csv(PROC / "rtdc_events_gated.csv.gz")
    ev["logE"] = np.log10(ev.emodulus)
    A2, A3s, A5 = read(P2 / "A2_variance_partition.csv"), read(P2 / "A3_shift_functions.csv"), read(P2 / "A5_reproducibility_icc.csv")
    A6, A7 = read(P2 / "A6_precision_vs_cells.csv"), read(P2 / "A7_bag_discrimination.csv")
    fig = figure(183, 165)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 1, 1])
    a = fig.add_subplot(gs[0, :2])
    meas = ev.drop_duplicates("measurement").set_index("measurement")[["date", "line", "cond"]]
    order = meas.assign(c=pd.Categorical(meas.cond, ["Serum", "EGF", "FGFJI"])).sort_values(["c", "line", "date"]).index
    grid = np.linspace(-0.45, 0.55, 300)
    from scipy.stats import gaussian_kde
    for i, m in enumerate(order):
        x = ev.loc[ev.measurement == m, "logE"].values
        k = gaussian_kde(x, bw_method=0.25)(grid)
        k = k / k.max() * 0.9
        col = C_COND[meas.loc[m, "cond"]]
        a.fill_between(grid, i, i + k, color=col, alpha=0.55, lw=0)
        a.plot(grid, i + k, color=col, lw=0.5)
        a.plot([np.median(x)] * 2, [i, i + 0.9], color=BLACK, lw=0.6)
        a.text(0.56, i + 0.3, f"{meas.loc[m, 'line']}  {str(meas.loc[m, 'date'])[2:]}  n={len(x)}", fontsize=4.4, va="center")
    a.set_yticks([]); a.spines["left"].set_visible(False)
    a.set_xlim(-0.45, 0.86)
    conds = [meas.loc[m, "cond"] for m in order]
    for cnd in ("Serum", "EGF", "FGFJI"):
        idx = [i for i, c_ in enumerate(conds) if c_ == cnd]
        if idx:
            a.text(0.83, (min(idx) + max(idx) + 0.9) / 2, cnd, rotation=90, va="center", ha="center", fontsize=6,
                   color=C_COND[cnd], fontweight="bold")
            a.plot([0.805, 0.805], [min(idx) + 0.1, max(idx) + 0.8], color=C_COND[cnd], lw=0.8)
    a.set_xlabel("log$_{10}$ apparent Young's modulus (kPa), single cells")
    a.set_title("RT-DC single-cell stiffness: 3 primary GBM lines x 3 culture states x 3 days (bar: median)",
                fontsize=6, loc="right")
    b = fig.add_subplot(gs[0, 2])
    if A3s is not None:
        for con, col in (("EGF vs Serum", ORANGE), ("FGFJI vs Serum", SKY), ("FGFJI vs EGF", PURPLE)):
            g = A3s[A3s.contrast == con]
            b.fill_between(g.decile, g.lo, g.hi, color=col, alpha=0.18, lw=0)
            b.plot(g.decile, g.shift_log10E, "o-", color=col, ms=2, label=con)
            b.axhline(g.loc[g.decile == 50, "shift_log10E"].iloc[0], color=col, lw=0.4, ls=":")
        b.axhline(0, **REF)
        b.set_xticks([10, 30, 50, 70, 90])
        b.set_xlabel("Decile of the stiffness distribution")
        b.set_ylabel("Shift in log$_{10}$ E (paired within line x day)")
        b.legend(fontsize=5, loc="lower left")
        b.set_title("Shift functions: flat = pure location shift", fontsize=6)
    c = fig.add_subplot(gs[1, :2])
    if A2 is not None:
        d_ = A2[A2.descriptor.isin(list(DESC_NAMES))].set_index("descriptor").loc[
            [k for k in DESC_NAMES if k in set(A2.descriptor)]]
        y = np.arange(len(d_))
        left = np.zeros(len(d_))
        for comp, col, lab in (("seqshare_condition", ORANGE, "Culture state"), ("seqshare_line", GREEN, "Cell line"),
                               ("seqshare_date", SKY, "Measurement day"), ("seqshare_residual", LGREY, "Residual")):
            c.barh(y, d_[comp], left=left, color=col, height=0.65, label=lab)
            left += d_[comp].values
        for yy, r in zip(y, d_.itertuples()):
            stars = "".join(s for s, p in (("c", r.p_perm_condition), ("l", r.p_perm_line), ("d", r.p_perm_date))
                            if p < 0.05)
            if stars:
                c.text(1.01, yy, stars, va="center", fontsize=5, transform=c.get_yaxis_transform())
        c.set_yticks(y, [DESC_NAMES[k] for k in d_.index], fontsize=5.2)
        c.invert_yaxis()
        c.set_xlim(0, 1)
        c.set_xlabel("Share of between-measurement variance (sequential sums of squares; letters: permutation P < 0.05)")
        c.legend(ncol=4, fontsize=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    dd = fig.add_subplot(gs[1, 2])
    if A5 is not None:
        d_ = A5.set_index("descriptor").loc[[k for k in DESC_NAMES if k in set(A5.descriptor)]]
        y = np.arange(len(d_))
        dd.errorbar(d_.icc1, y, xerr=[d_.icc1 - d_.lo, d_.hi - d_.icc1], fmt="o", ms=2.4, color=BLACK, elinewidth=0.6)
        dd.axvline(0, **REF)
        dd.set_yticks(y, [DESC_NAMES[k] for k in d_.index], fontsize=5.2)
        dd.invert_yaxis()
        dd.set_xlim(-1, 1)
        dd.set_xlabel("ICC across days (line x state)")
        dd.set_title("Day-to-day reproducibility", fontsize=6)
    e = fig.add_subplot(gs[2, :2])
    if A6 is not None:
        g6 = A6[A6.assay == "RT-DC"]
        for k, col in (("median", BLACK), ("q90", DGREY), ("iqr", ORANGE), ("upper_tail", VERMIL),
                       ("lower_tail", SKY), ("mad", GREEN), ("bowley", PURPLE)):
            g = g6[g6.descriptor == k].sort_values("n_cells")
            e.plot(g.n_cells, g.reliability, "o-", ms=2, color=col, label=DESC_NAMES[k])
        e.axhline(0.8, **REF)
        logx(e, [10, 30, 100, 300, 1000])
        e.set_ylim(-0.03, 1.03)
        e.set_xlabel("Cells measured per sample")
        e.set_ylabel("Reliability")
        e.legend(ncol=1, fontsize=5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        e.set_title("How many cells make a descriptor a reliable sample-level marker?", fontsize=6)
    f = fig.add_subplot(gs[2, 2])
    if A7 is not None:
        for con, col in (("EGF vs Serum", ORANGE), ("FGFJI vs Serum", SKY), ("FGFJI vs EGF", PURPLE)):
            g = A7[A7.task == con].sort_values("cells_per_bag")
            f.plot(g.cells_per_bag, g.auroc_location, "o-", color=col, ms=2, label=con)
            f.plot(g.cells_per_bag, g.auroc_distribution, "s--", color=col, ms=2, mfc="white", lw=0.7)
            f.plot(g.cells_per_bag, g.auroc_shape_only, "^:", color=col, ms=2, lw=0.7)
        f.axhline(0.5, **REF)
        logx(f, [30, 100, 300, 1000])
        f.set_ylim(0.2, 1.02)
        f.set_xlabel("Cells per bag")
        f.set_ylabel("AUROC (leave-one-day-out)")
        h = [Line2D([], [], color=BLACK, marker="o", ms=2, label="Location (mean, median)"),
             Line2D([], [], color=BLACK, marker="s", mfc="white", ms=2, ls="--", label="Location + shape"),
             Line2D([], [], color=BLACK, marker="^", ms=2, ls=":", label="Shape only")]
        f.legend(handles=h + [Line2D([], [], color=c_, label=l_) for l_, c_ in
                              (("EGF vs Serum", ORANGE), ("FGFJI vs Serum", SKY), ("FGFJI vs EGF", PURPLE))],
                 fontsize=4.4, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2)
        f.set_title("Discriminating culture states", fontsize=6)
    letters(fig, [(a, "a"), (b, "b"), (c, "c"), (dd, "d"), (e, "e"), (f, "f")])
    save(fig, "Fig5_single_cell_mechanics")


# ================================================================================================ Fig. 6
SCOL = {"S0 no association": GREY, "S1 location (IDH-anchored)": BLACK, "S2 spread (LGG IQR x1.25)": ORANGE,
        "S3 stiff tail (10% of LGG cells, 2-fold)": VERMIL, "S4 location + spread": BLUE,
        "S5 redundant with image + report": GREEN, "S2b spread, primary-culture variation": ORANGE,
        "S3b stiff tail, primary-culture variation": VERMIL}
SLS = {k: ("--" if k.startswith(("S2b", "S3b")) else "-") for k in SCOL}


def slabel(s):
    return s.split(" ", 1)[1]


def fig6():
    M, G = read(P2 / "plasmode_main.csv"), read(P2 / "plasmode_grid.csv")
    if M is None:
        print("Fig. 6 skipped: plasmode results not found")
        return
    fig = figure(183, 118)
    gs = fig.add_gridspec(2, 3)
    a = fig.add_subplot(gs[0, 0])
    for s, col in SCOL.items():
        g = M[(M.scenario == s) & (M.cells_per_tumour == 300)].sort_values("n_per_class")
        if not len(g):
            continue
        a.errorbar(g.n_per_class, g.power_M2_vs_M1, yerr=1.96 * g.power_M2_vs_M1_mcse, fmt="o", ls=SLS[s], ms=2,
                   color=col, mfc="white" if SLS[s] == "--" else col, elinewidth=0.5, label=slabel(s))
    a.axhline(0.8, **REF); a.axhline(0.05, color=LGREY, lw=0.5)
    logx(a, [25, 50, 100, 200, 400])
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel("Patients per class"); a.set_ylabel("Power, shape beyond average\n(LRT M2 vs M1, alpha = 0.05)")
    a.set_title("300 cells per tumour", fontsize=6)
    b = fig.add_subplot(gs[0, 1])
    for s, col in SCOL.items():
        g = M[(M.scenario == s) & (M.n_per_class == 100)].sort_values("cells_per_tumour")
        if not len(g):
            continue
        b.errorbar(g.cells_per_tumour, g.power_M2_vs_M1, yerr=1.96 * g.power_M2_vs_M1_mcse, fmt="o", ls=SLS[s], ms=2,
                   color=col, mfc="white" if SLS[s] == "--" else col, elinewidth=0.5)
    b.axhline(0.8, **REF); b.axhline(0.05, color=LGREY, lw=0.5)
    logx(b, [30, 100, 300, 1000])
    b.set_ylim(-0.02, 1.02)
    b.set_xlabel("Cells per tumour"); b.set_ylabel("Power, shape beyond average")
    b.set_title("100 patients per class", fontsize=6)
    c = fig.add_subplot(gs[0, 2])
    g = M[(M.n_per_class == 200) & (M.cells_per_tumour == 300)].set_index("scenario")
    g = g.loc[[k for k in SCOL if k in g.index]]
    y = np.arange(len(g))
    c.barh(y - 0.18, g.delta_M1_vs_M0, height=0.34, color=DGREY, label="+ average stiffness (M1 - M0)")
    c.barh(y + 0.18, g.delta_M2_vs_M0, height=0.34, color=BLUE, label="+ average and shape (M2 - M0)")
    c.errorbar(g.delta_M2_vs_M0, y + 0.18, xerr=[g.delta_M2_vs_M0 - g.delta_M2_vs_M0_lo,
                                                 g.delta_M2_vs_M0_hi - g.delta_M2_vs_M0], fmt="none", ecolor=BLACK,
               elinewidth=0.5)
    c.axvline(0, color=BLACK, lw=0.5)
    c.set_yticks(y, [slabel(s) for s in g.index], fontsize=4.8)
    c.invert_yaxis()
    c.set_xlabel("Gain in cross-validated AUROC")
    c.legend(fontsize=4.6, loc="upper left", bbox_to_anchor=(0.0, -0.22))
    c.set_title(f"200 per class, 300 cells\n(image + report AUROC {g.auc_M0.mean():.3f})", fontsize=6)
    for k, ts in enumerate((0.09, 0.47)):
        ax = fig.add_subplot(gs[1, k])
        if G is None:
            note(ax, "grid not available"); continue
        gg = G[(np.isclose(G.tau_scale, ts)) & (G.n_per_class == 200) & (G.cells_per_tumour == 1000)]
        piv = gg.pivot_table(index="delta", columns="r", values="power_M2_vs_M1")
        im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color="white" if v > 0.6 else BLACK)
        ax.set_xticks(range(piv.shape[1]), [f"{r:g}" for r in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{d:.3f}" for d in piv.index])
        ax.set_xlabel("LGG / GBM spread ratio (IQR)")
        ax.set_ylabel("GBM - LGG location difference (log$_{10}$ E)")
        ax.set_title(f"Power at 200 per class, 1,000 cells\nbetween-tumour s.d. of ln IQR {ts} "
                     f"({'RT-DC' if ts < 0.2 else 'primary-culture AFM'})", fontsize=5.5)
        ax.spines[:].set_visible(False)
        if k == 1:
            cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label("Power (M2 vs M1)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    e = fig.add_subplot(gs[1, 2])
    for s, t, col, mk in (("S0 no association", "M1_vs_M0", GREY, "o"), ("S5 redundant with image + report", "M1_vs_M0",
                                                                         GREEN, "o"),
                          ("S0 no association", "M2_vs_M1", GREY, "s"), ("S1 location (IDH-anchored)", "M2_vs_M1",
                                                                         BLACK, "s"),
                          ("S5 redundant with image + report", "M2_vs_M1", GREEN, "s")):
        g = M[(M.scenario == s) & (M.cells_per_tumour == 300)].sort_values("n_per_class")
        e.errorbar(g.n_per_class, g[f"power_{t}"], yerr=1.96 * g[f"power_{t}_mcse"], fmt=mk + "-", ms=2, color=col,
                   mfc="white" if t == "M2_vs_M1" else col, elinewidth=0.5, lw=0.7,
                   label=f"{s.split(' ', 1)[1].split(' (')[0]}: {t.replace('_vs_', ' vs ')}")
    e.axhline(0.05, color=BLACK, lw=0.5, ls="--")
    logx(e, [25, 50, 100, 200, 400])
    e.set_ylim(0, 0.3)
    e.set_xlabel("Patients per class"); e.set_ylabel("Rejection rate under the null")
    e.legend(fontsize=4.4, loc="upper left", bbox_to_anchor=(0.0, -0.25), ncol=1)
    e.set_title("Test size (300 cells per tumour)", fontsize=6)
    fig.legend(handles=[Line2D([], [], color=c_, marker="o", ms=2, ls=SLS[s], mfc="white" if SLS[s] == "--" else c_,
                               label=slabel(s)) for s, c_ in SCOL.items() if (M.scenario == s).any()],
               loc="upper center", ncol=4, fontsize=5, bbox_to_anchor=(0.5, 1.0))
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    letters(fig, [(a, "a"), (b, "b"), (c, "c"), (fig.axes[3], "d"), (e, "e")])
    save(fig, "Fig6_design_analysis")


# ================================================================================================ Extended Data
def ed1():
    if OOF is None:
        return
    fig = figure(183, 95)
    gs = fig.add_gridspec(2, 4)
    sets = [("Image models", [IMG, "image_phikon2_meanpool", "image_hipt_abmil", "image_hipt_meanpool",
                              "tissue_handcrafted"]),
            ("Report models and covariates", [REP, "report_bert_M1", "report_tfidf_M2", "report_unmasked_M0", "age"])]
    cols = [BLUE, SKY, GREEN, "#7FC97F", GREY]
    pairs = []
    for i, (c, s) in enumerate([("all", "patient"), ("all", "institution"), ("x40", "patient"), ("x40", "institution")]):
        for j, (title, models) in enumerate(sets):
            ax = fig.add_subplot(gs[j, i])
            for m, col in zip(models, cols):
                g = OOF[(OOF.cohort == c) & (OOF.scheme == s) & (OOF.model == m)]
                if not len(g):
                    continue
                fpr, tpr, _ = roc_curve(g.y, g.p)
                r = met(m, c, s)
                ax.plot(fpr, tpr, color=col, lw=0.7, label=f"{nm(m).split(': ')[-1][:26]} {r.auroc:.3f}")
            ax.plot([0, 1], [0, 1], **REF)
            ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend(fontsize=4.2, loc="lower right", handlelength=1)
            ax.set_title(f"{title}\n{'40x only' if c == 'x40' else 'matched'}, "
                         f"{'sites held out' if s == 'institution' else 'random folds'}", fontsize=5.2)
            if i == 0:
                ax.set_ylabel("Sensitivity")
            if j == 1:
                ax.set_xlabel("1 - specificity")
            pairs.append((ax, "abcdefgh"[len(pairs)]))
    letters(fig, pairs)
    save(fig, "ED1_roc_all_models")


def ed2():
    fv = read(P1 / "phase1_fold_variability.csv")
    conf = read(P1 / "phase1_confounding.csv")
    fig = figure(183, 70)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1, 1])
    a = fig.add_subplot(gs[0, 0])
    if fv is not None and len(fv):
        g = fv[fv.scheme == "institution"].copy()
        g["lab"] = g.source + " | " + g.cohort
        y = np.arange(len(g))
        a.errorbar(g.fold_auroc_mean, y, xerr=g.fold_auroc_sd, fmt="o", ms=2.4, color=C_INST, elinewidth=0.6)
        a.scatter(g.fold_auroc_min, y, marker="|", color=DGREY, s=14)
        a.scatter(g.fold_auroc_max, y, marker="|", color=DGREY, s=14)
        a.set_yticks(y, g.lab, fontsize=4.8); a.invert_yaxis()
        a.set_xlabel("Per-fold AUROC, mean +/- s.d. (bars: range)")
        a.set_title("Fold-to-fold variability, sites held out", fontsize=6)
    else:
        note(a, "fold logs not available")
    b = fig.add_subplot(gs[0, 1])
    st = read(RES / "slide_tiles_phikon2.csv")
    if st is not None and OOF is not None:
        pts = st.groupby("patient").n_tiles.sum()
        mm = OOF[(OOF.model == "fusion_primary") & (OOF.cohort == "all") & (OOF.scheme == "institution")]
        pts = pts[pts.index.isin(mm.patient)]
        y = cohort.loc[pts.index, "y_gbm"]
        bins = np.logspace(np.log10(max(pts.min(), 50)), np.log10(pts.max()), 30)
        b.hist(pts[y == 1], bins=bins, color=C_GBM, alpha=0.6, label="GBM")
        b.hist(pts[y == 0], bins=bins, color=C_LGG, alpha=0.6, label="LGG")
        b.set_xscale("log"); b.set_xlabel("Tissue tiles per patient (0.5 µm/px)"); b.set_ylabel("Patients")
        b.legend(fontsize=5)
        b.set_title("Tissue amount by class", fontsize=6)
    else:
        note(b, "tile counts not available")
    c = fig.add_subplot(gs[0, 2])
    if conf is not None and conf.test.str.contains("tertile").any():
        g = conf[conf.test.str.contains("tertile") & (conf.encoder == "phikon2")]
        x = np.arange(len(g))
        c.errorbar(x, g.auroc, yerr=[g.auroc - g.lo, g.hi - g.auroc], fmt="o", color=C_FUS, ms=3, elinewidth=0.6)
        c.set_xticks(x, [t.split()[-1] for t in g.test]); c.set_xlim(-0.5, len(g) - 0.5)
        c.set_ylim(0.5, 1.0); c.set_xlabel("Tertile of tissue amount"); c.set_ylabel("Fusion AUROC (sites held out)")
        c.set_title("Performance does not depend on tissue amount", fontsize=6)
    else:
        note(c, "not available")
    letters(fig, [(a, "a"), (b, "b"), (c, "c")])
    save(fig, "ED2_variability_tissue")


def ed3():
    fig = figure(183, 80)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
    ev = pd.read_csv(PROC / "rtdc_events_gated.csv.gz")
    meas = ev.drop_duplicates("measurement").set_index("measurement")[["date", "line", "cond"]].sort_index()
    A4 = read(P2 / "A4_shape_distance_tests.csv")
    pairs = []
    for k, mode in enumerate(("location", "location_scale")):
        ax = fig.add_subplot(gs[0, k])
        Dm = read(P2 / f"A4_distance_matrix_{mode}.csv")
        if Dm is None:
            note(ax, "not available"); continue
        Dm = Dm.set_index(Dm.columns[0])
        o = meas.assign(c=pd.Categorical(meas.cond, ["Serum", "EGF", "FGFJI"])).sort_values(["date", "line", "c"]).index
        M_ = Dm.loc[o, o].values
        im = ax.imshow(M_, cmap="viridis_r")
        ax.set_xticks(range(len(o)), [f"{meas.loc[m, 'line']} {meas.loc[m, 'cond']} {str(meas.loc[m, 'date'])[2:]}"
                                      for m in o], rotation=90, fontsize=3.8)
        ax.set_yticks(range(len(o)), [f"{meas.loc[m, 'line']} {meas.loc[m, 'cond']} {str(meas.loc[m, 'date'])[2:]}"
                                      for m in o], fontsize=3.8)
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
        cb.ax.tick_params(labelsize=4.5)
        cb.set_label("W$_1$ distance", fontsize=5)
        t = A4[A4.distance == mode] if A4 is not None else None
        sub = "; ".join(f"{r.factor}: ratio {r.between_over_within:.2f}, P = {r.p_perm:.3f}" for r in t.itertuples()) \
            if t is not None else ""
        ax.set_title(("Median removed" if mode == "location" else "Median and IQR removed") + f"\n{sub}", fontsize=4.8)
        pairs.append((ax, "ab"[k]))
    c = fig.add_subplot(gs[0, 2])
    afm = read(PROC / "afm_events.csv")
    if afm is not None:
        from scipy.stats import gaussian_kde
        grid = np.linspace(-0.5, 0.7, 300)
        for ln, col in (("LN229", ORANGE), ("U87", PURPLE)):
            x = afm.loc[afm.line == ln, "logE"].values
            c.plot(grid, gaussian_kde(x)(grid), color=col, label=f"{ln} (n = {len(x)})")
            c.plot(grid + np.median(x) * 0, gaussian_kde(x - np.median(x) + 0.0)(grid), color=col, ls=":", lw=0.7)
        c.set_xlabel("log$_{10}$ E (kPa); dotted: median-aligned")
        c.set_ylabel("Density")
        c.legend(fontsize=5)
        c.set_title("AFM, 2D culture (no replicate structure)", fontsize=6)
        pairs.append((c, "c"))
    letters(fig, pairs)
    save(fig, "ED3_shape_distances_afm")


def ed4():
    A3s, A3c, A6 = read(P2 / "A3_shift_functions.csv"), read(P2 / "A3_shape_contrasts.csv"), read(P2 / "A6_precision_vs_cells.csv")
    fig = figure(183, 70)
    gs = fig.add_gridspec(1, 3)
    a = fig.add_subplot(gs[0, 0])
    if A3s is not None:
        for con, col in (("X04 vs X01", GREEN), ("X08 vs X01", BLUE), ("X08 vs X04", VERMIL)):
            g = A3s[A3s.contrast == con]
            a.fill_between(g.decile, g.lo, g.hi, color=col, alpha=0.15, lw=0)
            a.plot(g.decile, g.shift_log10E, "o-", ms=2, color=col, label=con)
        a.axhline(0, **REF)
        a.set_xlabel("Decile"); a.set_ylabel("Shift in log$_{10}$ E (paired within state x day)")
        a.legend(fontsize=5)
        a.set_title("Cell lines (tumours of origin)", fontsize=6)
    b = fig.add_subplot(gs[0, 1])
    if A3c is not None:
        q = A3c[A3c.quantity.isin(["upper_minus_median", "lower_minus_median", "log_iqr_ratio"])].copy()
        q["lab"] = q.contrast + " | " + q.quantity.map({"upper_minus_median": "upper tail", "lower_minus_median":
                                                        "lower tail", "log_iqr_ratio": "log IQR ratio"})
        y = np.arange(len(q))
        b.errorbar(q["mean"], y, xerr=[q["mean"] - q.lo, q.hi - q["mean"]], fmt="o", ms=2, color=BLACK, elinewidth=0.5)
        b.axvline(0, **REF)
        b.set_yticks(y, q.lab, fontsize=4.2); b.invert_yaxis()
        b.set_xlabel("Shape contrast (log$_{10}$ units, 95% CI)")
        b.set_title("Shape contrasts beyond location", fontsize=6)
    c = fig.add_subplot(gs[0, 2])
    if A6 is not None:
        for k, col in (("median", BLACK), ("iqr", ORANGE), ("upper_tail", VERMIL), ("bowley", PURPLE)):
            for assay, ls in (("RT-DC", "-"), ("AFM_U87", "--")):
                g = A6[(A6.assay == assay) & (A6.descriptor == k)].sort_values("n_cells")
                c.plot(g.n_cells, g.sampling_sd, ls=ls, color=col, marker="o", ms=1.6, lw=0.7,
                       label=f"{DESC_NAMES[k]}" if assay == "RT-DC" else None)
        c.set_xscale("log"); c.set_yscale("log")
        c.set_xlabel("Cells per sample"); c.set_ylabel("Sampling s.d. of the descriptor")
        c.legend(fontsize=4.8, title="solid RT-DC, dashed AFM U87", title_fontsize=4.8)
        c.set_title("Sampling precision", fontsize=6)
    letters(fig, [(a, "a"), (b, "b"), (c, "c")])
    save(fig, "ED4_lines_precision")


def ed5():
    M = read(P2 / "plasmode_main.csv")
    if M is None:
        return
    scen = [k for k in SCOL if (M.scenario == k).any()]
    fig = figure(183, 105)
    gs = fig.add_gridspec(2, len(scen))
    pairs = []
    for j, s in enumerate(scen):
        for i, t in enumerate(("M1_vs_M0", "M2_vs_M1")):
            ax = fig.add_subplot(gs[i, j])
            piv = M[M.scenario == s].pivot_table(index="n_per_class", columns="cells_per_tumour", values=f"power_{t}")
            ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
            for a_ in range(piv.shape[0]):
                for b_ in range(piv.shape[1]):
                    v = piv.values[a_, b_]
                    ax.text(b_, a_, f"{v:.2f}", ha="center", va="center", fontsize=4.2,
                            color="white" if v > 0.6 else BLACK)
            ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns], fontsize=4.6)
            ax.set_yticks(range(piv.shape[0]), [str(c) for c in piv.index], fontsize=4.6)
            ax.spines[:].set_visible(False)
            if i == 1:
                ax.set_xlabel("Cells per tumour", fontsize=5)
            if j == 0:
                ax.set_ylabel(("Average stiffness\n(M1 vs M0)" if t == "M1_vs_M0" else "Shape beyond average\n(M2 vs M1)")
                              + "\npatients per class", fontsize=5)
            if i == 0:
                ax.set_title(slabel(s).replace(", ", ",\n").replace(" (", "\n("), fontsize=4.6)
            pairs.append((ax, None))
    save(fig, "ED5_plasmode_power_tables")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "ed1": ed1, "ed2": ed2,
        "ed3": ed3, "ed4": ed4, "ed5": ed5}
if __name__ == "__main__":
    want = sys.argv[1:] or list(FIGS)
    for k in want:
        try:
            FIGS[k]()
        except Exception as ex:                     # one failing figure must not stop the others
            import traceback
            traceback.print_exc()
            print(f"FIGURE_FAILED {k}: {ex}", flush=True)
    print("FIGURES_DONE")
