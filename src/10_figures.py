"""Step 10 - figures (PNG, light theme) from the result tables."""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import FIG, PROC, RAW, RES  # noqa: E402

# palette (validated categorical order, light surface) and chart ink
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


# ---------------------------------------------------------------- 1 inventory
inv = pd.read_csv(RES / "inventory.csv")
fig, ax = plt.subplots(figsize=(6.4, 2.6))
yy = np.arange(len(inv))[::-1]
ax.barh(yy, inv.GBM, height=0.5, color=ORANGE, label="GBM")
ax.barh(yy, inv.LGG, left=inv.GBM + 4, height=0.5, color=BLUE, label="Lower-grade glioma")
for y, (_, r) in zip(yy, inv.iterrows()):
    ax.text(r.GBM + r.LGG + 14, y, f"{r.patients:,}" if r.patients else "0  (no public source)", va="center", color=INK2)
ax.set_yticks(yy, inv.modality)
ax.set_xlabel("Patients")
ax.set_xlim(0, 1350)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", ncol=2)
ax.set_title("Which TCGA glioma patients have which input")
save(fig, "fig1_inventory.png")

# ---------------------------------------------------------------- 2 acquisition confounding
sl = pd.read_csv(PROC / "slides.csv")
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), gridspec_kw={"width_ratios": [2.4, 1, 1]})
ct = pd.crosstab(sl.scan_year.fillna(0).astype(int), sl.project)
ct = ct[ct.index > 0]
x = np.arange(len(ct))
w = 0.36
axes[0].bar(x - w / 2 - 0.01, ct["TCGA-GBM"], w, color=ORANGE, label="GBM")
axes[0].bar(x + w / 2 + 0.01, ct["TCGA-LGG"], w, color=BLUE, label="LGG")
axes[0].set_xticks(x, ct.index)
axes[0].set_ylabel("Slides")
axes[0].set_title("Scan year")
axes[0].legend(loc="upper left")
for ax, col, title in [(axes[1], "appmag", "Scan magnification"), (axes[2], "jpeg_q", "JPEG quality")]:
    c = pd.crosstab(sl[col], sl.project)
    xx = np.arange(len(c))
    ax.bar(xx - w / 2 - 0.01, c["TCGA-GBM"], w, color=ORANGE)
    ax.bar(xx + w / 2 + 0.01, c["TCGA-LGG"], w, color=BLUE)
    ax.set_xticks(xx, [f"{int(v)}x" if col == "appmag" else f"Q={int(v)}" for v in c.index])
    ax.set_title(title)
fig.suptitle("Acquisition differs by class: every 20x-scanned slide is a GBM", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
save(fig, "fig2_acquisition_confounding.png")

# ---------------------------------------------------------------- 3 AUROC overview (dot + CI)
rm = pd.read_csv(RES / "report_metrics.csv")
im = pd.read_csv(RES / "image_metrics.csv")
fm = pd.read_csv(RES / "fusion_metrics.csv")
fm = fm[fm.n > 0]
rows = []
lab = {"tfidf_M0": "Report, unmasked (M0)", "tfidf_M1": "Report, masked M1", "tfidf_M2": "Report, masked M2",
       "age": "Age only", "img_meanpool": "Image, HIPT mean-pool", "img_abmil": "Image, attention MIL",
       "handcrafted": "Image, hand-crafted", "acq_meta": "Acquisition metadata only", "stain_color": "Stain colour & sharpness only"}
for _, r in pd.concat([rm[rm.model.isin(lab)], im[im.model.isin(lab)]]).iterrows():
    rows.append({"label": lab[r.model], "cohort": r.cohort, "scheme": r.scheme, "auroc": r.auroc, "lo": r.auroc_lo,
                 "hi": r.auroc_hi, "src": "report" if r.model.startswith(("tfidf", "age")) else
                 ("nuis" if r.model in ("acq_meta", "stain_color") else "image")})
for _, r in fm[fm.model.isin(["late_image+report_M2", "early_image+report_M2"])].iterrows():
    rows.append({"label": {"late_image+report_M2": "Fusion: image + report M2 (late)",
                           "early_image+report_M2": "Fusion: image + report M2 (early)"}[r.model],
                 "cohort": r.cohort, "scheme": r.scheme, "auroc": r.auroc, "lo": r.auroc_lo, "hi": r.auroc_hi,
                 "src": "fusion", "matched": True})
T = pd.DataFrame(rows)
order = ["Report, unmasked (M0)", "Report, masked M1", "Report, masked M2", "Age only", "Image, HIPT mean-pool",
         "Image, attention MIL", "Image, hand-crafted", "Fusion: image + report M2 (late)",
         "Fusion: image + report M2 (early)", "Acquisition metadata only", "Stain colour & sharpness only"]
colors = {"report": VIOLET, "image": AQUA, "fusion": BLUE, "nuis": MUTED}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, cohort, title in [(axes[0], "all", "All slides"), (axes[1], "x40", "40x-scanned slides only")]:
    for i, name in enumerate(order[::-1]):
        for scheme, dy, mk in [("patient", 0.17, "o"), ("institution", -0.17, "D")]:
            r = T[(T.label == name) & (T.cohort == cohort) & (T.scheme == scheme)]
            if r.empty:
                continue
            r = r.iloc[0]
            c = colors[r.src]
            ax.plot([r.lo, r.hi], [i + dy] * 2, color=c, lw=2, solid_capstyle="round")
            ax.plot(r.auroc, i + dy, mk, ms=6.5 if mk == "o" else 5.5, color=c if scheme == "patient" else SURF,
                    mec=c, mew=1.8)
    ax.axvline(0.5, color=AXIS, lw=1)
    ax.set_xlim(0.3, 1.005)
    ax.set_title(title)
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(order)), order[::-1])
h1 = plt.Line2D([], [], marker="o", ls="", color=INK2, label="Random patient folds")
h2 = plt.Line2D([], [], marker="D", ls="", color=SURF, mec=INK2, mew=1.8, label="Whole institutions held out")
fig.legend(handles=[h1, h2], loc="lower center", ncol=2, bbox_to_anchor=(0.6, -0.06), fontsize=9)
fig.suptitle("GBM vs lower-grade glioma: every model, two cohorts, two validation schemes", x=0.01, ha="left",
             fontweight="bold")
fig.text(0.01, -0.09, "Fusion rows are evaluated on the matched image+report cohort; branch rows on each branch's full cohort.",
         color=MUTED, fontsize=8)
fig.tight_layout()
save(fig, "fig3_auroc_overview.png")

# ---------------------------------------------------------------- 4 masking
fig, ax = plt.subplots(figsize=(5.2, 3.0))
lv = ["tfidf_M0", "tfidf_M1", "tfidf_M2"]
for scheme, c, mk in [("patient", VIOLET, "o"), ("institution", ORANGE, "D")]:
    r = rm[(rm.cohort == "all") & (rm.scheme == scheme)].set_index("model").loc[lv]
    ax.plot(range(3), r.auroc, color=c, lw=2, marker=mk, ms=6, label={"patient": "Random patient folds",
                                                                      "institution": "Institutions held out"}[scheme])
    ax.fill_between(range(3), r.auroc_lo, r.auroc_hi, color=c, alpha=0.1, lw=0)
    ax.text(2.08, r.auroc.iloc[-1], f"{r.auroc.iloc[-1]:.3f}", va="center", color=INK2)
age = rm[(rm.cohort == "all") & (rm.scheme == "patient") & (rm.model == "age")].auroc.iloc[0]
ax.axhline(age, color=MUTED, lw=1)
ax.text(2.08, age - 0.009, f"age only {age:.3f}", va="center", color=MUTED, fontsize=8)
ax.set_xticks(range(3), ["M0 unmasked", "M1 no diagnosis /\ngrade / molecular", "M2 also no grading\ncriteria"])
ax.set_ylabel("AUROC")
ax.set_ylim(0.75, 1.0)
ax.set_xlim(-0.2, 2.55)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2)
ax.set_title("Report branch: how much survives masking")
save(fig, "fig4_report_masking.png")

# ---------------------------------------------------------------- 5 mechanics
ev = pd.read_csv(PROC / "rtdc_events_gated.csv.gz")
pm = pd.read_csv(RES / "mechanics_per_measurement.csv")
bag = pd.read_csv(RES / "mechanics_bag_classification.csv")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.3, 1.1, 1.2]})
cc = {"Serum": BLUE, "EGF": ORANGE, "FGFJI": AQUA}
for cond, c in cc.items():
    v = np.sort(np.log10(ev[ev.cond == cond].emodulus.values))
    axes[0].plot(10 ** v, np.linspace(0, 1, len(v)), color=c, lw=2, label=cond)
axes[0].set_xscale("log")
axes[0].set_xticks([0.4, 0.6, 1.0, 1.5, 2.5], ["0.4", "0.6", "1.0", "1.5", "2.5"])
axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_xlim(0.35, 3)
axes[0].set_xlabel("Young's modulus (kPa), RT-DC")
axes[0].set_ylabel("Cumulative fraction of cells")
axes[0].legend(loc="lower right")
axes[0].set_title("Single cells, 3 GBM lines pooled")
lines = ["X01", "X04", "X08"]
for j, cond in enumerate(cc):
    d = pm[pm.cond == cond]
    xs = np.array([lines.index(l) for l in d.line]) + (j - 1) * 0.22
    axes[1].plot(xs, 10 ** d.logE_median, "o", ms=6, color=cc[cond], mec=SURF, mew=1.5, label=cond)
axes[1].set_xticks(range(3), lines)
axes[1].set_ylabel("Median modulus per measurement (kPa)")
axes[1].set_title("Per measurement (3 dates; colours as left)")
tk = bag[(bag.task == "EGF_vs_FGFJI")]
fl = {"median_E_only": ("Median E", MUTED), "E_distribution": ("E distribution", VIOLET),
      "E_distribution+shape": ("E distribution + cell shape", GREEN)}
for f, (lbl, c) in fl.items():
    r = tk[tk.features == f].sort_values("bag_cells")
    axes[2].plot(r.bag_cells, r.bag_auroc, marker="o", color=c, lw=2, ms=6, label=lbl)
axes[2].set_xscale("log")
axes[2].set_xticks([50, 100, 300], ["50", "100", "300"])
axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[2].set_xlabel("Cells per sample")
axes[2].set_ylabel("AUROC, held-out date")
axes[2].set_ylim(0.5, 1.0)
axes[2].legend(loc="lower left", fontsize=8)
axes[2].set_title("EGF vs FGFJI from samples of cells")
fig.tight_layout()
save(fig, "fig5_mechanics.png")

# ---------------------------------------------------------------- 6 subgroups
sg = pd.read_csv(RES / "fusion_oof.csv")
co = pd.read_csv(PROC / "cohort.csv").set_index("patient")
d = sg[(sg.model == "late_image+report_M2") & (sg.cohort == "all") & (sg.scheme == "institution")].join(co, on="patient")
groups = [("GBM", "GBM_IDHwt", "GBM, IDH-wt"), ("GBM", "Astro_IDHmut", "GBM label, IDH-mutant"),
          ("LGG", "GBM_IDHwt", "LGG label, molecular GBM*"), ("LGG", "IDHwt_other", "LGG, IDH-wt other"),
          ("LGG", "Astro_IDHmut", "LGG, astro IDH-mut"), ("LGG", "Oligo_IDHmut_codel", "LGG, oligo 1p/19q")]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
rng = np.random.default_rng(0)
for i, (labl, cls, name) in enumerate(groups):
    g = d[(d.y_gbm == (labl == "GBM")) & (d.who2021_like == cls)]
    c = ORANGE if labl == "GBM" else BLUE
    ax.plot(g.p, i + rng.uniform(-0.18, 0.18, len(g)), "o", ms=3.5, color=c, alpha=0.55, mec="none")
    if len(g):
        ax.plot([g.p.median()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.text(1.02, i, f"n={len(g)}", va="center", color=INK2, fontsize=8)
ax.set_yticks(range(len(groups)), [g[2] for g in groups])
ax.invert_yaxis()
ax.axvline(0.5, color=AXIS, lw=1)
ax.set_xlim(0, 1.1)
ax.set_xlabel("Fused P(GBM), institutions held out")
ax.grid(axis="y", visible=False)
ax.set_title("Where the fused model puts each molecular group")
fig.text(0.01, -0.04, "*IDH-wild-type with TERT-promoter mutation or +7/-10: glioblastoma under WHO 2021. Bar = median.",
         color=MUTED, fontsize=8)
save(fig, "fig6_who2021_subgroups.png")

# ---------------------------------------------------------------- 7 attention tiles
att_f = RES / "abmil_attention_full.csv"
if att_f.exists():
    att = pd.read_csv(att_f)
    att = att[att.tile.str.contains("x20")]
    pick = []
    for lab_v in [1, 0]:
        pats = co[(co.y_gbm == lab_v)].index.intersection(att.patient.unique())
        sub = sg[(sg.model == "image_meanpool") & (sg.cohort == "all") & (sg.scheme == "patient")].set_index("patient").p
        pats = sorted(pats, key=lambda p: -sub.get(p, 0) if lab_v == 1 else sub.get(p, 1))[:3]
        for p in pats:
            t = att[att.patient == p].sort_values("attention", ascending=False).tile.iloc[0]
            pick.append((lab_v, p, t))
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.3))
    for ax, (lab_v, p, t) in zip(axes.ravel(), pick):
        ax.imshow(Image.open(RAW / "tiles" / t))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{'GBM' if lab_v else 'LGG'}  {p}", fontsize=8, fontweight="normal", color=INK2)
    fig.suptitle("Highest-attention 20x tile in confidently classified patients (256 µm field)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_attention_tiles.png")

# ---------------------------------------------------------------- 8 power
pw = RES / "power_matched_cohort.csv"
if pw.exists():
    P = pd.read_csv(pw)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    dc = {0.02: MUTED, 0.03: VIOLET, 0.05: BLUE, 0.08: AQUA}
    for ax, a0 in zip(axes, [0.85, 0.90, 0.95]):
        for dl, c in dc.items():
            r = P[(P.auroc_image_report == a0) & (P.delta_auroc == dl)]
            if len(r):
                ax.plot(r.n_per_class, r.power, marker="o", ms=5, lw=2, color=c, label=f"+{dl:.2f}")
        ax.axhline(0.8, color=AXIS, lw=1)
        ax.set_xscale("log")
        ax.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"Image + report AUROC {a0:.2f}")
        ax.set_xlabel("Matched patients per class")
    axes[0].set_ylabel("Power (DeLong, alpha 0.05)")
    h, l = axes[0].get_legend_handles_labels()
    axes[2].legend(h, l, title="AUROC gain from mechanics", loc="lower right", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    save(fig, "fig8_power.png")
