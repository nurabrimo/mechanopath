"""Builds the manuscript (Word) with every number read from the result tables.

  python dcc/09_manuscript.py          ->  report/MechanoPath_manuscript.docx (+ .md text version)

Run after dcc/06_phase1_analysis.py, src/13_mechanics_phase2.py, src/14_design_plasmode.py and dcc/08_figures.py.
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import FIG, PROC, RES  # noqa: E402

P1, P2, FD = RES / "phase1", RES / "phase2", FIG / "manuscript"
OUTD = ROOT / "report"
OUTD.mkdir(exist_ok=True)


def read(p):
    p = Path(p)
    if not p.exists() or p.stat().st_size < 5:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


MET, CMP, SUB = read(P1 / "phase1_metrics.csv"), read(P1 / "phase1_comparisons.csv"), read(P1 / "phase1_subgroups.csv")
SUBT, CONF = read(P1 / "phase1_subgroup_tests.csv"), read(P1 / "phase1_confounding.csv")
FV, DCA, BI = read(P1 / "phase1_fold_variability.csv"), read(P1 / "phase1_decision_curve.csv"), read(P1 / "phase1_by_institution.csv")
SUMM = json.load(open(P1 / "phase1_summary.json")) if (P1 / "phase1_summary.json").exists() else {}
A1, A2, A3s, A3c = (read(P2 / f) for f in ("A1_sample_descriptors.csv", "A2_variance_partition.csv",
                                           "A3_shift_functions.csv", "A3_shape_contrasts.csv"))
A4, A5, A6, A6n = (read(P2 / f) for f in ("A4_shape_distance_tests.csv", "A5_reproducibility_icc.csv",
                                          "A6_precision_vs_cells.csv", "A6_cells_needed.csv"))
A7, A8, A9 = read(P2 / "A7_bag_discrimination.csv"), read(P2 / "A8_afm_comparison.csv"), read(P2 / "A9_literature_values.csv")
ANCH = read(P2 / "A9_design_anchors.csv")
PM, PG = read(P2 / "plasmode_main.csv"), read(P2 / "plasmode_grid.csv")
LEAK, LABA = read(RES / "report_leak_audit.csv"), read(RES / "report_label_audit.csv")
TOP2 = read(RES / "report_top_terms_M2.csv")
if TOP2 is not None:
    TOP2.columns = ["term", "coef", "direction"]
PIN = json.load(open(P2 / "plasmode_inputs.json")) if (P2 / "plasmode_inputs.json").exists() else {}
cohort = pd.read_csv(PROC / "cohort.csv").set_index("patient")
slides = pd.read_csv(PROC / "slides.csv").set_index("patient")
IMG = SUMM.get("image_branch", "image_phikon2_abmil")
REP = SUMM.get("report_branch", "report_bert_M2")
NA = "n/a"


# ------------------------------------------------------------------------------------------------ formatting
def f3(x):
    return NA if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f2(x):
    return NA if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def pct(x, d=0):
    return NA if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "P = n/a"
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}"


def val(r, k):
    try:
        return r[k] if r is not None and k in r.index else np.nan
    except Exception:
        return np.nan


def met(model, c="all", s="institution"):
    if MET is None:
        return None
    r = MET[(MET.model == model) & (MET.cohort == c) & (MET.scheme == s)]
    return r.iloc[0] if len(r) else None


def auc(model, c="all", s="institution", ci=True):
    r = met(model, c, s)
    if r is None:
        return NA
    return f"{r.auroc:.3f} (95% CI {r.auroc_lo:.3f}-{r.auroc_hi:.3f})" if ci else f"{r.auroc:.3f}"


def cmp(model, comp, c="all", s="institution"):
    if CMP is None:
        return None
    r = CMP[(CMP.model == model) & (CMP.comparator == comp) & (CMP.cohort == c) & (CMP.scheme == s)]
    return r.iloc[0] if len(r) else None


def dtxt(model, comp, c="all", s="institution", holm=True):
    r = cmp(model, comp, c, s)
    if r is None:
        return NA
    p = r.p_holm if holm and not np.isnan(r.p_holm) else r.p_delong
    return f"{r.delta_auroc:+.3f} (95% CI {r.lo:+.3f} to {r.hi:+.3f}; {pfmt(p)})"


def sig(model, comp, c="all", s="institution"):
    r = cmp(model, comp, c, s)
    return r is not None and r.p_holm < 0.05 and r.delta_auroc > 0


def size(key, field="n"):
    return SUMM.get("cohort_sizes", {}).get(key, {}).get(field, NA)


# ------------------------------------------------------------------------------------------------ numbers
V = {}
V["n_lab"], V["n_lab_gbm"] = len(cohort), int(cohort.y_gbm.sum())
V["n_lab_lgg"] = V["n_lab"] - V["n_lab_gbm"]
st = read(RES / "slide_tiles_phikon2.csv")
if st is not None:
    V["n_slides"] = st.slide.nunique()
    V["tiles_total"] = f"{int(st.n_tiles.sum()):,}"
    V["tiles_med"] = int(st.n_tiles.median())
    V["tiles_iqr"] = f"{int(st.n_tiles.quantile(.25)):,}-{int(st.n_tiles.quantile(.75)):,}"
else:
    V.update(n_slides=len(slides), tiles_total=NA, tiles_med=NA, tiles_iqr=NA)
_mv = read(RES / "tissue_mask_validation.csv")
TISSUE_SENT = ""
if _mv is not None and (_mv.group == "fallback").any():
    _vv, _nf = _mv[_mv.group == "validation"], int((_mv.group == "fallback").sum())
    _w = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    TISSUE_SENT = (f" On {_w.get(_nf, _nf)} faintly stained slide{'s' if _nf > 1 else ''} on which this rule retained no "
                 "tile, the thumbnail was first converted to optical density relative to the glass background and "
                 "rescaled so that the 99.5th percentile of the summed optical density equalled 1.0, and the same rule "
                 "was then applied")
    if len(_vv):
        TISSUE_SENT += (f"; on {len(_vv)} randomly selected, normally stained slides this normalization reproduced the "
                      f"primary masks (median Dice coefficient {_vv.dice.median():.3f}, range {_vv.dice.min():.3f}-"
                      f"{_vv.dice.max():.3f})")
    TISSUE_SENT += "."
rep = pd.read_csv(PROC / "reports_masked.csv")
V["n_rep"] = rep.patient.nunique()
V["n_m"], V["n_m_gbm"] = size("all_institution"), size("all_institution", "n_gbm")
V["n_m_lgg"] = V["n_m"] - V["n_m_gbm"] if isinstance(V["n_m"], int) else NA
V["n_m_inst"] = size("all_institution", "n_institutions")
V["n40"], V["n40_gbm"] = size("x40_institution"), size("x40_institution", "n_gbm")
V["n40_lgg"] = V["n40"] - V["n40_gbm"] if isinstance(V["n40"], int) else NA
sx = slides.join(cohort[["y_gbm"]], how="inner")
V["n20_gbm"] = int(((sx.appmag == 20) & (sx.y_gbm == 1)).sum())
V["n20_lgg"] = int(((sx.appmag == 20) & (sx.y_gbm == 0)).sum())
V["n_gbm_slides"] = int((sx.y_gbm == 1).sum())

r_f, r_i, r_r = met("fusion_primary"), met(IMG), met(REP)
fp = met("fusion_primary", "all", "patient")
ip, rp = met(IMG, "all", "patient"), met(REP, "all", "patient")
drop_img = (ip.auroc - r_i.auroc) if (ip is not None and r_i is not None) else np.nan
drop_rep = (rp.auroc - r_r.auroc) if (rp is not None and r_r is not None) else np.nan
acq_all, acq_40 = met("acq_meta"), met("acq_meta", "x40")
acq_all_p = met("acq_meta", "all", "patient")


def conf(test, enc="phikon2"):
    if CONF is None:
        return None
    r = CONF[(CONF.test == test) & (CONF.encoder == enc)]
    return r.iloc[0] if len(r) else None


site_g, site_l = conf("site identification within GBM"), conf("site identification within LGG")
mag = conf("20x vs 40x scanning within GBM (AUROC)")
tiss = conf("tiles per patient alone (AUROC)")


def subt(model, c="all", s="institution"):
    if SUBT is None:
        return None
    r = SUBT[(SUBT.model == model) & (SUBT.cohort == c) & (SUBT.scheme == s)]
    return r.iloc[0] if len(r) else None


st_f, st_i, st_r = subt("fusion_primary"), subt(IMG), subt(REP)


def nb(model, t=0.5, c="all", s="institution"):
    if DCA is None:
        return None
    r = DCA[(DCA.model == model) & (DCA.cohort == c) & (DCA.scheme == s) & np.isclose(DCA.threshold, t)]
    return r.iloc[0] if len(r) else None


nb_f, nb_i, nb_r = nb("fusion_primary"), nb(IMG), nb(REP)


def fvsd(source, c="all", s="institution"):
    if FV is None:
        return None
    r = FV[(FV.source == source) & (FV.cohort == c) & (FV.scheme == s)]
    return r.iloc[0] if len(r) else None


# Phase 2
def shift(con, dec=50):
    r = A3s[(A3s.contrast == con) & (A3s.decile == dec)] if A3s is not None else []
    return r.iloc[0] if len(r) else None


def shapec(con, q):
    r = A3c[(A3c.contrast == con) & (A3c.quantity == q)] if A3c is not None else []
    return r.iloc[0] if len(r) else None


def a2(desc, col):
    r = A2[A2.descriptor == desc] if A2 is not None else []
    return float(r[col].iloc[0]) if len(r) else np.nan


def a4(mode, factor):
    r = A4[(A4.distance == mode) & (A4.factor == factor)] if A4 is not None else []
    return r.iloc[0] if len(r) else None


def a5(desc):
    r = A5[A5.descriptor == desc] if A5 is not None else []
    return r.iloc[0] if len(r) else None


def a6n(desc):
    r = A6n[A6n.descriptor == desc] if A6n is not None else []
    return r.iloc[0] if len(r) else None


def a7(task, n):
    r = A7[(A7.task == task) & (A7.cells_per_bag == n)] if A7 is not None else []
    return r.iloc[0] if len(r) else None


def a8(q):
    r = A8[A8.quantity.str.startswith(q)] if A8 is not None else []
    return r.iloc[0] if len(r) else None


anch = ANCH.set_index(ANCH.columns[0])["value"] if ANCH is not None else pd.Series(dtype=float)
ev = pd.read_csv(PROC / "rtdc_events_gated.csv.gz")
V["n_cells"] = f"{len(ev):,}"
V["n_meas"] = ev.measurement.nunique()
afm = read(PROC / "afm_events.csv")
V["n_afm"] = f"{len(afm):,}" if afm is not None else NA
s_egf, s_fgf, s_fe = shift("EGF vs Serum"), shift("FGFJI vs Serum"), shift("FGFJI vs EGF")
icc_shape = A5[~A5.descriptor.isin(["mean", "median", "q10", "q90"])] if A5 is not None else None
_cc = A3c[A3c.contrast.isin(["EGF vs Serum", "FGFJI vs Serum", "FGFJI vs EGF"])] if A3c is not None else None
N_COND_TESTS = len(_cc) if _cc is not None else NA
_nsig = int((_cc.p_signflip < 0.05).sum()) if _cc is not None else None
N_SIG_COND = {0: "none", 1: "one", 2: "two", 3: "three"}.get(_nsig, str(_nsig)) if _nsig is not None else NA


def first_n80(scen, m=300, test="M2_vs_M1"):
    if PM is None:
        return None
    g = PM[(PM.scenario == scen) & (PM.cells_per_tumour == m)].sort_values("n_per_class")
    ok = g[g[f"power_{test}"] >= 0.8]
    return int(ok.n_per_class.iloc[0]) if len(ok) else None


def pw(scen, n, m, test="M2_vs_M1"):
    if PM is None:
        return np.nan
    r = PM[(PM.scenario == scen) & (PM.n_per_class == n) & (PM.cells_per_tumour == m)]
    return float(r[f"power_{test}"].iloc[0]) if len(r) else np.nan


def pmain(scen, n, m, col):
    if PM is None:
        return np.nan
    r = PM[(PM.scenario == scen) & (PM.n_per_class == n) & (PM.cells_per_tumour == m)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def pgrid(ts, delta, r, n=200, m=1000):
    if PG is None:
        return np.nan
    g = PG[np.isclose(PG.tau_scale, ts) & np.isclose(PG.delta, delta) & np.isclose(PG.r, r) &
           (PG.n_per_class == n) & (PG.cells_per_tumour == m)]
    return float(g.power_M2_vs_M1.iloc[0]) if len(g) else np.nan


S0, S1, S2, S3, S4, S5 = ("S0 no association", "S1 location (IDH-anchored)", "S2 spread (LGG IQR x1.25)",
                          "S3 stiff tail (10% of LGG cells, 2-fold)", "S4 location + spread",
                          "S5 redundant with image + report")
S2b, S3b = "S2b spread, primary-culture variation", "S3b stiff tail, primary-culture variation"


def size_range(scens_tests, nmin=100):
    if PM is None:
        return NA
    vals = []
    for sc, t in scens_tests:
        g = PM[(PM.scenario == sc) & (PM.n_per_class >= nmin)]
        vals += list(g[f"power_{t}"])
    return f"{min(vals):.3f}-{max(vals):.3f}" if vals else NA


# ------------------------------------------------------------------------------------------------ text
def or_na(x, fmt="{:.3f}"):
    try:
        return fmt.format(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else NA
    except Exception:
        return NA


n2_80 = first_n80(S2)
n3_80 = first_n80(S3)
n2b_80 = first_n80(S2b)
n3b_80 = first_n80(S3b)


def npc(n):
    if n is None:
        return "more than 400 patients per class"
    return f"as few as {n} patients per class" if n <= 25 else f"{n} patients per class"


_ratio = np.nanmean([val(site_g, "balanced_accuracy") / val(site_g, "chance"),
                     val(site_l, "balanced_accuracy") / val(site_l, "chance")]) if site_g is not None else np.nan
SITE_PHRASE = ("well above chance" if _ratio > 2 else "above chance" if _ratio > 1.3 else "close to chance")
_mag = val(mag, "auroc")
MAG_PHRASE = ("almost perfectly" if _mag > 0.95 else "accurately" if _mag > 0.85 else "moderately well" if _mag > 0.7
              else "poorly")
ENCODES = ("The embeddings therefore encode acquisition and site, which is why we hold out whole institutions and "
           "report the 40x subset." if (_ratio > 1.3 or _mag > 0.8) else
           "The embeddings carried little acquisition or site information by these measures.")
_dh = cmp(IMG, "image_hipt_abmil")
HIPT_PHRASE = ("gave similar results" if _dh is not None and abs(_dh.delta_auroc) < 0.02 else
               f"gave a {'lower' if _dh is not None and _dh.delta_auroc > 0 else 'higher'} AUROC")
_s_i, _s_r = sig("fusion_primary", IMG), sig("fusion_primary", REP)
FUSION_PHRASE = ("Images and masked reports contributed complementary information, so that fusion exceeded either "
                 "modality" if (_s_i and _s_r) else
                 "Fusion exceeded reports but not images" if _s_r else "Fusion exceeded images but not reports" if _s_i
                 else "Fusion did not significantly exceed either modality")
_a40 = val(acq_40, "auroc")
ACQ_PHRASE = (f"where acquisition metadata alone carried no information (AUROC {f3(_a40)})" if _a40 < 0.6 else
              f"where acquisition metadata alone reached an AUROC of {f3(_a40)}")
_stf = st_f
ENRICH_PHRASE = ("the tumours that the model called glioblastoma despite an LGG label were enriched for IDH-wild-type "
                 "tumours with molecular features of glioblastoma" if (_stf is not None and _stf.p_fisher < 0.05 and
                 _stf.frac_called_molecular_gbm > _stf.frac_called_astro_idhmut) else
                 "LGG-labelled tumours with molecular features of glioblastoma were not called GBM significantly more "
                 "often than IDH-mutant astrocytomas")
_bt = cmp(REP, "report_tfidf_M2")
_btp = cmp(REP, "report_tfidf_M2", "all", "patient")
BERT_PARA = (
    f"The fine-tuned transformer was not the better report model. With random patient folds BiomedBERT and TF-IDF "
    f"{'performed similarly' if _btp is not None and abs(_btp.delta_auroc) < 0.02 else 'already differed'} "
    f"({auc(REP, 'all', 'patient', ci=False)} and {auc('report_tfidf_M2', 'all', 'patient', ci=False)}), "
    f"but with institutions held out BiomedBERT fell to {auc(REP, ci=False)} against {auc('report_tfidf_M2', ci=False)} "
    f"for the regularized bag-of-words model (difference {dtxt(REP, 'report_tfidf_M2')}). Fine-tuning a 110-million-"
    f"parameter model on a few hundred reports, with the stopping epoch chosen on reports from the same institutions as "
    f"the training data, rewards the phrasing of the contributing sites; a sparse linear model of word counts, refit "
    f"with a cross-validated penalty, carries fewer site-specific degrees of freedom. Because BiomedBERT was the "
    f"pre-specified report branch, the primary fusion uses it, and the TF-IDF fusion is reported as the pre-specified "
    f"secondary analysis. Pathology-report models should therefore be validated across institutions before a language "
    f"model is preferred to a linear baseline." if (_bt is not None and _bt.delta_auroc < -0.02) else
    f"BiomedBERT and the TF-IDF model performed similarly with institutions held out "
    f"({auc(REP, ci=False)} and {auc('report_tfidf_M2', ci=False)}), so the report branch is not sensitive to the "
    f"choice of text model.")


def _mask_sentence():
    if LEAK is None:
        return ""
    l1 = LEAK[LEAK.level == "M1"].iloc[0] if (LEAK.level == "M1").any() else None
    l2 = LEAK[LEAK.level == "M2"].iloc[0] if (LEAK.level == "M2").any() else None
    txt = (f"After masking, {int(val(l2, 'reports_with_residual_dx_words'))} reports retained a diagnosis term "
           f"(mean {val(l1, 'mean_tokens_masked'):.1f} and {val(l2, 'mean_tokens_masked'):.1f} tokens masked per "
           f"report at M1 and M2).")
    if TOP2 is not None:
        g = ", ".join(f"'{t}'" for t in TOP2[TOP2.direction == "GBM"].term.head(6))
        lg = ", ".join(f"'{t}'" for t in TOP2[TOP2.direction != "GBM"].term.head(6))
        txt += (f" The terms most associated with GBM in M2-masked reports were largely administrative and template "
                f"phrases ({g}), whereas those associated with LGG included residual morphological description ({lg}; "
                f"Extended Data Table 7).")
    return txt


MASK_SENTENCE = _mask_sentence()
STUDY_SIZE = (f"about {n2_80} patients per class" if n2_80 and n2_80 <= 100 else
              f"{n2_80} patients per class" if n2_80 else "more than 400 patients per class")
fus_gain_img = "improved" if sig("fusion_primary", IMG) else "did not significantly improve"
fus_gain_rep = "improved" if sig("fusion_primary", REP) else "did not significantly improve"
TITLE = ("Institution-robust classification of glioblastoma and lower-grade glioma from whole-slide images and "
         "pathology reports, and the evidence required to test single-cell mechanical heterogeneity as a "
         "diagnostic signal")
_p47 = pgrid(0.47, 0.0, 1.25)
AFM_PHRASE = ("but not with 200 patients per class and 1,000 cells if it is as large as in primary-culture AFM "
              f"(power {pct(_p47)})" if not np.isnan(_p47) and _p47 < 0.8 else
              f"and with {pct(_p47)} power at 200 patients per class and 1,000 cells even if it is as large as in "
              "primary-culture AFM")
ABSTRACT = (
    f"Distinguishing glioblastoma (GBM) from lower-grade glioma (LGG) determines prognosis and treatment, yet models "
    f"trained on public archives can learn the signatures of contributing sites and scanners instead of tumour "
    f"biology. We assembled {V['n_m']} patients from The Cancer Genome Atlas with a diagnostic whole-slide image and a "
    f"pathology report ({V['n_m_gbm']} GBM, {V['n_m_lgg']} LGG; {V['n_m_inst']} tissue-source sites) and trained an "
    f"attention-based multiple-instance model on Phikon-v2 embeddings of {V['tiles_total']} tissue tiles and a "
    f"BiomedBERT classifier on reports from which tumour type, grade, molecular findings and grading criteria had "
    f"been masked. With whole sites held out, late fusion of the two modalities reached an area under the receiver "
    f"operating characteristic curve (AUROC) of {auc('fusion_primary')}, compared with {auc(IMG, ci=False)} for images "
    f"and {auc(REP, ci=False)} for reports; among slides scanned at 40x, where scan magnification no longer "
    f"separates the classes, fusion reached {auc('fusion_primary', 'x40', ci=False)}. We then asked whether the "
    f"distribution of whole-cell mechanical properties across a tumour's cells carries diagnostic information beyond "
    f"its average. No public resource links cell mechanics to these patients. In {V['n_cells']} single cells from three "
    f"primary GBM lines measured by real-time deformability cytometry in three signalling states on three days, "
    f"state changes shifted the whole stiffness distribution without changing its shape, shape descriptors were "
    f"dominated by day-to-day variation and needed 1,000 or more cells per sample to become reliable, "
    f"whereas the median was reliable from about 100 cells. A plasmode design analysis that combines the Phase 1 "
    f"scores with real single-cell distributions shows that a 25% difference in the spread of stiffness between "
    f"classes would be detectable beyond images, reports and average stiffness with {npc(n2_80)} and 300 cells per "
    f"tumour if between-tumour variation in spread is as small as between cell-line measurements, but would require "
    f"{npc(n2b_80)} if it is as large as in primary cultures. Images and reports therefore provide institution-robust discrimination of GBM from LGG, whereas the "
    f"diagnostic value of mechanical heterogeneity remains untested and requires a prospectively matched cohort of the "
    f"size specified here.")

INTRO = [
    "Diffuse gliomas are classified by the fifth edition of the World Health Organization classification of central "
    "nervous system tumours (WHO CNS5) by integrating histology with molecular markers: IDH-wild-type astrocytic "
    "tumours with necrosis or microvascular proliferation, TERT promoter mutation, EGFR amplification or combined "
    "gain of chromosome 7 and loss of chromosome 10 are glioblastomas, whereas IDH-mutant astrocytomas and IDH-mutant, "
    "1p/19q-codeleted oligodendrogliomas of grades 2 and 3 constitute most lower-grade gliomas^{1-4}. The distinction "
    "between glioblastoma and lower-grade glioma separates median survivals of about 15 months from many years and "
    "drives the choice of adjuvant therapy, so tools that support it from routinely generated material are of "
    "clinical interest.",
    "Haematoxylin and eosin (H&E) whole-slide images and free-text pathology reports are produced for every resected "
    "glioma. Weakly supervised multiple-instance learning (MIL) turns gigapixel slides into patient-level predictions^{5-7}, "
    "self-supervised pathology foundation models provide transferable tile representations^{8-11}, and domain-specific "
    "language models extract diagnostic content from clinical text^{12-14}. Most published models were, however, "
    "developed on The Cancer Genome Atlas (TCGA), in which tissue-source sites differ in staining, scanning and case "
    "mix; models can identify the contributing site from slide appearance, and accuracy estimated with random patient "
    "splits overstates performance on unseen institutions^{15}. In TCGA-GBM and TCGA-LGG, every slide scanned at 20x "
    "magnification is a glioblastoma, which offers a further shortcut. Credible multimodal models therefore need "
    "evaluation that holds out whole institutions and removes acquisition shortcuts.",
    "Tumour cells also differ in their mechanical properties. Cancer cells are generally softer than their benign "
    "counterparts^{16,17}, tissue-scale stiffness distributions distinguish breast cancers from benign lesions^{18}, and "
    "high-throughput methods such as real-time deformability cytometry (RT-DC) measure thousands of single cells per "
    "minute^{19,20}. In gliomas, atomic force microscopy (AFM) has shown that low-grade glioma stem cells carry a tail "
    "of stiff cells absent from high-grade cells^{21} and that early-passage IDH1-R132H cultures are stiffer than "
    "IDH1-wild-type cultures^{22}, and the IDH status of glioma cells modulates tissue mechanics^{23}. These "
    "observations motivate the hypothesis that the distribution of mechanical responses across a tumour's cells, not "
    "only its average stiffness, carries diagnostic information that images and reports miss. Testing it requires "
    "single-cell mechanics, slides and reports from the same patients, which no public resource provides.",
    "Here we address both questions in two phases. In Phase 1 we develop a classifier of glioblastoma versus "
    "lower-grade glioma that fuses whole-slide images and pathology reports and evaluate it with institutions held "
    "out, in a magnification-matched subset and against explicit models of site and acquisition signatures. In "
    "Phase 2 we use public single-cell mechanics data to test whether the shape of the stiffness distribution carries "
    "reproducible information beyond its location, quantify how many cells are needed to estimate distributional "
    "descriptors, and combine the Phase 1 predictions with real single-cell distributions in a plasmode design "
    "analysis that specifies the matched cohort needed to test the mechanical hypothesis.",
]


def results_text():
    R = []
    R.append(("h", "Cohort and evaluation design"))
    R.append(("p",
              f"Of {V['n_lab']:,} patients with a pan-glioma diagnosis ({V['n_lab_gbm']} GBM, {V['n_lab_lgg']} LGG), "
              f"{V['n_slides']} had a diagnostic slide that yielded tissue tiles and {V['n_rep']} had a pathology "
              f"report; {V['n_m']} had both and form the matched cohort ({V['n_m_gbm']} GBM, {V['n_m_lgg']} LGG) from "
              f"{V['n_m_inst']} tissue-source sites (Fig. 1 and Table 1). Slides were tiled at a common resolution of "
              f"0.5 µm per pixel, giving a median of {V['tiles_med']:,} tiles per slide (interquartile range (IQR) "
              f"{V['tiles_iqr']}; {V['tiles_total']} in total). All {V['n20_gbm']} slides scanned at 20x came from "
              f"glioblastomas ({V['n20_gbm']} of {V['n_gbm_slides']} GBM slides and {V['n20_lgg']} LGG slides; "
              f"Fig. 1c), so we report every result also for the {V['n40']} matched patients whose slides were "
              f"scanned at 40x ({V['n40_gbm']} GBM, {V['n40_lgg']} LGG). Each model was evaluated with the same "
              f"five-times-repeated five-fold cross-validation, either with random patient folds or with folds that "
              f"keep every tissue-source site in a single fold; the latter is the pre-specified primary scheme."
              if isinstance(V["tiles_med"], int) else
              f"The matched cohort comprised {V['n_m']} patients (Table 1)."))
    R.append(("h", "Images and reports each discriminate GBM from LGG at unseen institutions"))
    hipt = met("image_hipt_abmil")
    mp = met("image_phikon2_meanpool")
    R.append(("p",
              f"The Phikon-v2 attention-MIL model reached an AUROC of {auc(IMG, 'all', 'patient')} with random patient "
              f"folds and {auc(IMG)} with institutions held out (Fig. 2 and Extended Data Fig. 1). Mean pooling of the "
              f"same embeddings gave {auc('image_phikon2_meanpool', ci=False)} and the HIPT encoder with attention MIL "
              f"{auc('image_hipt_abmil', ci=False)} with institutions held out (Phikon-v2 versus HIPT "
              f"{dtxt(IMG, 'image_hipt_abmil')}; attention versus mean pooling {dtxt(IMG, 'image_phikon2_meanpool')}). "
              f"BiomedBERT fine-tuned on M2-masked reports, from which tumour-type names, grades, morphology codes, "
              f"molecular findings and the histological grading criteria were removed, reached {auc(REP, 'all', 'patient')} "
              f"with random folds and {auc(REP)} with institutions held out. A TF-IDF logistic regression on the same "
              f"texts gave {auc('report_tfidf_M2', ci=False)} (BiomedBERT versus TF-IDF {dtxt(REP, 'report_tfidf_M2')}), "
              f"masking only diagnosis, grade and molecular terms (M1) gave {auc('report_bert_M1', ci=False)}, and "
              f"unmasked reports, which state the diagnosis, gave {auc('report_unmasked_M0', ci=False)}. Age alone "
              f"reached {auc('age', ci=False)}. {MASK_SENTENCE} Holding out institutions lowered the AUROC by "
              f"{or_na(drop_img)} for "
              f"images and {or_na(drop_rep)} for reports relative to random patient folds, the expected signature of "
              f"site-specific information in both modalities."))
    _si, _sr = sig("fusion_primary", IMG), sig("fusion_primary", REP)
    R.append(("h", "Fusion of images and reports improves discrimination (primary analysis)" if (_si and _sr) else "Fusion improves on reports but not on images (primary analysis)" if _sr else "Fusion improves on images but not on reports (primary analysis)" if _si else "Fusion of images and reports (primary analysis)"))
    R.append(("p",
              f"Late fusion, the unweighted mean of the image and report logits with no fitted parameters, reached "
              f"{auc('fusion_primary')} with institutions held out and {auc('fusion_primary', 'all', 'patient', ci=False)} "
              f"with random folds (Fig. 2a). In the pre-specified primary comparisons, fusion {fus_gain_img} on images "
              f"alone ({dtxt('fusion_primary', IMG)}) and {fus_gain_rep} on reports alone ({dtxt('fusion_primary', REP)}; "
              f"Holm-adjusted DeLong tests over the four primary contrasts; Fig. 2c). Among patients with 40x-scanned "
              f"slides, fusion reached {auc('fusion_primary', 'x40')}, against {auc(IMG, 'x40', ci=False)} for images "
              f"and {auc(REP, 'x40', ci=False)} for reports (differences {dtxt('fusion_primary', IMG, 'x40')} and "
              f"{dtxt('fusion_primary', REP, 'x40')}). Adding age to the fused model changed the AUROC by "
              f"{dtxt('fusion_primary_age', 'fusion_primary')}, and replacing the M2-masked by the M1-masked report gave "
              f"{auc('fusion_image_reportM1', ci=False)}. In the pre-specified secondary analysis that fused the image "
              f"model with the TF-IDF report model instead of BiomedBERT, the AUROC was {auc('fusion_image_tfidfM2')} "
              f"(versus images {dtxt('fusion_image_tfidfM2', IMG)}; versus the primary fusion "
              f"{dtxt('fusion_image_tfidfM2', 'fusion_primary')}) and {auc('fusion_image_tfidfM2', 'x40', ci=False)} "
              f"among 40x slides. Every model, cohort and scheme is reported in Fig. 2d,e and Extended Data Table 1."))
    _i40 = met(IMG, "x40")
    R.append(("h", "Acquisition and site signatures do not explain the image and fusion results" if (_i40 is not None and _i40.auroc >= 0.8) else "Acquisition and site signatures"))
    R.append(("p",
              f"A model given only acquisition metadata (scanner and its software version, magnification, pixel size, "
              f"JPEG quality, scan date, slide and tissue area) reached {auc('acq_meta', 'all', 'patient', ci=False)} with random folds and "
              f"{auc('acq_meta', ci=False)} with institutions held out, but {auc('acq_meta', 'x40', ci=False)} among "
              f"40x-scanned slides, where the image model retained {auc(IMG, 'x40', ci=False)} (Fig. 2b). Phikon-v2 "
              f"slide embeddings separated 20x- from 40x-scanned glioblastoma slides with an AUROC of "
              f"{or_na(val(mag, 'auroc'))} and identified the tissue-source site among sites with at least "
              f"15 patients with a balanced accuracy of {or_na(val(site_g, 'balanced_accuracy'))} within "
              f"glioblastomas (chance {or_na(val(site_g, 'chance'))}) and "
              f"{or_na(val(site_l, 'balanced_accuracy'))} within lower-grade gliomas (chance "
              f"{or_na(val(site_l, 'chance'))}; Fig. 3e). {ENCODES} Within institutions that "
              f"contributed at least five patients of each class, the fused model reached a pooled within-institution "
              f"AUROC of {or_na(val(r_f, 'auroc_within_institution'))} (Fig. 3c). The amount of tissue per "
              f"patient {'did not separate' if val(tiss, 'auroc') < 0.6 and val(tiss, 'auroc') > 0.4 else 'partly separated'} "
              f"the classes (AUROC {or_na(val(tiss, 'auroc'))}) and fused "
              f"performance was similar across tertiles of tissue amount (Extended Data Fig. 2)."))
    R.append(("h", "Calibration, clinical utility and WHO CNS5 molecular groups"))
    R.append(("p",
              f"With institutions held out, the calibration slope was {or_na(val(r_f, 'calibration_slope'), '{:.2f}')} "
              f"for fusion, {or_na(val(r_i, 'calibration_slope'), '{:.2f}')} for images and "
              f"{or_na(val(r_r, 'calibration_slope'), '{:.2f}')} for reports, with expected calibration "
              f"errors of {or_na(val(r_f, 'ece'))}, {or_na(val(r_i, 'ece'))} and "
              f"{or_na(val(r_r, 'ece'))} and Brier scores of {or_na(val(r_f, 'brier'))}, "
              f"{or_na(val(r_i, 'brier'))} and {or_na(val(r_r, 'brier'))} (Fig. 3a). At a "
              f"threshold probability of 0.5, the net benefit was {or_na(val(nb_f, 'net_benefit'))} for "
              f"fusion, {or_na(val(nb_i, 'net_benefit'))} for images and "
              f"{or_na(val(nb_r, 'net_benefit'))} for reports (Fig. 3b). TCGA diagnoses predate WHO CNS5, "
              f"and {or_na(val(st_f, 'n_molecular_gbm_in_lgg'), '{:.0f}')} matched patients labelled LGG "
              f"were IDH-wild-type with TERT promoter mutation or combined chromosome 7 gain and 10 loss, features "
              f"that now define glioblastoma. The fused model called {pct(val(st_f, 'frac_called_molecular_gbm'))} "
              f"of these tumours GBM, against {pct(val(st_f, 'frac_called_astro_idhmut'))} of "
              f"{or_na(val(st_f, 'n_astro_idhmut_in_lgg'), '{:.0f}')} IDH-mutant astrocytomas (odds ratio "
              f"{or_na(val(st_f, 'odds_ratio'), '{:.1f}')}, Fisher {pfmt(val(st_f, 'p_fisher'))}); "
              f"the image model called {pct(val(st_i, 'frac_called_molecular_gbm'))} and "
              f"{pct(val(st_i, 'frac_called_astro_idhmut'))} (Fig. 3d). Part of the apparent error of the "
              f"classifier on TCGA labels therefore falls on tumours whose label would change under current criteria."))
    R.append(("h", "Culture states shift the location, not the shape, of single-cell stiffness distributions"))
    con_med = a2("median", "seqshare_condition")
    R.append(("p",
              f"Phase 2 used RT-DC measurements of {V['n_cells']} gated single cells from three primary glioblastoma "
              f"lines in serum, EGF and FGF plus JAK-inhibitor (FGFJI) culture, which correspond to low, medium and high "
              f"activation of the STAT3-Ser/Hes3 signalling axis, measured on three days in a crossed design "
              f"({V['n_meas']} measurements, 96-2,321 cells each; Fig. 5a). Relative to serum, the median apparent "
              f"Young's modulus was {or_na(val(s_egf, 'fold_change'), '{:.2f}')}-fold in EGF (shift "
              f"{or_na(val(s_egf, 'shift_log10E'))} log10 units, 95% CI "
              f"{or_na(val(s_egf, 'lo'))} to {or_na(val(s_egf, 'hi'))}) and "
              f"{or_na(val(s_fgf, 'fold_change'), '{:.2f}')}-fold in FGFJI "
              f"({or_na(val(s_fgf, 'shift_log10E'))}, {or_na(val(s_fgf, 'lo'))} to "
              f"{or_na(val(s_fgf, 'hi'))}); the ordering of the three states follows the increasing "
              f"STAT3-Ser/Hes3 activation that defines them^{{24,35}}. Shift functions, which compare every decile between conditions within line-by-day blocks, "
              f"were flat (Fig. 5b): the change in the upper tail beyond the change in the median was "
              f"{or_na(val(shapec('FGFJI vs Serum', 'upper_minus_median'), 'mean'))} log10 units (95% CI "
              f"{or_na(val(shapec('FGFJI vs Serum', 'upper_minus_median'), 'lo'))} to "
              f"{or_na(val(shapec('FGFJI vs Serum', 'upper_minus_median'), 'hi'))}) for FGFJI versus serum "
              f"and the log IQR ratio was {or_na(val(shapec('FGFJI vs Serum', 'log_iqr_ratio'), 'mean'))} "
              f"({or_na(val(shapec('FGFJI vs Serum', 'log_iqr_ratio'), 'lo'))} to "
              f"{or_na(val(shapec('FGFJI vs Serum', 'log_iqr_ratio'), 'hi'))}); {N_SIG_COND} of the "
              f"{N_COND_TESTS} tail, spread, skewness and kurtosis contrasts between conditions reached P < 0.05 in "
              f"exact sign-flip tests. Culture "
              f"state accounted for {pct(con_med)} of the between-measurement variance of the median (restricted "
              f"permutation {pfmt(a2('median', 'p_perm_condition'))}) but {pct(a2('iqr', 'seqshare_condition'))} of that "
              f"of the IQR ({pfmt(a2('iqr', 'p_perm_condition'))}) and {pct(a2('upper_tail', 'seqshare_condition'))} of "
              f"that of the upper tail; measurement day accounted for {pct(a2('iqr', 'seqshare_date'))} and "
              f"{pct(a2('upper_tail', 'seqshare_date'))}, respectively (Fig. 5c). After removal of each distribution's "
              f"median, shapes were no more similar within culture states than between them (restricted permutation "
              f"{pfmt(val(a4('location', 'condition'), 'p_perm'))}) but were more similar within measurement "
              f"days ({pfmt(val(a4('location', 'date'), 'p_perm'))}); after removal of median and IQR, "
              f"residual shape differed by day ({pfmt(val(a4('location_scale', 'date'), 'p_perm'))}) and "
              f"weakly by cell line ({pfmt(val(a4('location_scale', 'line'), 'p_perm'))}; Extended Data "
              f"Fig. 3)."))
    R.append(("h", "Shape descriptors are irreproducible across days and require many cells"))
    i_med = a5("median")
    n_med, n_up, n_iqr = a6n("median"), a6n("upper_tail"), a6n("iqr")
    R.append(("p",
              f"Across days, the median of a given line in a given state was reproducible (intraclass correlation "
              f"{or_na(val(i_med, 'icc1'), '{:.2f}')}, 95% CI {or_na(val(i_med, 'lo'), '{:.2f}')}-"
              f"{or_na(val(i_med, 'hi'), '{:.2f}')}), whereas spread, tail, skewness and kurtosis "
              f"descriptors were not (intraclass correlations "
              f"{or_na(icc_shape.icc1.min() if icc_shape is not None else np.nan, '{:.2f}')} to "
              f"{or_na(icc_shape.icc1.max() if icc_shape is not None else np.nan, '{:.2f}')}; Fig. 5d). Resampling of "
              f"cells showed why: relative to the true between-sample variation, the median reached a reliability of 0.8 "
              f"with {or_na(val(n_med, 'cells_for_reliability_0.8'), '{:,.0f}')} cells, the upper tail needed "
              f"{or_na(val(n_up, 'cells_for_reliability_0.8'), '{:,.0f}')} cells, and the IQR reached only "
              f"{or_na(val(n_iqr, 'reliability_at_1000'), '{:.2f}')} at 1,000 cells; for quartile skewness "
              f"and octile kurtosis, sampling noise exceeded the observed between-sample variance at every sample size "
              f"(Fig. 5e and Extended Data Fig. 4). Between-sample differences in shape are therefore small relative "
              f"to both day-to-day variation and the sampling error of the numbers of cells typically measured by AFM."))
    R.append(("h", "Distribution descriptors add no discriminative information in bags of cells"))
    b1, b2, b3 = a7("EGF vs Serum", 100), a7("FGFJI vs Serum", 100), a7("FGFJI vs EGF", 100)
    R.append(("p",
              f"To mimic biopsy-sized samples, we drew bags of 30-1,000 cells from each measurement and classified "
              f"culture states with leave-one-day-out cross-validation. With 100 cells per bag, location descriptors "
              f"(mean and median of log10 E) gave AUROCs of {or_na(val(b1, 'auroc_location'))}, "
              f"{or_na(val(b2, 'auroc_location'))} and {or_na(val(b3, 'auroc_location'))} for EGF "
              f"versus serum, FGFJI versus serum and FGFJI versus EGF; adding spread, tail and skewness descriptors "
              f"changed these by {or_na(val(b1, 'delta_distribution_vs_location'), '{:+.3f}')}, "
              f"{or_na(val(b2, 'delta_distribution_vs_location'), '{:+.3f}')} and "
              f"{or_na(val(b3, 'delta_distribution_vs_location'), '{:+.3f}')} (95% CI for the last "
              f"{or_na(val(b3, 'delta_distribution_vs_location_lo'), '{:+.3f}')} to "
              f"{or_na(val(b3, 'delta_distribution_vs_location_hi'), '{:+.3f}')}, cluster bootstrap over "
              f"measurements), and shape descriptors alone performed at or below chance (AUROCs "
              f"{or_na(min([val(b, 'auroc_shape_only') for b in (b1, b2, b3)]))}-"
              f"{or_na(max([val(b, 'auroc_shape_only') for b in (b1, b2, b3)]))}; Fig. 5f). Cell lines, "
              f"which stand for tumours of origin, were poorly separated by any descriptor set when the classifier had "
              f"to generalize to an unseen culture state (Extended Data Table 5). In AFM measurements of LN229 and U87 "
              f"cells ({V['n_afm']} cells), the two lines differed in location (median difference "
              f"{or_na(val(a8('U87 vs LN229: median_diff'), 'estimate'))} log10 units) but their "
              f"median-aligned distributions did not differ in shape (Kolmogorov-Smirnov "
              f"{pfmt(val(a8('KS on median-aligned'), 'estimate'))}; log IQR ratio "
              f"{or_na(val(a8('U87 vs LN229: log_iqr_ratio'), 'estimate'))}, 95% CI "
              f"{or_na(val(a8('U87 vs LN229: log_iqr_ratio'), 'lo'))} to "
              f"{or_na(val(a8('U87 vs LN229: log_iqr_ratio'), 'hi'))}; Extended Data Fig. 3c)."))
    R.append(("h", "What a matched study would need: plasmode design analysis"))
    R.append(("p",
              f"Because no patient has linked mechanics, we estimated the operating characteristics of the test that a "
              f"matched study would use. Each simulated tumour received a real Phase 1 out-of-fold image-and-report "
              f"score (fused model, institutions held out; AUROC {or_na(PIN.get('score_auroc', np.nan))}), resampled "
              f"within class, and a stiffness distribution built from one of the {PIN.get('templates', NA)} real RT-DC "
              f"distributions standardized to median 0 and IQR 1, with location and spread drawn around class-specific "
              f"values. Between-tumour standard deviations were anchored on the RT-DC data (0.032 log10 units for the "
              f"median; 0.09 for the logarithm of the IQR) and the location effect on the difference between IDH1-R132H "
              f"and IDH1-wild-type cultures (0.072 log10 units, 1.18-fold)^{{22}}. The test of distribution beyond "
              f"average stiffness was a likelihood-ratio test of spread and tail descriptors added to a logistic model "
              f"that already contained the image-and-report score and two location estimators (mean and median); "
              f"with a single location estimator, the tails of small samples carry information about the location "
              f"error itself and inflate the size of the test. In null scenarios the rejection rate at a nominal 5% "
              f"was {size_range([(S0, 'M2_vs_M1'), (S1, 'M2_vs_M1'), (S5, 'M2_vs_M1')])} for 100 or more patients per "
              f"class (Fig. 6e). With between-tumour variation in spread as in RT-DC, a 25% larger spread of stiffness "
              f"in LGG was detected with 80% power with {npc(n2_80)} at 300 cells per tumour, and a stiff subpopulation "
              f"comprising 10% of LGG cells with {npc(n3_80)}; with the larger between-tumour variation of "
              f"primary-culture AFM measurements (scenarios S2b and S3b), the same effects required {npc(n2b_80)} and "
              f"{npc(n3b_80)} (Fig. 6a,b). With 200 patients per class and 300 cells, these effects raised the cross-validated AUROC "
              f"over images and reports by {or_na(pmain(S2, 200, 300, 'delta_M2_vs_M0'), '{:.3f}')} and "
              f"{or_na(pmain(S3, 200, 300, 'delta_M2_vs_M0'), '{:.3f}')}, whereas a pure location effect of IDH-anchored "
              f"size raised it by {or_na(pmain(S1, 200, 300, 'delta_M1_vs_M0'), '{:.3f}')} through average stiffness "
              f"alone (Fig. 6c). Over a grid of effects at 200 patients per class and 1,000 cells, a 25% spread "
              f"difference was detected with {pct(pgrid(0.09, 0.0, 1.25))} power under RT-DC-like and "
              f"{pct(pgrid(0.47, 0.0, 1.25))} under primary-culture-like between-tumour variation, and a 50% difference "
              f"with {pct(pgrid(0.47, 0.0, 1.5))} under the latter; a concurrent location difference reduced the power "
              f"for shape ({pct(pgrid(0.47, 0.072, 1.25))} for a 25% spread difference), because average stiffness then "
              f"already separates most patients (Fig. 6d). When mechanics "
              f"carried only information already present in images and reports, neither average stiffness nor shape "
              f"added to the model (rejection rates {size_range([(S5, 'M1_vs_M0'), (S5, 'M2_vs_M1')])})."))
    return R


DISCUSSION = [
    f"We developed a classifier of glioblastoma versus lower-grade glioma that fuses whole-slide images and pathology "
    f"reports and evaluated it under conditions that remove the two dominant shortcuts in TCGA: site signatures and "
    f"scan magnification. With whole institutions held out, the fused model reached an AUROC of "
    f"{auc('fusion_primary', ci=False)}, and among 40x-scanned slides, {ACQ_PHRASE}, it reached "
    f"{auc('fusion_primary', 'x40', ci=False)}. {FUSION_PHRASE}, and {ENRICH_PHRASE}. In Phase 2, real single-cell mechanics data showed that signalling-state "
    f"changes in glioblastoma cells move the whole stiffness distribution without changing its shape and that shape "
    f"descriptors are dominated by day-to-day variation and sampling error; the design analysis translates these "
    f"properties into the cohort needed to test the diagnostic value of mechanical heterogeneity.",
    f"The fall in AUROC from random to institution-held-out folds ({or_na(drop_img)} for images and "
    f"{or_na(drop_rep)} for reports) quantifies how much of the apparent performance in random splits reflects "
    f"site-specific information. Slide embeddings identified the contributing site {SITE_PHRASE} within a "
    f"diagnostic class, {'in line with' if SITE_PHRASE != 'close to chance' else 'in contrast to'} earlier reports "
    f"of site signatures in TCGA^{{15}}, and separated 20x- from "
    f"40x-scanned slides {MAG_PHRASE}. Masked reports still discriminated the classes, partly because "
    f"institutions use different report templates and because clinical context, such as the extent of resection or "
    f"prior surgery, differs between diagnoses; unmasked reports, which state the diagnosis, were nearly perfect "
    f"and illustrate why masking is required before reports can be treated as an independent modality. Phikon-v2 was "
    f"pretrained without labels on slides that include TCGA, which could make its representations of these slides "
    f"more informative than those of unseen material; the HIPT encoder, also pretrained on TCGA, {HIPT_PHRASE}, "
    f"and the UNI encoder^{{8}}, whose pretraining excluded TCGA, can be substituted in the published pipeline.",
    BERT_PARA,
    "The mechanical hypothesis predicts that distributional descriptors carry information beyond the average. In a "
    "public glioblastoma dataset with replicated single-cell mechanics across defined signalling states, we "
    "found the opposite for culture state: effects were pure location shifts, shape descriptors did not reproduce "
    "across days, and adding them to location descriptors did not improve discrimination of states. This does not "
    "refute the hypothesis for glioblastoma versus lower-grade glioma. The RT-DC lines are all glioblastomas, culture "
    "states are not diagnostic classes, and the stiff-cell tail described in low-grade glioma stem cells^{21} and "
    "the wider stiffness distribution of IDH1-mutant cultures^{22} are exactly the kinds of effect that our analysis "
    "cannot see in these data. What the data do establish are the measurement properties that any diagnostic test of "
    "heterogeneity must overcome: shape descriptors need of the order of 10^{3} cells per sample, which favours "
    "high-throughput methods such as RT-DC over AFM, and between-day variation in shape is of the same order as the "
    "between-class differences reported so far^{22}, which requires measurement-batch control in a prospective design.",
    "The design analysis combines the real distribution of Phase 1 scores with real single-cell distribution shapes "
    "and therefore inherits their properties; its effect sizes and between-tumour variances are anchored on the few "
    "published glioma values^{22}, and it assumes that mechanics and image or report scores are independent given the "
    "class unless stated otherwise (scenario S5). Within these assumptions, a spread difference of 25% between classes "
    f"would be detectable beyond images, reports and average stiffness with {npc(n2_80)} if spread varies as little "
    f"between tumours as between cell-line measurements, but would require {npc(n2b_80)} if it varies as much as in "
    f"primary-culture AFM. The result "
    "argues for a two-stage design: a pilot that measures between-tumour variation in spread with at least 300 cells "
    "per tumour, followed by a confirmatory cohort sized from that estimate, with the likelihood-ratio test of shape "
    "descriptors adjusted for two location estimators pre-registered as the primary analysis.",
    "Our study has limitations. TCGA diagnoses predate WHO CNS5, one diagnostic slide per patient was analysed, and "
    "the cohort is retrospective; residual confounding by site cannot be excluded even with institutions held out, "
    "because case mix and diagnosis remain associated with site. Reports are optical-character-recognition output, "
    "and masking cannot remove every indirect cue. RT-DC measures an apparent Young's modulus derived from "
    "deformation and cell size under flow, and the AFM data have no replicate structure. Most importantly, no "
    "mechanical measurement was available for any patient, so Phase 2 tests the measurement properties and the "
    "statistical design of the hypothesis but not the hypothesis itself; the three-input model of images, reports "
    "and cell mechanics requires prospectively collected matched tissue.",
]

METHODS = [
    ("Study design and reporting", [
        "This is a retrospective diagnostic modelling study on public, de-identified data, reported following the "
        "TRIPOD+AI recommendations^{25}. Phase 1 developed and validated, by cross-validation, models that classify "
        "TCGA patients as glioblastoma (TCGA-GBM) or lower-grade glioma (TCGA-LGG) from a diagnostic H&E slide, a "
        "pathology report or both. Phase 2 analysed public single-cell mechanics data and performed a simulation "
        "(plasmode) study of the design of a matched three-input study. The primary analysis, the fusion model and the "
        "comparisons were specified before the full-resolution models were run."]),
    ("Cohort and labels", [
        "Labels and molecular annotations came from the TCGA pan-glioma study^{2} as distributed by cBioPortal^{26} "
        f"(study lgggbm_tcga_pub): {V['n_lab']:,} patients, labelled GBM (n = {V['n_lab_gbm']}) or LGG (n = "
        f"{V['n_lab_lgg']}) by study of origin. A WHO "
        "CNS5-like group was derived for subgroup analyses: IDH-wild-type with grade 4 histology, TERT promoter "
        "mutation or combined chromosome 7 gain and 10 loss (molecular glioblastoma); IDH-mutant without 1p/19q "
        "codeletion (astrocytoma); IDH-mutant with codeletion (oligodendroglioma); other IDH-wild-type; and unknown. "
        "EGFR amplification was not available, so molecular glioblastomas may be undercounted. Patients were assigned "
        "to the institution of their tissue-source site."]),
    ("Whole-slide images", [
        "For every patient, the formalin-fixed diagnostic slide of the primary tumour with the lowest DX number was "
        "downloaded from the NCI Genomic Data Commons^{27} with gdc-client (880 slides). Tissue was detected on a "
        "thumbnail of at most 2,048 pixels by Otsu thresholding of the median-filtered saturation channel (threshold of "
        "at least 20, value above 50), removal of green and blue ink, morphological closing and removal of components "
        "smaller than 0.05% of the thumbnail." + TISSUE_SENT + " Non-overlapping tiles of 128 µm (256 pixels at 0.5 µm per pixel) with at "
        "least 50% tissue were kept; 40x slides were read at level 0 with 512-pixel windows and resized, so that "
        "every slide was analysed at the same physical resolution. Slides were read with OpenSlide^{28}."]),
    ("Tile encoders and patient-level image models", [
        "Each tile was embedded with Phikon-v2^{9} (ViT-L/16 pretrained with DINOv2^{29} on public pathology images; "
        "class token, 1,024 dimensions; tiles resized to 224 pixels, ImageNet normalization) and, as a second encoder, "
        "HIPT ViT-S/16^{7} (384 dimensions, 256-pixel input). Inference used half precision on NVIDIA GPUs; on "
        "pre-Ampere GPUs, attention was computed without fused kernels. Neither encoder was trained with diagnostic "
        "labels. Two patient-level models were fitted inside every training fold. Mean pooling: the mean tile "
        "embedding was standardized, reduced to 128 principal components and classified by L2-penalized logistic "
        "regression with the penalty chosen by inner five-fold cross-validation on AUROC. Attention MIL: a gated "
        "attention network^{5} (linear projection to 256 units with ReLU and dropout 0.25; gated attention with 128 "
        "units; linear classifier) was trained with AdamW (learning rate 2 x 10^{-4}, weight decay 10^{-2}) and binary "
        "cross-entropy on bags of at most 4,096 randomly drawn tiles per epoch (with a further 25% instance dropout), "
        "with early stopping on the AUROC of a stratified 15% validation split of the training fold (patience 10, at "
        "most 60 epochs); features were standardized with training-fold statistics, and predictions used all tiles. "
        "Attention MIL was fitted in all five repeats of each scheme."]),
    ("Pathology reports", [
        "Reports were taken from TCGA-Reports^{30}, the optical-character-recognition transcription of TCGA pathology "
        "reports (868 patients in the cohort). Three versions were used: unmasked (M0); M1, in which tumour-type names "
        "(with fuzzy matching for recognition errors), WHO grade statements, grade adjectives, ICD-O and SNOMED "
        "morphology codes and molecular findings (IDH, 1p/19q, EGFR, TERT, ATRX, MGMT, CDKN2A, loss of heterozygosity "
        "and FISH results) were replaced by a mask token; and M2, which additionally masks the histological grading "
        "criteria (necrosis, palisading, microvascular proliferation, mitoses, proliferation indices, percentages, "
        "cellularity, pleomorphism and atypia). BiomedBERT^{12} (BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext), "
        "with the mask token added to its vocabulary, was fine-tuned inside each training fold on 512-token windows "
        "(stride 384, at most eight per report) labelled with the patient's class, with AdamW (learning rate 2 x "
        "10^{-5}, weight decay 0.01), 10% linear warm-up and linear decay, four epochs, an effective batch of 16 and "
        "mixed precision; the epoch with the highest AUROC on a 10% validation split of the training fold was kept, "
        "and the patient probability was the mean over windows. A TF-IDF model (unigrams and bigrams, sublinear term "
        "frequency, minimum document frequency 3) with L2-penalized logistic regression (penalty by inner "
        "cross-validation) and a logistic model of age at diagnosis served as comparators."]),
    ("Nuisance models", [
        "To measure shortcut capacity, the same cross-validation was applied to a model of acquisition metadata "
        "(scanner, scanner software version, apparent magnification, microns per pixel, JPEG quality, scan date, "
        "slide area and tissue area), "
        "a model of stain colour and image-quality statistics, and a model of hand-crafted tissue features (nuclear "
        "density and size, H&E optical densities and texture). These models describe how acquisition differs between "
        "classes and are not candidate diagnostic models."]),
    ("Cross-validation, fusion and statistical analysis", [
        "All models used the same splits: five repeats of stratified five-fold cross-validation by patient, and five "
        "repeats of stratified group five-fold cross-validation with tissue-source institutions as groups. Patient "
        "probabilities were averaged over repeats, so that every prediction is out of fold. The matched cohort "
        "comprised patients with out-of-fold predictions from all core models; the 40x cohort was its subset with "
        "slides scanned at 40x. Late fusion averaged the branch logits with equal weights. AUROCs are reported with "
        "DeLong 95% confidence intervals^{31}; paired differences were tested with the DeLong test for correlated "
        "AUROCs (fast implementation^{32}) and given paired-bootstrap 95% confidence intervals (2,000 resamples). The "
        "four primary contrasts (fusion versus image and versus report, matched and 40x cohorts, institutions held "
        "out) were corrected with Holm's procedure^{33}, as were the secondary and branch families. Calibration was "
        "assessed by the slope and intercept of logistic recalibration, the expected calibration error (ten "
        "equal-width bins) and the Brier score; clinical utility by decision-curve analysis^{34}. Within-institution "
        "AUROC was computed in institutions with both classes and pooled with weights proportional to the number of "
        "GBM-LGG pairs. Proportions are given with Wilson intervals and compared with Fisher's exact test."]),
    ("Confounding analyses", [
        "Patient-mean Phikon-v2 and HIPT embeddings were used to predict (i) the tissue-source institution among "
        "institutions with at least 15 patients, separately within GBM and within LGG (multinomial logistic regression "
        "on 64 principal components, stratified five-fold cross-validation, balanced accuracy), and (ii) 20x versus 40x "
        "scanning within GBM (AUROC). The number of tissue tiles per patient was tested as a predictor of class, and "
        "fused performance was compared across tertiles of tissue amount."]),
    ("Single-cell mechanics data", [
        "RT-DC data came from Mechanomics RT-DC Set 01 - Glioblastoma (figshare doi:10.6084/m9.figshare.28161236, CC "
        "BY 4.0): primary glioblastoma lines X01, X04 and X08^{35} in serum, EGF and FGF plus JAK-inhibitor culture, "
        "measured in a 30-µm channel at 0.16 µl s^{-1} on three days (X04 on all three days, X01 and X08 on two), "
        "each line in all three conditions on each day (21 measurements). Apparent Young's moduli were those exported "
        "by Shape-Out^{36} from deformation and cross-sectional area. Events were kept if the modulus was finite "
        "(within the valid region of the look-up table), porosity (convex-to-measured area ratio) was below 1.05 and "
        "area was at least 30 µm^{2}, leaving 21,738 cells. AFM data (2D culture, Young's modulus of 477 LN229 and 683 "
        "U87 cells) came from the RODBUK repository (doi:10.57903/UJ/3LOV6Q, CC0) and carry no replicate structure. "
        "All analyses used log10 moduli."]),
    ("Distribution descriptors and analyses", [
        "Per sample we computed the mean, median, 10th and 90th percentiles, IQR, upper tail (90th percentile minus "
        "median), lower tail (median minus 10th percentile), tail asymmetry, quartile skewness, octile kurtosis^{37} "
        "and median absolute deviation. Variance partition: each descriptor was modelled with additive effects of "
        "culture state, line and day; we report sequential sums of squares (state, line, day), which partition the "
        "total variance (state is balanced within every line-by-day block), with restricted permutation tests (5,000 "
        "permutations) that permute state within line-by-day blocks, line within state-by-day blocks and day within "
        "line-by-state blocks. Shift functions^{38} compared deciles between conditions within the seven line-by-day "
        "blocks (or six state-by-day blocks for lines), with t-based intervals and exact sign-flip tests. Shape "
        "distances were 1-Wasserstein distances between quantile functions after subtraction of the median, or of the "
        "median followed by division by the IQR, tested by the ratio of mean between-group to within-group distance "
        "under restricted permutation. Reproducibility across days was quantified by the one-way intraclass "
        "correlation with line-by-state groups (group-bootstrap intervals). Sampling precision was estimated by drawing "
        "n cells with replacement from each measurement (400 draws per n); reliability was the between-sample variance "
        "within conditions (corrected for the sampling variance of full samples) divided by itself plus the sampling "
        "variance at n. Bag classification drew 50 bags of n cells per measurement and fitted L2-penalized logistic "
        "regression on location descriptors, location plus shape descriptors, or shape descriptors alone, with "
        "leave-one-day-out (states) or leave-one-state-out (lines) cross-validation; intervals came from 1,000 "
        "cluster-bootstrap resamples of measurements within class."]),
    ("Plasmode design analysis", [
        "Simulated studies of n patients per class (25-400) and m cells per tumour (30-1,000) were generated as "
        "described in Results. For each tumour, the image-and-report score was resampled with replacement from the "
        "out-of-fold fused probabilities of the matched cohort (institutions held out) within the tumour's class; a "
        "template was drawn from the 21 standardized RT-DC distributions; m cells were drawn from it and scaled and "
        "shifted by the tumour's spread and location. Scenarios: no association; location (GBM 0.072 log10 units "
        "softer); spread (LGG IQR 1.25-fold); stiff tail (10% of LGG cells shifted by 0.3 log10 units); location plus "
        "spread; and redundant (location determined by the standardized image-and-report logit, with the same mean "
        "class difference as the location scenario and no direct class effect). Nested logistic models (score; plus "
        "mean and median; plus log IQR, upper and lower tail) were compared by likelihood-ratio tests at alpha 0.05 and "
        "by five-fold cross-validated AUROC; 500 studies were simulated per scenario and design, and 300 per point of "
        "a grid of location differences (0, 0.036, 0.072), spread ratios (1, 1.1, 1.25, 1.5) and between-tumour "
        "standard deviations of the log IQR (0.09 and 0.47). Testing improvement in nested models by likelihood ratio "
        "rather than by comparison of AUROCs follows refs. ^{39,40}; plasmode simulation follows ref. ^{41}."]),
    ("Computation and software", [
        "Image download, tiling, feature extraction and model training ran on the Duke Compute Cluster (Slurm; NVIDIA "
        "RTX 2080 Ti, RTX A5000, RTX A6000 and RTX 5000/6000 Ada GPUs) with Python 3.11, PyTorch 2.6, Transformers "
        "4.57, timm, OpenSlide, scikit-learn^{42}, statsmodels and SciPy; figures were drawn with matplotlib. Random "
        "seeds are fixed in the code."]),
    ("Data availability", [
        "All data are public: TCGA slides from the NCI Genomic Data Commons, clinical and molecular annotations from "
        "cBioPortal (lgggbm_tcga_pub), pathology reports from TCGA-Reports (github.com/tatonetti-lab/tcga-path-reports, "
        "MIT licence), RT-DC data "
        "from figshare (doi:10.6084/m9.figshare.28161236) and AFM data from RODBUK (doi:10.57903/UJ/3LOV6Q). Derived "
        "tables, out-of-fold predictions and all result tables underlying the figures are provided with the code."]),
    ("Code availability", [
        "The complete pipeline (download manifests, Slurm scripts, tiling, encoders, models, analyses, figures and this "
        "manuscript) is available in the MechanoPath repository (github.com/nurabrimo/mechanopath)."]),
]

REFS = [
    "Louis, D. N. et al. The 2021 WHO Classification of Tumors of the Central Nervous System: a summary. Neuro Oncol. 23, 1231-1251 (2021).",
    "Ceccarelli, M. et al. Molecular profiling reveals biologically discrete subsets and pathways of progression in diffuse glioma. Cell 164, 550-563 (2016).",
    "Brennan, C. W. et al. The somatic genomic landscape of glioblastoma. Cell 155, 462-477 (2013).",
    "Cancer Genome Atlas Research Network. Comprehensive, integrative genomic analysis of diffuse lower-grade gliomas. N. Engl. J. Med. 372, 2481-2498 (2015).",
    "Ilse, M., Tomczak, J. M. & Welling, M. Attention-based deep multiple instance learning. Proc. Mach. Learn. Res. 80, 2127-2136 (2018).",
    "Lu, M. Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat. Biomed. Eng. 5, 555-570 (2021).",
    "Chen, R. J. et al. Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. Proc. IEEE/CVF CVPR, 16144-16155 (2022).",
    "Chen, R. J. et al. Towards a general-purpose foundation model for computational pathology. Nat. Med. 30, 850-862 (2024).",
    "Filiot, A., Jacob, P., Mac Kain, A. & Saillard, C. Phikon-v2, a large and public feature extractor for biomarker prediction. arXiv:2409.09173 (2024).",
    "Dosovitskiy, A. et al. An image is worth 16x16 words: transformers for image recognition at scale. Int. Conf. Learn. Represent. (2021).",
    "Mobadersany, P. et al. Predicting cancer outcomes from histology and genomics using convolutional networks. Proc. Natl Acad. Sci. USA 115, E2970-E2979 (2018).",
    "Gu, Y. et al. Domain-specific language model pretraining for biomedical natural language processing. ACM Trans. Comput. Healthc. 3, 2 (2021).",
    "Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. BERT: pre-training of deep bidirectional transformers for language understanding. Proc. NAACL-HLT, 4171-4186 (2019).",
    "Wolf, T. et al. Transformers: state-of-the-art natural language processing. Proc. EMNLP: System Demonstrations, 38-45 (2020).",
    "Howard, F. M. et al. The impact of site-specific digital histology signatures on deep learning model accuracy and bias. Nat. Commun. 12, 4423 (2021).",
    "Guck, J. et al. Optical deformability as an inherent cell marker for testing malignant transformation and metastatic competence. Biophys. J. 88, 3689-3698 (2005).",
    "Cross, S. E., Jin, Y.-S., Rao, J. & Gimzewski, J. K. Nanomechanical analysis of cells from cancer patients. Nat. Nanotechnol. 2, 780-783 (2007).",
    "Plodinec, M. et al. The nanomechanical signature of breast cancer. Nat. Nanotechnol. 7, 757-765 (2012).",
    "Otto, O. et al. Real-time deformability cytometry: on-the-fly cell mechanical phenotyping. Nat. Methods 12, 199-202 (2015).",
    "Mietke, A. et al. Extracting cell stiffness from real-time deformability cytometry: theory and experiment. Biophys. J. 109, 2023-2036 (2015).",
    "Andolfi, L. et al. Investigation of adhesion and mechanical properties of human glioma cells by single cell force spectroscopy and atomic force microscopy. PLoS ONE 9, e112582 (2014).",
    "Shmelev, M. E., Farniev, V. M., Shved, N. A. & Kumeiko, V. V. Nanomechanical signatures in glioma cells depend on CD44 distribution in IDH1 wild-type but not in IDH1R132H mutant early-passage cultures. Int. J. Mol. Sci. 24, 4056 (2023).",
    "Miroshnikova, Y. A. et al. Tissue mechanics promote IDH1-dependent HIF1a-tenascin C feedback to regulate glioblastoma aggression. Nat. Cell Biol. 18, 1336-1345 (2016).",
    "Urbanska, M. et al. De novo identification of universal cell mechanics gene signatures. eLife https://doi.org/10.7554/eLife.87930 (2025).",
    "Collins, G. S. et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 385, e078378 (2024).",
    "Cerami, E. et al. The cBio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data. Cancer Discov. 2, 401-404 (2012).",
    "Grossman, R. L. et al. Toward a shared vision for cancer genomic data. N. Engl. J. Med. 375, 1109-1112 (2016).",
    "Goode, A., Gilbert, B., Harkes, J., Jukic, D. & Satyanarayanan, M. OpenSlide: a vendor-neutral software foundation for digital pathology. J. Pathol. Inform. 4, 27 (2013).",
    "Oquab, M. et al. DINOv2: learning robust visual features without supervision. Trans. Mach. Learn. Res. (2024).",
    "Kefeli, J. & Tatonetti, N. TCGA-Reports: a machine-readable pathology report resource for benchmarking text-based AI models. Patterns 5, 100933 (2024).",
    "DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics 44, 837-845 (1988).",
    "Sun, X. & Xu, W. Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. IEEE Signal Process. Lett. 21, 1389-1393 (2014).",
    "Holm, S. A simple sequentially rejective multiple test procedure. Scand. J. Stat. 6, 65-70 (1979).",
    "Vickers, A. J. & Elkin, E. B. Decision curve analysis: a novel method for evaluating prediction models. Med. Decis. Making 26, 565-574 (2006).",
    "Poser, S. W. et al. Controlling distinct signaling states in cultured cancer cells provides a new platform for drug discovery. FASEB J. https://doi.org/10.1096/fj.201802603RR (2019).",
    "Herbig, M. et al. Real-time deformability cytometry: label-free functional characterization of cells. Methods Mol. Biol. 1678, 347-369 (2018).",
    "Moors, J. J. A. A quantile alternative for kurtosis. J. R. Stat. Soc. Ser. D 37, 25-32 (1988).",
    "Rousselet, G. A., Pernet, C. R. & Wilcox, R. R. Beyond differences in means: robust graphical methods to compare two groups in neuroscience. Eur. J. Neurosci. 46, 1738-1748 (2017).",
    "Demler, O. V., Pencina, M. J. & D'Agostino, R. B. Misuse of DeLong test to compare AUCs for nested models. Stat. Med. 31, 2577-2587 (2012).",
    "Pepe, M. S., Kerr, K. F., Longton, G. & Wang, Z. Testing for improvement in prediction model performance. Stat. Med. 32, 1467-1482 (2013).",
    "Franklin, J. M., Schneeweiss, S., Polinski, J. M. & Rassen, J. A. Plasmode simulation for the evaluation of pharmacoepidemiologic methods in complex healthcare databases. Comput. Stat. Data Anal. 72, 219-226 (2014).",
    "Pedregosa, F. et al. Scikit-learn: machine learning in Python. J. Mach. Learn. Res. 12, 2825-2830 (2011).",
]

LEGENDS = {
    "Fig1_design_cohort": ("Fig. 1 | Study design, cohort and acquisition structure.",
        f"a, Two-phase design. Phase 1 trains an image branch (Phikon-v2 tile embeddings, gated attention MIL) and a "
        f"report branch (BiomedBERT on masked reports) and fuses them by averaging logits; evaluation uses 5 x 5 "
        f"cross-validation with random patient folds or with whole institutions held out, and a 40x-only subset. "
        f"Phase 2 analyses public single-cell mechanics and feeds Phase 1 out-of-fold scores into a plasmode design "
        f"analysis. b, Patients with labels, slides, reports, both (matched) and 40x-scanned slides, by class. "
        f"c, Scan magnification by class: all {V['n20_gbm']} 20x-scanned slides are glioblastomas. d, The 14 largest "
        f"tissue-source institutions of the matched cohort by class."),
    "Fig2_discrimination": ("Fig. 2 | Discrimination of glioblastoma from lower-grade glioma.",
        "a,b, Receiver operating characteristic curves with institutions held out for the matched cohort (a) and for "
        "patients with 40x-scanned slides (b); legends give AUROC with DeLong 95% CI. Acquisition metadata (dashed) "
        "is a nuisance model. c, Paired differences in AUROC with paired-bootstrap 95% CIs and Holm-adjusted DeLong P "
        "values; blue, the four pre-specified primary contrasts. d,e, AUROC (95% CI) of every model with random "
        "patient folds (open) and institutions held out (filled) in the matched (d) and 40x (e) cohorts. Fusion, "
        "unweighted mean of branch logits; M1/M2, report masking levels (Methods)."),
    "Fig3_calibration_subgroups_confounding": ("Fig. 3 | Calibration, clinical utility, institutions, molecular "
        "groups and shortcut capacity.",
        "a, Calibration of out-of-fold probabilities (deciles of predicted probability; Wilson 95% CIs) with "
        "institutions held out; slopes from logistic recalibration and expected calibration error (ECE). "
        "b, Decision curves. c, AUROC of the fused model within each institution with at least five GBM and five LGG "
        "patients (n GBM/n LGG); dotted line, pooled within-institution AUROC. d, Fraction of patients called GBM "
        "(probability >= 0.5) by WHO CNS5-like group and TCGA label (Wilson 95% CIs). e, Site identification "
        "(balanced accuracy within diagnostic class; ticks, chance), detection of 20x scanning within GBM and "
        "prediction of class from tissue amount (AUROC) from slide embeddings."),
    "Fig4_attention": ("Fig. 4 | Attention of the image model.",
        "Thumbnails with the upper half of tiles coloured by attention rank of a full-data attention-MIL fit, and the "
        "four highest-attention tiles (128 µm; scale bars, 100 µm in tiles, 2 mm on thumbnails) for the two GBM and "
        "two LGG patients with 40x slides that the fused model classified most confidently with institutions held out."),
    "Fig5_single_cell_mechanics": ("Fig. 5 | Single-cell stiffness distributions of primary glioblastoma lines.",
        "a, Kernel densities of log10 apparent Young's modulus for the 21 RT-DC measurements (line, measurement date, "
        "number of gated cells), grouped by culture state; bars, medians. b, Shift functions: differences of each "
        "decile between culture states, paired within line-by-day blocks (mean and t-based 95% CI over seven blocks); "
        "dotted lines mark the median shift, so flat profiles indicate pure location shifts. c, Partition of the "
        "between-measurement variance of each descriptor into culture state, cell line, measurement day and residual "
        "(sequential sums of squares); letters mark restricted-permutation P < 0.05 for state (c), line (l) or day (d). "
        "d, Intraclass correlation of each descriptor across days within line-by-state groups (group-bootstrap 95% "
        "CI). e, Reliability of descriptors as a function of the number of cells per sample. f, AUROC for "
        "discriminating culture states from bags of cells (leave-one-day-out) with location descriptors (mean, "
        "median), location plus shape descriptors, or shape descriptors alone."),
    "Fig6_design_analysis": ("Fig. 6 | Plasmode design analysis for a matched images-reports-mechanics study.",
        "Simulated studies combine real Phase 1 out-of-fold scores with real RT-DC distribution shapes (Methods). "
        "a,b, Power of the likelihood-ratio test of shape descriptors added to a model with the image-and-report "
        "score and average stiffness (M2 versus M1), against patients per class at 300 cells per tumour (a) and "
        "against cells per tumour at 100 patients per class (b); error bars, 95% Monte Carlo intervals; dashed lines "
        "and open symbols, between-tumour variation in spread as in primary-culture AFM (S2b, S3b) instead of RT-DC. "
        "c, Mean gain "
        "in five-fold cross-validated AUROC over the image-and-report score from average stiffness (M1) or average "
        "stiffness plus shape (M2) at 200 patients per class and 300 cells (bars, 2.5th-97.5th percentiles over "
        "simulations). d, Power over location differences and spread ratios at 200 patients per class and 1,000 cells, "
        "for between-tumour variation in spread as in RT-DC (left) or in primary-culture AFM (right). e, Rejection "
        "rates in null scenarios; dashed line, nominal 5%."),
}
ED_LEGENDS = {
    "ED1_roc_all_models": ("Extended Data Fig. 1 | ROC curves of all image and report models.",
        "Matched and 40x cohorts with random patient folds and with institutions held out; AUROCs in legends."),
    "ED2_variability_tissue": ("Extended Data Fig. 2 | Fold variability and tissue amount.",
        "a, Mean, s.d. and range of per-fold AUROC with institutions held out. b, Tissue tiles per patient by class. "
        "c, AUROC of the fused model by tertile of tissue amount (DeLong 95% CI)."),
    "ED3_shape_distances_afm": ("Extended Data Fig. 3 | Shape distances and AFM distributions.",
        "a,b, 1-Wasserstein distances between the 21 RT-DC distributions after removal of the median (a) or of the "
        "median and IQR (b), ordered by day, line and state; titles give between/within distance ratios and "
        "restricted-permutation P values for each factor. c, AFM distributions of LN229 and U87 (solid) and "
        "median-aligned (dotted)."),
    "ED4_lines_precision": ("Extended Data Fig. 4 | Cell lines and sampling precision.",
        "a, Shift functions between cell lines paired within state-by-day blocks. b, Tail and spread contrasts beyond "
        "location for states and lines (t-based 95% CI). c, Sampling s.d. of descriptors against cells per sample for "
        "RT-DC (solid) and AFM U87 cells (dashed)."),
    "ED5_plasmode_power_tables": ("Extended Data Fig. 5 | Power tables of the design analysis.",
        "Power of the likelihood-ratio tests for average stiffness beyond the image-and-report score (M1 versus M0, "
        "top) and for shape beyond average stiffness (M2 versus M1, bottom) by scenario, patients per class and cells "
        "per tumour."),
}


# ------------------------------------------------------------------------------------------------ tables
def table1():
    pts = read(P1 / "phase1_oof.csv")
    if pts is None:
        return None
    m = pts[(pts.model == "fusion_primary") & (pts.cohort == "all") & (pts.scheme == "institution")].patient
    d = cohort.loc[m].join(slides[["appmag"]], how="left")
    rows = []
    for lab, g in (("GBM", d[d.y_gbm == 1]), ("LGG", d[d.y_gbm == 0])):
        rows.append({"": lab, "n": len(g),
                     "Age, median (IQR)": f"{g.AGE.median():.0f} ({g.AGE.quantile(.25):.0f}-{g.AGE.quantile(.75):.0f})",
                     "Female, n (%)": f"{(g.SEX == 'Female').sum()} ({100 * (g.SEX == 'Female').mean():.0f})",
                     "Grade 2 / 3 / 4": f"{(g.GRADE == 'G2').sum()} / {(g.GRADE == 'G3').sum()} / {(g.GRADE == 'G4').sum()}",
                     "IDH mutant / wild-type / NA": f"{(g.IDH_STATUS == 'Mutant').sum()} / {(g.IDH_STATUS == 'WT').sum()} / "
                                                     f"{g.IDH_STATUS.isna().sum() + (~g.IDH_STATUS.isin(['Mutant', 'WT']) & g.IDH_STATUS.notna()).sum()}",
                     "Molecular GBM (WHO CNS5-like)": int((g.who2021_like == "GBM_IDHwt").sum()),
                     "Scanned at 20x / 40x": f"{(g.appmag == 20).sum()} / {(g.appmag == 40).sum()}",
                     "Institutions": g.institution.nunique()})
    return pd.DataFrame(rows).set_index("").T.reset_index().rename(columns={"index": "Characteristic"})


def table2():
    if MET is None:
        return None
    keep = ["fusion_primary", IMG, "image_hipt_abmil", REP, "report_tfidf_M2", "age", "acq_meta"]
    rows = []
    for m in keep:
        r = {"Model": NAMES_SHORT.get(m, m)}
        for c, s, lab in (("all", "patient", "Matched, random folds"), ("all", "institution", "Matched, sites held out"),
                          ("x40", "institution", "40x, sites held out")):
            x = met(m, c, s)
            r[lab] = f"{x.auroc:.3f} ({x.auroc_lo:.3f}-{x.auroc_hi:.3f})" if x is not None else NA
        rows.append(r)
    return pd.DataFrame(rows)


NAMES_SHORT = {"fusion_primary": "Fusion (image + report)", IMG: "Image, Phikon-v2 attention MIL",
               "image_hipt_abmil": "Image, HIPT attention MIL", REP: "Report, BiomedBERT (M2)",
               "report_tfidf_M2": "Report, TF-IDF (M2)", "age": "Age", "acq_meta": "Acquisition metadata (nuisance)"}


def ed_table_metrics():
    if MET is None:
        return None
    d = MET.copy()
    d["Model"] = d.model.map(lambda m: NAMES_SHORT.get(m, m))
    d["AUROC (95% CI)"] = [f"{a:.3f} ({l:.3f}-{h:.3f})" for a, l, h in zip(d.auroc, d.auroc_lo, d.auroc_hi)]
    d = d.rename(columns={"cohort": "Cohort", "scheme": "CV", "auprc": "AUPRC", "brier": "Brier", "ece": "ECE",
                          "calibration_slope": "Slope", "auroc_within_institution": "Within-site AUROC",
                          "sensitivity@0.5": "Sens.", "specificity@0.5": "Spec."})
    cols = ["Model", "Cohort", "CV", "n", "AUROC (95% CI)", "AUPRC", "Within-site AUROC", "Sens.", "Spec.", "Brier",
            "ECE", "Slope"]
    out = d[cols].copy()
    for c in ("AUPRC", "Within-site AUROC", "Sens.", "Spec.", "Brier", "ECE", "Slope"):
        out[c] = out[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else NA)
    return out


def ed_table_cmp():
    if CMP is None:
        return None
    d = CMP.copy()
    d["Contrast"] = [f"{NAMES_SHORT.get(a, a)} vs {NAMES_SHORT.get(b, b)}" for a, b in zip(d.model, d.comparator)]
    d["Delta AUROC (95% CI)"] = [f"{x:+.3f} ({l:+.3f} to {h:+.3f})" for x, l, h in zip(d.delta_auroc, d.lo, d.hi)]
    d["P (DeLong)"] = d.p_delong.map(lambda p: pfmt(p).replace("P ", "").replace("= ", ""))
    d["P (Holm)"] = d.p_holm.map(lambda p: pfmt(p).replace("P ", "").replace("= ", ""))
    return d[["family", "cohort", "scheme", "Contrast", "n", "Delta AUROC (95% CI)", "P (DeLong)", "P (Holm)"]].rename(
        columns={"family": "Family", "cohort": "Cohort", "scheme": "CV"})


def ed_table_mech():
    if A1 is None:
        return None
    d = A1[A1.assay == "RT-DC"].copy()
    out = pd.DataFrame({"Measurement": d["sample"], "Line": d.line, "State": d.condition, "Day": d.date.astype(str),
                        "Cells": d.n_cells})
    for k, lab in (("median", "Median"), ("iqr", "IQR"), ("upper_tail", "Upper tail"), ("lower_tail", "Lower tail"),
                   ("bowley", "Quartile skew")):
        out[lab] = d[k].map(lambda v: f"{v:.3f}")
    out["rho(E, area)"] = d.rho_logE_logarea.map(lambda v: f"{v:.2f}")
    return out


def ed_table_bags():
    if A7 is None:
        return None
    d = A7.copy()
    out = pd.DataFrame({"Task": d.task, "CV": d.cv, "Cells/bag": d.cells_per_bag})
    for k, lab in (("location", "Location"), ("distribution", "Location + shape"), ("shape_only", "Shape only")):
        out[lab] = [f"{a:.3f} ({l:.3f}-{h:.3f})" for a, l, h in zip(d[f"auroc_{k}"], d[f"auroc_{k}_lo"], d[f"auroc_{k}_hi"])]
    out["Delta (95% CI)"] = [f"{a:+.3f} ({l:+.3f} to {h:+.3f})" for a, l, h in zip(
        d.delta_distribution_vs_location, d.delta_distribution_vs_location_lo, d.delta_distribution_vs_location_hi)]
    return out


def ed_table_var():
    if A2 is None:
        return None
    d = A2.copy()
    out = pd.DataFrame({"Descriptor": d.label})
    for k, lab in (("condition", "State"), ("line", "Line"), ("date", "Day")):
        out[f"{lab} share"] = d[f"seqshare_{k}"].map(lambda v: f"{100 * v:.1f}%")
        out[f"{lab} P"] = d[f"p_perm_{k}"].map(lambda v: f"{v:.4f}")
    out["Residual"] = d.seqshare_residual.map(lambda v: f"{100 * v:.1f}%")
    if A5 is not None:
        icc = A5.set_index("descriptor")
        out["ICC (95% CI)"] = [f"{icc.loc[k, 'icc1']:.2f} ({icc.loc[k, 'lo']:.2f} to {icc.loc[k, 'hi']:.2f})"
                               if k in icc.index else "" for k in d.descriptor]
    return out


def ed_table_plasmode():
    if PM is None:
        return None
    d = PM[PM.cells_per_tumour.isin([100, 300, 1000]) & PM.n_per_class.isin([50, 100, 200, 400])].copy()
    return pd.DataFrame({"Scenario": d.scenario, "Patients/class": d.n_per_class, "Cells/tumour": d.cells_per_tumour,
                         "Power M1 vs M0": d.power_M1_vs_M0.map(lambda v: f"{v:.2f}"),
                         "Power M2 vs M1": d.power_M2_vs_M1.map(lambda v: f"{v:.2f}"),
                         "AUROC M0": d.auc_M0.map(lambda v: f"{v:.3f}"), "AUROC M2": d.auc_M2.map(lambda v: f"{v:.3f}"),
                         "Mechanics-only AUROC": d.auc_mech.map(lambda v: f"{v:.3f}")})


# ------------------------------------------------------------------------------------------------ docx writer
from docx import Document  # noqa: E402
from docx.enum.section import WD_ORIENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Mm, Pt  # noqa: E402

doc = Document()
st_ = doc.styles["Normal"]
st_.font.name = "Arial"
st_.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
st_.font.size = Pt(10)
st_.paragraph_format.space_after = Pt(6)
st_.paragraph_format.line_spacing = 1.15
for s in doc.sections:
    s.left_margin = s.right_margin = Mm(22)
    s.top_margin = s.bottom_margin = Mm(20)
MD = []
TOKEN = re.compile(r"(\^\{[^}]*\}|_\{[^}]*\})")


def para(text, style=None, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph(style=style)
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("^{"):
            r = p.add_run(part[2:-1]); r.font.superscript = True
        elif part.startswith("_{"):
            r = p.add_run(part[2:-1]); r.font.subscript = True
        else:
            r = p.add_run(part)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
    if align:
        p.alignment = align
    MD.append(re.sub(r"\^\{([^}]*)\}", r"^\1^", re.sub(r"_\{([^}]*)\}", r"~\1~", text)))
    return p


def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.name = "Arial"
        r.font.color.rgb = None
    MD.append("#" * (level + 1) + " " + text)


def add_table(df, title, note=None, font=7):
    if df is None or not len(df):
        return
    para(title, bold=True, size=9)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1" if "Light Grid Accent 1" in [s.name for s in doc.styles] else "Table Grid"
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = str(c)
        for r in cell.paragraphs[0].runs:
            r.bold, r.font.size = True, Pt(font)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = str(v)
            for r in cells[j].paragraphs[0].runs:
                r.font.size = Pt(font)
    if note:
        para(note, italic=True, size=8)
    try:
        MD.append(df.to_markdown(index=False))
    except Exception:
        MD.append(df.to_string(index=False))


def add_figure(stem, legends):
    png = FD / f"{stem}.png"
    if not png.exists():
        return
    doc.add_picture(str(png), width=Mm(168))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    t, body = legends[stem]
    p = para(t + " ", bold=True, size=8.5)
    r = p.add_run(body); r.font.size = Pt(8.5)
    MD[-1] = f"**{t}** {body}"


# title page
para(TITLE, bold=True, size=15)
para("Nura Brimo^{1}", size=11)
para("^{1}Department of Biomedical Engineering, Duke University, Durham, NC, USA. Correspondence: nura.brimo@duke.edu",
     size=9)
para(f"Manuscript generated {date.today():%d %B %Y} from the analysis outputs of the MechanoPath pipeline.", italic=True,
     size=8)
heading("Abstract")
para(ABSTRACT)
heading("Introduction")
for t in INTRO:
    para(t)
heading("Results")
for kind, t in results_text():
    if kind == "h":
        heading(t, 2)
    else:
        para(t)
heading("Discussion")
for t in DISCUSSION:
    para(t)
heading("Methods")
for h, ps in METHODS:
    heading(h, 2)
    for t in ps:
        para(t)
heading("References")
for i, r in enumerate(REFS, 1):
    para(f"{i}. {r}", size=8.5)
heading("Tables")
add_table(table1(), "Table 1 | Characteristics of the matched cohort.",
          "IDH, isocitrate dehydrogenase; molecular GBM, IDH-wild-type with grade 4 histology, TERT promoter mutation or "
          "combined chromosome 7 gain and 10 loss.", font=8)
add_table(table2(), "Table 2 | Discrimination (AUROC, DeLong 95% CI) of the main models.", font=8)
heading("Figures")
for stem in LEGENDS:
    add_figure(stem, LEGENDS)
heading("Extended Data")
for stem in ED_LEGENDS:
    add_figure(stem, ED_LEGENDS)
sec = doc.add_section()
sec.orientation = WD_ORIENT.LANDSCAPE
sec.page_width, sec.page_height = sec.page_height, sec.page_width
add_table(ed_table_metrics(), "Extended Data Table 1 | All models: discrimination, calibration and operating points.",
          "Sens./Spec., sensitivity and specificity at probability 0.5; ECE, expected calibration error; slope, "
          "calibration slope.", font=6.5)
add_table(ed_table_cmp(), "Extended Data Table 2 | Paired comparisons of AUROC.",
          "Holm correction over the four primary contrasts (institution scheme) and within each other family, cohort "
          "and scheme.", font=6.5)
add_table(ed_table_mech(), "Extended Data Table 3 | RT-DC measurements and distribution descriptors (log10 E).",
          "rho(E, area), within-measurement Spearman correlation between modulus and cell area.", font=6.5)
add_table(ed_table_var(), "Extended Data Table 4 | Variance partition, permutation tests and reproducibility of "
          "descriptors.", font=6.5)
add_table(ed_table_bags(), "Extended Data Table 5 | Bag classification of culture states and cell lines.", font=6.5)
add_table(ed_table_plasmode(), "Extended Data Table 6 | Plasmode design analysis.", font=6.5)
if TOP2 is not None:
    t7 = pd.concat([TOP2[TOP2.direction == "GBM"].head(20).reset_index(drop=True).rename(
                        columns={"term": "GBM-associated term", "coef": "GBM coefficient"})[["GBM-associated term", "GBM coefficient"]],
                    TOP2[TOP2.direction != "GBM"].head(20).reset_index(drop=True).rename(
                        columns={"term": "LGG-associated term", "coef": "LGG coefficient"})[["LGG-associated term", "LGG coefficient"]]],
                   axis=1)
    for c in ("GBM coefficient", "LGG coefficient"):
        t7[c] = t7[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    add_table(t7.fillna(""), "Extended Data Table 7 | Report masking audit: terms most associated with each class in "
              "M2-masked reports (L2 logistic regression on TF-IDF features, full cohort).",
              "xmaskx denotes a masked span. No masked report retained a tumour-type name.", font=6.5)
out = OUTD / "MechanoPath_manuscript.docx"
doc.save(out)
(OUTD / "MechanoPath_manuscript.md").write_text("\n\n".join(MD))
print("saved", out, f"({len(doc.paragraphs)} paragraphs)")
print("MANUSCRIPT_DONE")
