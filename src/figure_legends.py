"""Titles and legends of the Nature-style figures, shared by src/12_nature_figures.py (writes
figures/nature/figure_legends.md) and report/build_content.py (Word captions).

Nature puts figure titles and footnotes in the legend, not on the artwork, so every title, suptitle and
footnote that the original figures (src/10_figures.py, src/11_report_figures.py) carried on the image is
kept here word for word; only left/middle/right became panel letters and x became the multiplication sign.
"""

# model names used in tables and supplementary figures (same as report/build_content.py)
MODEL_NAMES = {
    "img_meanpool": "Image, HIPT mean-pool", "img_abmil": "Image, attention MIL", "handcrafted": "Image, tissue features",
    "acq_meta": "Acquisition metadata only", "stain_color": "Stain colour + sharpness only",
    "tfidf_M0": "Report, unmasked M0", "tfidf_M1": "Report, masked M1", "tfidf_M2": "Report, masked M2", "age": "Age only",
    "tfidf_M1+age": "TF-IDF M1 + age",
    "image_meanpool": "Image, HIPT mean-pool", "image_abmil": "Image, attention MIL", "image_handcrafted": "Image, tissue features",
    "report_M1": "Report, masked M1", "report_M2": "Report, masked M2", "report_M0 (unmasked)": "Report, unmasked M0",
    "acq_meta (nuisance)": "Acquisition metadata only", "stain_color (nuisance)": "Stain colour + sharpness only",
    "late_image+report_M1": "Late fusion: image + report M1", "late_image+report_M2": "Late fusion: image + report M2",
    "late_image+report_M2+age": "Late fusion: image + report M2 + age",
    "late_image+acq_meta": "Late fusion: image + acquisition metadata",
    "early_image+report_M2": "Early fusion: image + report M2", "three_input_with_mechanics": "Three inputs incl. cell mechanics"}

# name -> (title, legend). Main figures keep the numbering and captions of the report; S figures plot
# results that the report otherwise gives only as tables.
LEGENDS = {
    "fig0_pipeline": (
        "Analysis pipeline.",
        "Solid boxes ran on linked TCGA patients; dashed boxes ran on public cell mechanics that are not linked to any "
        "patient, so the three-input test is specified but not yet runnable."),
    "fig1_inventory": (
        "Which TCGA glioma patients have which input.", ""),
    "fig13_institutions": (
        "Institutions with 10 or more patients: most sent only one class.",
        "Patients per institution (institutions with 10 or more patients)."),
    "fig2_acquisition_confounding": (
        "Acquisition differs by class: every 20×-scanned slide is a GBM.",
        "Scan year (a), scan magnification (b) and JPEG quality (c) by class."),
    "contact_TCGA-06-0156": (
        "QC contact sheet for one slide.",
        "a, Thumbnail with the sampled tile footprints (solid vermilion = haematoxylin-dense; dashed green = random; small "
        "boxes 20×, large boxes 5×). b, The 16 tiles (rows 1–3, 20×; row 4, 5×), framed in the colour of their sampling "
        "stratum. Slide TCGA-06-0156. Scale bars: 2 mm (a), 50 µm (20× tiles) and 200 µm (5× tiles)."),
    "fig4_report_masking": (
        "Report branch: how much survives masking.",
        "Report AUROC by masking level under random-patient and institution-held-out validation; grey line = age "
        "alone. Shading, 95% bootstrap CI."),
    "fig11_embedding_map": (
        "Patient-level HIPT embeddings (mean of 20× crops): scanner groups are as visible as diagnosis.",
        "UMAP of patient-level mean HIPT embeddings, coloured by diagnosis (a) and by scan magnification (b). The "
        "20×-scanned GBM slides form a separate island."),
    "fig12_tissue_features": (
        "Tissue features separate GBM from LGG at either scan magnification; sharpness tracks the scanner.",
        "Interpretable tissue features by class and scan magnification: nuclear area fraction (a), nuclei per mm² (b), "
        "nuclear size IQR (c) and sharpness (d, log scale). Bar = median, line = interquartile range; one point per "
        "patient (mean over QC-passing 20× tiles)."),
    "fig7_attention_tiles": (
        "Highest-attention 20× tile in confidently classified patients (256 µm field).",
        "Highest-attention 20× tile (attention MIL, full-data fit) in three confidently classified GBM (a–c) and three "
        "LGG (d–f) patients. Scale bars, 50 µm."),
    "fig3_auroc_overview": (
        "GBM vs lower-grade glioma: every model, two cohorts, two validation schemes.",
        "AUROC with 95% bootstrap CI for every model, all slides (a) and 40×-scanned slides only (b), under "
        "random-patient and institution-held-out validation. Fusion rows are evaluated on the matched image+report "
        "cohort; branch rows on each branch's full cohort."),
    "fig9_roc_curves": (
        "ROC curves with institutions held out on the matched cohort.",
        "All matched patients (a, n = 686) and 40×-scanned slides only (b, n = 556); AUROC in the key."),
    "fig10_calibration": (
        "Calibration of the fused model.",
        "Calibration of the late-fusion model (deciles of predicted probability)."),
    "fig6_who2021_subgroups": (
        "Where the fused model puts each molecular group.",
        "Fused P(GBM) for each WHO-2021-like molecular group, institutions held out. *IDH-wild-type with TERT-promoter "
        "mutation or +7/−10: glioblastoma under WHO 2021. Bar = median."),
    "fig5_mechanics": (
        "RT-DC mechanics.",
        "Single-cell modulus distributions by condition (a), per-measurement medians (b), and EGF vs FGFJI "
        "classification from bags of cells with a held-out date (c)."),
    "fig8_power": (
        "Power of a paired DeLong test for an AUROC gain from mechanics, by matched cohort size and baseline image + "
        "report AUROC (optimal combination weight assumed).",
        "Baseline image + report AUROC 0.85 (a), 0.90 (b) and 0.95 (c); dashed line, 80% power."),
    # ------------------------------------------------------------------ supplementary (results from tables)
    "figS1_report_roc": (
        "Report-branch ROC curves.",
        "Out-of-fold ROC curves of every report model on all slides (a, b) and on 40×-scanned slides only (c, d), with "
        "random patient folds (a, c) and institutions held out (b, d). AUROC in the key (same values as the "
        "report-branch table)."),
    "figS2_image_roc": (
        "Image-branch ROC curves.",
        "Out-of-fold ROC curves of every image model and nuisance baseline on all slides (a, b) and on 40×-scanned "
        "slides only (c, d), with random patient folds (a, c) and institutions held out (b, d). AUROC in the key (same "
        "values as the image-branch table)."),
    "figS3_tissue_feature_aurocs": (
        "Univariate AUROCs of interpretable tissue features.",
        "Same values as the univariate-feature table: GBM vs LGG on all slides and on 40×-scanned slides only, and 20× "
        "vs 40× scanning within GBM. AUROC < 0.5 means higher in LGG (or in 40×-scanned GBM); the last measure near 0.5 "
        "means the feature does not track the scanner."),
    "figS4_scanner_recognition": (
        "Can the representation recognise acquisition?",
        "Cross-validated AUROC (95% CI) for predicting 20× scanning, JPEG quality 70 and a scan before 2011 among GBM "
        "slides only (positives of n in brackets), from the HIPT embedding, the hand-crafted tissue features, or all "
        "hand-crafted features including colour and sharpness. Dashed line, chance."),
    "figS5_stratified_auroc": (
        "AUROC inside acquisition-homogeneous strata.",
        "Overall AUROC and AUROC within institution × scan-year strata (5 strata, 149 patients) and within JPEG-quality "
        "× software-version strata, with random patient folds (a) and institutions held out (b); all slides and "
        "40×-scanned slides only. Dashed line, chance."),
    "figS6_uf_lrt": (
        "University of Florida subset and likelihood-ratio tests.",
        "a, AUROC (95% CI) on the University of Florida patients only (15 GBM vs 15 LGG, same year and scanner "
        "settings), institutions held out. b, c, Likelihood-ratio tests on 40×-scanned slides: in-sample AUROC of scan "
        "year, JPEG quality and software version alone (open) and with the out-of-fold model score added (filled), with "
        "random patient folds (b) and institutions held out (c); LR χ² (1 df) at right, all p < 0.001."),
    "figS7_fusion_random_rocs_deltas": (
        "Fusion under random patient folds, and every paired AUROC comparison.",
        "a, b, ROC curves on the matched cohort with random patient folds, all matched patients (a, n = 686) and "
        "40×-scanned slides only (b, n = 556). c, d, AUROC change (95% paired-bootstrap CI) of each compared model over "
        "its baseline on the same patients, all slides (c) and 40×-scanned slides only (d); filled circles, random "
        "patient folds; open squares, institutions held out. Same values as the paired-comparison table."),
    "figS8_molecular_grade": (
        "Predictions by molecular group and histological grade.",
        "Mean P(GBM) (a) and percentage called GBM at P = 0.5 (b) for each project label and WHO-2021-like class, and "
        "the same by histological grade (c, d), for the image (HIPT mean-pool), report (masked M2) and late-fusion "
        "models; all slides, institutions held out. n at right (b) and under each grade (c, d)."),
    "figS9_mechanics_models": (
        "Mixed models, variance components and the heterogeneity test.",
        "a, Fold change in Young's modulus (95% CI) from mixed model A (per-measurement medians, random date) and "
        "model B (single cells, random measurement); reference levels serum, line X01, date 2014-11-26. b, Partial R² "
        "of condition, line and date for the per-measurement medians, and R² of the whole model. c–e, Bag AUROC (filled, "
        "solid) and measurement AUROC (open, dashed) by cells per bag, leave-one-date-out, for serum vs FGFJI (c), serum "
        "vs EGF (d) and EGF vs FGFJI (e)."),
}
