"""Swap the full-resolution DCC branch outputs into the files the fusion/confounding scripts read.

  python dcc/merge_dcc_outputs.py --encoder uni

results/image_oof.csv  <- nuisance baselines from the tile-sample run (acq_meta, stain_color,
                          handcrafted; encoder-independent) + img_meanpool/img_abmil from
                          results/image_oof_<encoder>.csv
results/report_oof.csv <- adds bert_M1 / bert_M2 rows if results/report_oof_bert_M*.csv exist
The original tile-sample files are kept as *_tilesample.csv.
"""
import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--encoder", default="uni")
a = ap.parse_args()

for name in ["image_oof", "report_oof"]:
    src, bak = RES / f"{name}.csv", RES / f"{name}_tilesample.csv"
    if not bak.exists():
        shutil.copy(src, bak)

base = pd.read_csv(RES / "image_oof_tilesample.csv")
new = pd.read_csv(RES / f"image_oof_{a.encoder}.csv").drop(columns=["encoder"], errors="ignore")
keep = base[base.model.isin(["acq_meta", "stain_color", "handcrafted"])]
pd.concat([keep, new]).to_csv(RES / "image_oof.csv", index=False)

r = pd.read_csv(RES / "report_oof_tilesample.csv")
for lv in ["M1", "M2"]:
    f = RES / f"report_oof_bert_{lv}.csv"
    if f.exists():
        r = pd.concat([r[r.model != f"bert_{lv}"], pd.read_csv(f)])
r.to_csv(RES / "report_oof.csv", index=False)
print("merged; now run: REPORT_MODEL_M2=bert_M2 python src/05_fusion.py && python src/07_confounding.py")
