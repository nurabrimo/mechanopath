"""Combine the per-task outputs of dcc/04_train_mil.py (array over cohort x scheme) into one table per encoder.

  python dcc/04b_merge_mil.py phikon2 hipt
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
EXPECT = 4                                      # cohort (all, x40) x scheme (patient, institution)
incomplete = []
for enc in sys.argv[1:] or ["phikon2", "hipt"]:
    for kind in ("image_oof", "image_metrics", "image_fold_log"):
        parts = sorted((RES / "mil_parts").glob(f"{kind}_{enc}_part*.csv"))
        if len(parts) < EXPECT:
            print(f"INCOMPLETE {enc}: {kind} has {len(parts)} of {EXPECT} parts")
            incomplete.append(enc)
            continue
        d = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        d.to_csv(RES / f"{kind}_{enc}.csv", index=False)
        print(f"{enc}: {kind} from {len(parts)} parts -> {len(d)} rows")
sys.exit(1 if incomplete else 0)
