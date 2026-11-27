"""Step 8 - how large must a matched (image + report + cell-mechanics) cohort be?

The central claim of the project needs a paired comparison on the same held-out patients:
AUROC(image + report + mechanics) vs AUROC(image + report). This simulation gives the power
of a two-sided DeLong test (alpha = 0.05) for that comparison under a binormal model:
  base score   S ~ N(mu_b * y, 1)                     (image + report, AUROC = A0)
  mechanics    M ~ N(mu_m * y, 1), independent of S given y
  combined     S + (mu_m / mu_b) * M ... the optimal (LDA) weight, so this is an upper bound:
               a weight learned on a small cohort loses some power.
mu_m is solved so that the combined AUROC equals A0 + delta.

Output: results/power_matched_cohort.csv and a printed table (n per class needed for 80% power).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import RES, SEED, delong_test  # noqa: E402


def auc_from_d(d):
    return stats.norm.cdf(d / np.sqrt(2))


def d_from_auc(a):
    return np.sqrt(2) * stats.norm.ppf(a)


rng = np.random.default_rng(SEED)
rows = []
N_SIM = 300
for A0 in [0.85, 0.90, 0.95]:
    db = d_from_auc(A0)
    for delta in [0.02, 0.03, 0.05, 0.08]:
        A1 = A0 + delta
        if A1 >= 0.995:
            continue
        dm = np.sqrt(max(d_from_auc(A1) ** 2 - db ** 2, 1e-9))   # independent signals add in quadrature
        for n in [25, 50, 100, 200, 400]:
            hits = 0
            for _ in range(N_SIM):
                y = np.r_[np.ones(n), np.zeros(n)].astype(int)
                S = rng.normal(db * y, 1)
                M = rng.normal(dm * y, 1)
                comb = S + (dm / db) * M
                _, _, _, p = delong_test(y, comb, S)
                hits += p < 0.05
            rows.append({"auroc_image_report": A0, "delta_auroc": delta, "n_per_class": n,
                         "mechanics_alone_auroc": auc_from_d(dm), "power": hits / N_SIM})
            print(rows[-1], flush=True)
R = pd.DataFrame(rows)
R.to_csv(RES / "power_matched_cohort.csv", index=False)
need = (R[R.power >= 0.8].groupby(["auroc_image_report", "delta_auroc"]).n_per_class.min()
        .unstack().reindex(index=[0.85, 0.90, 0.95]))
print("\nsmallest n per class (of 25/50/100/200/400) reaching 80% power:\n", need)
