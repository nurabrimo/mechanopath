"""Cell-level parameters of the Phase 2c simulation, moment-matched to measured single-cell distributions.

The simulation draws cells from a two-component mixture (soft majority + stiff minority of weight w0, separation
delta_m, within-component sd sigma_c). For a given source of measurements the parameters are set so that the
simulated within-patient distribution has the observed sd and skewness:
    sd^2   = sigma_c^2 + w0 (1 - w0) delta_m^2
    skew   = w0 (1 - w0) (1 - 2 w0) delta_m^3 / sd^3
with w0 fixed at 0.25 (assumption). tau_loc is the sd of measurement medians that is not explained by the class
or line labels (replicate-to-replicate location variation); tau_w (sd of the minority weight across patients) is
not identifiable from the data and is kept at 0.08 (assumption).

  python src/17_sim_params.py gbm_lines   -> results/phase2/sim_params_gbm_lines.csv   (21 RT-DC measurements, log10 E)
  python src/17_sim_params.py biopsy      -> results/phase2b/sim_params_biopsy.csv     (human biopsies, log10 deformation)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from common import RES  # noqa: E402


def solve(sd, skew, w0=0.25):
    f = w0 * (1 - w0) * (1 - 2 * w0)
    delta = (max(skew, 1e-3) * sd ** 3 / f) ** (1 / 3)
    sig2 = sd ** 2 - w0 * (1 - w0) * delta ** 2
    return delta, float(np.sqrt(max(sig2, 1e-6)))


def from_table(pm, sd_col, skew_col, med_col, formula):
    sd, skew = float(pm[sd_col].median()), float(pm[skew_col].median())
    delta, sig = solve(sd, skew)
    resid = smf.ols(f"{med_col} ~ {formula}", pm).fit().resid
    tau_loc = float(np.sqrt(resid.var(ddof=0) * len(pm) / max(len(pm) - len(smf.ols(f'{med_col} ~ {formula}', pm).fit().params), 1)))
    return {"tau_loc": tau_loc, "tau_w": 0.08, "w0": 0.25, "delta_m": delta, "sigma_c": sig, "obs_within_sd": sd, "obs_within_skew": skew,
            "obs_median_sd_between_measurements": float(pm[med_col].std()), "n_measurements": len(pm)}


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "gbm_lines":
        pm = pd.read_csv(RES / "phase2" / "per_measurement_v2.csv")
        r = from_table(pm, "logE_sd", "logE_skew", "logE_median", "C(line) + C(cond)")
        out = RES / "phase2" / "sim_params_gbm_lines.csv"
    else:
        pm = pd.read_csv(RES / "phase2b" / "per_measurement.csv")
        r = from_table(pm, "D_sd", "D_skew", "D_median", "C(stratum) + C(condition)")
        out = RES / "phase2b" / "sim_params_biopsy.csv"
    pd.DataFrame([r]).to_csv(out, index=False)
    print(pd.DataFrame([r]).round(4).T.to_string())
