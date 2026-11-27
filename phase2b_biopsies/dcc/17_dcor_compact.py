"""Compact single-cell table of the human tumour / healthy biopsy measurements (Phase 2b input).

Reads the per-measurement .npz files written by 06_dcor_extract.py, adds the apparent Young's modulus
(dclab, LE-2D-FEM-19 look-up table, CellCarrier B, 20 um channel, 0.06 ul/s, recorded temperature),
draws a fixed number of gated events per measurement and writes

  dcor_events_compact.csv.gz      one row per sampled event
  dcor_measurements_full.csv      one row per measurement: metadata, event counts, temperature handling
  dcor_measurement_summary.csv    moments and quantiles of the main features over all extracted events
  mechanopath_dcor_compact.tar.gz all three files + README (what is handed to the analysis container)

  python dcc/17_dcor_compact.py --dcor-dir $OUT_DIR/dcor --per-measurement 5000 --out $OUT_DIR/dcor_compact
"""
import argparse
import hashlib
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KEEP = ["time", "area_um", "deform", "area_ratio", "aspect", "volume", "inert_ratio_cvx", "bright_avg", "bright_sd",
        "fl1_max", "fl2_max", "fl3_max", "emodulus"]
SUMFEAT = ["area_um", "deform", "emodulus", "aspect", "volume", "bright_avg", "bright_sd"]
QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
T_RANGE = (15.0, 37.0)


def emodulus(area, deform, temp, px):
    from dclab.features import emodulus as em
    err = None
    for medium in ("CellCarrierB", "CellCarrier B"):
        try:
            e = em.get_emodulus(area_um=area.astype(np.float64), deform=deform.astype(np.float64), medium=medium,
                                channel_width=20.0, flow_rate=0.06, px_um=px, temperature=temp)
            return np.asarray(e, dtype=np.float32), medium
        except Exception as ex:                                            # noqa: BLE001
            err = ex
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcor-dir", required=True)
    ap.add_argument("--per-measurement", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, out = Path(a.dcor_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(src / "dcor_resource_table.csv")
    logs = pd.concat([pd.read_csv(f) for f in sorted(src.glob("dcor_extract_log_*.csv"))], ignore_index=True)
    t = t.merge(logs[["id", "n_total", "n_gated", "n_kept"]], on="id", how="left")
    valid_t = t.temperature[(t.temperature >= T_RANGE[0]) & (t.temperature <= T_RANGE[1])]
    t_fill = float(valid_t.median())
    print(f"{len(t)} measurements; recorded valid temperature in {len(valid_t)}; fallback {t_fill:.1f} C", flush=True)
    frames, summ, meta = [], [], []
    for _, r in t.iterrows():
        f = src / "events" / f"{r['id']}.npz"
        if not f.exists():
            print("missing", r["id"], flush=True)
            continue
        t0 = time.time()
        z = np.load(f)
        n = len(z["deform"])
        ev = {k: (z[k].astype(np.float32) if k in z.files else np.full(n, np.nan, np.float32)) for k in KEEP if k != "emodulus"}
        ev["event_index"] = z["event_index"]
        temp = r["temperature"]
        temp_ok = bool(np.isfinite(temp) and T_RANGE[0] <= temp <= T_RANGE[1])
        use_t = float(temp) if temp_ok else t_fill
        px = float(r["pixel_size"]) if np.isfinite(r["pixel_size"]) else 0.34
        try:
            ev["emodulus"], medium_used = emodulus(ev["area_um"], ev["deform"], use_t, px)
        except Exception as ex:                                            # noqa: BLE001
            print("emodulus failed:", type(ex).__name__, ex, flush=True)
            ev["emodulus"], medium_used = np.full(n, np.nan, np.float32), "none"
        df = pd.DataFrame(ev)
        # summaries over all extracted events
        s = {"id": r["id"], "n_events": n}
        for c in SUMFEAT:
            v = df[c].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            s[f"{c}_n"] = len(v)
            if len(v) > 10:
                s[f"{c}_mean"], s[f"{c}_sd"], s[f"{c}_skew"] = v.mean(), v.std(ddof=1), stats.skew(v)
                for q, val in zip(QS, np.quantile(v, QS)):
                    s[f"{c}_q{int(q * 100):02d}"] = val
        summ.append(s)
        meta.append({"id": r["id"], "temp_used": use_t, "temp_recorded_ok": temp_ok, "emodulus_medium": medium_used,
                     "frac_emodulus_valid": float(np.isfinite(df.emodulus).mean())})
        k = min(a.per_measurement, n)
        rng = np.random.default_rng(int(r["id"][:8], 16))
        sel = np.sort(rng.choice(n, k, replace=False))
        d2 = df.iloc[sel].copy()
        d2.insert(0, "measurement", r["id"])
        frames.append(d2)
        print(f"{r['name']:30s} n={n:6d} sampled={k:5d} T={use_t:4.1f}{'' if temp_ok else '*'} "
              f"emod valid {meta[-1]['frac_emodulus_valid']:.2f} {time.time() - t0:4.1f}s", flush=True)
    ev = pd.concat(frames, ignore_index=True)
    ev.to_csv(out / "dcor_events_compact.csv.gz", index=False, compression="gzip", float_format="%.6g")
    mt = t.merge(pd.DataFrame(meta), on="id").merge(ev.groupby("measurement").size().rename("n_sampled"), left_on="id", right_index=True)
    mt.to_csv(out / "dcor_measurements_full.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "dcor_measurement_summary.csv", index=False, float_format="%.6g")
    readme = (
        "MechanoPath Phase 2b input (generated on the Duke Compute Cluster by dcc/17_dcor_compact.py)\n"
        "Source: DCOR circle soteriou-kubankova (Soteriou, Kubankova et al., Nat Biomed Eng 2023), human biopsies, RT-FDC.\n"
        f"Events: gated (area_um >= 20, area_ratio < 1.2), up to {a.per_measurement} random events per measurement.\n"
        "Apparent Young's modulus: dclab LE-2D-FEM-19, CellCarrier B, 20 um channel, 0.06 ul/s, recorded temperature\n"
        f"(measurements without a valid recorded temperature use the median {t_fill:.1f} C; flag temp_recorded_ok).\n")
    (out / "README.txt").write_text(readme)
    tar = out / "mechanopath_dcor_compact.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for nm in ["dcor_events_compact.csv.gz", "dcor_measurements_full.csv", "dcor_measurement_summary.csv", "README.txt"]:
            tf.add(out / nm, arcname=nm)
    h = hashlib.sha256(tar.read_bytes()).hexdigest()
    print(f"events {len(ev)}  measurements {len(mt)}  archive {tar}  {tar.stat().st_size / 1e6:.1f} MB  sha256 {h}", flush=True)


if __name__ == "__main__":
    main()
