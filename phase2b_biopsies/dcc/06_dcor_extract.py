"""Single-cell RT-FDC features of human tumour and matched healthy tissue biopsies from DCOR.

Source: Soteriou, Kubankova et al., Nat Biomed Eng 2023 (doi:10.1038/s41551-023-01015-3); public
datasets (CC0) in the DCOR circle 'soteriou-kubankova' (https://dcor.mpl.mpg.de/organization/soteriou-kubankova).
Human data: frozen colon, fresh colon and fresh lung biopsies, one tumour and one matched healthy
measurement per patient. Only scalar per-event features are read (no images).

  python dcc/06_dcor_extract.py --out-dir $OUT_DIR/dcor --max-events 40000
  python dcc/06_dcor_extract.py --out-dir $OUT_DIR/dcor --combine

Per measurement the script writes <out-dir>/events/<resource id>.npz (all scalar features of a random
subset of the events passing a permissive debris/shape gate), and with --combine
  dcor_measurements.csv   one row per measurement (tissue, preservation, condition, patient, setup, counts)
  dcor_events.csv.gz      events of all measurements (float32)
The three frozen-colon "blind" measurements pairs are identified through the Extended Data Fig. 5
resources that carry identical event counts and acquisition time stamps.
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://dcor.mpl.mpg.de/api/3/action/"
ORG = "soteriou-kubankova"
FEATURES = ["area_um", "deform", "area_ratio", "aspect", "bright_avg", "bright_sd", "fl1_max", "fl2_max", "fl3_max",
            "volume", "inert_ratio_cvx", "inert_ratio_raw", "area_cvx", "size_x", "size_y", "frame", "time", "index_online"]


def ckan(action, **params):
    url = API + action + "?" + urllib.parse.urlencode(params)
    for k in range(5):
        try:
            return json.load(urllib.request.urlopen(url, timeout=120))["result"]
        except Exception as e:                                             # noqa: BLE001
            print("ckan retry", k, e, flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError(url)


def dataset_table():
    res = ckan("package_search", fq=f"organization:{ORG}", rows=100)["results"]
    rows = []
    for p in res:
        n = p["name"]
        if n.startswith("figure-5-") or n.startswith("extended-data-figure-5"):
            for r in p["resources"]:
                rows.append({"dataset": n, "name": r["name"], "id": r["id"], "size": r.get("size", 0),
                             "events": r.get("dc:experiment:event count"), "date": r.get("dc:experiment:date", ""),
                             "time": r.get("dc:experiment:time", ""), "channel_width": r.get("dc:setup:channel width"),
                             "flow_rate": r.get("dc:setup:flow rate"), "temperature": r.get("dc:setup:temperature"),
                             "pixel_size": r.get("dc:imaging:pixel size"), "medium": r.get("dc:setup:medium")})
    return pd.DataFrame(rows)


def annotate(t):
    """tissue, preservation, condition, patient id for each measurement."""
    t = t.copy()
    t["tissue"] = np.where(t.dataset.str.contains("lung"), "lung", "colon")
    t["preservation"] = np.where(t.dataset.str.contains("frozen"), "frozen", "fresh")
    t["condition"] = None
    t["patient"] = None
    t["role"] = "main"
    for i, r in t.iterrows():
        nm = r["name"]
        low = nm.lower()
        m = re.match(r"^S(\d+)", nm)
        if r.dataset.startswith("figure-5-") and "blind-samples" not in r.dataset:
            t.loc[i, "condition"] = "tumour" if "tumour" in low else "healthy"
            t.loc[i, "patient"] = f"{r.tissue}-{r.preservation}-S{m.group(1)}"
        elif "blind-samples" in r.dataset and m:                              # fresh blind sets: label is in the name
            t.loc[i, "condition"] = "tumour" if "tumour" in low else "healthy"
            t.loc[i, "patient"] = f"{r.tissue}-{r.preservation}-S{m.group(1)}"
            t.loc[i, "role"] = "blind-set"
    # frozen colon blind samples: matched to Extended Data Fig. 5 frozen resources by event count + time stamp
    ed = t[t.dataset.str.startswith("extended-data-figure-5") & t["name"].str.contains("frozen")]
    for i, r in t[t["name"].str.startswith("Blind_Colon")].iterrows():
        hit = ed[(ed.events == r.events) & (ed.date == r.date) & (ed.time == r.time)]
        if len(hit) == 1:
            nm = hit.iloc[0]["name"]
            t.loc[i, "condition"] = "healthy" if re.match(r"^S\d+H", nm) else "tumour"
            t.loc[i, "patient"] = "colon-frozen-ED" + re.match(r"^S(\d+)", nm).group(1)
            t.loc[i, "role"] = "blind-set"
    keep = t.condition.notna() & ~t.dataset.str.startswith("extended-data-figure-5")
    return t[keep].reset_index(drop=True)


def read_events(rid, max_events, seed):
    import dclab
    ds = dclab.new_dataset(rid)
    n_total = len(ds)
    avail = [f for f in FEATURES if f in ds]
    cols = {f: np.asarray(ds[f][:], dtype=np.float64) for f in avail}
    gate = np.isfinite(cols["area_um"]) & np.isfinite(cols["deform"]) & (cols["area_um"] >= 20)
    if "area_ratio" in cols:
        gate &= np.isfinite(cols["area_ratio"]) & (cols["area_ratio"] < 1.2)
    idx = np.flatnonzero(gate)
    n_gated = len(idx)
    if len(idx) > max_events:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_events, replace=False))
    out = {f: v[idx].astype(np.float32) for f, v in cols.items()}
    out["event_index"] = idx.astype(np.int32)
    return out, n_total, n_gated, avail


def extract(out_dir, max_events, only=None, shard=0, nshards=1):
    out_dir = Path(out_dir); (out_dir / "events").mkdir(parents=True, exist_ok=True)
    t = annotate(dataset_table())
    if shard == 0:
        t.to_csv(out_dir / "dcor_resource_table.csv", index=False)
    print(f"{len(t)} human measurements: " + str(t.groupby(['tissue', 'preservation', 'condition']).size().to_dict()), flush=True)
    log = []
    for i, r in t.iterrows():
        if only and r["id"] not in only:
            continue
        if i % nshards != shard:
            continue
        dst = out_dir / "events" / f"{r['id']}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        try:
            ev, n_total, n_gated, avail = read_events(r["id"], max_events, seed=int(r["id"][:8], 16))
            np.savez_compressed(dst, **ev)
            log.append({"id": r["id"], "n_total": n_total, "n_gated": n_gated, "n_kept": len(ev["deform"]),
                        "features": ",".join(avail), "seconds": round(time.time() - t0, 1)})
            print(f"[{i + 1}/{len(t)}] {r['name']:32s} total {n_total:7d} gated {n_gated:7d} kept {len(ev['deform']):6d}  "
                  f"{time.time() - t0:5.0f}s", flush=True)
        except Exception as e:                                             # noqa: BLE001
            print(f"[{i + 1}/{len(t)}] {r['name']} FAILED {type(e).__name__}: {e}", flush=True)
            log.append({"id": r["id"], "error": f"{type(e).__name__}: {e}"})
        lf = out_dir / f"dcor_extract_log_{shard}.csv"
        pd.DataFrame(log).to_csv(lf, mode="a", header=not lf.exists(), index=False)
        log = []


def combine(out_dir):
    out_dir = Path(out_dir)
    t = pd.read_csv(out_dir / "dcor_resource_table.csv")
    frames, meas = [], []
    for _, r in t.iterrows():
        f = out_dir / "events" / f"{r['id']}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        df = pd.DataFrame({k: z[k] for k in z.files})
        df.insert(0, "measurement", r["id"])
        frames.append(df)
        meas.append({**r.to_dict(), "n_events_kept": len(df)})
    ev = pd.concat(frames, ignore_index=True)
    pd.DataFrame(meas).to_csv(out_dir / "dcor_measurements.csv", index=False)
    ev.to_csv(out_dir / "dcor_events.csv.gz", index=False, compression="gzip", float_format="%.6g")
    print(f"combined {len(meas)} measurements, {len(ev)} events")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-events", type=int, default=40000)
    ap.add_argument("--only", default=None, help="comma-separated resource ids (tests)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.combine:
        combine(a.out_dir)
    else:
        extract(a.out_dir, a.max_events, set(a.only.split(",")) if a.only else None, a.shard, a.nshards)
