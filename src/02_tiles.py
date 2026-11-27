"""Step 2 - unpack slide tiles, tile QC and interpretable (hand-crafted) image features.

Tile set per patient (pulled from GDC's public tile service, one FFPE diagnostic
slide per patient, see harvest/gdc_tile_harvester.js):
  thumb.jpg                ~1/64 downsample of the whole slide (~32 um/px)
  x20_hdense_*.jpg  (6)    512x512 at ~0.5 um/px from the most haematoxylin-dense tissue
  x20_random_*.jpg  (6)    512x512 at ~0.5 um/px, uniformly random tissue tiles
  x5_hdense_*.jpg   (2)    512x512 at ~2 um/px (1 mm field), haematoxylin-dense
  x5_random_*.jpg   (2)    512x512 at ~2 um/px, random tissue
  meta.json                slide metadata (scanner, scan date, MPP, ...) and tile coordinates

Outputs
  data/processed/slides.csv          one row per slide: acquisition metadata (used as nuisance variables)
  data/processed/tile_features.csv   one row per tile: QC + hand-crafted features
  data/processed/handcrafted_patient.csv  patient-level aggregates
"""
import glob
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.color import rgb2hed
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RAW  # noqa: E402

TILE_DIR = RAW / "tiles"
UPLOADS = Path("/mnt/user-data/uploads/Downloads")


def unpack():
    recs = []
    for zp in sorted(UPLOADS.glob("mechanopath_gdc_tiles_part*.zip")):
        z = zipfile.ZipFile(zp)
        bad = z.testzip()
        assert bad is None, f"corrupt member {bad} in {zp.name}"
        pm = json.loads(z.read("part_manifest.json"))
        for m in z.namelist():
            if m.startswith("tiles/"):
                out = TILE_DIR / m[len("tiles/"):]
                if not out.exists():
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(m))
        for s in pm["slides"]:
            s["zip_part"] = zp.name
            recs.append(s)
    return recs


def slide_table(recs):
    rows = []
    for s in recs:
        r = {k: v for k, v in s.items() if k != "tiles"}
        r["n_tiles"] = len(s["tiles"])
        rows.append(r)
    df = pd.DataFrame(rows).drop_duplicates("patient")
    dt = pd.to_datetime(df.scan_date, format="%m/%d/%y", errors="coerce")
    df["scan_year"] = dt.dt.year
    df["scan_month_index"] = (dt.dt.year - 2000) * 12 + dt.dt.month
    df["aperio_major"] = df.aperio.str.extract(r"v(\d+)").astype(float)
    df["slide_area_mpx"] = df.W * df.H / 1e6
    df["tissue_area_mm2"] = df.thumb_tissue_frac * df.W * df.H * (df.mpp.fillna(0.5) ** 2) / 1e6
    return df


# ---------------------------------------------------------------------------
# per-tile features
# ---------------------------------------------------------------------------
def tissue_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    return (s > 0.07) & (v > 0.15)


def tile_features(path, mag, mpp):
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tis = tissue_mask(rgb)
    f = {"tissue_frac": tis.mean()}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f["blur_lapvar"] = cv2.Laplacian(gray, cv2.CV_64F).var()
    R, G, B = [rgb[..., i].astype(float) for i in range(3)]
    f["blood_frac"] = ((R > 140) & (G < 90) & (B < 120) & (R - G > 70)).mean()
    for i, c in enumerate("rgb"):
        f[f"mean_{c}"] = rgb[..., i][tis].mean() if tis.any() else np.nan
    hed = rgb2hed(rgb)
    H, E = hed[..., 0], hed[..., 1]
    if tis.sum() < 500:
        return f
    f["H_mean"], f["H_sd"] = H[tis].mean(), H[tis].std()
    f["E_mean"], f["E_sd"] = E[tis].mean(), E[tis].std()
    Hs = ndi.gaussian_filter(H, 1.0)
    thr = threshold_otsu(Hs[tis])
    nuc = (Hs > thr) & tis
    px_um2 = (mpp ** 2)
    min_px = int(round(12 / px_um2)) if mag == "x20" else 2      # >= ~12 um^2 at 20x
    nuc = ndi.binary_opening(nuc, iterations=1)
    nuc = ndi.binary_fill_holes(nuc)
    lab, n = ndi.label(nuc)
    sizes = ndi.sum(nuc, lab, range(1, n + 1)) if n else np.array([])
    keep = np.isin(lab, np.where(sizes >= min_px)[0] + 1)
    f["nuc_area_frac"] = keep.sum() / tis.sum()
    tissue_mm2 = tis.sum() * px_um2 / 1e6
    if mag == "x20":
        lab2, n2 = ndi.label(keep)
        props = regionprops(lab2)
        areas = np.array([p.area for p in props]) * px_um2
        f["nuc_density_per_mm2"] = n2 / tissue_mm2 if tissue_mm2 > 0 else np.nan
        if n2 >= 5:
            f["nuc_area_med_um2"] = np.median(areas)
            f["nuc_area_iqr_um2"] = np.subtract(*np.percentile(areas, [75, 25]))
            f["nuc_area_p90_um2"] = np.percentile(areas, 90)
            f["nuc_ecc_mean"] = np.mean([p.eccentricity for p in props])
            f["nuc_solidity_mean"] = np.mean([p.solidity for p in props])
            f["nuc_H_mean"] = np.mean([H[tuple(p.coords.T)].mean() for p in props])
        g8 = (gray // 32).astype(np.uint8)
        glcm = graycomatrix(g8, distances=[2, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=8,
                            symmetric=True, normed=True)
        for prop in ["contrast", "homogeneity", "energy", "correlation"]:
            f[f"glcm_{prop}"] = graycoprops(glcm, prop).mean()
        lbp = local_binary_pattern(gray, 8, 1, "uniform")
        hist = np.bincount(lbp[tis].astype(int), minlength=10) / tis.sum()
        f["lbp_entropy"] = -(hist[hist > 0] * np.log(hist[hist > 0])).sum()
    else:
        # low-power: pale eosinophilic, nucleus-poor tissue (a crude necrosis/oedema proxy)
        e_hi = E > np.percentile(E[tis], 50)
        h_lo = Hs < np.percentile(Hs[tis], 25)
        f["pale_eosin_lowH_frac"] = (e_hi & h_lo & tis).sum() / tis.sum()
    return f


def _job(args):
    patient, name, mag, stratum, mpp = args
    try:
        f = tile_features(TILE_DIR / name, mag, mpp)
    except Exception as e:  # keep going; record the failure
        f = {"error": str(e)[:80]}
    f.update({"patient": patient, "tile": name, "mag": mag, "stratum": stratum})
    return f


if __name__ == "__main__":
    recs = unpack()
    slides = slide_table(recs)
    slides.to_csv(PROC / "slides.csv", index=False)
    print("slides:", len(slides), slides.project.value_counts().to_dict())
    jobs = []
    for s in recs:
        for t in s["tiles"]:
            eff_mpp = (s.get("mpp") or 0.5) * t["downsample"]
            jobs.append((s["patient"], t["name"], t["mag"], t["stratum"], eff_mpp))
    idx = pd.DataFrame(jobs, columns=["patient", "tile", "mag", "stratum", "mpp_eff"])
    idx.to_csv(PROC / "tiles_index.csv", index=False)
    print("tiles:", len(jobs))
    if "--index-only" in sys.argv:
        sys.exit(0)
    with ProcessPoolExecutor(2) as ex:
        feats = list(ex.map(_job, jobs, chunksize=64))
    tf = pd.DataFrame(feats)
    tf.to_csv(PROC / "tile_features.csv", index=False)
    # QC flags
    tf["qc_ok"] = (tf.tissue_frac >= 0.5) & (tf.blood_frac < 0.3) & (tf.blur_lapvar > 15)
    print("tiles passing QC:", round(tf.qc_ok.mean(), 3), tf.groupby("mag").qc_ok.mean().round(3).to_dict())
    tf.to_csv(PROC / "tile_features.csv", index=False)
    # patient aggregates (QC-passing tiles only; fall back to all tiles if none pass)
    num = [c for c in tf.columns if c not in ("patient", "tile", "mag", "stratum", "qc_ok", "error")]
    out = []
    for p, g in tf.groupby("patient"):
        row = {"patient": p}
        for mag in ["x20", "x5"]:
            gm = g[(g.mag == mag) & g.qc_ok]
            if len(gm) == 0:
                gm = g[g.mag == mag]
            for c in num:
                v = gm[c].dropna()
                if len(v) == 0:
                    continue
                row[f"{mag}_{c}_mean"] = v.mean()
                if mag == "x20":
                    row[f"{mag}_{c}_sd"] = v.std()
                    row[f"{mag}_{c}_max"] = v.max()
            row[f"{mag}_n_qc_tiles"] = int(g[(g.mag == mag)].qc_ok.sum())
        out.append(row)
    hp = pd.DataFrame(out)
    hp.to_csv(PROC / "handcrafted_patient.csv", index=False)
    print("handcrafted patient features:", hp.shape)
