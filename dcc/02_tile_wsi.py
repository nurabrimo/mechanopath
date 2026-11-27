"""Tissue detection + tile coordinates for one whole-slide image (full-resolution DCC pipeline).

Every slide is tiled at the SAME physical resolution (default 0.5 um/px, i.e. 20x) whatever
it was scanned at: a 40x slide is read at level 0 with 512-px windows and resized to 256, a
20x slide is read with 256-px windows. Acquisition metadata are saved so the analysis can
stratify or restrict by scanner settings (all 20x-scanned TCGA-GBM/LGG slides are GBM).

  python dcc/02_tile_wsi.py --wsi /path/slide.svs --out-dir $TILE_DIR [--tile 256 --mpp 0.5 --min-tissue 0.5]

Tissue: saturation of the thumbnail, Otsu threshold (floor 20), pen-ink removal, closing and removal of
small components. Faintly stained slides on which this rule finds no tile are re-thresholded with the
same rule after optical-density normalisation of the thumbnail (od_normalise); the attribute 'mask_rule'
records which path was used. On normally stained slides both paths give the same mask (Dice >= 0.997).

Output: <out-dir>/<slide_stem>.h5 with dataset 'coords' (N x 2, level-0 x,y of each tile's
top-left corner) and attributes (read_level, read_size, tile, target_mpp, mpp0, appmag,
scanner, scan_date, jpeg_q, mask_rule, od_gain, ...); <out-dir>/qc/<slide_stem>.jpg thumbnail with the mask.
"""
import argparse
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide


def slide_metadata(sl):
    p = sl.properties
    meta = {"vendor": p.get("openslide.vendor", ""),
            "mpp0": float(p.get(openslide.PROPERTY_NAME_MPP_X, "nan") or "nan"),
            "appmag": float(p.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, "nan") or "nan"),
            "scanner": p.get("aperio.ScanScope ID", ""), "scan_date": p.get("aperio.Date", ""),
            "width0": sl.dimensions[0], "height0": sl.dimensions[1]}
    q = re.search(r"Q=(\d+)", p.get("tiff.ImageDescription", ""))
    meta["jpeg_q"] = int(q.group(1)) if q else -1
    if np.isnan(meta["mpp0"]) and not np.isnan(meta["appmag"]):
        meta["mpp0"] = 10.0 / meta["appmag"]          # 20x ~ 0.5 um/px, 40x ~ 0.25 um/px
    return meta


def tissue_mask(thumb_rgb):
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    blur = cv2.medianBlur(sat, 7)
    thr, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (blur > max(thr, 20)) & (val > 50)
    R, G, B = [thumb_rgb[..., i].astype(int) for i in range(3)]
    pen = ((G > R + 5) & (G >= B)) | ((R < 80) & (B > 120) & (B - R > 70))       # green / blue ink
    m &= ~pen
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    min_area = max(20, 0.0005 * m.size)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep.astype(bool)


def od_normalise(thumb_rgb, target=1.0, kmax=40.0):
    """Optical density relative to the glass background (the dominant bright grey level), rescaled so that the
    99.5th percentile of the summed OD equals `target`, and converted back to RGB (background -> white)."""
    g = cv2.medianBlur(cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY), 7)
    h = np.bincount(g.ravel(), minlength=256).astype(float)
    h[:150] = 0
    mode = int(np.argmax(h))
    bg = np.median(thumb_rgb[np.abs(g.astype(int) - mode) <= 2], axis=0) + 1.0
    od = np.clip(-np.log10((thumb_rgb.astype(float) + 1.0) / bg), 0, None)
    p = np.percentile(cv2.medianBlur(od.sum(-1).astype(np.float32), 5), 99.5)
    k = float(min(kmax, max(1.0, target / max(p, 1e-4))))
    return np.clip(255.0 * 10 ** (-od * k), 0, 255).astype(np.uint8), k


def tile_grid(mask, W, H, size0, min_tissue):
    """Level-0 top-left corners of the size0 x size0 tiles whose tissue fraction on the mask is >= min_tissue."""
    th, tw = mask.shape
    sx, sy = tw / W, th / H
    integ = cv2.integral(mask.astype(np.uint8))
    coords = []
    for y in range(0, H - size0 + 1, size0):
        for x in range(0, W - size0 + 1, size0):
            x0, y0 = int(x * sx), int(y * sy)
            x1, y1 = max(x0 + 1, int((x + size0) * sx)), max(y0 + 1, int((y + size0) * sy))
            area = (x1 - x0) * (y1 - y0)
            frac = (integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]) / area
            if frac >= min_tissue:
                coords.append((x, y))
    return np.asarray(coords, dtype=np.int64).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--mpp", type=float, default=0.5)
    ap.add_argument("--min-tissue", type=float, default=0.5)
    ap.add_argument("--thumb-max", type=int, default=2048)
    a = ap.parse_args()

    out = Path(a.out_dir); (out / "qc").mkdir(parents=True, exist_ok=True)
    stem = Path(a.wsi).name.split(".")[0]
    sl = openslide.OpenSlide(a.wsi)
    meta = slide_metadata(sl)
    W, H = sl.dimensions
    size0 = int(round(a.tile * a.mpp / meta["mpp0"]))          # tile edge in level-0 pixels
    read_level = sl.get_best_level_for_downsample(size0 / a.tile + 1e-6)
    ds = sl.level_downsamples[read_level]
    if ds > size0 / a.tile + 1e-6:                              # never read below the target resolution
        read_level, ds = 0, 1.0
    read_size = int(round(size0 / ds))

    thumb = np.asarray(sl.get_thumbnail((a.thumb_max, a.thumb_max)).convert("RGB"))
    mask = tissue_mask(thumb)
    coords = tile_grid(mask, W, H, size0, a.min_tissue)
    rule, gain = "primary", 1.0
    if len(coords) == 0:                                        # faint staining: same rule after OD normalisation
        thumb_n, gain = od_normalise(thumb)
        mask = tissue_mask(thumb_n)
        coords = tile_grid(mask, W, H, size0, a.min_tissue)
        rule = "od_normalised"

    with h5py.File(out / f"{stem}.h5", "w") as f:
        f.create_dataset("coords", data=coords)
        for k, v in {**meta, "read_level": read_level, "read_size": read_size, "size0": size0, "tile": a.tile,
                     "target_mpp": a.mpp, "wsi": str(a.wsi), "tissue_frac_thumb": float(mask.mean()),
                     "mask_rule": rule, "od_gain": gain}.items():
            f.attrs[k] = v
    qc = thumb.copy()
    qc[~mask] = (qc[~mask] * 0.4 + 150).astype(np.uint8)
    cv2.imwrite(str(out / "qc" / f"{stem}.jpg"), cv2.cvtColor(qc, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{stem}: {len(coords)} tiles, mpp0={meta['mpp0']:.4f}, appmag={meta['appmag']}, read_level={read_level}, "
          f"read_size={read_size}, mask={rule} (gain {gain:.1f})")


if __name__ == "__main__":
    main()
