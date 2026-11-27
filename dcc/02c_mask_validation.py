"""Agreement between the primary tissue rule and its optical-density-normalised fallback (dcc/02_tile_wsi.py).

  python dcc/02c_mask_validation.py [--n 20 --seed 7]

The fallback is applied only to slides on which the primary rule keeps no tile (faint staining). To check that it
detects the same tissue, both rules are applied to a random sample of normally stained slides. The primary rule is
re-run from the thumbnail, so the tile count must also equal the one stored in the slide's grid.

Output: results/tissue_mask_validation.csv, one row per slide: group ('validation' or 'fallback'), tile counts
(stored grid, primary rule, fallback rule), Dice coefficient of the two thumbnail masks and the OD gain.
"""
import argparse
import glob
import importlib.util
import os
import random
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tile_wsi", ROOT / "dcc" / "02_tile_wsi.py")
tw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tw)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--tile", type=int, default=256)
ap.add_argument("--mpp", type=float, default=0.5)
ap.add_argument("--min-tissue", type=float, default=0.5)
a = ap.parse_args()


def attr(f, k, default):
    v = f.attrs.get(k, default)
    return v.decode() if isinstance(v, bytes) else v


normal, fallback = [], []
for p in sorted(glob.glob(os.environ["TILE_DIR"] + "/*.h5")):
    with h5py.File(p, "r") as f:
        rec = (Path(p).stem, str(attr(f, "wsi", "")), int(f["coords"].shape[0]), float(attr(f, "od_gain", 1.0)))
        (fallback if str(attr(f, "mask_rule", "primary")) == "od_normalised" else normal).append(rec)
random.Random(a.seed).shuffle(normal)

rows = []
for stem, wsi, n, _ in normal[:a.n]:
    sl = openslide.OpenSlide(wsi)
    meta = tw.slide_metadata(sl)
    W, H = sl.dimensions
    size0 = int(round(a.tile * a.mpp / meta["mpp0"]))
    rgb = np.asarray(sl.get_thumbnail((2048, 2048)).convert("RGB"))
    m0 = tw.tissue_mask(rgb)
    rgbn, k = tw.od_normalise(rgb)
    m1 = tw.tissue_mask(rgbn)
    rows.append(dict(group="validation", slide=stem, tiles_grid=n,
                     tiles_primary=len(tw.tile_grid(m0, W, H, size0, a.min_tissue)),
                     tiles_fallback=len(tw.tile_grid(m1, W, H, size0, a.min_tissue)),
                     dice=float(2 * (m0 & m1).sum() / max(m0.sum() + m1.sum(), 1)), od_gain=k))
    print(stem, {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rows[-1].items() if k_ != "slide"},
          flush=True)
for stem, _, n, g in fallback:
    rows.append(dict(group="fallback", slide=stem, tiles_grid=n, tiles_primary=0, tiles_fallback=n, dice=np.nan,
                     od_gain=g))

d = pd.DataFrame(rows)
d.to_csv(ROOT / "results" / "tissue_mask_validation.csv", index=False)
v = d[d.group == "validation"]
r = v.tiles_fallback / v.tiles_primary
print(f"validation slides {len(v)} | grid reproduced {int((v.tiles_grid == v.tiles_primary).sum())}/{len(v)} | "
      f"Dice min {v.dice.min():.3f} median {v.dice.median():.3f} | tile ratio fallback/primary "
      f"{r.min():.3f}-{r.max():.3f} | slides using the fallback {int((d.group == 'fallback').sum())}")
