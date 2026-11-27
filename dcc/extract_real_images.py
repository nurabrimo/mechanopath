"""Extract real images for the workflow figure (runs on DCC, env /work/nb295/conda/mechanopath).
  * one GBM and one LGG TCGA diagnostic slide (40x): thumbnail + three tissue tiles at 0.5 um/px
  * single-cell RT-FDC images (tumour and matched healthy colon tissue) from the public DCOR dataset of Soteriou et al. 2023
Output: ~/mechanopath_full/results/real_images/*.png
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd, h5py
from PIL import Image

OUT = Path.home() / "mechanopath_full/results/real_images"; OUT.mkdir(parents=True, exist_ok=True)
cohort = pd.read_csv(Path.home() / "mechanopath_full/data/processed/cohort.csv").set_index("patient")
slides = pd.read_csv(Path.home() / "mechanopath_full/data/processed/slides.csv")
FEAT = Path("/cwork/nb295/mechanopath/features/phikon2"); WSI = Path("/cwork/nb295/mechanopath/wsi")
import openslide

rng = np.random.default_rng(7); log = {}
for cls, y in [("GBM", 1), ("LGG", 0)]:
    pats = [p for p in slides[slides.appmag == 40].patient if p in cohort.index and cohort.loc[p, "y_gbm"] == y]
    for p in rng.permutation(pats)[:20]:
        svs = glob.glob(str(WSI / "**" / f"{p}*.svs"), recursive=True) or glob.glob(str(WSI / f"{p}*.svs"))
        h5 = glob.glob(str(FEAT / f"{p}*.h5"))
        if not svs or not h5:
            continue
        s = openslide.OpenSlide(svs[0])
        th = Image.fromarray(np.asarray(s.get_thumbnail((900, 900)).convert("RGB"))); th.save(OUT / f"slide_{cls}.png")
        with h5py.File(h5[0], "r") as f:
            C = f["coords"][:]; at = dict(f.attrs)
        mpp = float(s.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25)); size0 = int(round(128 / mpp))   # 128 um tile
        k = 0
        for i in rng.permutation(len(C)):
            x, yy = int(C[i][0]), int(C[i][1])
            t = Image.fromarray(np.asarray(s.read_region((x, yy), 0, (size0, size0)).convert("RGB").resize((256, 256), Image.LANCZOS)))
            a = np.asarray(t).astype(float)
            if (a.mean(2) < 215).mean() > 0.75 and a.std() > 20:          # mostly tissue, not blank
                t.save(OUT / f"tile_{cls}_{k}.png"); k += 1
            if k == 3:
                break
        log[cls] = {"patient": p, "slide": Path(svs[0]).name, "mpp": mpp, "coords_attrs": {kk: str(v) for kk, v in at.items()}}
        print(cls, p, "mpp", mpp, "tiles", k, flush=True)
        if k == 3:
            break

# ---------------------------------------------------------------- RT-FDC cell images from DCOR
import dclab
pairs = pd.read_csv(Path.home() / "mechanopath_full/results/phase2b_pairs_for_images.csv") if (Path.home() / "mechanopath_full/results/phase2b_pairs_for_images.csv").exists() else None
gf = pd.read_csv("/work/nb295/mechanopath/code/results/phase2b/gating_flow.csv")
u = gf[gf.stratum == "colon-fresh"].groupby("unit").filter(lambda g: len(g) == 2).unit.iloc[0]
for cond in ["tumour", "healthy"]:
    rid = gf[(gf.unit == u) & (gf.condition == cond)].measurement.iloc[0]
    try:
        ds = dclab.new_dataset(rid)
        area = ds["area_um"]; ar = ds["area_ratio"]; asp = ds["aspect"]
        ok = np.where((area >= 25) & (ar < 1.05) & (asp <= 2) & (area > 40) & (area < 150))[0]
        sel = rng.choice(ok, min(8, len(ok)), replace=False)
        for j, ev in enumerate(sel):
            img = np.asarray(ds["image"][ev]).astype(np.uint8)
            cx, cy = ds["pos_x"][ev] / ds.config["imaging"]["pixel size"], ds["pos_y"][ev] / ds.config["imaging"]["pixel size"]
            h, w = img.shape; half = 40
            x0 = int(np.clip(cx - half, 0, w - 2 * half)); y0 = int(np.clip(cy - half, 0, h - 2 * half))
            Image.fromarray(img[y0:y0 + 2 * half, x0:x0 + 2 * half]).resize((240, 240), Image.LANCZOS).save(OUT / f"cell_{cond}_{j}.png")
        log[f"cells_{cond}"] = {"unit": u, "dcor_resource": rid, "n": int(len(sel)), "deform_median": float(np.median(ds["deform"][ok]))}
        print("cells", cond, rid, len(sel), flush=True)
    except Exception as e:
        log[f"cells_{cond}"] = {"error": repr(e)}; print("cells error", cond, repr(e), flush=True)
json.dump(log, open(OUT / "provenance.json", "w"), indent=1)
print("REAL_IMAGES_DONE")
