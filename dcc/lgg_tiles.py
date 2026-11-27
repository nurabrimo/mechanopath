import glob, numpy as np, h5py, openslide
from PIL import Image
from pathlib import Path
OUT = Path.home() / "mechanopath_full/results/real_images"
p = "TCGA-R8-A6ML"
svs = glob.glob(f"/cwork/nb295/mechanopath/wsi/**/{p}*.svs", recursive=True)[0]
h5 = glob.glob(f"/cwork/nb295/mechanopath/features/phikon2/{p}*.h5")[0]
s = openslide.OpenSlide(svs); mpp = float(s.properties[openslide.PROPERTY_NAME_MPP_X]); size0 = int(round(128 / mpp))
C = h5py.File(h5, "r")["coords"][:]
rng = np.random.default_rng(11); scored = []
for i in rng.permutation(len(C))[:600]:
    t = np.asarray(s.read_region((int(C[i][0]), int(C[i][1])), 0, (size0, size0)).convert("RGB").resize((256, 256)))
    a = t.astype(float); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    tissue = (a.mean(2) < 215).mean(); blood = ((R > B + 45) & (R > G + 60)).mean(); nuclei = ((B > R - 5) & (a.mean(2) < 140)).mean()
    if tissue > 0.6 and blood < 0.15:
        scored.append((nuclei, i, t))
scored.sort(key=lambda z: -z[0])
pick = [scored[k] for k in (0, min(3, len(scored)-2), min(8, len(scored)-1))]   # cellular but not the most extreme
for k, (n, i, t) in enumerate(pick):
    Image.fromarray(t).save(OUT / f"tile_LGG_{k}.png"); print(k, round(n, 3))
print("LGG_TILES_DONE", len(scored))
