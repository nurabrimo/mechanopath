"""Step 3 - deep tile embeddings with a pathology self-supervised encoder.

Encoder: HIPT ViT-S/16 (vit256_small_dino.pth, Chen et al., CVPR 2022), DINO-pretrained
on 256x256 TCGA patches at 20x. It is the strongest openly downloadable pathology
encoder reachable from this workspace (UNI/CONCH/Virchow need gated Hugging Face
access; the DCC pipeline in dcc/ can use them).

Caveat: HIPT's self-supervised pretraining used TCGA slides (no labels). The encoder
may therefore have seen unlabeled images from these same slides. This does not leak
GBM/LGG labels, but external validation on non-TCGA slides is still required.

Each 512x512 tile is split into four 256x256 crops; each crop gives a 384-d CLS
embedding. Output: data/processed/embeddings_hipt.npz with arrays
  emb (n_crops x 384, float16), and index columns patient / tile / mag / stratum / crop.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, RAW  # noqa: E402

sys.path.insert(0, str(RAW / "models"))
import hipt_vision_transformer as vits  # noqa: E402  (HIPT_4K/vision_transformer.py)

torch.set_num_threads(2)
WEIGHTS = RAW / "models" / "vit256_small_dino.pth"


def load_encoder():
    model = vits.vit_small(patch_size=16)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd = sd.get("teacher", sd)
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [m for m in missing if not m.startswith("head")], missing
    model.eval()
    return model


def to_tensor(img: Image.Image):
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5                                  # HIPT eval transform
    return torch.from_numpy(a.transpose(2, 0, 1))


def crops(path):
    im = Image.open(path).convert("RGB")
    return [im.crop((x, y, x + 256, y + 256)) for y in (0, 256) for x in (0, 256)]


if __name__ == "__main__":
    tiles = pd.read_csv(PROC / "tiles_index.csv")[["patient", "tile", "mag", "stratum"]]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if limit:
        tiles = tiles.head(limit)
    model = load_encoder()
    out_idx, out_emb = [], []
    batch, meta = [], []
    t0 = time.time()

    def flush():
        with torch.inference_mode():
            e = model(torch.stack(batch)).numpy().astype(np.float16)
        out_emb.append(e)
        out_idx.extend(meta)
        batch.clear(); meta.clear()

    for i, r in enumerate(tiles.itertuples()):
        for c, im in enumerate(crops(RAW / "tiles" / r.tile)):
            batch.append(to_tensor(im))
            meta.append((r.patient, r.tile, r.mag, r.stratum, c))
        if len(batch) >= 64:
            flush()
        if i % 500 == 0:
            print(f"{i}/{len(tiles)} tiles  {time.time() - t0:.0f}s", flush=True)
    if batch:
        flush()
    emb = np.concatenate(out_emb)
    idx = pd.DataFrame(out_idx, columns=["patient", "tile", "mag", "stratum", "crop"])
    np.savez_compressed(PROC / ("embeddings_hipt.npz" if not limit else "embeddings_hipt_test.npz"),
                        emb=emb, **{c: np.asarray(idx[c].astype(str).tolist(), dtype="U") for c in idx.columns})
    print("done", emb.shape, f"{time.time() - t0:.0f}s")
