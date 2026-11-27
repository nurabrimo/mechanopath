"""Tile embeddings for every tissue tile of a slide (full-resolution DCC pipeline).

  python dcc/03_extract_features.py --coords $TILE_DIR/<stem>.h5 [...] --encoder phikon2 --out-dir $FEAT_DIR

Encoders (all self-supervised pathology models; none was trained with diagnostic labels)
  phikon2  owkin/phikon-v2, ViT-L/16 DINOv2, 1024-d CLS token; 224-px input, ImageNet normalisation.
           Open weights (non-commercial licence). Pretraining data include TCGA slides (no labels).
  hipt     HIPT ViT-S/16 DINO (Chen et al., CVPR 2022), 384-d; 256-px input. Pretrained on TCGA (no labels).
  uni      MahmoodLab/UNI, ViT-L/16, 1024-d. Gated: needs `hf auth login` by the account holder.
           Pretraining excluded TCGA, so it serves as the leakage-free sensitivity encoder.
Every slide is read at 0.5 um/px whatever its scan magnification (see 02_tile_wsi.py).

Output: <out-dir>/<encoder>/<stem>.h5 with 'features' (N x D, float16), 'coords' and the slide attributes.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class Tiles(Dataset):
    def __init__(self, h5path, transform):
        with h5py.File(h5path, "r") as f:
            self.coords = f["coords"][:]
            self.attrs = dict(f.attrs)
        self.wsi, self.t, self.sl = self.attrs["wsi"], transform, None
        self.rs, self.lvl, self.tile = int(self.attrs["read_size"]), int(self.attrs["read_level"]), int(self.attrs["tile"])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self.sl is None:                                    # one OpenSlide handle per worker process
            self.sl = openslide.OpenSlide(self.wsi)
        x, y = self.coords[i]
        im = self.sl.read_region((int(x), int(y)), self.lvl, (self.rs, self.rs)).convert("RGB")
        if self.rs != self.tile:
            im = im.resize((self.tile, self.tile), Image.BILINEAR)
        return self.t(im)


class HFCls(torch.nn.Module):
    """Wraps a transformers vision backbone and returns the CLS token."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).last_hidden_state[:, 0]


def turing_safe():
    """Fused SDPA kernels (memory-efficient attention) raise illegal-memory-access errors in fp16 on pre-Ampere
    GPUs (RTX 2080 Ti, compute capability 7.5) with torch 2.6; use plain attention there."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        return "eager"
    return "sdpa"


def build_encoder(name, weights_dir):
    import torchvision.transforms as T
    attn = turing_safe()
    if name in ("phikon2", "phikon"):
        from transformers import AutoModel
        repo = "owkin/phikon-v2" if name == "phikon2" else "owkin/phikon"
        m = HFCls(AutoModel.from_pretrained(repo, attn_implementation=attn))
        return m, T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(*IMAGENET)])
    if name == "uni":
        import timm
        if attn == "eager":
            timm.layers.set_fused_attn(False)
        m = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        return m, T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(*IMAGENET)])
    if name == "hipt":
        sys.path.insert(0, str(weights_dir))
        import hipt_vision_transformer as vits
        m = vits.vit_small(patch_size=16)
        sd = torch.load(Path(weights_dir) / "vit256_small_dino.pth", map_location="cpu", weights_only=False)
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.get("teacher", sd).items()}
        missing, _ = m.load_state_dict(sd, strict=False)
        assert not [k for k in missing if not k.startswith("head")], missing
        return m, T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", nargs="+", required=True)
    ap.add_argument("--encoder", default="phikon2", choices=["phikon2", "phikon", "uni", "hipt"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weights-dir", default=str(ROOT / "data" / "raw" / "models"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--allow-cpu", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available() and not a.allow_cpu:            # a broken GPU node must not crawl on CPU
        print("NO_GPU: CUDA is not available on this node", flush=True)
        sys.exit(3)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tfm = build_encoder(a.encoder, a.weights_dir)
    model = model.to(dev).eval()
    out_dir = Path(a.out_dir) / a.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    for h5 in a.coords:
        stem = Path(h5).stem
        dst = out_dir / f"{stem}.h5"
        if dst.exists():
            continue
        t0 = time.time()
        try:
            ds = Tiles(h5, tfm)
            if len(ds) == 0:
                print(stem, "no tissue tiles", flush=True)
                continue
            dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=dev.type == "cuda",
                            persistent_workers=False)
            feats = []
            with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                for xb in dl:
                    feats.append(model(xb.to(dev, non_blocking=True)).float().cpu().numpy().astype(np.float16))
            F = np.concatenate(feats)
            tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp")
            with h5py.File(tmp, "w") as f:
                f.create_dataset("features", data=F)
                f.create_dataset("coords", data=ds.coords)
                for k, v in ds.attrs.items():
                    f.attrs[k] = v
                f.attrs["encoder"] = a.encoder
            if dst.exists():                                      # another task finished this slide first
                tmp.unlink()
            else:
                tmp.rename(dst)                                   # never leave a half-written file behind
            n_done += 1
            dt = time.time() - t0
            print(f"{stem} {F.shape} {dt:.0f}s {len(F) / max(dt, 1e-6):.0f} tiles/s", flush=True)
        except Exception as e:                                     # a corrupt slide must not stop the array task
            print(f"FEATURE_FAILED {stem}: {str(e)[:200]}", flush=True)
            if "CUDA" in str(e):                                   # the CUDA context is unusable after this
                sys.exit(2)
    print(f"done {n_done} slides", flush=True)


if __name__ == "__main__":
    main()
