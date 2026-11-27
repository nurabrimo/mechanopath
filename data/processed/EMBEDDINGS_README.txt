HIPT ViT-S/16 embeddings (55,864 crops x 384, float16) are shipped as two files because of a 30 MB upload limit:
  mechanopath_embeddings_hipt_part1.npz, mechanopath_embeddings_hipt_part2.npz  (split by patient)
Rebuild data/processed/embeddings_hipt.npz with:
  python - <<'PY'
  import numpy as np
  a, b = np.load('mechanopath_embeddings_hipt_part1.npz'), np.load('mechanopath_embeddings_hipt_part2.npz')
  np.savez_compressed('data/processed/embeddings_hipt.npz', **{k: np.concatenate([a[k], b[k]]) for k in a.files})
  PY
or recompute them with src/03_embed.py (about 2.5 h on 2 CPU cores).
