#!/bin/bash
#SBATCH --job-name=mp_env
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/env_%j.out
source dcc/config.sh
set -x
if [ ! -x "$CONDA_ENV/bin/python" ]; then
  conda create -y -p "$CONDA_ENV" python=3.11 pip
fi
source activate "$CONDA_ENV"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scikit-learn scipy scikit-image opencv-python-headless statsmodels dclab umap-learn \
            matplotlib pillow pyarrow h5py tifffile openslide-python openslide-bin timm huggingface_hub \
            transformers accelerate torchstain
python - <<'PY'
import torch, openslide, timm, transformers, sklearn, pandas
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| openslide", openslide.__library_version__,
      "| timm", timm.__version__, "| transformers", transformers.__version__, "| sklearn", sklearn.__version__, "| pandas", pandas.__version__)
PY
echo ENV_OK
