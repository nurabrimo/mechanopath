export NETID=nb295
export MECHANOPATH_ROOT=$HOME/mechanopath_full
export WSI_DIR=/cwork/$NETID/mechanopath/wsi
export TILE_DIR=/cwork/$NETID/mechanopath/coords
export FEAT_DIR=/cwork/$NETID/mechanopath/features
export HF_HOME=/work/$NETID/hf_cache
export CONDA_ENV=/work/$NETID/conda/mechanopath
export PIP_CACHE_DIR=/work/$NETID/pip_cache
module load Anaconda3/2024.02 >/dev/null 2>&1
mkdir -p "$WSI_DIR" "$TILE_DIR" "$FEAT_DIR" "$HF_HOME" "$MECHANOPATH_ROOT/logs" "$PIP_CACHE_DIR" 2>/dev/null || true
