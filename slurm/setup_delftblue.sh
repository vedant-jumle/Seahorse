#!/bin/bash
# One-off setup on the DelftBlue login node. Run from the repo root:
#   bash slurm/setup_delftblue.sh
# Builds the `seahorse` conda env on scratch and pre-downloads the model, so jobs
# can run with HF_HUB_OFFLINE=1 (compute-node internet access is not assumed).
set -eo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
mkdir -p "$HF_HOME" "/scratch/$USER/logs" "/scratch/$USER/seahorse_runs"

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"

if conda env list | grep -qE "^seahorse\s"; then
  echo "updating existing seahorse env"
  conda env update -n seahorse -f environment.yml --prune
else
  echo "creating seahorse env"
  conda env create -f environment.yml
fi
conda activate seahorse
pip install -e .

python - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("$MODEL"))
PY
echo "setup done: env=seahorse, model=$MODEL cached in $HF_HOME"
