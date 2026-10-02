#!/bin/bash
#SBATCH --job-name=seahorse
#SBATCH --partition=gpu-v100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/scratch/%u/logs/seahorse_%j.out
#SBATCH --error=/scratch/%u/logs/seahorse_%j.err

# V100 variant of run.slurm (gpu-v100: tesla_v100s 32GB, sm_70, 2-day limit). V100 has no bf16
# and fp16 overflows on Qwen, so experiments run in fp32 (the seahorse default). From the repo root:
#   EXP=think_v1 EXTRA_ARGS="--stage 1" OUT=/scratch/$USER/seahorse_runs/x sbatch [--time=..] slurm/run_v100.slurm
# Runs experiments/$EXP/run.py; outputs go to $OUT (default /scratch/$USER/seahorse_runs/${EXP}_<jobid>).
# No internet on compute nodes: the model must already be in HF_HOME (download on the login node).
set -eo pipefail

EXP=${EXP:?set EXP, e.g. EXP=think_v1}
REPO=${REPO:-/scratch/$USER/Seahorse}
OUT=${OUT:-/scratch/$USER/seahorse_runs/${EXP}_$SLURM_JOB_ID}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
module load cuda/12.9

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$REPO"
echo "EXP=$EXP commit=$(git log -1 --oneline) node=$(hostname) out=$OUT"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
pytest -q tests
echo "pytest passed"
python "experiments/$EXP/run.py" --out "$OUT" ${EXTRA_ARGS:-}
echo "Done: $OUT"
