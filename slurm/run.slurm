#!/bin/bash
#SBATCH --job-name=seahorse
#SBATCH --partition=gpu-a100-small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7900M
#SBATCH --output=/scratch/%u/logs/seahorse_%j.out
#SBATCH --error=/scratch/%u/logs/seahorse_%j.err

# Generic experiment job. From the repo root on DelftBlue:
#   EXP=v0_1 sbatch slurm/run.slurm
#   EXP=v0_1 EXTRA_ARGS="--layers 23 --alphas 2" sbatch slurm/run.slurm
# Runs experiments/$EXP/run.py; outputs go to /scratch/$USER/seahorse_runs/${EXP}_<jobid>.
# gpu-a100-small: 10GB MIG slice, <=2 CPUs, <=8000MB/CPU, 4h max. No internet on
# compute nodes: the model must already be in HF_HOME (download on the login node).
set -eo pipefail

EXP=${EXP:?set EXP, e.g. EXP=v0_1}
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
echo "EXP=$EXP commit=$(git log -1 --oneline)"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
pytest -q tests
python "experiments/$EXP/run.py" --out "$OUT" ${EXTRA_ARGS:-}
echo "Done: $OUT"
