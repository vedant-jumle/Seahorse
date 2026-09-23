#!/bin/bash
#SBATCH --job-name=seahorse_v0
#SBATCH --partition=gpu-v100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/scratch/%u/logs/seahorse_v0_%j.out
#SBATCH --error=/scratch/%u/logs/seahorse_v0_%j.err

# Submit from the repo root on DelftBlue:  sbatch slurm/v0.slurm
# Extra run.py flags:  EXTRA_ARGS="--layers 14 17 --alphas 0 1" sbatch slurm/v0.slurm
# The log directory must exist before submission (setup_delftblue.sh creates it).
set -eo pipefail

REPO=${REPO:-/scratch/$USER/Seahorse}
OUT=${OUT:-/scratch/$USER/seahorse_runs/v0_$SLURM_JOB_ID}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
module load cuda/12.9

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$REPO"
git log -1 --oneline
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
pytest -q tests
python experiments/v0/run.py --out "$OUT" ${EXTRA_ARGS:-}
echo "Done: $OUT"
