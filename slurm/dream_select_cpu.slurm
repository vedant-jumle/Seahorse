#!/bin/bash
#SBATCH --job-name=ds_analyze
#SBATCH --partition=compute-p1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --output=/scratch/%u/logs/ds_analyze_%j.out
#SBATCH --error=/scratch/%u/logs/ds_analyze_%j.err

# dream_select CPU job (no GPU; PREREG addendum 1): run.py --stage analyze-gen / summary on the saved answers and labels,
# after the whole dream_select test file, in one Python process (experiments/dream_select/jobs.py).
#   MODEL=qwen25_7b SPECS="analyze-gen|/scratch/$USER/seahorse_runs/<run>/qwen25_7b|" sbatch slurm/dream_select_cpu.slurm
set -eo pipefail

MODEL=${MODEL:?set MODEL}
SPECS=${SPECS:?set SPECS}
REPO=${REPO:-/scratch/$USER/Seahorse}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$REPO"
echo "dream_select CPU MODEL=$MODEL SPECS=$SPECS commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
export SPECS
python experiments/dream_select/jobs.py --model "$MODEL" --device cpu \
  --pytest "-q --import-mode=importlib -p no:cacheprovider experiments/dream_select/test_dream_select.py"
