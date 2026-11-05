#!/bin/bash
#SBATCH --job-name=dc_cpu
#SBATCH --partition=compute-p1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/scratch/%u/logs/dc_cpu_%j.out
#SBATCH --error=/scratch/%u/logs/dc_cpu_%j.err

# dream_compare CPU job (no GPU; copied from slurm/dream_select_cpu.slurm): run.py stages such as judge-tasks /
# judge-summary / analyses, after pytest (TESTS, default the dream_compare test file), in one Python process (jobs.py).
#   MODEL=qwen25_7b SPECS="judge-summary|/scratch/$USER/seahorse_runs/<run>/judge_smoke|" sbatch slurm/dream_compare_cpu.slurm
set -eo pipefail

MODEL=${MODEL:?set MODEL}
SPECS=${SPECS:?set SPECS}
REPO=${REPO:-/scratch/$USER/Seahorse}
TESTS=${TESTS:-experiments/dream_compare/test_dream_compare.py}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$REPO"
echo "dream_compare CPU MODEL=$MODEL SPECS=$SPECS TESTS=$TESTS commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
export SPECS
python experiments/dream_compare/jobs.py --model "$MODEL" --device cpu \
  --pytest "-q --import-mode=importlib -p no:cacheprovider $TESTS"
