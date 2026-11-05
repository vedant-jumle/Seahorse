#!/bin/bash
#SBATCH --job-name=dream_compare
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7900M
#SBATCH --output=/scratch/%u/logs/dream_compare_%j.out
#SBATCH --error=/scratch/%u/logs/dream_compare_%j.err

# dream_compare GPU job (experiments/dream_compare/run.py through jobs.py), copied from slurm/dream_select.slurm.
# Partition, time, CPUs, memory, GPUs and dependencies come from the submit script (experiments/dream_compare/submit_*.sh):
# Gemma-4-26B-A4B in fp32 needs 2 x A100 (--gpus-per-task=2; the model is spread with device_map "auto"). Environment:
#   MODEL   a key of experiments/dream_compare/models.py SPECS
#   SPECS   ';'-separated run.py calls "STAGE|OUT|EXTRA", all in ONE Python process after pytest (jobs.py)
#   TESTS   the pytest targets (default: the dream_compare test file; Stage A / B jobs add their own files)
# No internet on compute nodes: models must already be in HF_HOME.
set -eo pipefail

MODEL=${MODEL:?set MODEL}
SPECS=${SPECS:?set SPECS}
REPO=${REPO:-/scratch/$USER/Seahorse}
TESTS=${TESTS:-experiments/dream_compare/test_dream_compare.py}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
module load cuda/12.9 2>/dev/null || true

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
echo "dream_compare MODEL=$MODEL SPECS=$SPECS TESTS=$TESTS commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || true
export SPECS
python experiments/dream_compare/jobs.py --model "$MODEL" --device cuda \
  --pytest "-q --import-mode=importlib -p no:cacheprovider $TESTS"
