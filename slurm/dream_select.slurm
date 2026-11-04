#!/bin/bash
#SBATCH --job-name=dream_select
#SBATCH --partition=gpu-a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=7900M
#SBATCH --output=/scratch/%u/logs/dream_select_%j.out
#SBATCH --error=/scratch/%u/logs/dream_select_%j.err

# dream_select GPU job (PREREG addendum 1; experiments/dream_select/run.py through jobs.py). Partition, time, CPUs, memory
# and dependencies normally come from experiments/dream_select/submit.sh. Environment:
#   MODEL   qwen25_15b | qwen25_7b | llama32_3b
#   SPECS   ';'-separated run.py calls "STAGE|OUT|EXTRA" (tf, gen, judge, analyze-gen, compare-sel), all in ONE Python
#           process after pytest (experiments/dream_select/jobs.py; slow imports on /scratch, see dream_bind/multi.py)
# No internet on compute nodes: models must already be in HF_HOME.
set -eo pipefail

MODEL=${MODEL:?set MODEL (qwen25_15b | qwen25_7b | llama32_3b)}
SPECS=${SPECS:?set SPECS}
REPO=${REPO:-/scratch/$USER/Seahorse}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
module load cuda/12.9 2>/dev/null || true

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation on the 10 GB slice

cd "$REPO"
echo "dream_select MODEL=$MODEL SPECS=$SPECS commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || true
export SPECS
python experiments/dream_select/jobs.py --model "$MODEL" --device cuda \
  --pytest "-q --import-mode=importlib -p no:cacheprovider experiments/dream_select/test_dream_select.py"
