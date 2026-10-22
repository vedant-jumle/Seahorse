#!/bin/bash
#SBATCH --job-name=dream_bind
#SBATCH --partition=gpu-a100-small
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=6000M
#SBATCH --output=/scratch/%u/logs/dream_bind_%j.out
#SBATCH --error=/scratch/%u/logs/dream_bind_%j.err

# dream_bind GPU job (experiments/dream_bind/run.py). Defaults: gpu-a100-small = one 10 GB A100 MIG slice (<= 2
# CPUs, <= 8000 MB per CPU, 4 h); sbatch flags override them (the judge stage runs on gpu-a100). Qwen3.5-2B fits
# in fp32 only as the text decoder (run.py loads it without the vision tower). From the repo root:
#   MODEL=qwen35_2b STAGES=patch OUT=/scratch/$USER/seahorse_runs/<run>/qwen35_2b sbatch slurm/dream_bind.slurm
#   EXTRA: extra run.py arguments. No internet on compute nodes: models must already be in HF_HOME.
set -eo pipefail

#   SPECS (instead of STAGES / OUT / EXTRA): several run.py calls after one pytest, ';'-separated, each
#   "STAGES|OUT|EXTRA" (e.g. the addendum-7 smokes, and the judge over several model dirs).
MODEL=${MODEL:?set MODEL (qwen35_2b | qwen25_15b | llama32_3b | qwen35_9b)}
if [ -z "${SPECS:-}" ]; then
  STAGES=${STAGES:?set STAGES (patch | knockout | prefix | analytic | gen | judge | ...)}
  OUT=${OUT:?set OUT}
  SPECS="$STAGES|$OUT|${EXTRA:-}"
fi
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
echo "dream_bind MODEL=$MODEL SPECS=$SPECS commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || true
pytest -q tests
echo "pytest passed"
IFS=';' read -ra RUNS <<< "$SPECS"
for spec in "${RUNS[@]}"; do
  IFS='|' read -r st out ex <<< "$spec"
  echo "=== run.py --stage $st --out $out $ex"
  python experiments/dream_bind/run.py --model "$MODEL" --stage $st --out "$out" --device cuda --dtype fp32 $ex
  echo "Done: $out"
done
