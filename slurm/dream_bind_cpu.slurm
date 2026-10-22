#!/bin/bash
#SBATCH --job-name=db_analyze
#SBATCH --partition=compute-p1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --output=/scratch/%u/logs/db_analyze_%j.out
#SBATCH --error=/scratch/%u/logs/db_analyze_%j.err

# dream_bind CPU job (no GPU): run.py --stage analyze on the saved answers, judge labels and panel labels.
#   MODEL=qwen25_15b OUT=/scratch/$USER/seahorse_runs/<run>/qwen25_15b sbatch slurm/dream_bind_cpu.slurm
set -eo pipefail

#   OUTS: several model dirs (space-separated) instead of OUT (addendum 7: base, _refusal, _pool dirs)
MODEL=${MODEL:?set MODEL}
OUTS=${OUTS:-${OUT:?set OUT or OUTS}}
REPO=${REPO:-/scratch/$USER/Seahorse}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$REPO"
echo "dream_bind analyze MODEL=$MODEL OUTS=$OUTS commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
pytest -q tests/test_dream_bind.py
echo "pytest passed"
for OUT in $OUTS; do
  python experiments/dream_bind/run.py --model "$MODEL" --stage analyze --out "$OUT" --device cpu
  echo "Done: $OUT/analysis"
done
