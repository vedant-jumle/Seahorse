#!/bin/bash
#SBATCH --job-name=core_v1
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/core_v1_%j.out
#SBATCH --error=/scratch/%u/logs/core_v1_%j.err

# One core_v1 job. Partition, time, CPUs, memory and GPUs come from the sbatch command line
# (experiments/core_v1/submit.sh builds the whole chain). Environment:
#   STAGE  smoke | prep | gen | score (run.py) | analyze (analyze.py --run) | combine (analyze.py --combine)
#   CFG    experiments/core_v1/config_<model>.yaml        (GPU stages)
#   RUN    the model's run dir, e.g. /scratch/$USER/seahorse_runs/core_v1_<ts>/qwen35_2b
#          (combine: the run root; it combines $RUN/qwen35_2b and $RUN/qwen35_9b)
#   EXTRA  extra arguments for run.py (e.g. --judge-smoke)
# No internet on compute nodes: models come from HF_HOME (downloaded on the login node).
set -eo pipefail

STAGE=${STAGE:?set STAGE}
RUN=${RUN:?set RUN}
REPO=${REPO:-/scratch/$USER/Seahorse}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
module load cuda/12.9 2>/dev/null || true

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$REPO"
echo "core_v1 STAGE=$STAGE CFG=${CFG:-} RUN=$RUN commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || true
pytest -q tests
echo "pytest passed"
case "$STAGE" in
  analyze) python experiments/core_v1/analyze.py --run "$RUN" ;;
  combine) python experiments/core_v1/analyze.py --combine "$RUN/qwen35_2b" "$RUN/qwen35_9b" --out "$RUN/analysis" ;;
  *) python experiments/core_v1/run.py --config "${CFG:?set CFG}" --stage "$STAGE" --out "$RUN" ${EXTRA:-} ;;
esac
echo "Done: $STAGE -> $RUN"
