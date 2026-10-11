#!/bin/bash
#SBATCH --job-name=dream_refs
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/dream_refs_%j.out
#SBATCH --error=/scratch/%u/logs/dream_refs_%j.err

# One dream_refs (experiment 2) job. Partition, time, CPUs, memory, GPUs and dependencies come from the sbatch
# command line (experiments/dream_refs/submit_A.sh / submit_B.sh). Environment:
#   STAGE  dream | validate | prep_lens | gen | score | analyze
#   RUN    the run root, e.g. /scratch/$USER/seahorse_runs/dream_refs_<ts> (the 2B stages use $RUN/qwen35_2b)
#   EXTRA  extra arguments for run.py (gen: --items ... / --sections ...; analyze: --panel $RUN/panel)
# No internet on compute nodes: models come from HF_HOME (already downloaded). Every job runs pytest first.
set -eo pipefail

STAGE=${STAGE:?set STAGE}
RUN=${RUN:?set RUN}
REPO=${REPO:-/scratch/$USER/Seahorse}
CFG=experiments/dream_refs/config_qwen35_2b.yaml

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
module load cuda/12.9 2>/dev/null || true

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$REPO"
echo "dream_refs STAGE=$STAGE RUN=$RUN EXTRA=${EXTRA:-} commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || true
pytest -q tests
echo "pytest passed"
case "$STAGE" in
  dream)     python experiments/dream_refs/dream.py --config experiments/dream_refs/config_dream.yaml --out "$RUN/dream" ${EXTRA:-} ;;
  validate)  python experiments/dream_refs/validate.py --run "$RUN" ;;
  prep_lens) python experiments/dream_refs/run.py --config "$CFG" --stage prep lens --out "$RUN/qwen35_2b" ${EXTRA:-} ;;
  gen|score|analyze) python experiments/dream_refs/run.py --config "$CFG" --stage "$STAGE" --out "$RUN/qwen35_2b" ${EXTRA:-} ;;
  *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
echo "Done: $STAGE -> $RUN"
