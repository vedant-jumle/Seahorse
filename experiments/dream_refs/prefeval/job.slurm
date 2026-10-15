#!/bin/bash
#SBATCH --job-name=dr_pe
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/dr_pe_%j.out
#SBATCH --error=/scratch/%u/logs/dr_pe_%j.err

# One job of the PrefEval benchmark (experiment 2 addendum). Partition, time, resources and dependencies come from
# the sbatch command line (submit_A.sh / submit_B.sh). Environment:
#   STAGE    dream | validate | prep | gen | score | analyze
#   RUN      the run root, e.g. /scratch/$USER/seahorse_runs/dream_refs_pe_<ts> (the 2B stages use $RUN/qwen35_2b)
#   VALENCE  prep: the dream_refs v3 run's qwen35_2b/prep/state.pt (the valence direction v_l)
#   EXTRA    extra arguments (gen: --shard k/n | --sections unrel; analyze: --panel $RUN/panel)
# No internet on compute nodes: models come from HF_HOME. Every job runs pytest first.
set -eo pipefail

STAGE=${STAGE:?set STAGE}
RUN=${RUN:?set RUN}
REPO=${REPO:-/scratch/$USER/Seahorse}
PE=experiments/dream_refs/prefeval

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
module load cuda/12.9 2>/dev/null || true

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$REPO"
echo "dr_pe STAGE=$STAGE RUN=$RUN EXTRA=${EXTRA:-} commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || true
pytest -q tests
echo "pytest passed"
case "$STAGE" in
  dream)    python $PE/dream.py --out "$RUN/dream" ${EXTRA:-} ;;
  validate) python $PE/validate.py --run "$RUN" ;;
  prep)     python $PE/run.py --stage prep --out "$RUN/qwen35_2b" --valence "${VALENCE:?set VALENCE}" ${EXTRA:-} ;;
  gen|score|analyze) python $PE/run.py --stage "$STAGE" --out "$RUN/qwen35_2b" ${EXTRA:-} ;;
  *) echo "unknown STAGE $STAGE"; exit 2 ;;
esac
echo "Done: $STAGE -> $RUN"
