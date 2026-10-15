#!/bin/bash
#SBATCH --job-name=dr_pe_panel
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/dr_pe_panel_%j.out
#SBATCH --error=/scratch/%u/logs/dr_pe_panel_%j.err

# PrefEval benchmark (experiment 2 addendum) panel judge, one judge per job: prefeval/panel.py (PrefEval violation
# + helpfulness rubrics; Ollama set-up as slurm/panel.slurm). Resources from the sbatch command line
# (prefeval/submit_B.sh). Environment:
#   RUN         the PrefEval run root (holds qwen35_2b/)
#   JUDGE       gptoss | gemma4
#   DEADLINE_MIN  optional: stop starting new prompts after this many minutes (resumable)
#   WORKERS     concurrent requests (default 16)
set -eo pipefail

RUN=${RUN:?set RUN}
JUDGE=${JUDGE:?set JUDGE}
REPO=${REPO:-/scratch/$USER/Seahorse}
WORKERS=${WORKERS:-16}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$RUN/panel"
echo "dr_pe_panel RUN=$RUN JUDGE=$JUDGE commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
pytest -q tests/test_panel.py tests/test_dream_refs_prefeval.py
echo "pytest passed"

nvidia-smi --query-gpu=name,memory.total --format=csv
export OLLAMA_MODELS=/scratch/$USER/ollama/models
export LD_LIBRARY_PATH=/scratch/$USER/ollama/lib/ollama:${LD_LIBRARY_PATH:-}
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_NUM_PARALLEL=$WORKERS OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=2h OLLAMA_FLASH_ATTENTION=1
OLLAMA=/scratch/$USER/ollama/bin/ollama
$OLLAMA serve > "$RUN/panel/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OPID=$!
trap 'kill $OPID 2>/dev/null || true; wait $OPID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1 && break
  sleep 1
done
echo "ollama up on port $PORT"

python experiments/dream_refs/prefeval/panel.py --run "$RUN" --judge "$JUDGE" --out "$RUN/panel" --url "http://127.0.0.1:$PORT" \
  --workers "$WORKERS" ${DEADLINE_MIN:+--deadline-min $DEADLINE_MIN}
echo "Done: panel $JUDGE -> $RUN/panel"
