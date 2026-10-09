#!/bin/bash
#SBATCH --job-name=cv1_panel
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/cv1_panel_%j.out
#SBATCH --error=/scratch/%u/logs/cv1_panel_%j.err

# core_v1 multi-judge robustness check (experiments/core_v1/panel_judge.py, analyze_panel.py).
# Partition, time, CPUs, memory, GPU and dependencies come from the sbatch command line. Environment:
#   STAGE     smoke | judge | analyze
#   RUN       core_v1 run root, e.g. /scratch/$USER/seahorse_runs/core_v1_20261007_1820
#   JUDGES    space-separated judge tags (gptoss gemma4 llama31 qwen36); smoke / judge stages
#   SUBSET    1 = only the answers the pre-registered C2/C3/C5 use (judge stage)
#   WORKERS   concurrent requests (default 16); PARALLEL = OLLAMA_NUM_PARALLEL (default = WORKERS)
#   N_SMOKE   smoke / ftest prompts per judge (default 64)
#   DEADLINE_MIN  judge stage: stop starting new prompts after this many minutes (the pre-registered subset goes first)
#   STAGE ftest: throughput of schema / json / none output constraints; picks format_<judge>.txt used by the judge stage
# Ollama runs on the GPU node (127.0.0.1, random free port); compute nodes have no internet, so only models
# already in OLLAMA_MODELS are used. The server is killed on exit.
set -eo pipefail

STAGE=${STAGE:?set STAGE}
RUN=${RUN:?set RUN}
REPO=${REPO:-/scratch/$USER/Seahorse}
PANEL=${PANEL:-$RUN/panel}
WORKERS=${WORKERS:-16}
PARALLEL=${PARALLEL:-$WORKERS}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$PANEL"
echo "panel STAGE=$STAGE RUN=$RUN JUDGES=${JUDGES:-} commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
if [ "$STAGE" = analyze ]; then   # the tests import torch/transformers (slow on a cold node); GPU stages skip them
  pytest -q tests/test_panel.py
  echo "pytest passed"
fi

if [ "$STAGE" = analyze ]; then
  python experiments/core_v1/analyze_panel.py --run "$RUN" --panel "$PANEL" --out "$PANEL"
  echo "Done: analyze -> $PANEL"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
export OLLAMA_MODELS=/scratch/$USER/ollama/models
export LD_LIBRARY_PATH=/scratch/$USER/ollama/lib/ollama:${LD_LIBRARY_PATH:-}
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_NUM_PARALLEL=$PARALLEL OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=2h OLLAMA_FLASH_ATTENTION=1
OLLAMA=/scratch/$USER/ollama/bin/ollama
$OLLAMA serve > "$PANEL/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OPID=$!
trap 'kill $OPID 2>/dev/null || true; wait $OPID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1 && break
  sleep 1
done
curl -s "http://127.0.0.1:$PORT/api/tags" | head -c 400; echo
echo "ollama up on port $PORT (parallel $PARALLEL)"

rc=0
for J in $JUDGES; do
  echo "=== judge $J $(date -Is)"
  if [ "$STAGE" = ftest ]; then   # throughput of the output constraint: schema vs any-JSON vs none
    for F in schema json none; do
      python experiments/core_v1/panel_judge.py --run "$RUN" --judge "$J" --out "$PANEL/ftest" --url "http://127.0.0.1:$PORT" \
        --limit "${N_SMOKE:-64}" --stratified --workers "$WORKERS" --format $F --file "${J}_$F.jsonl" || rc=1
    done
    python experiments/core_v1/panel_judge.py --run "$RUN" --out "$PANEL" --judge "$J" --pick-format "$PANEL/ftest" "$J" || true
  elif [ "$STAGE" = smoke ]; then
    python experiments/core_v1/panel_judge.py --run "$RUN" --judge "$J" --out "$PANEL/smoke" --url "http://127.0.0.1:$PORT" \
      --limit "${N_SMOKE:-64}" --stratified --workers "$WORKERS" || rc=1
  else
    python experiments/core_v1/panel_judge.py --run "$RUN" --judge "$J" --out "$PANEL" --url "http://127.0.0.1:$PORT" \
      ${SUBSET:+--subset} --workers "$WORKERS" --format "$(cat "$PANEL/format_$J.txt" 2>/dev/null || echo schema)" \
      ${DEADLINE_MIN:+--deadline-min $DEADLINE_MIN} || rc=1
  fi
  $OLLAMA ps || true
done
echo "Done: $STAGE (rc=$rc) -> $PANEL"
exit $rc
