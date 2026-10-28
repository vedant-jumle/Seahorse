#!/bin/bash
#SBATCH --job-name=db_panel
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/db_panel_%j.out
#SBATCH --error=/scratch/%u/logs/db_panel_%j.err

# dream_bind panel judge, one judge per job: experiments/dream_bind/panel.py with core_v1's panel code (the
# Ollama server set-up is slurm/panel.slurm's). Resources come from the sbatch command line
# (experiments/dream_bind/submit_gen.sh). Environment:
#   RUN         the run root (holds qwen25_15b/)
#   JUDGE       gptoss | gemma4
#   FORMAT_DIR  optional: a dir with format_<judge>.txt (core_v1's chosen output constraint); default schema
#   WORKERS     concurrent requests (default 16)
#   MODELS      the model dir(s) under RUN (default qwen25_15b)
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
echo "db_panel RUN=$RUN JUDGE=$JUDGE commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
pytest -q tests/test_panel.py
echo "pytest passed"

nvidia-smi --query-gpu=name,memory.total --format=csv
export OLLAMA_MODELS=/scratch/$USER/ollama/models
export LD_LIBRARY_PATH=/scratch/$USER/ollama/lib/ollama:${LD_LIBRARY_PATH:-}
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_NUM_PARALLEL=$WORKERS OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=2h OLLAMA_FLASH_ATTENTION=1
OLLAMA=/scratch/$USER/ollama/bin/ollama
# Ollama's GPU discovery has a 30 s watchdog; with the slow /scratch libraries it can time out, and the server then
# runs on the CPU (total_vram="0 B") and every request hangs until the time limit (job 1118903, 2026-10-28).
# Wait for the discovery line, restart up to 3 times if it found no GPU, and fail fast otherwise.
OPID=
trap 'kill $OPID 2>/dev/null || true; wait $OPID 2>/dev/null || true' EXIT
GPU_OK=0
for attempt in 1 2 3; do
  LOG="$RUN/panel/ollama_${SLURM_JOB_ID}_$attempt.log"
  $OLLAMA serve > "$LOG" 2>&1 &
  OPID=$!
  for i in $(seq 1 300); do
    grep -q 'msg="vram-based default context"' "$LOG" 2>/dev/null && break
    sleep 1
  done
  if grep 'msg="vram-based default context"' "$LOG" | grep -qv 'total_vram="0 B"'; then GPU_OK=1; break; fi
  echo "ollama attempt $attempt: no GPU found ($(grep -o 'total_vram="[^"]*"' "$LOG" | tail -1)), restarting"
  kill $OPID 2>/dev/null || true; wait $OPID 2>/dev/null || true
  sleep 10
done
[ "$GPU_OK" = 1 ] || { echo "ollama found no GPU after 3 attempts"; exit 1; }
for i in $(seq 1 120); do
  curl -s "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1 && break
  sleep 1
done
echo "ollama up on port $PORT ($(grep -o 'total_vram="[^"]*"' "$LOG" | tail -1))"
FMT=$(cat "${FORMAT_DIR:-/nonexistent}/format_$JUDGE.txt" 2>/dev/null || echo schema)
python experiments/dream_bind/panel.py --run "$RUN" --judge "$JUDGE" --out "$RUN/panel" --url "http://127.0.0.1:$PORT" \
  --workers "$WORKERS" --format "$FMT" --models ${MODELS:-qwen25_15b}
echo "Done: panel $JUDGE -> $RUN/panel"
