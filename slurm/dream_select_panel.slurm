#!/bin/bash
#SBATCH --job-name=ds_panel
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/ds_panel_%j.out
#SBATCH --error=/scratch/%u/logs/ds_panel_%j.err

# dream_select panel judges (PREREG addendum 1): experiments/dream_select/panel.py through Ollama on the GPU node, one model
# dir. The Ollama set-up (random free port, GPU-discovery watchdog with up to 3 restarts) is slurm/dream_bind_panel.slurm's.
# Resources and dependencies come from experiments/dream_select/submit.sh. Environment:
#   MODEL_DIR   RUN/<model> (holds gen/items/*.jsonl.gz); the output is MODEL_DIR/panel/<judge>.jsonl
#   JUDGES      space-separated, run in turn (gptoss | gemma4)
#   LIMIT       optional: the first N unique prompts per judge (the judge smoke)
#   WORKERS     concurrent requests (default 16)
#   PRE_SPECS / POST_SPECS + MODEL: optional run.py calls (jobs.py, "STAGE|OUT|EXTRA;...") before Ollama starts (the
#               original judge in the judge smoke) and after the panel (its analysis)
set -eo pipefail

MODEL_DIR=${MODEL_DIR:?set MODEL_DIR}
JUDGES=${JUDGES:?set JUDGES}
REPO=${REPO:-/scratch/$USER/Seahorse}
WORKERS=${WORKERS:-16}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$MODEL_DIR/panel"
echo "ds_panel MODEL_DIR=$MODEL_DIR JUDGES=$JUDGES LIMIT=${LIMIT:-} commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ -n "${PRE_SPECS:-}" ]; then
  SPECS="$PRE_SPECS" python experiments/dream_select/jobs.py --model "${MODEL:?set MODEL with PRE_SPECS}" --device cuda \
    --pytest "-q --import-mode=importlib -p no:cacheprovider experiments/dream_select/test_dream_select.py"
else
  # the torch-free tests (prompt, parse, panel client, job specs)
  python -m pytest -q --import-mode=importlib -p no:cacheprovider experiments/dream_select/test_dream_select.py \
    -k "panel or judge or jobs"
  echo "pytest passed"
fi

# Ollama v0.40.13 (reinstalled 2026-11-04; the old soname shim is no longer needed), run from a node-local copy:
# see slurm/ollama_local.sh.
source slurm/ollama_local.sh
OPID=
trap 'kill $OPID 2>/dev/null || true; wait $OPID 2>/dev/null || true' EXIT
for JUDGE in $JUDGES; do
  PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
  export OLLAMA_HOST=127.0.0.1:$PORT
  export OLLAMA_NUM_PARALLEL=$WORKERS OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=2h OLLAMA_FLASH_ATTENTION=1
  # Ollama's GPU discovery has a 30 s watchdog; with the slow /scratch libraries it can time out and the server then runs
  # on the CPU (total_vram="0 B"; dream_bind job 1118903). Wait for the discovery line, restart up to 3 times.
  GPU_OK=0
  for attempt in 1 2 3; do
    LOG="$MODEL_DIR/panel/ollama_${SLURM_JOB_ID}_${JUDGE}_$attempt.log"
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
  echo "ollama up on port $PORT ($(grep -o 'total_vram="[^"]*"' "$LOG" | tail -1)) for $JUDGE"
  python experiments/dream_select/panel.py --model-dir "$MODEL_DIR" --judge "$JUDGE" --url "http://127.0.0.1:$PORT" \
    --workers "$WORKERS" ${LIMIT:+--limit $LIMIT}
  echo "Done: panel $JUDGE -> $MODEL_DIR/panel"
  kill $OPID 2>/dev/null || true; wait $OPID 2>/dev/null || true
  OPID=
done

if [ -n "${POST_SPECS:-}" ]; then
  SPECS="$POST_SPECS" python experiments/dream_select/jobs.py --model "${MODEL:?set MODEL with POST_SPECS}" --device cpu
fi
