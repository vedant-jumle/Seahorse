#!/bin/bash
#SBATCH --job-name=dc_panel
#SBATCH --ntasks=1
#SBATCH --output=/scratch/%u/logs/dc_panel_%j.out
#SBATCH --error=/scratch/%u/logs/dc_panel_%j.err

# dream_compare judges on one tasks file (copied from slurm/dream_select_panel.slurm): the original judge (Qwen3.5-9B via
# run.py --stage judge-local, before Ollama starts) and the Ollama panel (experiments/dream_compare/judges.py), then a
# summary. The Ollama set-up (node-local copy via slurm/ollama_local.sh, random port, GPU-discovery watchdog with up to 3
# restarts) is dream_select's. Resources and dependencies come from the submit script. Environment:
#   JDIR        the judge dir: holds tasks.jsonl (written by PRE_SPECS' judge-tasks); outputs orig.jsonl, <judge>.jsonl
#   JUDGES      space-separated Ollama judges, run in turn (gptoss | gemma4)
#   LIMIT       optional: the first N tasks per judge
#   WORKERS     concurrent requests (default 16)
#   PRE_SPECS / POST_SPECS + MODEL: run.py calls (jobs.py, "STAGE|OUT|EXTRA;...") before Ollama (judge-tasks,
#               judge-local) and after the panel (judge-summary)
#   TESTS       pytest targets (default the dream_compare test file)
set -eo pipefail

JDIR=${JDIR:?set JDIR}
JUDGES=${JUDGES:?set JUDGES}
REPO=${REPO:-/scratch/$USER/Seahorse}
WORKERS=${WORKERS:-16}
TESTS=${TESTS:-experiments/dream_compare/test_dream_compare.py}

module load 2025
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate seahorse
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$REPO"
mkdir -p "$JDIR"
echo "dc_panel JDIR=$JDIR JUDGES=$JUDGES LIMIT=${LIMIT:-} TESTS=$TESTS commit=$(git log -1 --oneline) node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ -n "${PRE_SPECS:-}" ]; then
  SPECS="$PRE_SPECS" python experiments/dream_compare/jobs.py --model "${MODEL:?set MODEL with PRE_SPECS}" --device cuda \
    --pytest "-q --import-mode=importlib -p no:cacheprovider $TESTS"
else
  python -m pytest -q --import-mode=importlib -p no:cacheprovider $TESTS -k "judge or panel or bm25 or bench or porter"
  echo "pytest passed"
fi

source slurm/ollama_local.sh
OPID=
trap 'kill $OPID 2>/dev/null || true; wait $OPID 2>/dev/null || true' EXIT
for JUDGE in $JUDGES; do
  PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
  export OLLAMA_HOST=127.0.0.1:$PORT
  export OLLAMA_NUM_PARALLEL=$WORKERS OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=2h OLLAMA_FLASH_ATTENTION=1
  GPU_OK=0
  for attempt in 1 2 3; do
    LOG="$JDIR/ollama_${SLURM_JOB_ID}_${JUDGE}_$attempt.log"
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
  python experiments/dream_compare/judges.py --tasks "$JDIR/tasks.jsonl" --judge "$JUDGE" --url "http://127.0.0.1:$PORT" \
    --out "$JDIR/$JUDGE.jsonl" --workers "$WORKERS" ${LIMIT:+--limit $LIMIT}
  echo "Done: panel $JUDGE -> $JDIR/$JUDGE.jsonl"
  kill $OPID 2>/dev/null || true; wait $OPID 2>/dev/null || true
  OPID=
done

if [ -n "${POST_SPECS:-}" ]; then
  SPECS="$POST_SPECS" python experiments/dream_compare/jobs.py --model "${MODEL:?set MODEL with POST_SPECS}" --device cpu
fi
