# Source from a panel job (after cd to the repo): copies Ollama to the node's local disk and exports OLLAMA,
# OLLAMA_MODELS and LD_LIBRARY_PATH.
#
# Why: Ollama's GPU discovery has a 30 s watchdog per backend. Started from /scratch it often runs out of time
# loading the CUDA libraries and falls back to the CPU (dream_bind job 1118903; check job 1200115 on 2026-11-04,
# where every backend timed out). Copied to local disk with only the CUDA 12 backend (CUDA 13 has no Volta
# support, so trying it first only costs a watchdog window on the V100s), discovery found the GPU on the first
# try (check job 1200411: library=CUDA, compute 7.0). The copy takes about 30 s (1.3 GB). Models stay on /scratch.
OLLAMA_SRC=/scratch/$USER/ollama
OLLAMA_LOCAL=${TMPDIR:-/tmp}/ollama_local_${SLURM_JOB_ID:-$$}
mkdir -p "$OLLAMA_LOCAL/bin" "$OLLAMA_LOCAL/lib/ollama"
cp "$OLLAMA_SRC/bin/ollama" "$OLLAMA_LOCAL/bin/"
find "$OLLAMA_SRC/lib/ollama" -maxdepth 1 \( -type f -o -type l \) -exec cp -P {} "$OLLAMA_LOCAL/lib/ollama/" \;
cp -rP "$OLLAMA_SRC/lib/ollama/cuda_v12" "$OLLAMA_LOCAL/lib/ollama/"
export OLLAMA_MODELS=$OLLAMA_SRC/models
export LD_LIBRARY_PATH=$OLLAMA_LOCAL/lib/ollama:${LD_LIBRARY_PATH:-}
OLLAMA=$OLLAMA_LOCAL/bin/ollama
echo "ollama: local copy at $OLLAMA_LOCAL ($($OLLAMA --version 2>&1 | tail -1))"
