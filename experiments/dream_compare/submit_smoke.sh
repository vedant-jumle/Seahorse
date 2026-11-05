#!/bin/bash
# dream_compare Stage 0 cluster smoke (PREREG.md "Stage 0"): one GPU job per model (pytest first, then run.py --stage
# smoke: load, identity checks on the real model, Gemma's sliding window at 1024, one item per benchmark adapter, peak
# memory), then one judge smoke (gpu-a100: judge-tasks from the smokes' answers + a sanity set, the original judge
# Qwen3.5-9B, gpt-oss and Gemma-4 through Ollama, judge-summary with parse rates) after all of them (afterany: it also
# runs if one model fails, on the answers that exist + the sanity set).
#   qwen25_15b  gpu-a100-small (10 GB MIG)   qwen25_7b  gpu-a100 (1 x 80 GB)   llama32_3b  gpu-v100 (32 GB)
#   gemma4_26b_a4b  gpu-a100, 2 GPUs (fp32 text decoder ~101 GB, device_map "auto", 66 GiB per GPU)
#   granite42_30b   gpu-a100, 2 GPUs, only if it is in the HF cache (optional model)
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull` and the prep scripts (data/external/*/processed):
#   bash experiments/dream_compare/submit_smoke.sh [RUN_ROOT]
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_compare_smoke_$(date +%Y%m%d_%H%M)}
S=$RUN/smoke
J=$RUN/judge_smoke
for f in longmemeval/processed/oracle.json locomo/processed/locomo10.json personamem/processed/32k.json beam/processed/100K.json; do
  [ -f "data/external/$f" ] || { echo "data/external/$f missing: run the prep scripts first"; exit 1; }
done
need() { ls /scratch/$USER/hf_cache/hub | grep -q "^$1\$"; }
for m in models--Qwen--Qwen2.5-1.5B-Instruct models--Qwen--Qwen2.5-7B-Instruct models--meta-llama--Llama-3.2-3B-Instruct \
         models--google--gemma-4-26B-A4B-it models--Qwen--Qwen3.5-9B; do
  need $m || { echo "$m not in the HF cache"; exit 1; }
done
mkdir -p "$S" "$J" "/scratch/$USER/logs"

MIG="--partition=gpu-a100-small --cpus-per-task=2 --mem-per-cpu=7900M --gpus-per-task=1"
A100="--partition=gpu-a100 --cpus-per-task=6 --mem-per-cpu=7900M --gpus-per-task=1"
A100x2="--partition=gpu-a100 --cpus-per-task=8 --mem-per-cpu=7900M --gpus-per-task=2"
V100="--partition=gpu-v100 --cpus-per-task=2 --mem-per-cpu=5300M --gpus-per-task=1"
GPU=slurm/dream_compare.slurm
PANEL=slurm/dream_compare_panel.slurm

sub() {  # script, name, resources+time+deps..., then the env as the last argument
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$script"
}

{
  echo "dream_compare Stage 0 smoke $RUN, submitted $(date -Is) at commit $(git rev-parse HEAD)"
  ids=""
  q15=$(sub $GPU dc_q15_smoke $MIG --time=01:30:00 "MODEL=qwen25_15b,SPECS=smoke|$S/qwen25_15b|")
  q7=$(sub $GPU dc_q7_smoke $A100 --time=01:30:00 "MODEL=qwen25_7b,SPECS=smoke|$S/qwen25_7b|")
  ll=$(sub $GPU dc_ll_smoke $V100 --time=01:30:00 "MODEL=llama32_3b,SPECS=smoke|$S/llama32_3b|")
  g4=$(sub $GPU dc_g4_smoke $A100x2 --time=03:00:00 "MODEL=gemma4_26b_a4b,SPECS=smoke|$S/gemma4_26b_a4b|--gb-per-gpu 66")
  ids="$q15:$q7:$ll:$g4"
  echo "smokes: qwen25_15b $q15 (gpu-a100-small 1:30), qwen25_7b $q7 (gpu-a100 1:30), llama32_3b $ll (gpu-v100 1:30), gemma4_26b_a4b $g4 (2 x gpu-a100 3:00)"
  if need models--ibm-granite--granite-4.2-30b; then
    gr=$(sub $GPU dc_gr_smoke $A100x2 --time=03:00:00 "MODEL=granite42_30b,SPECS=smoke|$S/granite42_30b|--gb-per-gpu 70")
    ids="$ids:$gr"
    echo "optional: granite42_30b $gr (2 x gpu-a100 3:00)"
  fi
  js=$(sub $PANEL dc_judge_smoke --partition=gpu-a100 --cpus-per-task=4 --mem-per-cpu=7900M --gpus-per-task=1 \
       --time=02:30:00 --dependency=afterany:$ids \
       "MODEL=qwen25_7b,JDIR=$J,JUDGES=gptoss gemma4,PRE_SPECS=judge-tasks|$J|--tasks-from $S;judge-local|$J|,POST_SPECS=judge-summary|$J|")
  echo "judge smoke $js (gpu-a100 2:30) after $ids (afterany)"
} | tee -a "$RUN/jobs.txt"
