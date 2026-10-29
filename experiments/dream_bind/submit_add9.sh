#!/bin/bash
# dream_bind PREREG addendum 9 (storage order with N = 3 / 5 stored facts), one fresh run root, chained with afterok.
# Per model:
#   smoke (--tiny with the real model: pytest (not slow) + collide add9 on g3c_sister, g3d_1, g5c_dog, g5d_1 + the add9
#   analysis on it, fit check; built-in checks: RN-sep-ordered with zero dummies == RN-sep-shift, RN-joint == T-N, the
#   kept-token asserts, one-token values and dummy names, the groups-file counts)
#     -> collide add9 (30 groups, 114 members, 6 conditions: 6,840 answers + the TF readout)
#     -> original judge + gpt-oss panel (gpu-a100) -> Gemma-4 panel after gpt-oss -> analyze (compute-p1)
#   qwen25 and 2b on gpu-a100 (full A100), llama on gpu-v100 (fp32). At most 6 jobs run at once (judge + gpt-oss per model).
# Times (PREREG addendum 9, run plan): addendum 8's measured durations (sacct, run dream_bind_add8_20261025_1812): the
# add8pool collide jobs spent 36 / 57 / 55 min generating (Qwen2.5 / 2B / Llama; 8,840 answers + 3,536 yes/no) plus
# 22-30 min of imports, pytest and loading; scaled to 6,840 answers, + 30%, + up to 60 min start-up -> 2:00 / 2:30 / 2:30.
# Smoke 1:30 (addendum 8: 17-34 min); judge 1:30; gpt-oss and Gemma-4 3:00 each (scaled ~2:15); analyze 2:00.
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_bind/submit_add9.sh [RUN_ROOT]          # PARTS="qwen25 2b llama" by default
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_bind_add9_$(date +%Y%m%d_%H%M)}
PARTS=${PARTS:-qwen25 2b llama}
mkdir -p "$RUN" "/scratch/$USER/logs"
V100="--partition=gpu-v100 --cpus-per-task=2 --mem-per-cpu=5300M --gpus-per-task=1"
A100J="--partition=gpu-a100 --cpus-per-task=4 --mem-per-cpu=7900M --gpus-per-task=1"
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"

sub() {  # script, name, resources+time+deps..., then the env as the last argument
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$script"
}
GPU=slurm/dream_bind.slurm
PANEL=slurm/dream_bind_panel.slurm
ANA=slurm/dream_bind_cpu.slurm

chain() {  # model key, tag, resources, partition name, fit limit, smoke time, collide time
  local M=$1 tag=$2 R=$3 pn=$4 L="--max-reserved-gb $5" TS=$6 TC=$7
  local S=$RUN/smoke sm c9 j pa pb an
  sm=$(sub $GPU a9_${tag}_smoke $R --time=$TS "MODEL=$M,SPECS=collide|$S/${M}_g9|--tiny --collide-set add9 $L;analyze|$S/${M}_g9|")
  c9=$(sub $GPU a9_${tag}_collide $R --time=$TC --dependency=afterok:$sm "MODEL=$M,STAGES=collide,OUT=$RUN/${M}_g9,EXTRA=--collide-set add9")
  j=$(sub $GPU a9_${tag}_judge $A100J --time=01:30:00 --dependency=afterok:$c9 "MODEL=$M,SPECS=judge|$RUN/${M}_g9|")
  pa=$(sub $PANEL a9_${tag}_gptoss $A100J --time=03:00:00 --dependency=afterok:$c9 "RUN=$RUN,JUDGE=gptoss,MODELS=${M}_g9_collide")
  pb=$(sub $PANEL a9_${tag}_gemma4 $A100J --time=03:00:00 --dependency=afterok:$c9,afterany:$pa "RUN=$RUN,JUDGE=gemma4,MODELS=${M}_g9_collide")
  an=$(sub $ANA a9_${tag}_analyze $CPU --time=02:00:00 --dependency=afterok:$j:$pa:$pb "MODEL=$M,OUTS=$RUN/${M}_g9")
  echo "$M ($pn): smoke $sm ($TS) -> collide add9 $c9 ($TC) -> judge $j (1:30), gptoss $pa (3:00), gemma4 $pb (3:00, after gptoss) (gpu-a100) -> analyze $an (2:00)"
}

{
  echo "dream_bind addendum 9 run $RUN, parts '$PARTS', submitted $(date -Is) at commit $(git rev-parse HEAD)"
  for P in $PARTS; do
    case $P in
    qwen25) chain qwen25_15b q25 "$A100J" gpu-a100 70 01:30:00 02:00:00 ;;
    2b)     chain qwen35_2b 2b "$A100J" gpu-a100 70 01:30:00 02:30:00 ;;
    llama)
      ls /scratch/$USER/hf_cache/hub | grep -q "models--meta-llama--Llama-3.2-3B-Instruct" || { echo "Llama-3.2-3B-Instruct not in the HF cache"; exit 1; }
      chain llama32_3b ll "$V100" gpu-v100 28 01:30:00 02:30:00 ;;
    *) echo "unknown part $P"; exit 1 ;;
    esac
  done
} | tee -a "$RUN/jobs.txt"
