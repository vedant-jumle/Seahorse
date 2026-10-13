#!/bin/bash
# dream_bind PREREG addendum 5 on both models: (1) the plain-text controls (--gen-set text) and (2) the two-fact
# collision test (--stage collide; output in the sibling dir <model>_collide). Per model, chained with afterok:
#   smoke (--tiny, real model: R-full / R-full-state check, R2-joint == T2 check, fit check)
#     -> gen (gen --gen-set text + collide, coherence NLL)
#       -> judge (Qwen3.5-9B bf16, both dirs; gpu-a100), panel gpt-oss, panel Gemma-4 (both dirs; gpu-a100)
#         -> analyze (both dirs; compute-p1)
# Qwen2.5 runs on gpu-a100-small (its arm-R generation fit there; the smoke's fit check confirms <= 9 GB reserved);
# the 2B on gpu-v100 (it does not fit the 10 GB slice). Times: measured durations of the earlier chains scaled to
# this run's counts (Qwen2.5 ~3,600 judged answers, 2B ~4,050), + 30%.
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_bind/submit_add5.sh [RUN_ROOT]    # default /scratch/$USER/seahorse_runs/dream_bind_add5_<ts>
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_bind_add5_$(date +%Y%m%d_%H%M)}
mkdir -p "$RUN" "/scratch/$USER/logs"
SMALL="--partition=gpu-a100-small --cpus-per-task=2 --mem-per-cpu=6000M --gpus-per-task=1"
V100="--partition=gpu-v100 --cpus-per-task=2 --mem-per-cpu=5300M --gpus-per-task=1"
A100J="--partition=gpu-a100 --cpus-per-task=4 --mem-per-cpu=7900M --gpus-per-task=1"
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"

sub() {  # script, name, resources+time+deps..., then the env as the last argument
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$script"
}

chain() {  # model, gpu resources, smoke limit GB, gen time, gptoss time, gemma time
  local M=$1 RES=$2 LIM=$3 GT=$4 PT1=$5 PT2=$6 tag=${1:0:5}
  local OUT=$RUN/$M
  sm=$(sub slurm/dream_bind.slurm a5_${tag}_smoke $RES --time=00:30:00 "MODEL=$M,STAGES=gen collide,OUT=$RUN/smoke/$M,EXTRA=--tiny --gen-set text --max-reserved-gb $LIM")
  g=$(sub slurm/dream_bind.slurm a5_${tag}_gen $RES --time=$GT --dependency=afterok:$sm "MODEL=$M,STAGES=gen collide,OUT=$OUT,EXTRA=--gen-set text")
  j=$(sub slurm/dream_bind.slurm a5_${tag}_judge $A100J --time=00:40:00 --dependency=afterok:$g "MODEL=$M,STAGES=judge,OUT=$OUT")
  pa=$(sub slurm/dream_bind_panel.slurm a5_${tag}_gptoss $A100J --time=$PT1 --dependency=afterok:$g "RUN=$RUN,JUDGE=gptoss,MODELS=$M ${M}_collide")
  pb=$(sub slurm/dream_bind_panel.slurm a5_${tag}_gemma4 $A100J --time=$PT2 --dependency=afterok:$g "RUN=$RUN,JUDGE=gemma4,MODELS=$M ${M}_collide")
  an=$(sub slurm/dream_bind_cpu.slurm a5_${tag}_analyze $CPU --time=00:20:00 --dependency=afterok:$j:$pa:$pb "MODEL=$M,OUT=$OUT")
  echo "$M: smoke $sm -> gen $g -> judge $j, panel gptoss $pa, panel gemma4 $pb -> analyze $an"
}

{
  echo "dream_bind addendum 5 run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD)"
  chain qwen25_15b "$SMALL" 9.0 01:45:00 01:00:00 00:50:00
  chain qwen35_2b "$V100" 28 01:20:00 01:00:00 00:50:00
} | tee -a "$RUN/jobs.txt"
