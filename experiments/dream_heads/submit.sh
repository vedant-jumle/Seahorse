#!/bin/bash
# dream_heads (PREREG.md) on DelftBlue: one run root, per-model chains with afterok.
#   smoke  (pytest: dream_heads' tests + Stage A's; h-run --tiny: 4 items, 4 comply pairs, every check: R-full == T-full,
#           keep-all == R-full, KVzip-sh mask == splice, answer scores == Stage A's; then h-analyze on the smoke)
#   -> run (h-run: d_comply, anchors, folds / L* / licence heads / d_told, every condition; resumable per item)
#   -> analyze (CPU: report.md, verdict.json) ; finally summary (afterany over the analyses).
# Stage A's selections and judged labels are read from $SA/<model> (never written).
# Qwen2.5-1.5B and Llama-3.2-3B on gpu-v100, Qwen2.5-7B on gpu-a100. 4 jobs per model + 1 summary = 13.
# From the repo on DelftBlue (/scratch/$USER/Seahorse) after `git pull`:
#   bash experiments/dream_heads/submit.sh [RUN_ROOT]
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_heads_$(date +%Y%m%d_%H%M)}
SA=${SA:-/scratch/$USER/seahorse_runs/dream_compare_a_20261105_1822}
MODELS=${MODELS:-"qwen25_15b llama32_3b qwen25_7b"}
S=$RUN/smoke
[ -f data/external/longmemeval/processed/oracle.json ] || { echo "LongMemEval oracle not prepared"; exit 1; }
for m in $MODELS; do
  [ -d "$SA/$m/tf/items" ] || { echo "Stage A records missing: $SA/$m/tf/items"; exit 1; }
done
mkdir -p "$S" "/scratch/$USER/logs"

A100="--partition=gpu-a100 --cpus-per-task=4 --mem-per-cpu=7900M --gpus-per-task=1"
V100="--partition=gpu-v100 --cpus-per-task=2 --mem-per-cpu=5300M --gpus-per-task=1"
CPU="--partition=compute-p1 --cpus-per-task=4 --mem-per-cpu=3G"
GPU=slurm/dream_compare.slurm
ANA=slurm/dream_compare_cpu.slurm
T_H=experiments/dream_heads/test_dream_heads.py
T_SMOKE="$T_H experiments/dream_compare/test_stage_a.py"
RUNNER="DC_RUNNER=../dream_heads/run"

sub() {  # script, name, resources+time+deps..., then the env as the last argument
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$RUNNER,$env" "$script"
}

{
  echo "dream_heads run $RUN, submitted $(date -Is) at commit $(git rev-parse HEAD); Stage A $SA"
  ans=""
  for M in $MODELS; do
    D=$RUN/$M
    case $M in
      qwen25_15b) tag=q15; R=$V100; fit="--max-reserved-gb 30"; t_sm=01:30:00; t_run=08:00:00 ;;
      llama32_3b) tag=ll;  R=$V100; fit="--max-reserved-gb 30"; t_sm=01:30:00; t_run=10:00:00 ;;
      qwen25_7b)  tag=q7;  R=$A100; fit="--max-reserved-gb 75"; t_sm=01:30:00; t_run=10:00:00 ;;
      *) echo "unknown model $M"; exit 1 ;;
    esac
    sm=$(sub $GPU dh_${tag}_smoke $R --time=$t_sm "MODEL=$M,TESTS=$T_SMOKE,SPECS=h-run|$S/$M|--tiny --stage-a $SA/$M $fit;h-analyze|$S/$M|--stage-a $SA/$M")
    rn=$(sub $GPU dh_${tag}_run $R --time=$t_run --dependency=afterok:$sm "MODEL=$M,TESTS=$T_H,SPECS=h-run|$D|--stage-a $SA/$M $fit")
    an=$(sub $ANA dh_${tag}_analyze $CPU --time=00:45:00 --dependency=afterok:$rn "MODEL=$M,TESTS=$T_H,SPECS=h-analyze|$D|--stage-a $SA/$M")
    ans="$ans:$an"
    echo "$M: smoke $sm ($t_sm) -> run $rn ($t_run) -> analyze $an"
  done
  su=$(sub $ANA dh_summary $CPU --time=00:20:00 --dependency=afterany$ans "MODEL=qwen25_7b,TESTS=$T_H,SPECS=h-summary|$RUN|")
  echo "summary $su (afterany$ans) -> $RUN/summary.md"
} | tee -a "$RUN/jobs.txt"
