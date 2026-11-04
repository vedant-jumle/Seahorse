#!/bin/bash
# dream_select PREREG addendum 1: (re)submit the judging part of a run whose gen jobs are already queued or done
# (submit.sh's judge smoke 1195912 failed: the Ollama install had lost its soname links; slurm/dream_select_panel.slurm now
# adds them in a private shim dir). Infrastructure only: the same jobs, scripts, times and dependencies as submit.sh.
#   judge smoke: gpt-oss and Gemma-4 on 24 prompts each of the 7B smoke's answers + analyze-gen (the original judge part
#   passed in 1195912: 150 unique prompts, parsed 1.000) -> per model: judge, gpt-oss, Gemma-4 (after gpt-oss), each after
#   the model's gen job and the judge smoke -> analyze-gen -> the cross-model summary.
#   bash experiments/dream_select/submit_judges.sh RUN Q15_GEN_JOB Q7_GEN_JOB LL_GEN_JOB
set -euo pipefail

RUN=${1:?run root}
declare -A GEN=([qwen25_15b]=${2:?qwen25_15b gen job} [qwen25_7b]=${3:?qwen25_7b gen job} [llama32_3b]=${4:?llama32_3b gen job})
S=$RUN/smoke
A100J="--partition=gpu-a100 --cpus-per-task=4 --mem-per-cpu=7900M --gpus-per-task=1"
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"
GPU=slurm/dream_select.slurm
PANEL=slurm/dream_select_panel.slurm
ANA=slurm/dream_select_cpu.slurm

sub() {  # script, name, resources+time+deps..., then the env as the last argument
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$script"
}

{
  echo "dream_select judging (re)submitted $(date -Is) at commit $(git rev-parse HEAD) for $RUN; gen jobs: ${GEN[*]}"
  jsm=$(sub $PANEL ds_judge_smoke $A100J --time=01:30:00 "MODEL=qwen25_7b,MODEL_DIR=$S/qwen25_7b,JUDGES=gptoss gemma4,LIMIT=24,POST_SPECS=analyze-gen|$S/qwen25_7b|")
  echo "judge smoke (panel only) $jsm (gpu-a100, 1:30)"
  ans=""
  for M in qwen25_15b qwen25_7b llama32_3b; do
    case $M in qwen25_15b) tag=q15 ;; qwen25_7b) tag=q7 ;; llama32_3b) tag=ll ;; esac
    D=$RUN/$M
    g=${GEN[$M]}
    j=$(sub $GPU ds_${tag}_judge $A100J --time=02:00:00 --dependency=afterok:$g:$jsm "MODEL=$M,SPECS=judge|$D|")
    pa=$(sub $PANEL ds_${tag}_gptoss $A100J --time=03:00:00 --dependency=afterok:$g:$jsm "MODEL_DIR=$D,JUDGES=gptoss")
    pb=$(sub $PANEL ds_${tag}_gemma4 $A100J --time=03:00:00 --dependency=afterok:$g:$jsm,afterany:$pa "MODEL_DIR=$D,JUDGES=gemma4")
    an=$(sub $ANA ds_${tag}_analyze $CPU --time=01:00:00 --dependency=afterok:$j:$pa:$pb "MODEL=$M,SPECS=analyze-gen|$D|")
    ans="$ans:$an"
    echo "$M: gen $g -> judge $j (gpu-a100, 2:00), gptoss $pa (gpu-a100, 3:00), gemma4 $pb (gpu-a100, 3:00, after gptoss) -> analyze $an (compute-p1, 1:00); judges also wait for $jsm"
  done
  su=$(sub $ANA ds_summary $CPU --time=00:45:00 --dependency=afterok$ans "MODEL=qwen25_7b,SPECS=summary|$RUN|")
  echo "summary $su (compute-p1, 0:45) after$ans -> $RUN/summary.md"
} | tee -a "$RUN/jobs.txt"
