#!/bin/bash
# Phase A of experiment 2 (dream_refs) on DelftBlue: dream -> validate -> prep + lens, chained with afterok.
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_refs/submit_A.sh [RUN]        # default /scratch/$USER/seahorse_runs/dream_refs_<YYYYMMDD_HHMM>
#
#   dream      Qwen3.5-9B bf16 on a full A100 (gpu-a100): the references, from the experience sentences alone
#   validate   CPU (compute-p1): selection + validators -> validate/refs.json, validation.json, spotcheck.md
#   prep_lens  Qwen3.5-2B fp32 on gpu-a100-small (PREP_RES="$V100" to use a V100): the shifts (reusing core_v1's prep/state.pt), the valence direction, the
#              signs, the logit lens, S1 and S3 -> report_A.md
# Phase B (gen / score / panel / analyze: submit_B.sh) is NOT submitted here: it waits for the user's spot check of
# spotcheck.md and a read of report_A.md.
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_refs_$(date +%Y%m%d_%H%M)}
mkdir -p "$RUN" "/scratch/$USER/logs"
S=experiments/dream_refs/job.slurm
A100="--partition=gpu-a100 --cpus-per-task=8 --mem-per-cpu=7900M --gpus-per-task=1"
V100="--partition=gpu-v100 --cpus-per-task=6 --mem-per-cpu=5300M --gpus-per-task=1"
A100S="--partition=gpu-a100-small --cpus-per-task=2 --mem-per-cpu=8000M --gpus-per-task=1"   # 2B fp32 fits the 10 GB slice
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"
test -f /scratch/$USER/seahorse_runs/core_v1_20261007_1820/qwen35_2b/prep/state.pt || { echo "core_v1 state.pt missing"; exit 1; }

sub() {  # name, resources+time+deps..., then the env as the last argument
  local name=$1; shift
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$S"
}

d=$(sub dr_dream $A100 --time=00:45:00 "STAGE=dream,RUN=$RUN")
v=$(sub dr_validate $CPU --time=00:20:00 --dependency=afterok:$d "STAGE=validate,RUN=$RUN")
P=${PREP_RES:-$A100S}
p=$(sub dr_prep_lens $P --time=01:15:00 --dependency=afterok:$v "STAGE=prep_lens,RUN=$RUN")

{
  echo "dream_refs phase A run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD)"
  echo "dream $d (gpu-a100) -> validate $v (compute-p1) -> prep_lens $p (${PREP_RES:-gpu-a100-small})"
  echo "outputs: $RUN/spotcheck.md (after validate), $RUN/report_A.md (after prep_lens)"
} | tee -a "$RUN/jobs.txt"
