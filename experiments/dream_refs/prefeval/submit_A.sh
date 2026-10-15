#!/bin/bash
# Phase A of the PrefEval benchmark (experiment 2 addendum) on DelftBlue: dream -> validate -> prep (+ the gate
# report), chained with afterok. From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_refs/prefeval/submit_A.sh DR_RUN [DEP] [RUN]
#     DR_RUN  the dream_refs v3 (29-item) run root: its qwen35_2b/prep/state.pt holds the valence direction v_l
#     DEP     optional job id that writes that state (its prep_lens job); prep waits for it (afterok)
#     RUN     default /scratch/$USER/seahorse_runs/dream_refs_pe_<YYYYMMDD_HHMM>
#
#   dream     Qwen3.5-9B bf16 on a full A100: frame, opposite, alternatives (prompts v3) + follow-ups, 200 items
#   validate  CPU: selection, checks, dreaming failures -> validate/refs.json, spotcheck.md
#   prep      Qwen3.5-2B fp32 on a V100 (core_v1's calibration does not fit the 10 GB slice): calibration (checked against core_v1),
#             writes, memories, thresholds, the gate on every test question and on the 50 unrelated prompts,
#             valence signs, logit lens -> report_A.md
# Phase B (submit_B.sh) waits for the user's spot check and a read of report_A.md.
set -euo pipefail

DR_RUN=${1:?usage: submit_A.sh DR_RUN [DEP] [RUN]}
DEP=${2:-}
RUN=${3:-/scratch/$USER/seahorse_runs/dream_refs_pe_$(date +%Y%m%d_%H%M)}
VALENCE=$DR_RUN/qwen35_2b/prep/state.pt
mkdir -p "$RUN" "/scratch/$USER/logs"
S=experiments/dream_refs/prefeval/job.slurm
A100="--partition=gpu-a100 --cpus-per-task=8 --mem-per-cpu=7900M --gpus-per-task=1"
V100="--partition=gpu-v100 --cpus-per-task=6 --mem-per-cpu=5300M --gpus-per-task=1"   # the calibration OOMs on the 10 GB slice
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"
test -n "$DEP" || test -f "$VALENCE" || { echo "no $VALENCE and no DEP job"; exit 1; }

sub() {
  local name=$1; shift
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$S"
}

P=${PREP_RES:-$V100}
# smoke first: 4 items through dream -> validate -> prep (real models, real settings) in $RUN/smoke
SM=$RUN/smoke
SMI=$(python -c "import json; print(' '.join(i['id'] for i in json.load(open('experiments/dream_refs/prefeval/sample.json'))['items'][:4]))")
sd=$(sub pe_smoke_dream $A100 --time=00:30:00 "STAGE=dream,RUN=$SM,EXTRA=--items $SMI")
sv=$(sub pe_smoke_validate $CPU --time=00:15:00 --dependency=afterok:$sd "STAGE=validate,RUN=$SM")
sp=$(sub pe_smoke_prep $P --time=00:40:00 --dependency=afterok:$sv${DEP:+:$DEP} "STAGE=prep,RUN=$SM,VALENCE=$VALENCE")
d=$(sub pe_dream $A100 --time=01:50:00 --dependency=afterok:$sv "STAGE=dream,RUN=$RUN")
v=$(sub pe_validate $CPU --time=00:20:00 --dependency=afterok:$d "STAGE=validate,RUN=$RUN")
p=$(sub pe_prep $P --time=01:00:00 --dependency=afterok:$v:$sp${DEP:+:$DEP} "STAGE=prep,RUN=$RUN,VALENCE=$VALENCE")

{
  echo "PrefEval phase A run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD); valence from $VALENCE"
  echo "smoke (4 items, $SM): dream $sd -> validate $sv -> prep $sp"
  echo "dream $d (after smoke validate) -> validate $v -> prep $p (${PREP_RES:-gpu-v100}; after smoke prep${DEP:+ and $DEP})"
} | tee -a "$RUN/jobs.txt"
