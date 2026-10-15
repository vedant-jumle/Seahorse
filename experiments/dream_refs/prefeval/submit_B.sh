#!/bin/bash
# Phase B of the PrefEval benchmark (experiment 2 addendum): gen (the fixed grid) -> score (coherence + the
# original judge: j_lean and PrefEval's four rubrics) -> panel (gpt-oss, Gemma-4: violation + helpfulness) ->
# analyze. Submit ONLY after the user has spot-checked RUN/spotcheck.md and read RUN/report_A.md:
#   bash experiments/dream_refs/prefeval/submit_B.sh RUN
#
# Size (200 items x 11 conditions x 5 answers = 11,000 answers, cap 200): gen two V100 shards (~30 min each by
# core_v1's rate of ~57 ms per decode step at ~50 rows) + unrelated prompts (~25 min); score ~2 h on an A100
# (~11k j_lean + ~36k PrefEval prompts); panel 22,000 prompts per judge (gpt-oss ~0.5-0.85 /s: 7-12 h; Gemma-4
# ~1 /s: ~6 h), primary cells (nomem, ctx, alpha 1) first. Time limits are the estimate + 50%; the panel is sized
# at gpt-oss's measured ~0.85 prompts/s: 21.6k prompts = 7.1 h -> 10 h 40, new prompts stop at 610 min (resumable).
# Each judge writes its own file under an exclusive lock.
set -euo pipefail

RUN=${1:?usage: submit_B.sh RUN}
test -f "$RUN/qwen35_2b/prep/state.pt" || { echo "no phase A prep in $RUN"; exit 1; }
S=experiments/dream_refs/prefeval/job.slurm
P=experiments/dream_refs/prefeval/panel.slurm
A100="--partition=gpu-a100 --cpus-per-task=8 --mem-per-cpu=7900M --gpus-per-task=1"
V100="--partition=gpu-v100 --cpus-per-task=6 --mem-per-cpu=5300M --gpus-per-task=1"
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"

sub() {
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$script"
}

g0=$(sub $S pe_gen0 $V100 --time=01:15:00 "STAGE=gen,RUN=$RUN,EXTRA=--sections main --shard 0/2")
g1=$(sub $S pe_gen1 $V100 --time=01:15:00 "STAGE=gen,RUN=$RUN,EXTRA=--sections main --shard 1/2")
gu=$(sub $S pe_genU $V100 --time=01:00:00 "STAGE=gen,RUN=$RUN,EXTRA=--sections unrel")
sc=$(sub $S pe_score $A100 --time=03:30:00 --dependency=afterok:$g0:$g1:$gu "STAGE=score,RUN=$RUN")
pa=$(sub $P pe_panel_gptoss $A100 --time=10:40:00 --dependency=afterok:$sc "RUN=$RUN,JUDGE=gptoss,DEADLINE_MIN=610")
pb=$(sub $P pe_panel_gemma4 $A100 --time=10:40:00 --dependency=afterok:$sc "RUN=$RUN,JUDGE=gemma4,DEADLINE_MIN=610")
an=$(sub $S pe_analyze $CPU --time=00:30:00 --dependency=afterok:$pa:$pb "STAGE=analyze,RUN=$RUN,EXTRA=--panel $RUN/panel")

{
  echo "PrefEval phase B run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD)"
  echo "gen $g0 $g1 (items) $gu (unrelated) (gpu-v100) -> score $sc (gpu-a100) -> panel $pa $pb (gpu-a100) -> analyze $an"
} | tee -a "$RUN/jobs.txt"
