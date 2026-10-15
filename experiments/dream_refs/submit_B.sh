#!/bin/bash
# Phase B of experiment 2 (dream_refs): generation of the FIXED grid (PREREG.md) -> score (coherence + original
# judge) -> panel judges (gpt-oss, Gemma-4) -> analyze. Submit only after the user has spot-checked
# RUN/spotcheck.md and read RUN/report_A.md. From the repo on DelftBlue, after `git pull`:
#   bash experiments/dream_refs/submit_B.sh RUN          # the phase A run root (it holds qwen35_2b/prep)
#
# gen is split by item over three V100 jobs (the whole grid is ~18k answers, ~1 h of V100 time by core_v1's rate)
# plus one job for the unrelated prompts. score on gpu-a100 (the judge is Qwen3.5-9B bf16). Time limits = estimate
# + 50%; the panel at gpt-oss's measured ~0.85 prompts/s: ~18k prompts = 5.9 h -> 9 h, and each panel job stops
# starting new prompts at 510 min (resumable). Each judge writes its own file under an exclusive lock.
set -euo pipefail

RUN=${1:?usage: submit_B.sh RUN}
test -f "$RUN/qwen35_2b/prep/state.pt" || { echo "no phase A prep in $RUN"; exit 1; }
S=experiments/dream_refs/job.slurm
P=experiments/dream_refs/panel.slurm
A100="--partition=gpu-a100 --cpus-per-task=8 --mem-per-cpu=7900M --gpus-per-task=1"
V100="--partition=gpu-v100 --cpus-per-task=6 --mem-per-cpu=5300M --gpus-per-task=1"
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"

sub() {
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$script"
}

G1="vegetarian vegan lactose_intolerant celiac diabetes loves_hiking tight_budget early_riser prefers_quiet eco_conscious"
G2="has_young_kids loves_cooking loves_cold bookworm history_buff hates_jazz no_alcohol hates_horror hates_spicy hates_hiking"
G3="hates_seafood hates_beach hates_sweets loves_jazz norway singapore japanese_food tennis horror"
g1=$(sub $S dr_gen1 $V100 --time=01:00:00 "STAGE=gen,RUN=$RUN,EXTRA=--sections main yn --items $G1")
g2=$(sub $S dr_gen2 $V100 --time=01:00:00 "STAGE=gen,RUN=$RUN,EXTRA=--sections main yn --items $G2")
g3=$(sub $S dr_gen3 $V100 --time=01:00:00 "STAGE=gen,RUN=$RUN,EXTRA=--sections main yn --items $G3")
gu=$(sub $S dr_genU $V100 --time=01:00:00 "STAGE=gen,RUN=$RUN,EXTRA=--sections unrel")
sc=$(sub $S dr_score $A100 --time=01:30:00 --dependency=afterok:$g1:$g2:$g3:$gu "STAGE=score,RUN=$RUN")
FMT=/scratch/$USER/seahorse_runs/core_v1_20261007_1820/panel   # core_v1's chosen output constraint per judge
pa=$(sub $P dr_panel_gptoss $A100 --time=09:00:00 --dependency=afterok:$sc "RUN=$RUN,JUDGE=gptoss,FORMAT_DIR=$FMT,DEADLINE_MIN=510")
pb=$(sub $P dr_panel_gemma4 $A100 --time=09:00:00 --dependency=afterok:$sc "RUN=$RUN,JUDGE=gemma4,FORMAT_DIR=$FMT,DEADLINE_MIN=510")
an=$(sub $S dr_analyze $CPU --time=00:30:00 --dependency=afterok:$pa:$pb "STAGE=analyze,RUN=$RUN,EXTRA=--panel $RUN/panel")

{
  echo "dream_refs phase B run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD)"
  echo "gen $g1 $g2 $g3 (items) $gu (unrelated) (gpu-v100) -> score $sc (gpu-a100) -> panel $pa $pb (gpu-a100) -> analyze $an"
} | tee -a "$RUN/jobs.txt"
