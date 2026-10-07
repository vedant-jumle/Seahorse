#!/bin/bash
# Submit the core_v1 chains for both models on DelftBlue (afterok dependencies; nothing heavy runs here).
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/core_v1/submit.sh [RUN_ROOT]        # default /scratch/$USER/seahorse_runs/core_v1_<YYYYMMDD_HHMM>
#
# Per model: smoke -> prep (calibrate, write, analytic, checks) -> gen -> score (coherence + judge) -> analyze.
#   qwen35_2b: smoke, prep, gen on gpu-v100 (fp32); score on gpu-a100 (the judge is Qwen3.5-9B in bf16, and
#              it also waits for the 9B smoke, which tests the judge); analyze on compute-p1 (CPU)
#   qwen35_9b: smoke (+ judge smoke), prep, gen, score on gpu-a100 (A100 80GB, fp32); analyze on compute-p1
# Then one combine job (C7 table + the hand-audit sample) after both analyses.
# Every job runs pytest first. The account allows 8 running jobs; every job is far below the 1-day cap.
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/core_v1_$(date +%Y%m%d_%H%M)}
mkdir -p "$RUN" "/scratch/$USER/logs"
S=slurm/core_v1.slurm
V100="--partition=gpu-v100 --cpus-per-task=6 --mem-per-cpu=5300M --gpus-per-task=1"
A100="--partition=gpu-a100 --cpus-per-task=8 --mem-per-cpu=7900M --gpus-per-task=1"
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"
C2=experiments/core_v1/config_qwen35_2b.yaml
C9=experiments/core_v1/config_qwen35_9b.yaml
R2=$RUN/qwen35_2b
R9=$RUN/qwen35_9b

sub() {  # name, resources+time+deps..., then the env as the last argument
  local name=$1; shift
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$S"
}

# --- qwen35_9b (A100) -------------------------------------------------------------------------
s9=$(sub cv1_9b_smoke $A100 --time=03:00:00 "STAGE=smoke,CFG=$C9,RUN=$R9,EXTRA=--judge-smoke")
p9=$(sub cv1_9b_prep $A100 --time=03:00:00 --dependency=afterok:$s9 "STAGE=prep,CFG=$C9,RUN=$R9")
g9=$(sub cv1_9b_gen $A100 --time=14:00:00 --dependency=afterok:$p9 "STAGE=gen,CFG=$C9,RUN=$R9")
c9=$(sub cv1_9b_score $A100 --time=08:00:00 --dependency=afterok:$g9 "STAGE=score,CFG=$C9,RUN=$R9")
a9=$(sub cv1_9b_analyze $CPU --time=02:00:00 --dependency=afterok:$c9 "STAGE=analyze,RUN=$R9")

# --- qwen35_2b (V100; scoring on A100) --------------------------------------------------------
s2=$(sub cv1_2b_smoke $V100 --time=01:30:00 "STAGE=smoke,CFG=$C2,RUN=$R2")
p2=$(sub cv1_2b_prep $V100 --time=02:00:00 --dependency=afterok:$s2 "STAGE=prep,CFG=$C2,RUN=$R2")
g2=$(sub cv1_2b_gen $V100 --time=08:00:00 --dependency=afterok:$p2 "STAGE=gen,CFG=$C2,RUN=$R2")
c2=$(sub cv1_2b_score $A100 --time=04:00:00 --dependency=afterok:$g2:$s9 "STAGE=score,CFG=$C2,RUN=$R2")
a2=$(sub cv1_2b_analyze $CPU --time=02:00:00 --dependency=afterok:$c2 "STAGE=analyze,RUN=$R2")

# --- both ---------------------------------------------------------------------------------------
cb=$(sub cv1_combine $CPU --time=02:00:00 --dependency=afterok:$a2:$a9 "STAGE=combine,RUN=$RUN")

{
  echo "core_v1 run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD)"
  echo "qwen35_9b: smoke $s9 -> prep $p9 -> gen $g9 -> score $c9 -> analyze $a9"
  echo "qwen35_2b: smoke $s2 -> prep $p2 -> gen $g2 -> score $c2 (also after $s9) -> analyze $a2"
  echo "combine: $cb (after $a2, $a9)"
} | tee "$RUN/jobs.txt"
