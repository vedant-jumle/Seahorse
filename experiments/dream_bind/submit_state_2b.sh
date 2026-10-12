#!/bin/bash
# dream_bind addendum 4 on Qwen3.5-2B (what the DeltaNet state contributes; exploratory), chained with afterok:
#   smoke (--tiny, the real 2B, --gen-set state: the R-full-state == prior-turn check and the state checks)
#     -> gen (--gen-set state; generation + coherence NLL; gpu-v100 32 GB, fp32, text decoder only)
#       -> judge (Qwen3.5-9B bf16; gpu-a100), panel gpt-oss, panel Gemma-4 (Ollama; gpu-a100)
#         -> analyze (compute-p1)
# Times = the addendum-3 chain's (gen 38 min, judge 6.5, panels ~18, analyze 3) x ~1.05 (2,916 vs 2,811 answers) + 30%.
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_bind/submit_state_2b.sh [RUN_ROOT]   # default /scratch/$USER/seahorse_runs/dream_bind_state2b_<ts>
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_bind_state2b_$(date +%Y%m%d_%H%M)}
OUT=$RUN/qwen35_2b
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

sm=$(sub slurm/dream_bind.slurm dbs_smoke $V100 --time=00:30:00 "MODEL=qwen35_2b,STAGES=gen,OUT=$RUN/smoke/qwen35_2b,EXTRA=--tiny --gen-set state --max-reserved-gb 28")
g=$(sub slurm/dream_bind.slurm dbs_gen $V100 --time=01:00:00 --dependency=afterok:$sm "MODEL=qwen35_2b,STAGES=gen,OUT=$OUT,EXTRA=--gen-set state")
j=$(sub slurm/dream_bind.slurm dbs_judge $A100J --time=00:30:00 --dependency=afterok:$g "MODEL=qwen35_2b,STAGES=judge,OUT=$OUT")
pa=$(sub slurm/dream_bind_panel.slurm dbs_panel_gptoss $A100J --time=00:40:00 --dependency=afterok:$g "RUN=$RUN,JUDGE=gptoss,MODELS=qwen35_2b")
pb=$(sub slurm/dream_bind_panel.slurm dbs_panel_gemma4 $A100J --time=00:40:00 --dependency=afterok:$g "RUN=$RUN,JUDGE=gemma4,MODELS=qwen35_2b")
an=$(sub slurm/dream_bind_cpu.slurm dbs_analyze $CPU --time=00:20:00 --dependency=afterok:$j:$pa:$pb "MODEL=qwen35_2b,OUT=$OUT")

{
  echo "dream_bind addendum 4 (2B, DeltaNet state) run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD)"
  echo "smoke $sm -> gen $g (gpu-v100) -> judge $j, panel gptoss $pa, panel gemma4 $pb (gpu-a100) -> analyze $an (compute-p1)"
} | tee -a "$RUN/jobs.txt"
