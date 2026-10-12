#!/bin/bash
# dream_bind arm R on Qwen3.5-2B (PREREG addendum 3), chained with afterok dependencies:
#   smoke (--tiny, the real 2B: R-full-state == prior-turn check + fit check, peak reserved <= 9 GB)
#     -> gen (generation + coherence NLL; gpu-a100-small, fp32, text decoder only)
#       -> judge (Qwen3.5-9B bf16; gpu-a100), panel gpt-oss, panel Gemma-4 (Ollama; gpu-a100)
#         -> analyze (compute-p1)
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_bind/submit_gen_2b.sh [RUN_ROOT]   # default /scratch/$USER/seahorse_runs/dream_bind_gen2b_<ts>
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_bind_gen2b_$(date +%Y%m%d_%H%M)}
OUT=$RUN/qwen35_2b
mkdir -p "$RUN" "/scratch/$USER/logs"
SMALL="--partition=gpu-a100-small --cpus-per-task=2 --mem-per-cpu=6000M --gpus-per-task=1"
A100J="--partition=gpu-a100 --cpus-per-task=4 --mem-per-cpu=7900M --gpus-per-task=1"
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"

sub() {  # script, name, resources+time+deps..., then the env as the last argument
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$script"
}

sm=$(sub slurm/dream_bind.slurm db2_smoke $SMALL --time=00:45:00 "MODEL=qwen35_2b,STAGES=gen,OUT=$RUN/smoke/qwen35_2b,EXTRA=--tiny --max-reserved-gb 9.0")
g=$(sub slurm/dream_bind.slurm db2_gen $SMALL --time=02:30:00 --dependency=afterok:$sm "MODEL=qwen35_2b,STAGES=gen,OUT=$OUT")
j=$(sub slurm/dream_bind.slurm db2_judge $A100J --time=01:00:00 --dependency=afterok:$g "MODEL=qwen35_2b,STAGES=judge,OUT=$OUT")
pa=$(sub slurm/dream_bind_panel.slurm db2_panel_gptoss $A100J --time=01:30:00 --dependency=afterok:$g "RUN=$RUN,JUDGE=gptoss,MODELS=qwen35_2b")
pb=$(sub slurm/dream_bind_panel.slurm db2_panel_gemma4 $A100J --time=01:00:00 --dependency=afterok:$g "RUN=$RUN,JUDGE=gemma4,MODELS=qwen35_2b")
an=$(sub slurm/dream_bind_cpu.slurm db2_analyze $CPU --time=00:30:00 --dependency=afterok:$j:$pa:$pb "MODEL=qwen35_2b,OUT=$OUT")

{
  echo "dream_bind 2B arm R run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD)"
  echo "smoke $sm -> gen $g (gpu-a100-small) -> judge $j, panel gptoss $pa, panel gemma4 $pb (gpu-a100) -> analyze $an (compute-p1)"
} | tee -a "$RUN/jobs.txt"
