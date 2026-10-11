#!/bin/bash
# dream_bind arm R + subj generation on Qwen2.5-1.5B (PREREG addendum 2), chained with afterok dependencies:
#   gen (generation + coherence NLL; gpu-a100-small, fp32)
#     -> judge (the original judge, Qwen3.5-9B bf16; gpu-a100)
#     -> panel gpt-oss, panel Gemma-4 (Ollama; gpu-a100)
#   -> analyze (compute-p1, after the judge and both panels)
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_bind/submit_gen.sh [RUN_ROOT]    # default /scratch/$USER/seahorse_runs/dream_bind_gen_<ts>
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_bind_gen_$(date +%Y%m%d_%H%M)}
OUT=$RUN/qwen25_15b
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

g=$(sub slurm/dream_bind.slurm db_gen $SMALL --time=01:30:00 "MODEL=qwen25_15b,STAGES=gen,OUT=$OUT")
j=$(sub slurm/dream_bind.slurm db_judge $A100J --time=01:00:00 --dependency=afterok:$g "MODEL=qwen25_15b,STAGES=judge,OUT=$OUT")
pa=$(sub slurm/dream_bind_panel.slurm db_panel_gptoss $A100J --time=02:00:00 --dependency=afterok:$g "RUN=$RUN,JUDGE=gptoss")
pb=$(sub slurm/dream_bind_panel.slurm db_panel_gemma4 $A100J --time=01:15:00 --dependency=afterok:$g "RUN=$RUN,JUDGE=gemma4")
an=$(sub slurm/dream_bind_cpu.slurm db_analyze $CPU --time=00:30:00 --dependency=afterok:$j:$pa:$pb "MODEL=qwen25_15b,OUT=$OUT")

{
  echo "dream_bind gen run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD)"
  echo "gen $g (gpu-a100-small) -> judge $j, panel gptoss $pa, panel gemma4 $pb (gpu-a100) -> analyze $an (compute-p1)"
} | tee -a "$RUN/jobs.txt"
