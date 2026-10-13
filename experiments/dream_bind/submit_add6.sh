#!/bin/bash
# dream_bind PREREG addendum 6 / 6b, chained with afterok:
#   Qwen2.5 (A) binding-ID repair + (B) G-key retrieval/capacity:
#     smoke (--tiny, both stages, fit check) -> idrepair job, gkey job (parallel; gpu-a100-small)
#       -> judge (original, both dirs), panel gpt-oss, panel Gemma-4 (both dirs; gpu-a100) -> analyze (compute-p1)
#   Qwen3.5-2B (C) DeltaNet conv buffer vs recurrent state (--gen-set decomp):
#     smoke (--tiny, fit check) -> gen (gpu-a100-small) -> judge, panels (gpu-a100) -> analyze (compute-p1)
# Times: local 4060 tiny runs and the earlier chains' measured durations, scaled to the counts, + 30%.
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_bind/submit_add6.sh [RUN_ROOT]    # default /scratch/$USER/seahorse_runs/dream_bind_add6_<ts>
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_bind_add6_$(date +%Y%m%d_%H%M)}
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

Q=$RUN/qwen25_15b
sq=$(sub slurm/dream_bind.slurm a6_q25_smoke $SMALL --time=00:40:00 "MODEL=qwen25_15b,STAGES=idrepair gkey,OUT=$RUN/smoke/qwen25_15b,EXTRA=--tiny --max-reserved-gb 9.0")
ga=$(sub slurm/dream_bind.slurm a6_q25_idrepair $SMALL --time=01:15:00 --dependency=afterok:$sq "MODEL=qwen25_15b,STAGES=idrepair,OUT=$Q")
gb=$(sub slurm/dream_bind.slurm a6_q25_gkey $SMALL --time=00:45:00 --dependency=afterok:$sq "MODEL=qwen25_15b,STAGES=gkey,OUT=$Q")
jq=$(sub slurm/dream_bind.slurm a6_q25_judge $A100J --time=00:40:00 --dependency=afterok:$ga:$gb "MODEL=qwen25_15b,STAGES=judge,OUT=$Q")
p1=$(sub slurm/dream_bind_panel.slurm a6_q25_gptoss $A100J --time=00:45:00 --dependency=afterok:$ga:$gb "RUN=$RUN,JUDGE=gptoss,MODELS=qwen25_15b_idrepair qwen25_15b_gkey")
p2=$(sub slurm/dream_bind_panel.slurm a6_q25_gemma4 $A100J --time=00:40:00 --dependency=afterok:$ga:$gb "RUN=$RUN,JUDGE=gemma4,MODELS=qwen25_15b_idrepair qwen25_15b_gkey")
aq=$(sub slurm/dream_bind_cpu.slurm a6_q25_analyze $CPU --time=00:20:00 --dependency=afterok:$jq:$p1:$p2 "MODEL=qwen25_15b,OUT=$Q")

T=$RUN/qwen35_2b
s2=$(sub slurm/dream_bind.slurm a6_2b_smoke $SMALL --time=00:40:00 "MODEL=qwen35_2b,STAGES=gen,OUT=$RUN/smoke/qwen35_2b,EXTRA=--tiny --gen-set decomp --max-reserved-gb 9.3")
g2=$(sub slurm/dream_bind.slurm a6_2b_gen $SMALL --time=01:15:00 --dependency=afterok:$s2 "MODEL=qwen35_2b,STAGES=gen,OUT=$T,EXTRA=--gen-set decomp")
j2=$(sub slurm/dream_bind.slurm a6_2b_judge $A100J --time=00:30:00 --dependency=afterok:$g2 "MODEL=qwen35_2b,STAGES=judge,OUT=$T")
q1=$(sub slurm/dream_bind_panel.slurm a6_2b_gptoss $A100J --time=00:30:00 --dependency=afterok:$g2 "RUN=$RUN,JUDGE=gptoss,MODELS=qwen35_2b")
q2=$(sub slurm/dream_bind_panel.slurm a6_2b_gemma4 $A100J --time=00:30:00 --dependency=afterok:$g2 "RUN=$RUN,JUDGE=gemma4,MODELS=qwen35_2b")
a2=$(sub slurm/dream_bind_cpu.slurm a6_2b_analyze $CPU --time=00:20:00 --dependency=afterok:$j2:$q1:$q2 "MODEL=qwen35_2b,OUT=$T")

{
  echo "dream_bind addendum 6/6b run $RUN submitted $(date -Is) at commit $(git rev-parse HEAD)"
  echo "qwen25_15b (A)+(B): smoke $sq -> idrepair $ga, gkey $gb -> judge $jq, panel gptoss $p1, panel gemma4 $p2 -> analyze $aq"
  echo "qwen35_2b (C): smoke $s2 -> gen $g2 -> judge $j2, panel gptoss $q1, panel gemma4 $q2 -> analyze $a2"
} | tee -a "$RUN/jobs.txt"
