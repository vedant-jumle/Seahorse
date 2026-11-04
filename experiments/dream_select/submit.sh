#!/bin/bash
# dream_select PREREG addendum 1 (DelftBlue replication + generation), one fresh run root, chained with afterok. Per model:
#   smoke (--tiny with the real model: pytest + tf (3 items, k = 4) + gen + analyze-gen; built-in checks: R-full == T-full
#   in tf and in gen, the kept-token asserts, Paris, the re-derived head / content positions == the tf run's, the fit check)
#     -> tf (7B and Llama: 64 items, every selector and budget)
#     -> gen (5,320 answers: 17 / 14 conditions x 5 answers per item; the 1.5B from the LOCAL run's selections, copied to
#        <run>/qwen25_15b/tf)
#     -> original judge (Qwen3.5-9B bf16) + gpt-oss panel (gpu-a100) -> Gemma-4 panel after gpt-oss -> analyze-gen (CPU)
#   One judge smoke (gpu-a100, after the 7B smoke): the original judge on the 7B smoke's answers, gpt-oss and Gemma-4 on
#   24 prompts each, analyze-gen. Every judge / panel job also waits for it.
#   Finally: the cross-model summary (CPU) -> <run>/summary.md with the replication verdict.
#   1.5B on gpu-a100-small (10 GB slice; local peak 6.7 GB reserved; fit check 9 GB), 7B on gpu-a100 (fp32 ~30 GB of
#   weights; fit check 75 GB), Llama on gpu-v100 (fp32; fit check 30 GB).
# Times: dream_bind add8 / add9's measured rates (jobs.txt + sacct: Qwen2.5 on the MIG slice ~60 answers/min, Llama on a
# V100 ~100/min, the judge ~134 answers/min with 64 new tokens, the panels ~79/min at the slowest) and the local 1.5B run
# (tf 12 min for 64 items; gen ~113 answers/min on the RTX 4060), scaled to 5,320 answers per model, + 30%, + up to
# 60 min of Python start-up on /scratch. Smoke 1:30; tf 3:00; gen 3:00; judge 2:00; gpt-oss / Gemma-4 3:00 each;
# analyze 1:00; judge smoke 2:00; summary 0:45.
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull` and after copying the local run
# (results/dream_select_20261104_0032: rows.jsonl.gz, meta.json, report.md, verdict.json) to $LOCAL_TF:
#   bash experiments/dream_select/submit.sh [RUN_ROOT]
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_select_cluster_$(date +%Y%m%d_%H%M)}
LOCAL_TF=${LOCAL_TF:-/scratch/$USER/seahorse_runs/dream_select_20261104_0032}
S=$RUN/smoke
for f in rows.jsonl.gz meta.json verdict.json; do
  [ -f "$LOCAL_TF/$f" ] || { echo "the local run is not at $LOCAL_TF ($f missing)"; exit 1; }
done
for m in models--Qwen--Qwen2.5-1.5B-Instruct models--Qwen--Qwen2.5-7B-Instruct models--meta-llama--Llama-3.2-3B-Instruct \
         models--Qwen--Qwen3.5-9B; do
  ls /scratch/$USER/hf_cache/hub | grep -q "^$m\$" || { echo "$m not in the HF cache"; exit 1; }
done
mkdir -p "$RUN/qwen25_15b/tf" "$S" "/scratch/$USER/logs"
cp "$LOCAL_TF"/{rows.jsonl.gz,meta.json,verdict.json,report.md} "$RUN/qwen25_15b/tf/"

MIG="--partition=gpu-a100-small --cpus-per-task=2 --mem-per-cpu=7900M --gpus-per-task=1"
A100_7B="--partition=gpu-a100 --cpus-per-task=6 --mem-per-cpu=7900M --gpus-per-task=1"
V100="--partition=gpu-v100 --cpus-per-task=2 --mem-per-cpu=5300M --gpus-per-task=1"
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
  echo "dream_select addendum 1 run $RUN, submitted $(date -Is) at commit $(git rev-parse HEAD); local tf copied from $LOCAL_TF"
  # --- smokes
  sm15=$(sub $GPU ds_q15_smoke $MIG --time=01:30:00 "MODEL=qwen25_15b,SPECS=tf|$S/qwen25_15b/tf|--tiny --max-reserved-gb 9;compare-sel|$S/qwen25_15b/tf|--sel-from $RUN/qwen25_15b/tf;gen|$S/qwen25_15b|--tiny --sel-from $RUN/qwen25_15b/tf --max-reserved-gb 9;analyze-gen|$S/qwen25_15b|--sel-from $RUN/qwen25_15b/tf")
  sm7=$(sub $GPU ds_q7_smoke $A100_7B --time=01:30:00 "MODEL=qwen25_7b,SPECS=tf|$S/qwen25_7b/tf|--tiny --max-reserved-gb 75;gen|$S/qwen25_7b|--tiny --max-reserved-gb 75;analyze-gen|$S/qwen25_7b|")
  smll=$(sub $GPU ds_ll_smoke $V100 --time=01:30:00 "MODEL=llama32_3b,SPECS=tf|$S/llama32_3b/tf|--tiny --max-reserved-gb 30;gen|$S/llama32_3b|--tiny --max-reserved-gb 30;analyze-gen|$S/llama32_3b|")
  echo "smokes: qwen25_15b $sm15 (gpu-a100-small, 1:30), qwen25_7b $sm7 (gpu-a100, 1:30), llama32_3b $smll (gpu-v100, 1:30)"
  # --- the judge smoke: original judge + both panel judges (24 prompts each) + analyze-gen on the 7B smoke's answers
  jsm=$(sub $PANEL ds_judge_smoke $A100J --time=02:00:00 --dependency=afterok:$sm7 "MODEL=qwen25_7b,MODEL_DIR=$S/qwen25_7b,JUDGES=gptoss gemma4,LIMIT=24,PRE_SPECS=judge|$S/qwen25_7b|,POST_SPECS=analyze-gen|$S/qwen25_7b|")
  echo "judge smoke $jsm (gpu-a100, 2:00) after $sm7"
  ans=""
  for M in qwen25_15b qwen25_7b llama32_3b; do
    case $M in
      qwen25_15b) tag=q15; R=$MIG; pn=gpu-a100-small; sm=$sm15 ;;
      qwen25_7b)  tag=q7;  R=$A100_7B; pn=gpu-a100; sm=$sm7 ;;
      llama32_3b) tag=ll;  R=$V100; pn=gpu-v100; sm=$smll ;;
    esac
    D=$RUN/$M
    if [ $M = qwen25_15b ]; then
      tf=""
      gen=$(sub $GPU ds_${tag}_gen $R --time=03:00:00 --dependency=afterok:$sm "MODEL=$M,SPECS=gen|$D|--sel-from $D/tf")
      pre="smoke $sm -> (tf: the local run) -> gen $gen (3:00)"
    else
      tf=$(sub $GPU ds_${tag}_tf $R --time=03:00:00 --dependency=afterok:$sm "MODEL=$M,SPECS=tf|$D/tf|")
      gen=$(sub $GPU ds_${tag}_gen $R --time=03:00:00 --dependency=afterok:$tf "MODEL=$M,SPECS=gen|$D|")
      pre="smoke $sm -> tf $tf (3:00) -> gen $gen (3:00)"
    fi
    j=$(sub $GPU ds_${tag}_judge $A100J --time=02:00:00 --dependency=afterok:$gen:$jsm "MODEL=$M,SPECS=judge|$D|")
    pa=$(sub $PANEL ds_${tag}_gptoss $A100J --time=03:00:00 --dependency=afterok:$gen:$jsm "MODEL_DIR=$D,JUDGES=gptoss")
    pb=$(sub $PANEL ds_${tag}_gemma4 $A100J --time=03:00:00 --dependency=afterok:$gen:$jsm,afterany:$pa "MODEL_DIR=$D,JUDGES=gemma4")
    an=$(sub $ANA ds_${tag}_analyze $CPU --time=01:00:00 --dependency=afterok:$j:$pa:$pb "MODEL=$M,SPECS=analyze-gen|$D|")
    ans="$ans:$an"
    echo "$M ($pn): $pre -> judge $j (gpu-a100, 2:00), gptoss $pa (gpu-a100, 3:00), gemma4 $pb (gpu-a100, 3:00, after gptoss) -> analyze $an (compute-p1, 1:00); judges also wait for $jsm"
  done
  su=$(sub $ANA ds_summary $CPU --time=00:45:00 --dependency=afterok$ans "MODEL=qwen25_7b,SPECS=summary|$RUN|")
  echo "summary $su (compute-p1, 0:45) after$ans -> $RUN/summary.md"
} | tee -a "$RUN/jobs.txt"
