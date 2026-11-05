#!/bin/bash
# dream_compare Stage A (selectors; PREREG.md "Stage A") on DelftBlue: one fresh run root, per-model chains with afterok.
#   smoke (pytest: Stage 0's + Stage A's tests; a-tf --tiny (4 items: 2 reused + 2 fresh, k4) -> a-gen --tiny ->
#          a-judge-tasks -> a-analyze; built-in asserts: R-full == T-full, per-head keep-all == R-full, the same picks in
#          every head == the splice, k x E entries per slice, the reuse identity, the fit check)
#   -> tf (133 items; Gemma-4 / Granite: 2 parts by item, resumable)
#   -> gen (27k answers: 41 conditions x 5; Gemma-4 / Granite: 3 parts)
#   -> the original judge (Qwen3.5-9B; a-judge-tasks + judge-local) and gpt-oss (Ollama) -> Gemma-4 judge (after
#      gpt-oss) -> a-analyze (CPU); finally a-summary over the models with results (afterany).
# Gates: qwen25_7b / gemma4_26b_a4b / granite42_30b smokes wait for their Stage 0 smokes (1214590 / 1214592 / 1214593,
# afterok); every judge job waits for the Stage 0 judge smoke 1214594 (afterok). Qwen2.5-1.5B and Llama-3.2-3B passed
# Stage 0 (1215658, 1214591). A model whose gate job failed is skipped (SKIP="gemma4_26b_a4b granite42_30b" etc.).
# Times from dream_select addendum 1 (sacct: tf 28-30 min / 64 items; gen 45-77 min / 5,320 answers; judge 18-30 min,
# gpt-oss 56-97 min, Gemma-4 89-95 min per 5,320), scaled: 133 items with ~2.5x the passes per item; 27k answers
# (5.1x); Gemma-4 (MoE, fp32, 2 GPUs) and Granite (30B dense fp32, 64 layers) assumed ~4x slower than the 7B; + margin.
# Qwen2.5-1.5B runs on gpu-v100: gpu-a100-small (the MIG slice) caps jobs at 4 h, below the tf / gen limits.
# Jobs: 7 per small model, 10 per 2-GPU model, 1 summary = 42 (Stage A's share of the ~100-job cap: <= 45).
# From the repo on DelftBlue (/scratch/$USER/Seahorse) after `git pull`:
#   bash experiments/dream_compare/submit_stage_a.sh [RUN_ROOT]
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_compare_a_$(date +%Y%m%d_%H%M)}
REUSE=${REUSE:-/scratch/$USER/seahorse_runs/dream_select_cluster_20261104_0129}
SKIP=${SKIP:-}
GATE_Q7=1214590; GATE_G4=1214592; GATE_GR=1214593; GATE_JUDGE=1214594
S=$RUN/smoke
[ -f data/external/longmemeval/processed/oracle.json ] || { echo "LongMemEval oracle not prepared"; exit 1; }
for m in qwen25_15b qwen25_7b llama32_3b; do
  [ -f "$REUSE/$m/tf/meta.json" ] || { echo "dream_select addendum 1 meta missing: $REUSE/$m/tf/meta.json"; exit 1; }
done
need() { ls /scratch/$USER/hf_cache/hub | grep -q "^$1\$"; }
for m in models--Qwen--Qwen2.5-1.5B-Instruct models--Qwen--Qwen2.5-7B-Instruct models--meta-llama--Llama-3.2-3B-Instruct \
         models--google--gemma-4-26B-A4B-it models--ibm-granite--granite-4.2-30b models--Qwen--Qwen3.5-9B; do
  need $m || { echo "$m not in the HF cache"; exit 1; }
done
mkdir -p "$S" "/scratch/$USER/logs"

MIG="--partition=gpu-a100-small --cpus-per-task=2 --mem-per-cpu=7900M --gpus-per-task=1"
A100="--partition=gpu-a100 --cpus-per-task=6 --mem-per-cpu=7900M --gpus-per-task=1"
A100x2="--partition=gpu-a100 --cpus-per-task=8 --mem-per-cpu=7900M --gpus-per-task=2"
V100="--partition=gpu-v100 --cpus-per-task=2 --mem-per-cpu=5300M --gpus-per-task=1"
A100J="--partition=gpu-a100 --cpus-per-task=4 --mem-per-cpu=7900M --gpus-per-task=1"
CPU="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=4G"
GPU=slurm/dream_compare.slurm
PANEL=slurm/dream_compare_panel.slurm
ANA=slurm/dream_compare_cpu.slurm
T_A=experiments/dream_compare/test_stage_a.py
T_SMOKE="experiments/dream_compare/test_dream_compare.py $T_A"

sub() {  # script, name, resources+time+deps..., then the env as the last argument
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,$env" "$script"
}
skip() { [[ " $SKIP " == *" $1 "* ]]; }

{
  echo "dream_compare Stage A run $RUN, submitted $(date -Is) at commit $(git rev-parse HEAD); reuse $REUSE; skip '$SKIP'"
  ans=""
  for M in qwen25_15b llama32_3b qwen25_7b gemma4_26b_a4b granite42_30b; do
    if skip $M; then echo "$M: skipped (SKIP)"; continue; fi
    D=$RUN/$M
    case $M in
      qwen25_15b)     tag=q15; R=$V100;   gate="";                     fit="--max-reserved-gb 30";  ex=""; ru="--reuse $REUSE"
                      t_sm=01:30:00; t_tf=06:00:00; t_gen=14:00:00; ntf=1; ngen=1 ;;
      llama32_3b)     tag=ll;  R=$V100;   gate="";                     fit="--max-reserved-gb 30";  ex=""; ru="--reuse $REUSE"
                      t_sm=01:30:00; t_tf=06:00:00; t_gen=14:00:00; ntf=1; ngen=1 ;;
      qwen25_7b)      tag=q7;  R=$A100;   gate="--dependency=afterok:$GATE_Q7"; fit="--max-reserved-gb 75"; ex=""; ru="--reuse $REUSE"
                      t_sm=02:00:00; t_tf=08:00:00; t_gen=16:00:00; ntf=1; ngen=1 ;;
      gemma4_26b_a4b) tag=g4;  R=$A100x2; gate="--dependency=afterok:$GATE_G4"; fit=""; ex="--gb-per-gpu 66"; ru=""
                      t_sm=04:00:00; t_tf=20:00:00; t_gen=22:00:00; ntf=2; ngen=3 ;;
      granite42_30b)  tag=gr;  R=$A100x2; gate="--dependency=afterok:$GATE_GR"; fit=""; ex="--gb-per-gpu 70"; ru=""
                      t_sm=04:00:00; t_tf=20:00:00; t_gen=22:00:00; ntf=2; ngen=3 ;;
    esac
    sm=$(sub $GPU dca_${tag}_smoke $R --time=$t_sm $gate "MODEL=$M,TESTS=$T_SMOKE,SPECS=a-tf|$S/$M|--tiny $ru $fit $ex;a-gen|$S/$M|--tiny $fit $ex;a-judge-tasks|$S/$M|;a-analyze|$S/$M|")
    tfs=""
    for p in $(seq 0 $((ntf - 1))); do
      j=$(sub $GPU dca_${tag}_tf$p $R --time=$t_tf --dependency=afterok:$sm "MODEL=$M,TESTS=$T_A,SPECS=a-tf|$D|--part $p --nparts $ntf $ru $fit $ex")
      tfs="$tfs:$j"
    done
    gens=""
    for p in $(seq 0 $((ngen - 1))); do
      j=$(sub $GPU dca_${tag}_gen$p $R --time=$t_gen --dependency=afterok$tfs "MODEL=$M,TESTS=$T_A,SPECS=a-gen|$D|--part $p --nparts $ngen $fit $ex")
      gens="$gens:$j"
    done
    jo=$(sub $GPU dca_${tag}_judge $A100J --time=08:00:00 --dependency=afterok$gens:$GATE_JUDGE "MODEL=$M,TESTS=$T_A,SPECS=a-judge-tasks|$D|;judge-local|$D/judge|")
    pa=$(sub $PANEL dca_${tag}_gptoss $A100J --time=16:00:00 --dependency=afterok$gens:$GATE_JUDGE "MODEL=$M,TESTS=$T_A,JDIR=$D/judge,JUDGES=gptoss,PRE_SPECS=a-judge-tasks|$D|")
    pb=$(sub $PANEL dca_${tag}_gemma4 $A100J --time=16:00:00 --dependency=afterok$gens:$GATE_JUDGE,afterany:$pa "MODEL=$M,TESTS=$T_A,JDIR=$D/judge,JUDGES=gemma4,PRE_SPECS=a-judge-tasks|$D|")
    an=$(sub $ANA dca_${tag}_analyze $CPU --time=01:00:00 --dependency=afterok:$jo:$pa:$pb "MODEL=$M,TESTS=$T_A,SPECS=a-analyze|$D|")
    ans="$ans:$an"
    echo "$M: smoke $sm ($t_sm${gate:+, $gate}) -> tf${tfs//:/ } ($ntf x $t_tf) -> gen${gens//:/ } ($ngen x $t_gen) -> judge $jo (8:00) + gptoss $pa (16:00) -> gemma4 $pb (16:00, after gptoss); judges also wait for $GATE_JUDGE -> analyze $an"
  done
  su=$(sub $ANA dca_summary $CPU --time=00:30:00 --dependency=afterany$ans "MODEL=qwen25_7b,TESTS=$T_A,SPECS=a-summary|$RUN|")
  echo "summary $su (afterany$ans) -> $RUN/summary_a.md"
} | tee -a "$RUN/jobs.txt"
