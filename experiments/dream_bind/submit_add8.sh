#!/bin/bash
# dream_bind PREREG addendum 8, one fresh run root, chained with afterok. Per model:
#   smoke (--tiny with the real model, every new code path, fit check; built-in checks: R-full(-state) == prior turn,
#   R2-joint == T2, R2-seq == R2-joint, the single-fact re-encode identity, probe R-full(-state) == T-full, Paris, and new:
#   R2-sep-tagged with an empty dummy == R2-sep-shift, the tagged-layout asserts, the pool split == the file / disjoint /
#   covering, dirs8 picked on half A only, refusal8 items in half B, one-token dummies / lures / pool-pair values)
#     -> dp: directions (--dir-set add8: told + comply on pool half A) + probe (primary, --probe-set add8)
#            + probe (--items pool --probe-set add8)                  [parts 2a, 2b, 3, 6]
#        -> rf8: gen refusal8 (pool half B)  -> cp: gen comply (primary)   [parts 2b, 3]
#     -> c1: collide add8 (the addendum-5 pairs) -> c2: collide add8pool (the 34 pool pairs)   [parts 1, 2c]
#     -> original judge + gpt-oss panel (gpu-a100) -> Gemma-4 panel after gpt-oss -> analyze (compute-p1)
#   Chains are sequential within a model (at most 2 GPU jobs per model at a time, at most 8 jobs running overall).
#   qwen25 and 2b on gpu-a100 (full A100; the MIG slice runs one job at a time and the 2B refusal gen OOMed there in
#   addendum 7); llama on gpu-v100 (fp32).
# Times: addendum 7's measured durations (jobs.txt + sacct: Qwen2.5 MIG ~60 answers/min, the 2B on a full A100 ~90/min,
# Llama on a V100 ~100/min; the judge ~134 answers/min and the panels ~79/min at the slowest), scaled to the counts in
# PREREG addendum 8, + 30%, + ~30 min per job for the slow Python start-up on /scratch seen on 2026-10-25 (one
# interpreter per job via experiments/dream_bind/multi.py; the first smokes, at 0:45-0:50, timed out in 7 start-ups).
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_bind/submit_add8.sh [RUN_ROOT]          # PARTS="qwen25 2b llama" by default
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_bind_add8_$(date +%Y%m%d_%H%M)}
PARTS=${PARTS:-qwen25 2b llama}
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
GPU=slurm/dream_bind.slurm
PANEL=slurm/dream_bind_panel.slurm
ANA=slurm/dream_bind_cpu.slurm

judges() {  # model, tag, deps, judge time, gptoss time, gemma time, panel model dirs, judge SPECS -> "j pa pb"
  local M=$1 tag=$2 dep=$3 JT=$4 P1=$5 P2=$6 MD=$7 JS=$8
  local j pa pb
  j=$(sub $GPU a8_${tag}_judge $A100J --time=$JT --dependency=afterok:$dep "MODEL=$M,SPECS=$JS")
  pa=$(sub $PANEL a8_${tag}_gptoss $A100J --time=$P1 --dependency=afterok:$dep "RUN=$RUN,JUDGE=gptoss,MODELS=$MD")
  pb=$(sub $PANEL a8_${tag}_gemma4 $A100J --time=$P2 --dependency=afterok:$dep,afterany:$pa "RUN=$RUN,JUDGE=gemma4,MODELS=$MD")
  echo "$j $pa $pb"
}

chain() {  # model key, tag, resources, partition name, fit limit, smoke / dp / rf8 / cp / c1 / c2 times
  local M=$1 tag=$2 R=$3 pn=$4 L="--max-reserved-gb $5" TS=$6 TD=$7 TR=$8 TC=$9 T1=${10} T2=${11}
  local S=$RUN/smoke sm dp rf cp c1 c2 j pa pb an
  sm=$(sub $GPU a8_${tag}_smoke $R --time=$TS "MODEL=$M,SPECS=directions probe|$S/$M|--tiny --dir-set add8 --probe-set add8 $L;probe|$S/$M|--tiny --items pool --probe-set add8 $L;gen|$S/${M}_refusal8|--tiny --gen-set refusal8 $L;gen|$S/${M}_comply|--tiny --gen-set comply $L;collide|$S/$M|--tiny --collide-set add8 $L;collide|$S/${M}_pp|--tiny --collide-set add8pool $L")
  dp=$(sub $GPU a8_${tag}_dirprobe $R --time=$TD --dependency=afterok:$sm "MODEL=$M,SPECS=directions probe|$RUN/$M|--dir-set add8 --probe-set add8;probe|$RUN/$M|--items pool --probe-set add8")
  rf=$(sub $GPU a8_${tag}_refusal8 $R --time=$TR --dependency=afterok:$dp "MODEL=$M,STAGES=gen,OUT=$RUN/${M}_refusal8,EXTRA=--gen-set refusal8")
  cp=$(sub $GPU a8_${tag}_comply $R --time=$TC --dependency=afterok:$rf "MODEL=$M,STAGES=gen,OUT=$RUN/${M}_comply,EXTRA=--gen-set comply")
  c1=$(sub $GPU a8_${tag}_collide $R --time=$T1 --dependency=afterok:$sm "MODEL=$M,STAGES=collide,OUT=$RUN/$M,EXTRA=--collide-set add8")
  c2=$(sub $GPU a8_${tag}_poolpairs $R --time=$T2 --dependency=afterok:$c1 "MODEL=$M,STAGES=collide,OUT=$RUN/${M}_pp,EXTRA=--collide-set add8pool")
  read -r j pa pb <<< "$(judges $M $tag $cp:$c2 03:15:00 05:00:00 05:00:00 "${M}_refusal8 ${M}_comply ${M}_collide ${M}_pp_collide" \
    "judge|$RUN/${M}_refusal8|;judge|$RUN/${M}_comply|;judge|$RUN/$M|;judge|$RUN/${M}_pp|")"
  an=$(sub $ANA a8_${tag}_analyze $CPU --time=02:00:00 --dependency=afterok:$j:$pa:$pb:$dp "MODEL=$M,OUTS=$RUN/$M $RUN/${M}_refusal8 $RUN/${M}_comply $RUN/${M}_pp")
  echo "$M ($pn): smoke $sm ($TS) -> dirs8+probe+probepool $dp ($TD) -> refusal8 $rf ($TR) -> comply $cp ($TC); smoke -> collide add8 $c1 ($T1) -> add8pool $c2 ($T2) -> judge $j (3:15), gptoss $pa (5:00), gemma4 $pb (5:00, after gptoss) (gpu-a100) -> analyze $an (2:00)"
}

{
  echo "dream_bind addendum 8 run $RUN, parts '$PARTS', submitted $(date -Is) at commit $(git rev-parse HEAD)"
  for P in $PARTS; do
    case $P in
    qwen25) chain qwen25_15b q25 "$A100J" gpu-a100 70 01:30:00 01:30:00 01:45:00 01:00:00 02:15:00 04:30:00 ;;
    2b)     chain qwen35_2b 2b "$A100J" gpu-a100 70 01:30:00 01:45:00 02:00:00 01:00:00 02:00:00 04:00:00 ;;
    llama)
      ls /scratch/$USER/hf_cache/hub | grep -q "models--meta-llama--Llama-3.2-3B-Instruct" || { echo "Llama-3.2-3B-Instruct not in the HF cache"; exit 1; }
      chain llama32_3b ll "$V100" gpu-v100 28 01:30:00 01:30:00 01:45:00 01:00:00 02:00:00 04:00:00 ;;
    *) echo "unknown part $P"; exit 1 ;;
    esac
  done
} | tee -a "$RUN/jobs.txt"
