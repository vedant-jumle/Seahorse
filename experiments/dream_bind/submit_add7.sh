#!/bin/bash
# dream_bind PREREG addendum 7, one fresh run root, chained with afterok. Per model:
#   smoke (--tiny with the real model, every new code path, fit check; built-in checks: R-full(-state) == prior turn,
#   R2-joint == T2, R2-seq == R2-joint, the single-fact re-encode identity, probe R-full(-state) == T-full, Paris)
#     -> stage jobs -> original judge (gpu-a100) + panels gpt-oss and Gemma-4 (gpu-a100, Ollama) -> analyze (compute-p1)
#   qwen25 (gpu-a100-small): directions + probe (A, B1-2) -> refusal gen (B3); pool gen (C); collide add7 (G);
#          stage 0 on pool facts (D, patch + knockout)
#   2b     (gpu-a100-small; Q2B_PART=v100 if the smoke's fit check fails): directions + probe; refusal; collide add7
#          (G + R2-state-only-J); stage 0 on pool facts (patch)
#   9b     (gpu-a100, fp32): probe only (F); no judges
#   llama  (gpu-v100, fp32): stage 0 primary + pool (D, E); probe (A); gen-set text (E); collide add7 (E, G); pool (C)
#          ONLY once meta-llama/Llama-3.2-3B-Instruct is in /scratch/$USER/hf_cache (gated; the user downloads it)
# Times: the earlier chains' measured rates (MIG: Qwen2.5 ~55 answers/min, 2B ~26-40/min; V100 ~11/min), scaled to
# the counts in PREREG addendum 7, + 30%; every MIG job <= 4 h.
# From the repo on DelftBlue (/scratch/$USER/Seahorse), after `git pull`:
#   bash experiments/dream_bind/submit_add7.sh [RUN_ROOT]                 # PARTS="qwen25 2b 9b" by default
#   PARTS=llama bash experiments/dream_bind/submit_add7.sh <same RUN_ROOT>  # later, adds the Llama chain
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_bind_add7_$(date +%Y%m%d_%H%M)}
PARTS=${PARTS:-qwen25 2b 9b}
mkdir -p "$RUN" "/scratch/$USER/logs"
SMALL="--partition=gpu-a100-small --cpus-per-task=2 --mem-per-cpu=6000M --gpus-per-task=1"
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
  j=$(sub $GPU a7_${tag}_judge $A100J --time=$JT --dependency=afterok:$dep "MODEL=$M,SPECS=$JS")
  pa=$(sub $PANEL a7_${tag}_gptoss $A100J --time=$P1 --dependency=afterok:$dep "RUN=$RUN,JUDGE=gptoss,MODELS=$MD")
  pb=$(sub $PANEL a7_${tag}_gemma4 $A100J --time=$P2 --dependency=afterok:$dep "RUN=$RUN,JUDGE=gemma4,MODELS=$MD")
  echo "$j $pa $pb"
}

{
  echo "dream_bind addendum 7 run $RUN, parts '$PARTS', submitted $(date -Is) at commit $(git rev-parse HEAD)"
  for P in $PARTS; do
    case $P in
    qwen25)
      M=qwen25_15b; S=$RUN/smoke; L="--max-reserved-gb 9.0"
      sm=$(sub $GPU a7_q25_smoke $SMALL --time=00:50:00 "MODEL=$M,SPECS=directions probe gen|$S/${M}_refusal|--tiny --gen-set refusal $L;gen|$S/${M}_pool|--tiny --gen-set pool $L;collide|$S/$M|--tiny --collide-set add7 $L;patch knockout|$S/${M}_s0pool|--tiny --items pool $L")
      dp=$(sub $GPU a7_q25_dirprobe $SMALL --time=00:50:00 --dependency=afterok:$sm "MODEL=$M,STAGES=directions probe,OUT=$RUN/$M")
      rf=$(sub $GPU a7_q25_refusal $SMALL --time=01:15:00 --dependency=afterok:$dp "MODEL=$M,STAGES=gen,OUT=$RUN/${M}_refusal,EXTRA=--gen-set refusal")
      po=$(sub $GPU a7_q25_pool $SMALL --time=01:45:00 --dependency=afterok:$sm "MODEL=$M,STAGES=gen,OUT=$RUN/${M}_pool,EXTRA=--gen-set pool")
      co=$(sub $GPU a7_q25_collide $SMALL --time=01:00:00 --dependency=afterok:$sm "MODEL=$M,STAGES=collide,OUT=$RUN/$M,EXTRA=--collide-set add7")
      s0=$(sub $GPU a7_q25_s0pool $SMALL --time=00:40:00 --dependency=afterok:$sm "MODEL=$M,STAGES=patch knockout,OUT=$RUN/${M}_s0pool,EXTRA=--items pool")
      read -r j pa pb <<< "$(judges $M q25 $rf:$po:$co 01:30:00 01:45:00 01:30:00 "${M}_refusal ${M}_pool ${M}_collide" \
        "judge|$RUN/${M}_refusal|;judge|$RUN/${M}_pool|;judge|$RUN/$M|")"
      an=$(sub $ANA a7_q25_analyze $CPU --time=00:30:00 --dependency=afterok:$j:$pa:$pb:$dp:$s0 "MODEL=$M,OUTS=$RUN/$M $RUN/${M}_refusal $RUN/${M}_pool")
      echo "$M (gpu-a100-small): smoke $sm -> dirs+probe $dp (0:50) -> refusal $rf (1:15); pool $po (1:45); collide $co (1:00); s0pool $s0 (0:40) -> judge $j, gptoss $pa, gemma4 $pb (gpu-a100) -> analyze $an"
      ;;
    2b)
      M=qwen35_2b; S=$RUN/smoke
      if [ "${Q2B_PART:-small}" = v100 ]; then R2=$V100; L="--max-reserved-gb 28"; pn=gpu-v100; else R2=$SMALL; L="--max-reserved-gb 9.4"; pn=gpu-a100-small; fi
      sm=$(sub $GPU a7_2b_smoke $R2 --time=00:50:00 "MODEL=$M,SPECS=directions probe gen|$S/${M}_refusal|--tiny --gen-set refusal $L;collide|$S/$M|--tiny --collide-set add7 $L;patch|$S/${M}_s0pool|--tiny --items pool $L")
      dp=$(sub $GPU a7_2b_dirprobe $R2 --time=01:00:00 --dependency=afterok:$sm "MODEL=$M,STAGES=directions probe,OUT=$RUN/$M")
      # the full refusal gen OOMs on the 10 GB slice although the --tiny smoke fits (job 1052696, 2026-10-22): full A100
      rf=$(sub $GPU a7_2b_refusal $A100J --time=03:00:00 --dependency=afterok:$dp "MODEL=$M,STAGES=gen,OUT=$RUN/${M}_refusal,EXTRA=--gen-set refusal")
      co=$(sub $GPU a7_2b_collide $R2 --time=01:45:00 --dependency=afterok:$sm "MODEL=$M,STAGES=collide,OUT=$RUN/$M,EXTRA=--collide-set add7")
      s0=$(sub $GPU a7_2b_s0pool $R2 --time=00:30:00 --dependency=afterok:$sm "MODEL=$M,STAGES=patch,OUT=$RUN/${M}_s0pool,EXTRA=--items pool")
      read -r j pa pb <<< "$(judges $M 2b $rf:$co 01:00:00 01:15:00 01:00:00 "${M}_refusal ${M}_collide" \
        "judge|$RUN/${M}_refusal|;judge|$RUN/$M|")"
      an=$(sub $ANA a7_2b_analyze $CPU --time=00:30:00 --dependency=afterok:$j:$pa:$pb:$dp:$s0 "MODEL=$M,OUTS=$RUN/$M $RUN/${M}_refusal")
      echo "$M ($pn): smoke $sm -> dirs+probe $dp (1:00) -> refusal $rf (1:30); collide $co (1:45); s0pool $s0 (0:30) -> judge $j, gptoss $pa, gemma4 $pb (gpu-a100) -> analyze $an"
      ;;
    9b)
      M=qwen35_9b
      sm=$(sub $GPU a7_9b_smoke $A100J --time=00:45:00 "MODEL=$M,STAGES=probe,OUT=$RUN/smoke/$M,EXTRA=--tiny --max-reserved-gb 75")
      pr=$(sub $GPU a7_9b_probe $A100J --time=01:30:00 --dependency=afterok:$sm "MODEL=$M,STAGES=probe,OUT=$RUN/$M")
      an=$(sub $ANA a7_9b_analyze $CPU --time=00:20:00 --dependency=afterok:$pr "MODEL=$M,OUTS=$RUN/$M")
      echo "$M (gpu-a100, fp32): smoke $sm -> probe $pr (1:30) -> analyze $an"
      ;;
    llama)
      M=llama32_3b; S=$RUN/smoke; L="--max-reserved-gb 28"
      ls /scratch/$USER/hf_cache/hub | grep -q "models--meta-llama--Llama-3.2-3B-Instruct" || { echo "Llama-3.2-3B-Instruct not in the HF cache: download it first"; exit 1; }
      sm=$(sub $GPU a7_ll_smoke $V100 --time=01:00:00 "MODEL=$M,SPECS=patch knockout probe gen collide|$S/$M|--tiny --gen-set text --collide-set add7 $L;gen|$S/${M}_pool|--tiny --gen-set pool $L;patch knockout|$S/${M}_s0pool|--tiny --items pool $L")
      s0=$(sub $GPU a7_ll_stage0 $V100 --time=01:00:00 --dependency=afterok:$sm "MODEL=$M,SPECS=patch knockout|$RUN/$M|;patch knockout|$RUN/${M}_s0pool|--items pool")
      pr=$(sub $GPU a7_ll_probe $V100 --time=00:45:00 --dependency=afterok:$sm "MODEL=$M,STAGES=probe,OUT=$RUN/$M")
      tx=$(sub $GPU a7_ll_text $V100 --time=03:45:00 --dependency=afterok:$sm "MODEL=$M,STAGES=gen,OUT=$RUN/$M,EXTRA=--gen-set text")
      co=$(sub $GPU a7_ll_collide $V100 --time=04:45:00 --dependency=afterok:$sm "MODEL=$M,STAGES=collide,OUT=$RUN/$M,EXTRA=--collide-set add7")
      po=$(sub $GPU a7_ll_pool $V100 --time=05:15:00 --dependency=afterok:$sm "MODEL=$M,STAGES=gen,OUT=$RUN/${M}_pool,EXTRA=--gen-set pool")
      read -r j pa pb <<< "$(judges $M ll $tx:$co:$po 02:00:00 02:15:00 02:00:00 "$M ${M}_collide ${M}_pool" \
        "judge|$RUN/$M|;judge|$RUN/${M}_pool|")"
      an=$(sub $ANA a7_ll_analyze $CPU --time=00:30:00 --dependency=afterok:$j:$pa:$pb:$s0:$pr "MODEL=$M,OUTS=$RUN/$M $RUN/${M}_pool")
      echo "$M (gpu-v100): smoke $sm -> stage0 $s0 (1:00); probe $pr (0:45); text $tx (3:45); collide $co (4:45); pool $po (5:15) -> judge $j, gptoss $pa, gemma4 $pb (gpu-a100) -> analyze $an"
      ;;
    *) echo "unknown part $P"; exit 1 ;;
    esac
  done
} | tee -a "$RUN/jobs.txt"
