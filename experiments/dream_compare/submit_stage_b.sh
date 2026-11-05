#!/bin/bash
# dream_compare Stage B (PREREG.md "Stage B": memory paradigms end to end) on DelftBlue. Per model, one chain:
#   smoke   b-smoke: pytest (Stage 0 + Stage B tests) + one item per benchmark split through every paradigm,
#           the identity checks (P3 of every turn of a session == the text; ragged == per-row), peak memory
#   gen x4  b-run: LongMemEval oracle + s120 | LoCoMo | PersonaMem 32k | BEAM 100K    (afterok: smoke)
#   judge   b-tasks + b-judge-local (the original judge, Qwen3.5-9B)                  (afterany: gens; afterok: judge smoke)
#   gptoss  the Ollama panel judge gpt-oss on the same tasks                         (afterany: gens; afterok: judge smoke)
#   gemma4  the Ollama panel judge Gemma-4                                            (afterany: gptoss)
#   analyze b-analyze -> <model>/report_b.md, verdict_b.json (CPU)                     (afterany: judge, gptoss, gemma4)
# then one cross-model b-summary (CPU, afterany: every analyze). 9 jobs per model (10 for Gemma-4 and Granite, whose
# LongMemEval oracle and s120 run as two jobs) + 1 = 48 jobs.
# Gating (Stage 0 smokes, PREREG): qwen25_7b afterok 1214590, gemma4_26b_a4b afterok 1214592, granite42_30b afterok
# 1214593; every judge job afterok the judge smoke 1214594. Qwen2.5-1.5B and Llama-3.2-3B are ungated.
# No caches go to disk: every context's stored K,V live in the job's CPU RAM (sized per model below) and are freed
# after the context; the storage numbers come from the accounting (paradigms.storage).
#   bash experiments/dream_compare/submit_stage_b.sh [RUN_ROOT] [MODELS...]
set -euo pipefail

RUN=${1:-/scratch/$USER/seahorse_runs/dream_compare_b_$(date +%Y%m%d_%H%M)}
shift || true
MODELS=${*:-qwen25_15b llama32_3b qwen25_7b gemma4_26b_a4b granite42_30b}
G_Q7=${G_Q7:-1214590}; G_G4=${G_G4:-1214592}; G_GR=${G_GR:-1214593}; G_JS=${G_JS:-1214594}
for f in longmemeval/processed/oracle.json longmemeval/processed/s120.json locomo/processed/locomo10.json \
         personamem/processed/32k.json beam/processed/100K.json; do
  [ -f "data/external/$f" ] || { echo "data/external/$f missing: run the prep scripts first"; exit 1; }
done
mkdir -p "$RUN" "/scratch/$USER/logs"

GPU=slurm/dream_compare.slurm
CPU=slurm/dream_compare_cpu.slurm
PANEL=slurm/dream_compare_panel.slurm
TESTS_B="experiments/dream_compare/test_dream_compare.py experiments/dream_compare/test_stage_b.py"
A100J="--partition=gpu-a100 --cpus-per-task=4 --mem-per-cpu=7900M --gpus-per-task=1"
CPUR="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"

sub() {  # script, name, resources+time+deps..., then the env as the last argument
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,DC_RUNNER=stage_b,$env" "$script"
}

# per model: resources, extra b-run flags, the smoke's / gens' / judges' time limits (from the local 1.5B smoke and
# dream_select's measured rates, PREREG "Stage B: jobs"), the Stage 0 gate
res() {
  case $1 in
    qwen25_15b)     echo "--partition=gpu-v100 --cpus-per-task=8 --mem-per-cpu=5300M --gpus-per-task=1";;
    llama32_3b)     echo "--partition=gpu-v100 --cpus-per-task=12 --mem-per-cpu=5300M --gpus-per-task=1";;
    qwen25_7b)      echo "--partition=gpu-a100 --cpus-per-task=12 --mem-per-cpu=7900M --gpus-per-task=1";;
    gemma4_26b_a4b) echo "--partition=gpu-a100 --cpus-per-task=24 --mem-per-cpu=7900M --gpus-per-task=2";;
    granite42_30b)  echo "--partition=gpu-a100 --cpus-per-task=24 --mem-per-cpu=7900M --gpus-per-task=2";;
  esac
}
flags() {
  case $1 in
    qwen25_15b)     echo "--p1-kv-gb 12 --batch-kv-gb 6 --mask-gb 2";;
    llama32_3b)     echo "--p1-kv-gb 8 --batch-kv-gb 4 --mask-gb 1.5";;
    qwen25_7b)      echo "--p1-kv-gb 20 --batch-kv-gb 12 --mask-gb 4";;
    gemma4_26b_a4b) echo "--gb-per-gpu 66 --p1-kv-gb 16 --batch-kv-gb 10 --mask-gb 3";;
    granite42_30b)  echo "--gb-per-gpu 70 --p1-kv-gb 12 --batch-kv-gb 8 --mask-gb 3";;
  esac
}
gate() {  # the Stage 0 smoke a model waits for; none if it already completed (Slurm rejects afterok on finished jobs)
  local j
  case $1 in
    qwen25_7b) j=$G_Q7;; gemma4_26b_a4b) j=$G_G4;; granite42_30b) j=$G_GR;; *) j="";;
  esac
  [ -z "$j" ] && { echo ""; return; }
  local st
  st=$(sacct -j "$j" -X -n -o State | head -1 | tr -d ' ')
  case $st in
    COMPLETED) echo "";;
    PENDING|RUNNING|REQUEUED|CONFIGURING) echo "$j";;
    *) echo "FAILED:$st";;
  esac
}
judge_dep() {  # afterok on the judge smoke unless it already completed
  local st
  st=$(sacct -j "$G_JS" -X -n -o State | head -1 | tr -d ' ')
  [ "$st" = COMPLETED ] && echo "" || echo ",afterok:$G_JS"
}
GEN_T=23:59:00
MAXH=22.5

{
  echo "dream_compare Stage B $RUN, submitted $(date -Is) at commit $(git rev-parse HEAD)"
  ans="${EXTRA_ANS:-}"
  JS=$(judge_dep)
  for M in $MODELS; do
    R=$(res $M); F=$(flags $M); g=$(gate $M)
    case $g in FAILED:*) echo "$M: Stage 0 smoke ${g#FAILED:}: not submitted"; continue;; esac
    D=$RUN/$M; G=$D/gen; JD=$D/judge
    mkdir -p "$G" "$JD"
    dep=${g:+--dependency=afterok:$g}
    sm=$(sub $GPU b_${M}_smoke $R --time=04:00:00 $dep "MODEL=$M,TESTS=$TESTS_B,SPECS=b-smoke|$D/smoke|$F")
    case $M in   # the two 2-GPU models (slower per token): LongMemEval oracle and s120 in separate jobs
      gemma4_26b_a4b|granite42_30b)
        g1a=$(sub $GPU b_${M}_lmeo $R --time=$GEN_T --dependency=afterok:$sm "MODEL=$M,TESTS=experiments/dream_compare/test_stage_b.py,SPECS=b-run|$G|--bench longmemeval:oracle --max-hours $MAXH $F")
        g1b=$(sub $GPU b_${M}_lmes $R --time=$GEN_T --dependency=afterok:$sm "MODEL=$M,TESTS=experiments/dream_compare/test_stage_b.py,SPECS=b-run|$G|--bench longmemeval:s120 --max-hours $MAXH $F")
        g1="$g1a:$g1b";;
      *)
        g1=$(sub $GPU b_${M}_lme $R --time=$GEN_T --dependency=afterok:$sm "MODEL=$M,TESTS=experiments/dream_compare/test_stage_b.py,SPECS=b-run|$G|--bench longmemeval:oracle+longmemeval:s120 --max-hours $MAXH $F");;
    esac
    g2=$(sub $GPU b_${M}_loc $R --time=$GEN_T --dependency=afterok:$sm "MODEL=$M,TESTS=experiments/dream_compare/test_stage_b.py,SPECS=b-run|$G|--bench locomo:locomo10 --max-hours $MAXH $F")
    g3=$(sub $GPU b_${M}_pm $R --time=$GEN_T --dependency=afterok:$sm "MODEL=$M,TESTS=experiments/dream_compare/test_stage_b.py,SPECS=b-run|$G|--bench personamem:32k --max-hours $MAXH $F")
    g4=$(sub $GPU b_${M}_beam $R --time=$GEN_T --dependency=afterok:$sm "MODEL=$M,TESTS=experiments/dream_compare/test_stage_b.py,SPECS=b-run|$G|--bench beam:100K --max-hours $MAXH $F")
    gens="$g1:$g2:$g3:$g4"
    jo=$(sub $GPU b_${M}_judge $A100J --time=12:00:00 --dependency=afterany:$gens$JS "MODEL=$M,TESTS=experiments/dream_compare/test_stage_b.py,SPECS=b-tasks|$JD|--from $G;b-judge-local|$JD|")
    pa=$(sub $PANEL b_${M}_gptoss $A100J --time=23:00:00 --dependency=afterany:$jo$JS "JDIR=$JD,JUDGES=gptoss")
    pb=$(sub $PANEL b_${M}_gemma4 $A100J --time=16:00:00 --dependency=afterany:$pa "JDIR=$JD,JUDGES=gemma4")
    an=$(sub $CPU b_${M}_analyze $CPUR --time=02:00:00 --dependency=afterany:$jo:$pa:$pb "MODEL=$M,TESTS=experiments/dream_compare/test_stage_b.py,SPECS=b-analyze|$D|--from $G --judge-dir $JD")
    ans="$ans:$an"
    echo "$M: smoke $sm${g:+ (afterok $g)}; gens lme $g1 loc $g2 pm $g3 beam $g4 (afterok $sm); judge $jo (afterany gens$JS); gptoss $pa; gemma4 $pb; analyze $an"
  done
  su=$(sub $CPU b_summary $CPUR --time=01:00:00 --dependency=afterany$ans "MODEL=qwen25_7b,TESTS=experiments/dream_compare/test_stage_b.py,SPECS=b-summary|$RUN|")
  echo "summary $su (afterany$ans)"
} | tee -a "$RUN/jobs.txt"
