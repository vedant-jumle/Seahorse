#!/bin/bash
# dream_compare Stage B repair, 2026-11-07 (PREREG.md "Stage B run record"; infrastructure only, no rule changed).
#   Llama-3.2-3B: PersonaMem (1219646) and BEAM (1219647) ran out of memory on the V100. Resumed on the V100 with the
#     memory fixes (lazy / freed P1 history, budgeted P4 selection batches and turn-store chunks, chunked MemArt
#     scores), same flags; finished contexts are kept (b-run skips them). Its pending panel / analyze jobs (1219649-51)
#     are cancelled and resubmitted after the new gens; the original judge resumes by prompt hash (only new prompts).
#   Granite-4.2-30B: BEAM (1217774) ran out of memory on an A100 in a dream batch; resumed (nothing was written) with
#     smaller batch budgets. Every split gets afterany continuation jobs (each exits before loading the model when its
#     split is complete); the pending judge 1217775 is made to wait for all of them (scontrol).
# Run from the repo on DelftBlue after `git pull`:  bash experiments/dream_compare/resubmit_b_20261107.sh
set -euo pipefail
RUN=/scratch/$USER/seahorse_runs/dream_compare_b_20261105_1915
GPU=slurm/dream_compare.slurm; CPU=slurm/dream_compare_cpu.slurm; PANEL=slurm/dream_compare_panel.slurm
TB=experiments/dream_compare/test_stage_b.py
V100="--partition=gpu-v100 --cpus-per-task=12 --mem-per-cpu=5300M --gpus-per-task=1"
A2="--partition=gpu-a100 --cpus-per-task=24 --mem-per-cpu=7900M --gpus-per-task=2"
A100J="--partition=gpu-a100 --cpus-per-task=4 --mem-per-cpu=7900M --gpus-per-task=1"
CPUR="--partition=compute-p1 --cpus-per-task=2 --mem-per-cpu=3G"
NICE=${NICE:-5000000}     # behind the Gemma diagnostic / smoke among this user's jobs (reset afterwards)
sub() {
  local script=$1 name=$2; shift 2
  local env=${!#}
  local args=("${@:1:$#-1}")
  sbatch --parsable --job-name="$name" "${args[@]}" --export="ALL,DC_RUNNER=stage_b,$env" "$script"
}
{
  echo "dream_compare Stage B repair $RUN, submitted $(date -Is) at commit $(git rev-parse HEAD)"
  # ---- Llama-3.2-3B
  M=llama32_3b; D=$RUN/$M; G=$D/gen; JD=$D/judge
  F="--p1-kv-gb 8 --batch-kv-gb 4 --mask-gb 1.5"
  scancel 1219649 1219650 1219651 || true
  pm=$(sub $GPU b_${M}_pm2 $V100 --time=23:59:00 --nice=$NICE "MODEL=$M,TESTS=$TB,SPECS=b-run|$G|--bench personamem:32k --max-hours 22.5 $F")
  bm=$(sub $GPU b_${M}_beam2 $V100 --time=23:59:00 --nice=$NICE "MODEL=$M,TESTS=$TB,SPECS=b-run|$G|--bench beam:100K --max-hours 22.5 $F")
  jo=$(sub $GPU b_${M}_judge2 $A100J --time=12:00:00 --dependency=afterany:$pm:$bm "MODEL=$M,TESTS=$TB,SPECS=b-tasks|$JD|--from $G;b-judge-local|$JD|")
  pa=$(sub $PANEL b_${M}_gptoss2 $A100J --time=23:00:00 --dependency=afterany:$jo "JDIR=$JD,JUDGES=gptoss")
  pb=$(sub $PANEL b_${M}_gemma4_2 $A100J --time=16:00:00 --dependency=afterany:$pa "JDIR=$JD,JUDGES=gemma4")
  an=$(sub $CPU b_${M}_analyze2 $CPUR --time=02:00:00 --dependency=afterany:$jo:$pa:$pb "MODEL=$M,TESTS=$TB,SPECS=b-analyze|$D|--from $G --judge-dir $JD")
  echo "$M: cancelled 1219649 1219650 1219651; gens pm $pm beam $bm (V100, resume); judge $jo (afterany gens; resumes by prompt hash); gptoss $pa; gemma4 $pb; analyze $an"
  # ---- Granite-4.2-30B
  M=granite42_30b; D=$RUN/$M; G=$D/gen
  F="--gb-per-gpu 70 --p1-kv-gb 12 --batch-kv-gb 8 --mask-gb 3"
  FB="--gb-per-gpu 70 --p1-kv-gb 12 --batch-kv-gb 6 --mask-gb 2 --attn-gb 3 --mlp-gb 3"   # BEAM: nothing written yet
  chain() {  # tag bench flags first-dependency n -> the job ids
    local tag=$1 bench=$2 fl=$3 prev=$4 n=$5 ids="" dep
    for c in $(seq 1 "$n"); do
      dep=${prev:+--dependency=afterany:$prev}
      prev=$(sub $GPU b_${M}_${tag}_c$c $A2 --time=23:59:00 --nice=$NICE $dep "MODEL=$M,TESTS=$TB,SPECS=b-run|$G|--bench $bench --max-hours 22.5 $fl")
      ids="$ids:$prev"
    done
    echo "${ids#:}"
  }
  c_lmeo=$(chain lmeo longmemeval:oracle "$F" 1217770 2)
  c_lmes=$(chain lmes longmemeval:s120 "$F" 1217771 1)
  c_loc=$(chain loc locomo:locomo10 "$F" 1217772 1)
  c_pm=$(chain pm personamem:32k "$F" 1217773 3)
  c_beam=$(chain beam beam:100K "$FB" "" 3)
  all="1217770:1217771:1217772:1217773:$c_lmeo:$c_lmes:$c_loc:$c_pm:$c_beam"
  scontrol update jobid=1217775 dependency=afterany:$all
  echo "$M: continuations lmeo $c_lmeo, lmes $c_lmes, loc $c_loc, pm $c_pm; BEAM resubmitted $c_beam (budgets $FB); judge 1217775 dependency -> afterany:$all (1217776-78 follow it unchanged)"
} | tee -a "$RUN/jobs.txt"
