# dream_refs: experiment 2, dreaming as scaffold removal

Can a model write the references a memory needs (an opposite, alternatives, a general like/dislike direction),
so that no human writes them? And does storing the concept and the sign separately ("concept × sign") stop
dislikes from being recalled as likes?
- Spec: [docs/dreaming/02-scaffold.md](../../docs/dreaming/02-scaffold.md).
- [PREREG.md](PREREG.md): the claims S1–S5, the fixed generation grid and every rule, written before the run.

In one example: for "I can't stand jazz music." a person used to write the opposite ("I'm a huge jazz fan.") and
eight alternatives ("I can't stand rock music.", …). Here Qwen3.5-9B writes them from the sentence alone, offline,
and the 2B memory is built from those.

## Files

| File | What |
|---|---|
| `PREREG.md` | claims, grid, rules, checks, limitations (committed before any job) |
| `dream_prompts.yaml` | the dreamer's prompts (frame, opposite, alternatives, paraphrases, null values, topics) |
| `config_dream.yaml` | the dreamer: Qwen3.5-9B bf16, sampler, how many to keep, the valence frames and follow-up |
| `config_qwen35_2b.yaml` | the 2B run: everything core_v1's, plus the grid (doses, seed set 2, primary α) and the path of core_v1's reused state |
| `refs.py` | pure functions: parsing, validators, selection rules, nulls, valence pairs, sign readers, S1 / S3 rules, concept × sign |
| `dream.py` | the dreaming pass (9B, A100); `--mock` writes canned outputs for `--tiny` pipelines |
| `validate.py` | CPU: raw outputs → `validate/refs.json`, `validation.json`, and `spotcheck.md` for the user |
| `run.py` | 2B stages `prep`, `lens` (phase A), `gen`, `score`, `analyze` (phase B) |
| `analyze.py` | phase B tables and the S2 / S4 rules, under the original judge, each panel judge and the majority |
| `panel.py`, `panel.slurm` | phase B: gpt-oss and Gemma-4 re-judge every answer (core_v1's panel code) |
| `job.slurm`, `submit_A.sh`, `submit_B.sh` | the DelftBlue jobs and chains |
| `../../tests/test_dream_refs.py` | unit tests (validators, selection, valence, sign readers, S1 incl. indeterminate, S3, concept × sign, the grid, S2/S4 rules, no hand reference in any prompt) |

## Run layout

```
RUN/                         e.g. /scratch/vvjumle/seahorse_runs/dream_refs_<ts>
  jobs.txt                   job ids
  dream/                     raw.jsonl.gz, dream.json, config.json          (dream.py)
  validate/                  refs.json, validation.json                     (validate.py)
  spotcheck.md               per item: hand vs generated opposite / alternatives, flags   <- for the user
  report_A.md                S5, S1, S3, the valence direction, the logit lens            <- for the user
  qwen35_2b/prep/            state.pt (core_v1's calibration + every shift + v_l + signs), items.json, checks.json
  qwen35_2b/lens/            lens.jsonl.gz, s1.json, s3.json, valence.json, report_A.md
  qwen35_2b/gen|score|analysis/   phase B
  panel/                     phase B: gptoss.jsonl, gemma4.jsonl
```

## Phase A (dream → validate → prep + lens)

```
# local: commit + push; then on DelftBlue
ssh delftblue
cd /scratch/vvjumle/Seahorse && git pull
bash experiments/dream_refs/submit_A.sh            # -> RUN/jobs.txt
```

| Job | Partition | What |
|---|---|---|
| `dr_dream` | gpu-a100 | Qwen3.5-9B bf16 writes every item's frame, opposite, alternatives, paraphrases, null values, and the valence topics (~30–40 min) |
| `dr_validate` | compute-p1 | selection and validators → `validate/refs.json`, `validation.json`, `spotcheck.md` |
| `dr_prep_lens` | gpu-v100 | Qwen3.5-2B: reuses core_v1's state; computes the generated / null / paraphrase shifts, v_l, the signs, concept × sign; the hand-shift reproduction check; the logit lens, S1, S3 → `report_A.md` |

Then **stop**: the user spot-checks `spotcheck.md` and reads S1 / S3 in `report_A.md`.

Bring the results back:
```
rsync -a delftblue:/scratch/vvjumle/seahorse_runs/dream_refs_<ts> results/
# then read results/dream_refs_<ts>/spotcheck.md and results/dream_refs_<ts>/report_A.md
```

## Phase B (after the spot check)

```
ssh delftblue
cd /scratch/vvjumle/Seahorse && git pull
bash experiments/dream_refs/submit_B.sh /scratch/vvjumle/seahorse_runs/dream_refs_<ts>
```
Three gen jobs by item plus one for the unrelated prompts (gpu-v100, ≤ 4 h each; ~18k answers in all) → score
(gpu-a100: coherence + the original judge) → panel (gpt-oss and Gemma-4, gpu-a100, Ollama) → analyze
(compute-p1) → `qwen35_2b/analysis/report.md`, `claims.json`. analyze can be rerun locally:
`conda run -n torch python experiments/dream_refs/run.py --config experiments/dream_refs/config_qwen35_2b.yaml --stage analyze --out results/dream_refs_<ts>/qwen35_2b --panel results/dream_refs_<ts>/panel`.

## Local checks (CPU / no Qwen3.5)

```
PYTHONPATH=src:<scratch with pytest> conda run -n torch python -m pytest -q tests/test_dream_refs.py
# the whole pipeline on a tiny random model, with canned dreamer outputs:
PYTHONPATH=src conda run -n torch python experiments/dream_refs/dream.py --mock --out /tmp/dr/dream
PYTHONPATH=src conda run -n torch python experiments/dream_refs/validate.py --run /tmp/dr
PYTHONPATH=src conda run -n torch python experiments/dream_refs/run.py --config experiments/dream_refs/config_qwen35_2b.yaml \
    --stage prep lens gen score analyze --tiny --random-model qwen2 --out /tmp/dr/qwen35_2b
# the real dreamer's plumbing on a tiny random model (outputs are noise; validate then fails, as it should):
PYTHONPATH=src conda run -n torch python experiments/dream_refs/dream.py --tiny --random-model qwen2 --out /tmp/drr/dream
```
In `--tiny` mode the "core_v1 state" is made by core_v1's own prep on the tiny subset, and the judge labels are
deterministic stand-ins (marked FAKE), so every stage runs end to end.

## The PrefEval benchmark (`prefeval/`, PREREG addendum)

The same pipeline on 200 preferences sampled from PrefEval's explicit split (stratified by topic and polarity),
where no hand-written reference exists. The dreamer (prompts v3, unchanged) writes each preference's opposite and
alternatives, plus neutral follow-ups that carry the write and the keys (the test question is never used for
either). Conditions: nomem, ctx, gen:opposite, gen:centroid, cs:concept_x_sign at alpha 0.5 / 1 / 2. Measure:
PrefEval's preference-following accuracy (their violation and helpfulness prompts, `prefeval/upstream/`, CC BY-NC
4.0) under the original judge, gpt-oss, Gemma-4 and their majority.

| File | What |
|---|---|
| `prefeval/upstream/` | PrefEval's judge prompts, the data (1000 rows) and their licence, verbatim (`NOTICE.md`) |
| `prefeval/bench.py` | polarity label, stratified sampler (`python bench.py` writes `sample.json`), follow-up checks, judge parsing, PrefEval's accuracy, majority |
| `prefeval/sample.json` | the pre-registered sample |
| `prefeval/prompts.yaml` | the follow-up prompt (the only new prompt) |
| `prefeval/dream.py`, `validate.py` | dreaming and its checks -> `validate/refs.json`, `spotcheck.md` |
| `prefeval/run.py` | `prep` (writes, memories, the gate on each test question and on the unrelated prompts -> `report_A.md`), `gen`, `score`, `analyze` |
| `prefeval/panel.py`, `panel.slurm` | gpt-oss / Gemma-4 on the violation and helpfulness rubrics |
| `prefeval/job.slurm`, `submit_A.sh`, `submit_B.sh` | DelftBlue chains (phase A has a 4-item smoke first) |

```
bash experiments/dream_refs/prefeval/submit_A.sh <dream_refs v3 run root> [<its prep_lens job id>]
# spot check RUN/spotcheck.md, read RUN/report_A.md, then:
bash experiments/dream_refs/prefeval/submit_B.sh RUN
# local smoke: dream.py --mock --tiny --out /tmp/pe/dream; validate.py --run /tmp/pe;
#   run.py --stage prep gen score analyze --tiny --random-model qwen2 --out /tmp/pe/qwen35_2b
```
