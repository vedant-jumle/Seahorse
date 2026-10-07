# core_v1: one consolidated, reproducible run of the core claims, on two model scales

This is the run behind the short write-up. Every number comes from one design, one config per model, fixed
seeds, and one analysis script that works from the saved files.
- [PREREG.md](PREREG.md): the claims C1–C7 and their decision rules, written before the run.
- [items.yaml](items.yaml): every item.
- [judge_prompts.yaml](judge_prompts.yaml): the judge rubrics.

## Files

| File | What |
|---|---|
| `items.yaml`, `items.py` | 15 leanings, 8 dislikes, 6 one-of-many, 12 facts, 50 unrelated prompts; loader and validator (bench_v1's validator per item, plus core_v1's rules) |
| `config_qwen35_2b.yaml`, `config_qwen35_9b.yaml` | one config per model: model, precision, layers, every seed, doses, sampling, batch sizes |
| `core.py` | pure helpers: the layer-mapping rule, conditions, dose and control fields, the batched two-pass hook, batched generation with fixed per-row random streams, the judge parser |
| `run.py` | GPU stages `smoke`, `prep`, `gen`, `score` (below) |
| `analyze.py` | every table, CSV and claim decision from the saved files (no GPU) |
| `judge_prompts.yaml` | the judge rubrics (Qwen3.5-9B, bf16, thinking off) |
| `submit.sh`, `../../slurm/core_v1.slurm` | the DelftBlue chains |
| `../../tests/test_core_v1.py` | unit tests (CPU): layer rule, conditions, controls, batched == unbatched generation, bootstrap, judge parsing, items |

## Precision and hardware

| Model | Precision | Hardware | Layers (block outputs) | Types |
|---|---|---|---|---|
| Qwen3.5-2B (24 blocks) | fp32 | gpu-v100 (judge-scoring job on gpu-a100) | 20, 21, 23 | linear, linear, full |
| Qwen3.5-9B (32 blocks) | fp32 | gpu-a100 (A100 80GB) | 27, 28, 31 | full, linear, full |

- **The 9B's layers** come from the 2B's by relative depth, `round((l+1)·32/24) − 1`. The `l/(n−1)`
  convention gives the same set.
  - The set keeps the 2B set's defining feature: it includes the last full-attention block.
  - It does not keep the block types: 27 is full attention.
  - A type-matched alternative, {28, 29, 31}, was not used. The pre-registered rule is "same relative
    depth".
- **Both models run in fp32**, because a full A100 80GB holds the 9B in fp32. So no fp16 or bf16 path is
  used for the experiment models, and the planned 2B fp16 precision control is unnecessary.
- **The judge** is Qwen3.5-9B in bf16, on an A100.
- **Thinking** is off everywhere. The 9B chat template defaults to thinking ON, so run.py forces
  `enable_thinking=False` in every chat-template call and asserts the template tail.

## Stages (one job each; `--out` is the model's run dir)

```
python experiments/core_v1/run.py --config experiments/core_v1/config_qwen35_2b.yaml --stage smoke --out RUN/qwen35_2b
python experiments/core_v1/run.py --config ... --stage prep  --out RUN/qwen35_2b   # -> RUN/qwen35_2b/prep
python experiments/core_v1/run.py --config ... --stage gen   --out RUN/qwen35_2b   # -> .../gen  (reads prep/state.pt)
python experiments/core_v1/run.py --config ... --stage score --out RUN/qwen35_2b   # -> .../score (coherence + judge)
python experiments/core_v1/analyze.py --run RUN/qwen35_2b                          # -> .../analysis
python experiments/core_v1/analyze.py --combine RUN/qwen35_2b RUN/qwen35_9b --out RUN/analysis   # C7 + audit sample
```

### smoke
The whole pipeline on one item per group, with tiny settings and the real model. It also checks that the
no-memory greedy answers are finite and coherent.
- `--judge-smoke` (9B job) also loads and runs the judge.
- Output: `smoke/smoke_report.json`.
- The job fails (and its chain stops) if the answers are incoherent or the judge doesn't parse.

### prep
- **Calibration:** mu and PCA-256 whitening at the read layers, the generic match thresholds per item and
  layer, and the median residual norm of every block.
- **Writes:** for every item and follow-up, the plain, opposite, centroid and placebo shifts at every
  block, plus the write keys.
- **Memories:** delta rule.
- **Logit lens** of every stored shift at every block (`lens.jsonl.gz`).
- **The fact analytic:** Δlog P(target) and specificity at the measure prefix, per condition
  (`facts_analytic.jsonl.gz`).
- **Checks** (`checks.json`):
  - exact recall
  - the write equals `diag_keys.unit_writes`
  - cached KV generation equals the full-sequence two-pass reader
  - two-pass equals one-pass for one layer
  - fields equal xlayer_v1's
  - the rand norm
  - batched vs unbatched generation, and equality with think_v1's generator
  - 2B: the fact shifts and thresholds reproduce xlayer_v1 Stage A
- **Saved:** `state.pt`, `items.json`, `validation.json`.

### gen
Per item, written to `gen/items/<id>.jsonl.gz`; a restarted job skips finished items.
- **Main prompts:** every condition of a prompt goes in ONE batch with the same random streams; ctx goes
  in its own batch.
- **Yes/no probes:** greedy, with first-token margins.

Per unrelated prompt (`gen/unrel/uNN.jsonl.gz`):
- Rows: nomem; `gate_on` for every item; the gated conditions of the items whose gate opens somewhere.
- Answers of closed gates are exactly nomem's and are not regenerated.
- KL(nomem ‖ memory) along the nomem greedy answer.
- The gate open rates.

### score
- `coherence.jsonl.gz`: the mean token NLL of every answer under the no-memory model (same model and
  precision).
- `judge.jsonl.gz`: the raw judge output and the parsed labels for every main answer.

### analyze
Writes `report.md`, `claims.json`, and the CSVs:
- `pref`, `facts`, `facts_tf`, `yn`, `mech`, `mech_matched`
- `selectivity`, `false_fire`
- `lens_facts`, `lens_prefs`
- `seed2`, `per_item`, `flags`

The combine step writes `C7.md`, `claims_all.json`, `audit_sample.jsonl` (about 200 answers with empty
human-label fields) and `audit_key.jsonl` (the automatic labels, kept separate so the audit is blind).

## Reproduce on DelftBlue

```
# local
git push
# login node (no heavy compute there): models into the offline cache once
ssh delftblue
cd /scratch/$USER/Seahorse && git pull
HF_HOME=/scratch/$USER/hf_cache python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-2B'); s('Qwen/Qwen3.5-9B')"
bash experiments/core_v1/submit.sh            # prints and saves the job ids to RUN/jobs.txt
# afterwards, locally
rsync -a delftblue:/scratch/vvjumle/seahorse_runs/core_v1_<ts> results/
conda run -n torch python experiments/core_v1/analyze.py --combine results/core_v1_<ts>/qwen35_2b results/core_v1_<ts>/qwen35_9b --out results/core_v1_<ts>/analysis
```

Each run's `config.json` records the commit, torch and transformers versions, the GPU, and the items-file
hash.

**Local checks (CPU):**
- `pytest tests`.
- A dry run of the whole pipeline on a tiny random model:
  `python experiments/core_v1/run.py --config experiments/core_v1/config_qwen35_2b.yaml --stage smoke --random-model qwen2 --out /tmp/cv1`.
- Item validation with the tokenizer:
  `python experiments/core_v1/items.py --model Qwen/Qwen3.5-2B`.
