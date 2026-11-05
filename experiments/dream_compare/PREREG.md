# dream_compare: pre-registration

The full comparison run: does a stored-cache memory beat text memory on anything that matters (paradigms), and which
way of choosing stored positions works best at matched storage (selectors)? Plan: three stages, each with its own
section here, committed before that stage's jobs. Each stage stops after its smoke tests pass.

- **Stage 0: infrastructure** (this section; no scientific rules)
- **Stage A: selectors** (below; committed 2026-11-05 before its jobs)
- **Stage B: paradigms** (placeholder below; filled in by the Stage B executor before its jobs)

Models: Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Llama-3.2-3B-Instruct, google/gemma-4-26B-A4B-it (and, optional,
ibm-granite/granite-4.2-30b), all in fp32. Benchmarks: LongMemEval (oracle + a 120-item S sample), LoCoMo
(locomo10), PersonaMem v1 (32k), BEAM (100K).

## Stage 0: infrastructure

Built in `experiments/dream_compare/`. Existing experiments are imported, never edited: dream_select's and dream_bind's
code, configs and reports are byte-identical (checked, below).

### What was built

1. **`models.py`: the model adapter.** One spec per model replaces dream_select's hard-coded `MODELS` and
   `N_BLOCKS = 28`. Each spec gives: the layers and their types; the KV heads and head_dim of every layer; the rotary
   embedding of every layer type (base, type, partial rotation); the sliding window; which layers have K = V; the final
   logit soft-cap; chat-template quirks (Llama's pinned date; thinking off for Gemma 4 and Granite); dtype; GPUs; the
   maximum context.
   - The three dream_select models load through dream_select's own model blocks and dream_bind's loader, so their
     settings are exactly dream_select's.
   - Gemma 4 loads its text decoder only (the checkpoint is multimodal): `from_pretrained` with the text config and a
     key mapping, spread over 2 A100s with `device_map="auto"`. The loader asserts that nothing is missing and that
     only vision-tower keys are skipped.
   - Every spec is checked against the model's real `config.json` in the tests.

   Gemma 4 facts, verified against its `config.json` (2026-11-05):

   | | sliding layers (25) | global layers (5: 5, 11, 17, 23, 29) |
   |---|---|---|
   | KV heads x head_dim | 8 x 256 | 2 x 512, K = V |
   | RoPE | base 1e4, full rotation | base 1e6, "proportional", 25% of each head rotated |
   | window | 1024 | none |

   Query heads: 16. Final logit soft-cap: 30. MoE: 128 experts, 8 active. One stored position costs 25 x 8 + 5 x 2 =
   210 K,V entries (Qwen2.5-1.5B: 56; 7B: 112; Llama-3.2-3B: 224).

2. **Cache utilities**, wrappers that dispatch on the adapter:
   - `kv_list` / `make_cache` for per-layer shapes.
   - `Prefix`: the stored entries, plus the RoPE position of each entry.
   - `rope_shift(C, K, delta, layer)`: uses that layer's own rotary embedding, so partial rotation is handled. On
     Gemma's global layers, the 75% of dims with zero frequency are left untouched.
   - `compose_prefix` over mixed layer types; `harvest_kv`, `forward`, `gen_batch`, `answer_score`, `session_pieces`.
   - `C.lm_head` is wrapped with Gemma's soft-cap. dream_bind's and dream_select's code, which calls `C.lm_head`
     directly, therefore sees the model's real logits.

   **Sliding windows.** transformers 5's default cache for Gemma *evicts* sliding-layer entries beyond the window (it
   keeps 1023). Its sliding mask is also computed from cache *slots*, not RoPE positions. With a sparse restored
   prefix, the slot distance is smaller than the position distance, so entries that the model would never see become
   visible. The adapter avoids both:
   - every restored or harvested memory lives in a plain `DynamicCache` (nothing is evicted);
   - on sliding models, every pass gets explicit masks built from positions: a restored entry is visible to a query
     iff `0 <= q_pos - kv_pos < window`.

   Tiny Gemma-4 test (window 8), sparse prefix whose distances exceed the window: position masks match the reference
   (a full pass with the dropped slots masked) to 3e-6, while transformers' slot masks are off by 3.7.

3. **`headmask.py`: per-head masking.** This follows the selector-baseline design. A
   `register_forward_pre_hook(with_kwargs=True)` on every attention module rewrites `attention_mask`, per (layer, KV
   head), over the stored slots. The mask is expanded to the query heads of each GQA group. It handles float and bool
   masks, None (SDPA's implicit causal mask) and Gemma's per-layer-type masks, and works in transformers 4.46 and 5.18,
   with eager and SDPA attention.

4. **Benchmark adapters** (`benchmarks/`). A common schema (`schema.py`): contexts made of sessions made of turns
   (role, text, date, turn id), plus the question, answers, evidence (turn ids; session ids for PersonaMem), category,
   abstention flag, options and rubric. `validate()` checks the invariants.
   - Prep scripts (`prep_*.py`) download pinned revisions into `data/external/<bench>/` (git-ignored; parsed, never
     executed), write the processed split to `data/external/<bench>/processed/`, and commit a manifest (counts,
     sha256, item ids) plus a README with the licence and citation to `data/<bench>/`.

   | split | items | per category |
   |---|---|---|
   | LongMemEval oracle | 500 (30 abstention) | multi-session 133, temporal-reasoning 133, knowledge-update 78, single-session-user 70, single-session-assistant 56, single-session-preference 30 |
   | LongMemEval s120 | 120 (10 abstention) | 20 per type; `random.Random(7)` over the sorted ids of each type |
   | LoCoMo locomo10 | 1,986 (446 adversarial) | single-hop 841, temporal 321, multi-hop 282, open-domain 96, adversarial 446 |
   | PersonaMem 32k | 589 (4 options each) | track_full_preference_evolution 139, recall_user_shared_facts 129, recalling_the_reasons_behind_previous_updates 99, suggest_new_ideas 93, generalizing_to_new_scenarios 57, provide_preference_aligned_recommendations 55, recalling_facts_mentioned_by_the_user 17 |
   | BEAM 100K | 320 (summarisation, instruction-following dropped) | 40 each: abstention, contradiction_resolution, event_ordering, information_extraction, knowledge_update, multi_session_reasoning, preference_following, temporal_reasoning |

   **LoCoMo's category codes were verified, and they do not follow the paper's numbered list:** 1 multi-hop,
   2 temporal, 3 open-domain, 4 single-hop, 5 adversarial. Sources: the official scorer, the QA prompt code, the
   questions themselves, and repo issues #6 / #29.

   Two LongMemEval facts for Stage A: 182 oracle items have exactly one labelled evidence turn. Counting user turns
   only (dream_select's rule) gives 133, of which the 64 single-session-user items are dream_select's set.

5. **`judges.py`.** Each benchmark's official prompt, reproduced verbatim (extracted with `ast` from the source file,
   never executed) and pinned by sha256 in the tests:
   - LongMemEval: the five templates of `evaluate_qa.py`. The file is identical at dream_select's pinned commit and
     at the current one; the shared template equals dream_select's pinned prompt.
   - LoCoMo: Mem0's J prompt (`evaluation/metrics/llm_judge.py`, Chhikara et al. 2025), plus the official token F1
     (vendored nltk Porter stemmer, since the cluster env has no nltk; checked equal to nltk).
   - BEAM: the unified rubric judge, averaged over rubric items.
   - PersonaMem: no judge. The option with the highest log-probability wins.

   Panel: the original judge (Qwen3.5-9B) plus gpt-oss and Gemma-4 via `slurm/ollama_local.sh`. The majority goes to
   the original judge on a tie (as dream_select). `panel_label` also gives the majority without the same-family judge
   (for the Gemma subject model: the original + gpt-oss).

6. **`retrieval.py`.** Pure-Python Okapi BM25 over turns (k1 1.5, b 0.75, Lucene idf, stop list, Porter stems),
   returning the top-m turns in time order, with recall@m against the labelled evidence. Local recall@10: LongMemEval
   oracle 0.875, LoCoMo 0.621, PersonaMem (session level) 0.886, BEAM 0.494.

7. **Tests** (`test_dream_compare.py`; `-m "not cluster"` deselects the cluster-only checks). They cover:
   - specs vs the real configs;
   - identity tests on tiny random Qwen2 / Llama / Gemma-4: full stored turn == text; `rope_shift` == recompute per
     layer type (partial rotary incl.); per-head keep-all == full; same-positions-in-every-head == physical splice
     (eager and SDPA, teacher-forced and multi-step decoding); GQA expansion; the masked generation loop == dream_bind's
     `gen_batch`; the sliding window by position; soft-cap and K = V;
   - the text-only loader on a tiny saved multimodal Gemma-4;
   - each adapter on hand-written fixtures, plus the processed splits against their manifests;
   - the judge-prompt hashes and parsers, LoCoMo's F1, BM25 against a hand computation, and the panel runner.

8. **Slurm.** `slurm/dream_compare{,_cpu,_panel}.slurm`, copied from dream_select's, with a `TESTS` variable and
   multi-GPU resources from the submit script. `submit_smoke.sh` runs the Stage 0 cluster smoke.

### Verification

- **Local:** all old and new tests pass, transformers 4.46 (Gemma skipped) and 5.18 (scratch venv, Gemma included).
- **dream_select regression:** re-running its analyses reproduces the reports byte for byte. Checked on copies: the
  local tf run's `report.md` / `verdict.json` (run at the same relative path), the three models' `report_gen.md` /
  `verdict_gen.json`, and `summary.md` / `summary.json`.
- **Cluster smoke** (`submit_smoke.sh`). Per model:
  - load;
  - identity on the real model with one short turn: full == text (<= 0.01 logits, same greedy tokens), per-head
    keep-all == full, shared == splice, `rope_shift` == recompute per layer type (relative <= 1e-4);
  - Gemma only: the sliding window at 1024 on a ~1,150-token sequence;
  - peak memory per GPU;
  - one item per benchmark adapter through harvest -> restore (R-full == text asserted; R-evidence; R-shift via
    `compose_prefix` + `rope_shift`) -> teacher-forced answer score, plus BM25 recall.

  Then one judge smoke: every new prompt kind on the smokes' answers, plus a sanity set with known labels, through the
  original judge, gpt-oss and Gemma-4, reporting parse rates. Results: see "Stage 0 results" below.

### Stage 0 results

Run root `/scratch/vvjumle/seahorse_runs/dream_compare_smoke_20261105_1530` (`jobs.txt` lists the jobs).

Passed (2026-11-05, commit a403e0f; every job ran the full test file first):

- **Qwen2.5-1.5B** (job 1215658, A100 MIG 10 GB), peak 6.0 GB reserved:
  - full == text: max |dlogit| 3e-4, same greedy tokens
  - per-head keep-all == full: 0
  - shared == splice: 4e-5, same tokens
  - `rope_shift` vs recompute: relative 1e-5
  - each benchmark item, R-full vs text: <= 8e-4
- **Llama-3.2-3B** (job 1214591, V100), peak 12.6 GB:
  - full == text: 1e-4
  - shared == splice: 5e-5
  - `rope_shift` vs recompute: relative 1e-6
  - each benchmark item, R-full vs text: <= 1.3e-4
- On both models, the BM25 recall of the four smoke items is 1.0 at m = 10. The first run of the 1.5B (job 1214589)
  failed in pytest only: a tiny-model tolerance (1.07e-5 vs 1e-5). The tolerance was widened to 5e-5.

Queued: Qwen2.5-7B (1214590), Gemma-4 on 2 x A100 (1214592), Granite (1214593) and the judge smoke (1214594).
gpu-a100 was full when this was written; Slurm estimated starts on 2026-11-08. Their results go here when they run.

## Stage A: selectors

*Written and committed 2026-11-05, before any Stage A job. No Stage A number exists yet; the code is checked on tiny
random models and a local `--tiny` smoke (Qwen2.5-1.5B, 4 items) only. Code: `stage_a.py` (torch side), `baselines.py`
(the scorers and selectors, numpy), `analyze_a.py` (tables, rules), `test_stage_a.py`, `submit_stage_a.sh`. It follows
the earlier selector-baseline design (`dream_select addendum 2 (baselines)`, never run), extended to 133 items and five
models.*

### Question

At a matched storage budget, which way of choosing the stored positions of a memory turn restores answering best: ours
(dreamed-question attention, leave-one-out), or the published KV-cache selectors (KVzip, SnapKV, H2O, VATP, Expected
Attention, StreamingLLM)? Do per-head positions beat one shared set? Is the "licence gap" of dream_select (a slice
that looks fine teacher-forced but makes the model refuse or miss in generation) a property of selectors in general?

### Data

LongMemEval oracle (`data/external/longmemeval/processed/oracle.json`, Stage 0's adapter), **the 133 items whose
labelled evidence contains exactly one user turn** (`benchmarks/longmemeval.single_turn_ids(user_only=True)`; file
order). The stored memory is that user turn's text (as dream_select; no date, no other turn). Chosen over the 182-item
alternative (assistant evidence turns stored as assistant turns) because the dreamer prompt and every earlier result
are about user turns.

| category | items | of which abstention |
|---|---|---|
| single-session-user | 64 (= dream_select's 64; turn, question and answer checked identical) | 0 |
| temporal-reasoning | 39 | 4 |
| single-session-preference | 19 | 0 |
| single-session-assistant | 5 | 0 |
| multi-session | 4 | 4 |
| knowledge-update | 2 | 0 |

- **Primary set (all rules): the 106 items that are neither abstention nor single-session-preference.** A preference
  item's "answer" is a rubric and an abstention item's is an explanation, so neither has an answer to score
  teacher-forced. The preference items (judged with LongMemEval's preference prompt) and the abstention items (its
  abstention prompt) are run in every condition and reported separately. The 64 single-session-user items are also
  reported alone, for comparison with dream_select.
- 70 of the 133 are verbatim (the answer string is in the turn; gold / gold@k exist only for these). Turn lengths
  (Qwen tokenizer): 20 to 1,040 tokens, median 61; 2 turns have more than 200 content tokens, so S-loo (capped at 200,
  as dream_select) is missing on those two. On Gemma the 1,040-token turn exceeds the sliding window: every pass uses
  the adapter's position masks.

### Setup (as dream_select, through the Stage 0 adapter)

- Harvest one pass over [chat head][the turn as a user turn]; K,V of every layer. Selectable = the turn's content
  positions only (not the head, not the role / end tokens). Every slice keeps the head (not counted) plus its picks at
  their stored RoPE positions; the question turn starts at n_head + n_seg.
- Anchors: `nomem`, `ctx`, `T-full` (teacher-forced only), `R-full`, `R-content`. Asserted per item: R-full == T-full
  (logits within 0.01; tf at every scored row, gen at the last prompt row) and per-head keep-all == R-full.
- **Shared** conditions are the physical splice (dream_select's). **Per-head** conditions keep the whole stored turn
  in the cache and mask, per (layer, KV head), the stored slots that head does not keep (Stage 0's `headmask.py`; the
  mask is expanded to the query heads of each GQA group). Tested: keep-all == full cache; the same picks in every head
  == the splice (teacher-forced and multi-step decoding).
- **Budgets, matched in stored K,V entries.** One stored position costs E = sum over layers of the KV heads = 56
  (Qwen2.5-1.5B), 112 (Qwen2.5-7B), 224 (Llama-3.2-3B), 210 (Gemma-4: 25 x 8 + 5 x 2), 512 (Granite-4.2: 64 x 8)
  entries. Budget k in {k2, k4, k8, k10 = max(2, round(0.10 x n_content))} (capped at n_content) means k x E entries:
  shared = k positions in every (layer, KV head); per-head uniform = k positions per (layer, KV head); KVzip's global
  form = the top k x E (layer, KV head, position) entries anywhere. Asserted per slice: exactly k x E entries besides
  the head. (On Gemma the global layers' entries are 512-dimensional and the sliding layers' 256: matched in entries,
  not numbers; disclosed.) Teacher-forced: all four budgets; generation: k4 and k8.

### Conditions (18 selections per budget)

None sees the benchmark question or answer, except SnapKV-oracle (flagged, a ceiling) and gold@k (the label). Every
selector function takes (C, store[, dreams]) only; the tests check the signatures.

**Ours** (shared positions): `S-dream`, `S-attn`, `S-recon`, `S-loo`, `random` (3 seeds teacher-forced, seed 0 in
generation), `gold@k` (verbatim items) — dream_select's definitions. **Reuse:** on qwen25_15b, qwen25_7b and
llama32_3b, the 64 single-session-user items take addendum 1's saved dreams, own answers and S-attn / S-dream /
S-recon / S-loo scores (`dream_select_cluster_20261104_0129/<model>/tf/meta.json`); the fresh harvest must give the
same head, turn and content positions (asserted), and S-attn / S-dream / S-recon are recomputed beside as a check
(information only). The other 69 items, and every item on Gemma-4 and Granite, are dreamed fresh with dream_select's
dreamer (same prompt, few-shot examples, greedy, strict retry). S-dream-split is not run (its bands assume 28 layers).

**Baselines** (`baselines.py`; "ph" = per-head uniform top-k, "sh" = shared top-k of the per-head score averaged over
all (layer, KV head)):

1. **KVzip** (Kim et al. 2025): a faithful reimplementation of the official `attention/score.py::_get_score`, because
   the official code (snu-mllab/KVzip @5d84729) needs flash-attn (fp16/bf16 only, Ampere only: no fp32, no V100), a
   custom CUDA kernel and transformers 4.51. After the whole stored turn: a new user turn "Repeat the previous context
   exactly." + the assistant header + the turn's own text, teacher-forced. Score of a stored entry (layer, KV head,
   position) = the maximum softmax attention it receives over the query heads of its GQA group and over all input rows
   (repeat prompt and reconstruction). The softmax runs over the whole key set (head, turn, input): one chunk, since the
   longest turn (1,040 tokens) is below KVzip's 2K chunk. **KVzip-g** = the official non-uniform head budget (global
   top k x E entries over all layers and KV heads, KVzip's `_threshold`); **KVzip-ph**; **KVzip-sh**. Differences from
   the official code, disclosed: the repeat request is a new user turn (our stored turn is a closed turn), fp32 eager,
   no chunking, our template head is the sink.
2. **SnapKV** (Li et al. 2024): window = the question turn (its last <= 32 rows); vote = the sum over window rows of
   the softmax attention to each stored entry, mean over the GQA group (KVPress), max-pooled over the stored axis
   (kernel 7, stride 1, padding 3, as KVzip's appendix), per-head top-k. **SnapKV-dream**: the window is each dreamed
   question in turn, after the whole stored turn; pooled votes summed over the 3 dreams (ph and sh).
   **SnapKV-oracle-ph**: the real benchmark question (a ceiling that cheats; flagged; excluded from H4's "best
   baseline").
3. **H2O** (Zhang et al. 2023): accumulated attention = the column sum over all rows of the harvest pass, mean over the
   GQA group; the budget split evenly: the last k//2 content positions (recent) + the top k - k//2 of the others
   (heavy). ph and sh.
4. **VATP** (Guo et al. 2024): the H2O score x the l1 norm of the entry's value vector, the same recent / heavy split;
   ph.
5. **Expected Attention** (KVPress `expected_attention_press.py`, the authors' reference): implemented. Pre-RoPE
   queries of every layer's attention input over the harvest pass (the first 4 rows dropped, KVPress's n_sink; q_norm
   applied where the model has one: Gemma-4); per query head mean mu and covariance Sigma; the RoPE rotation averaged
   over the 512 positions after the turn (each layer's own rotary embedding, partial rotation included); score
   z_j = s mu'.k_j + (s^2 / 2) k_j' Sigma' k_j with s = the layer's own attention scaling (1/sqrt(d) on Qwen and Llama,
   which is KVPress's formula; 1 on Gemma-4; 1/128 on Granite); softmax over the turn's keys, mean over the GQA group,
   x the l2 norm of the value; **EA-ph**. Disclosed: KVPress's head-adaptive allocation (AdaKV) is not used; s is the
   model's real scaling instead of KVPress's fixed 1/sqrt(d).
6. **StreamingLLM** (Xiao et al. 2024): the first k//2 + the last k - k//2 content positions (the head is the sink).
7. **S-dream-VATP** (exploratory): S-dream's rows (each dreamed question's content tokens + its last prompt row), per
   (layer, KV head) the mean attention x the value's l1 norm, averaged over layers and heads, summed over the dreams;
   shared top-k.

### Measures

- Teacher-forced (all budgets): log P(gold answer | question) with the answer starting the reply (sum; per-token
  mean), the first-token refusal mass P(I) + P(Sorry) + P(As), gap closed (cond - nomem) / (R-full - nomem) per item
  (items with R-full - nomem < 0.5 nats dropped), hit rate (verbatim items).
- Generation (k4, k8): 1 greedy + 4 samples per condition, <= 64 new tokens, dream_bind's sampler, the random streams
  of dream_select addendum 1 (seed tag "dream_select"). Judged with LongMemEval's official prompt for the item's
  question type (`judges.lme_prompt`; abstention items: its abstention prompt) by the original judge (Qwen3.5-9B) +
  gpt-oss + Gemma-4 (Ollama). Label = the three-judge majority (a tie goes to the original judge). For the Gemma-4
  subject model every rule is also reported with the majority of the original judge and gpt-oss only (no same-family
  judge; tie to the original). Per item: the mean over its 5 answers. Also refusal (dream_bind addendum 7's regex),
  loop rate, has-answer (verbatim), each judge alone, and judged gap closed (cond - nomem) / (R-full - nomem).
- Cost: selection seconds per selector per item; stored entries per slice.

### Pre-registered rules

All at **k = 8, generation, three-judge majority, primary set (106 items)**, 95% percentile item bootstrap (2000
replicates, seed 7), paired on the items present in both conditions. "Holds" = the CI's lower bound > 0. Per model;
the teacher-forced analogues are reported beside.

- **H1: the real KVzip beats S-recon** (our KVzip-like proxy). KVzip's granularity = whichever of KVzip-g / -ph / -sh
  has the higher mean judged correctness at k8 on that model (chosen on the same data; disclosed); KVzip-g (the
  official form) vs S-recon is reported beside.
- **H2: per-head beats shared positions.** Per method: KVzip-ph vs KVzip-sh, SnapKV-dream-ph vs -sh, H2O-ph vs -sh.
  Holds on a model if it holds for >= 2 of the 3 methods.
- **H3: the licence gap is general.** Per item and method: d = gap_tf - gap_gen, where gap_tf = (s_c - s_nomem) /
  (s_Rfull - s_nomem) on the answer score (item dropped if the denominator < 0.5 nats) and gap_gen = (c_c - c_nomem) /
  (c_Rfull - c_nomem) on judged correctness (dropped if the denominator < 0.4). Methods at their native per-head form:
  KVzip-ph, SnapKV-dream-ph, H2O-ph, VATP-ph. A model passes if >= 3 of the 4 methods have a CI of mean d with lower
  bound > 0. **The rule holds if >= 3 of the models with results pass.** The ratio-of-means version and medians are
  reported beside.
- **H4: S-loo vs the best baseline** (no direction): S-loo@k8 - the non-oracle baseline condition with the highest mean
  judged correctness at k8 (one of the 11 baseline conditions), paired CI; reported as better / worse /
  indistinguishable.
- **Paired S-loo vs gold@k** (the earlier comparison was unpaired): at k8 (and k4 beside), on the verbatim primary items
  where both exist; two-sided.
- Cross-model: H1, H2 and H4 are also counted over models ("holds on n of m models with results"); H3 has its own
  count above. Models: qwen25_15b, qwen25_7b, llama32_3b, gemma4_26b_a4b, granite42_30b (all five are run; a model
  whose Stage 0 smoke failed is not run and is reported as missing).

### Run plan

Per model: smoke (`--tiny`: 4 items, 2 reused + 2 fresh, k4; pytest first) -> tf -> gen -> the original judge and
gpt-oss -> Gemma-4 judge -> analyze; then a cross-model summary. Gemma-4 and Granite split tf and gen into parts (by
item, resumable per item) to stay under 24 h per job. Gates: qwen25_7b, Gemma-4 and Granite wait for their Stage 0
smokes (1214590, 1214592, 1214593); every judge job waits for the Stage 0 judge smoke (1214594). Outputs per model:
`report_a.md`, `verdict_a.json`; the run root gets `summary_a.md`.

### Stage A run (record; no rule changed)

- Code: commit 56af7f1 (+ b24e9cc, ec9cec4: submit-script resources only). Local checks before submission: the Stage A
  tests pass on transformers 4.46 (25, Gemma skipped) and 5.18 (28, Gemma included); the Stage 0 and dream_select
  suites still pass; dream_select's committed reports re-derive byte for byte; a local `--tiny` smoke on Qwen2.5-1.5B
  (RTX 4060) passed: R-full == T-full <= 3e-4, keep-all == R-full 0, same picks in every head == splice 2e-5, the
  recomputed S-attn / S-dream / S-recon equal addendum 1's saved scores to 1e-7 (same top-4 picks), peak 6.6 GB.
- Shared-module changes (additive, tested): `run.py` dispatches `--stage a-*` to `stage_a.py`; `models.gen_batch`
  takes the masked loop on models spread over 2 GPUs (dream_bind's loop leaves the logits on lm_head's device; this
  also applies to the pending Stage 0 Granite smoke).
- Qwen2.5-1.5B runs on gpu-v100 (gpu-a100-small caps jobs at 4 h).
- Run root `/scratch/vvjumle/seahorse_runs/dream_compare_a_20261105_1822` (`jobs.txt` lists the 42 jobs), submitted
  2026-11-05 at ec9cec4.

## Stage B: paradigms

Written and committed before any Stage B job (2026-11-05). Code: `paradigms.py` (memory layout, store, conditions,
ragged batches, P4 selectors, storage accounting), `memart.py` (P5), `stage_b.py` (runner), `analyze_b.py`,
`test_stage_b.py`, `submit_stage_b.sh`.

### Question

Does a stored-cache memory beat text memory on anything that matters (accuracy, refusals, storage, latency) on real
memory benchmarks, with retrieval held fixed? Per question, the whole history is the memory.

### Data

All items of: LongMemEval oracle (500) and s120 (120); LoCoMo locomo10 (1,986: 1,540 judged + F1, the 446 adversarial
as an abstention test); PersonaMem v1 32k (589, multiple choice); BEAM 100K (320). Models: qwen25_15b, qwen25_7b,
llama32_3b, gemma4_26b_a4b, granite42_30b (fp32; a model whose Stage 0 smoke fails is not run and is reported missing).

### The memory (identical for every paradigm)

- **Turns -> chat messages.** Each turn is one message; its content is `"[<session date>] <text>"` when the turn has a
  date (LongMemEval, LoCoMo, BEAM sessions with a time anchor), else the text. Text is stripped and cut at 16,000
  characters (`" [...]"` appended; 3 BEAM turns, 1 LongMemEval-S turn).
- **Roles.** LongMemEval, BEAM: as given. PersonaMem: as given; its system (persona) turns become user messages (a chat
  template's system slot is the head).
- **LoCoMo frame (neutral).** Stage 0 found Llama answering in a speaker's voice when the two speakers map to user /
  assistant. Here the transcript is a log the user provides: **every** LoCoMo turn is a user message
  `"[<date>] <Speaker>: <text> [shared an image: <caption>]"` (neither speaker is the assistant), and the question is
  asked by a third party:
  `I'm reading a log of conversations between <A> and <B>. <question> Answer with a short phrase from the log. If the log
  does not say, answer "Not mentioned in the conversation".`
- **Other question turns.** LongMemEval: `Today is <question_date>. <question>`. BEAM, PersonaMem: the question as given
  (PersonaMem: the options are scored as the assistant's reply).
- **Encoding.** Each session is rendered with the chat template (`session_pieces`: the head + one template piece per
  turn, asserted to tokenise as the whole conversation) and encoded **once** (session-level prefill, SDPA, plain cache,
  position masks on Gemma). A session longer than 16,384 tokens is encoded in segments of whole turns (each with the
  head; BEAM only). The store keeps every memory token's K, V on the CPU at its session position. Eager attention only
  where scores are needed (S-dream's passes).

### Paradigms (conditions)

| | at rest | at question time |
|---|---|---|
| P0 | nothing | head + question |
| P1 | text | head + the whole visible history as text + question; only where it fits (below) |
| P2-m | text | head + the BM25 top-m turns' pieces (time order) as text + question |
| P3-m | text (index) + full K, V | the **same** top-m turns restored from the store, each moved to its P2 position (`rope_shift` per layer type); the question at P2's position. Same tokens and positions as P2; only the encoding context differs |
| P4{loo,dream}-m | text (index) + k = 8 positions per turn | P3-m's prefix reduced to the head + each turn's 8 picked content positions, at their P3 positions; the question where P3's starts |
| P5-m | full K, V in 16-token blocks | MemArt-style: per layer, the top-k blocks by the question's attention, re-encoded contiguously after the head; k = ceil(P3-m's restored turn tokens / 16) |
| P5s-m | same | as P5 with k = ceil(P4-m's kept entries / 16) (matched to P4) |

- **Retrieval:** Stage 0's BM25 (`retrieval.py`) over the item's visible turns, query = the raw question; m in {3, 10}
  turns. PersonaMem is session-level (its evidence is): BM25 over the visible sessions (document = their turns' text),
  **m in {1, 3} sessions** (deviation from {3, 10}: contexts have 5.3 sessions on average, so m = 10 would be the whole
  history), all visible turns of the retrieved sessions.
- **P1** runs where head + history + question <= min(the model's context - 512, the tokens whose plain cache fits
  `--p1-kv-gb`: 1.5B 12 GB, Llama 8, 7B 20, Gemma 16, Granite 12). N skipped is reported per split. The history is
  prefilled once per context (chunks of 4,096) and shared by its items (PersonaMem: the prefix up to each item's
  visible end).
- **P4 selectors, fixed in advance** (Stage A runs in parallel and is not waited for): **P4loo = S-loo, P4dream =
  S-dream, k = 8 per turn**, dream_select's dreamer prompt and selectors unchanged, re-implemented device-safely
  (equal on one device: tested). Selection is query-free: each turn is harvested **alone** (head + its own piece,
  role header included) and scored there; the picks (content positions = the tokens spelling its rendered content)
  are applied to its session-level stored entries. Turns with <= 8 content tokens keep all of them; the template
  tokens of a turn are dropped (as dream_select's slices). The dreamer sees every turn as given (assistant turns and
  LoCoMo lines included). Picks are cached per (role, content); dreams, own answers and leave-one-out rows are batched
  (ragged rows; == per turn, tested).
- **S-loo's scope (amended before any Stage B job, 2026-11-05, after the local smoke).** S-loo costs 3 forward passes
  per content token (measured locally: ~0.1 s per content token per turn on Qwen2.5-1.5B, O(n^2) per turn); on every
  retrieved turn of every item it would need days per model. So: **P4loo runs on the 100-item subsets only**
  (LongMemEval oracle, LoCoMo, BEAM; not on s120 or PersonaMem), with dream_select's own cap (S-loo only on turns with
  <= 200 content tokens, `LOO_MAX_CONTENT`; longer turns take S-dream's picks, counted). **P4dream runs on every
  item.** B2 / B3 / B5 for P4loo are computed on the subsets. P5s's budget = P4dream's kept entries (both selectors
  keep min(8, content) per turn, so the budgets are equal).
- **P4-kvzip is not included**: Stage A's `baselines.py` was not committed when this section was first committed
  (385c47f); it landed afterwards (56af7f1), after Stage B's design was fixed. A later addendum may add per-head KVzip
  or Stage A's winner.
- **P5 (MemArt-style), from the paper (no public code).** Blocks of BS = 16 consecutive turn tokens per session /
  segment (head excluded, always kept; a session's last block may be shorter). Relevance I(q, K) = max over the block's
  keys of q . k with q and k **without RoPE** (stored keys rotated back to position 0 per entry; q = q_proj (+ q_norm)
  of the layer's input), exact (no AABB pre-filter). Per (query token, query head): reciprocal-rank normalisation
  1 / (rank + 60) over the candidate blocks; Max over all of them (RR-Max, the paper's Qwen2.5-7B setting); top-k
  blocks (ties -> earlier), the same for every head, **per layer** inside the question's prefill (layer l's queries
  have seen the blocks retrieved at layers < l); chronological order; block j re-encoded at positions nh + 16 j ..;
  the question at nh + 16 k. Departures: exact scoring; c = 60 (the paper gives none); per-layer selection inferred
  from its per-attention Algorithm 1; the query is the question turn (user header + question + generation prompt);
  short blocks padded with slots masked in that layer. PersonaMem: only blocks inside the visible turns.

### Answers

- One greedy answer per item and condition (PersonaMem: the four options' summed log-probabilities, no generation),
  max new tokens LongMemEval 128, LoCoMo 48, BEAM 256.
- Plus 4 samples per generative condition (dream_bind's sampler: temperature 1, top-k 20, top-p 0.95, presence 1.5;
  one stream per row seeded by (item, condition, sample)) on a fixed 100-item subset per benchmark:
  `sorted(random.Random(7).sample(sorted(item ids), 100))` of LongMemEval oracle, LoCoMo and BEAM (PersonaMem has no
  generation; s120 has none).
- Batching: rows with different prefixes are left-padded into one batch (masks from RoPE positions, pad slots masked);
  identical to each row alone (tested; checked on the first context of every job, see below).

### Measures (per model x split x condition)

- **Judged correctness** (primary): LongMemEval's per-type prompts (abstention prompt for `_abs`); LoCoMo: Mem0's J on
  the 1,540 non-adversarial items (+ the official F1); BEAM: the rubric judge per rubric item, item = mean; PersonaMem:
  the option with the highest summed log-probability (mean per token beside). Panel = the original judge (Qwen3.5-9B)
  + gpt-oss + Gemma-4: boolean kinds by majority (tie -> the original judge); BEAM rubric items by the median of the
  judges' scores (two parsed: their mean). Gemma-4 subject model: also without the Gemma-4 judge.
- **Refusal rate**: dream_bind's refusal regex or "not mentioned" / "no information" / "not specified|stated|provided",
  on the greedy answers (all items) and the samples (subset).
- **Abstention accuracy**: LongMemEval `_abs` (judged), LoCoMo adversarial (the official category-5 rule: "not
  mentioned" / "no information available" in the answer), BEAM abstention (rubric).
- **Knowledge-update accuracy**: LongMemEval knowledge-update, BEAM knowledge_update.
- **Evidence recall@m** of the retrieval (turns; PersonaMem sessions).
- **Storage per conversation** (at rest, mean over the items' visible histories): text = UTF-8 bytes of the rendered
  turns (P1, P2; also the BM25 index P3 / P4 need); KV = (head + all turn tokens) x K,V numbers per position x 4 bytes
  (P3: + text; P5: + MemArt's AABB index, 2 x head_dim per KV head per layer per block, though our scoring is exact);
  P4 = (head + min(8, content tokens) per turn) x (numbers x 4 + a 4-byte position) + text. Gemma: both as stored
  (K and V) and V-only on its K = V global layers. Caches are not kept on disk; the numbers are from this accounting.
- **Question-time cost**: prefill tokens (P0 head + question; P1 + history; P2 + retrieved text; P3-P5 the question
  only) and restored entries; latency on the first 20 subset items (PersonaMem: the first 20 ids), each condition
  alone (B = 1, synchronised): time to the first token including the question-time work (P1: the history's prefill;
  P3 / P4: compose from the CPU store; P5: block scoring in the prefill) and decode time per token.

### Pre-registered rules

Per model x split (LongMemEval oracle, LongMemEval s120, LoCoMo, PersonaMem, BEAM) x m; 95% percentile item bootstrap
(2000 replicates, seed 7), paired on the items present in both conditions.

- **B1 (format): P3-m vs P2-m judged correctness is equivalent within +-0.03** (TOST: the 90% CI of the paired mean
  difference inside (-0.03, 0.03)). Else "different" if the 95% CI excludes 0, else "inconclusive". Also pooled per m:
  the mean over model x split cells of the cell's mean difference, items resampled within cells (per model in its
  report; over every model x split in the summary).
- **B2: P4 vs P3** (per selector, m): the correctness difference with its CI, and the storage saving factor P3 KV bytes
  / P4 KV bytes. No direction.
- **B3: P4 vs P5 at matched storage**: P4 vs P5s (matched restored entries at question time; at-rest bytes of both
  reported); P4 vs P5 beside.
- **B4: Pareto frontier** of correctness vs at-rest storage bytes, vs question-time prefill tokens and vs time to first
  token: the non-dominated conditions.
- **B5: refusal rate per condition**, with paired differences P3 - P2, P4 - P3 (both selectors), P5 - P3. The licence gap
  "appears end to end" on a cell if P4loo - P3 refusal has a 95% CI above 0; counted over cells.

### Checks that fail a job

On the first context of every split: P3 with every turn of the first session restored == that session as text
(teacher-forced logits at the question rows <= 0.01, same greedy tokens); ragged rows == the adapter's `gen_batch` per
row (greedy tokens, 4 row kinds). In tests: the same identities on tiny Qwen2 / Llama / Gemma-4 (Gemma on the
cluster's transformers 5.18), compose / `rope_shift` ordering, MemArt's block scoring on toy tensors, the hook with
every block == P3 at the same positions, storage accounting by hand, BM25 integration, the judges' routing per
benchmark and category, the TOST / Pareto helpers.

### Run plan

Per model: smoke (pytest + one item per split through every condition + the checks) -> 4 generation jobs (LongMemEval
oracle + s120; LoCoMo; PersonaMem; BEAM; resumable per item, `--max-hours 22.5` inside a 24 h limit) -> the original
judge (`b-tasks` + `b-judge-local`, left-padded batches checked against run.py's unpadded path on the first 8 prompts)
-> gpt-oss -> Gemma-4 judge -> analyze (`report_b.md`, `verdict_b.json`); then `summary_b.md`. Gates: qwen25_7b,
Gemma-4 and Granite wait (afterok) for their Stage 0 smokes 1214590 / 1214592 / 1214593; every judge job waits for the
judge smoke 1214594; Qwen2.5-1.5B and Llama-3.2-3B are ungated. Gemma-4 and Granite (2 GPUs, slower) run LongMemEval
oracle and s120 as two jobs. Stage B uses 48 queue slots (9 per model, 10 for the two 2-GPU models, + 1 summary).

### Stage B run record

- **Local checks (2026-11-05).** `test_stage_b.py`: 22 passed locally (transformers 4.46; the 5 Gemma-4 cases skip) and
  26 passed on a DelftBlue CPU node (transformers 5.18, Gemma-4 tiny models included). The old suites
  (`test_dream_compare.py`, `test_dream_select.py`): 102 passed, 15 skipped. Tiny random-model dry runs of every split
  (CPU) and a real Qwen2.5-1.5B run of one LongMemEval oracle item (RTX 4060): P3 of a whole session vs its text
  max |dlogit| 1.9e-4, same greedy tokens; ragged rows == per-row generation (4 / 4). The laptop GPU (8 GB) cannot hold
  the run's batches, so its times are not used for sizing.
- **Run root** `/scratch/vvjumle/seahorse_runs/dream_compare_b_20261105_1915` (`jobs.txt`), commits 275444a / 2d93767.
  Qwen2.5-7B's gate (1214590) had completed before submission, so its smoke has no dependency (Slurm rejects afterok on
  a finished job); Gemma-4 and Granite wait for 1214592 / 1214593, every judge job for 1214594.

| model | smoke | gens | judge | gpt-oss | Gemma-4 | analyze |
|---|---|---|---|---|---|---|
| qwen25_15b (V100) | 1218492 | 1218493-96 | 1218497 | 1218499 | 1218500 | 1218501 |
| llama32_3b (V100) | 1219643 | 1219644-47 | 1219648 | 1219649 | 1219650 | 1219651 |
| qwen25_7b (A100) | 1217750 | 1217751-54 | 1217755 | 1217756 | 1217757 | 1217758 |
| gemma4_26b_a4b (2 x A100) | 1217759 | 1217760-64 | 1217765 | 1217766 | 1217767 | 1217768 |
| granite42_30b (2 x A100) | 1217769 | 1217770-74 | 1217775 | 1217776 | 1217777 | 1217778 |

  Summary 1217779 (after every analyze; its dependency list updated with `scontrol`).
  First smokes, both on V100, failed after passing pytest and the identity checks on every split they reached
  (P3-all vs text 1.5e-4 to 8.7e-4, ragged == per-row 4 / 4): Llama 1217740 ran out of memory on LoCoMo's P1 row
  (four copies of the 24k-token history alive at once), the 1.5B 1217731 on BEAM's one-pass 16k-token session
  encoding (fp32 SDPA with a mask takes the math path on V100: T x T scores per head). Fixed (df98948, 6739282: P1
  rows use views of the shared history, the ragged cache is built layer by layer, session encoding and long
  prompts run in 2,048-token chunks with an attention-memory budget; tested chunked == one pass) and both chains
  resubmitted; no science changed. The second 1.5B smoke (1218150) passed LongMemEval oracle / s120, LoCoMo and
  PersonaMem, then ran out of memory in the identity check's one-pass text reference of a 16k-token BEAM segment;
  the check now uses the segment's leading turns (<= 2,048 tokens; 4d04fb8). Both chains were resubmitted again (the
  rows above; the smokes resume from the items already written). The 1.5B smoke 1218492 PASSED (every split; BEAM
  identity 2.9e-4 on 7 turns; peak 15.1 / 16.3 GB allocated / reserved on the V100). Llama 1218502 passed
  LongMemEval, LoCoMo and PersonaMem's answers, then ran out of memory in PersonaMem's latency pass (P1's own prefill
  while the shared history was still on the GPU); fixed (34cfa50) and its chain resubmitted. Gens order per model: LongMemEval (one job; Gemma / Granite: oracle, s120),
  LoCoMo, PersonaMem, BEAM.
