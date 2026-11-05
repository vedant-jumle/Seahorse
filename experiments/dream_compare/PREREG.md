# dream_compare: pre-registration

The full comparison run: does a stored-cache memory beat text memory on anything that matters (paradigms), and which
way of choosing stored positions works best at matched storage (selectors)? Plan: three stages, each with its own
section here, committed before that stage's jobs. Each stage stops after its smoke tests pass.

- **Stage 0: infrastructure** (this section; no scientific rules)
- **Stage A: selectors** (placeholder below; filled in by the Stage A executor before its jobs)
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

(Filled in after the cluster smoke.)

## Stage A: selectors

(Placeholder: the Stage A executor writes the data, selectors, budgets and pre-registered rules H1-H4 here, from the
earlier selector-baseline design, before its jobs.)

## Stage B: paradigms

(Placeholder: the Stage B executor writes the paradigms P0-P5, retrieval sizes, measures and rules B1-B5 here before its
jobs.)
