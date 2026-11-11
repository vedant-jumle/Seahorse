# dream_heads: pre-registration

*Written and committed 2026-11-11, before any code ran on a real model and before any job. No number below exists yet.
Teacher-forced only: no generation, no judges. The outcome "judged correctness" is Stage A's saved three-judge majority
per item x condition (dream_compare, `results/dream_compare_a_20261105_1822/<model>/`), reused, never re-judged.*

## Question

In dream_compare Stage A, **KVzip-g** (KVzip's official global budget: the top k x E (layer, KV head, position) entries
over all layers and KV heads, by each entry's own maximum attention while the model repeats the turn) nearly matched the
whole stored turn on judged correctness on Qwen2.5-7B and Llama-3.2-3B, with little licence gap. The shared-position
selectors (S-loo, S-dream, KVzip-sh) at the same budget refuse much more. On Qwen2.5-1.5B KVzip-g also fails (TF gap
closed 0.59, judged 0.21). **Why does KVzip-g's sparse slice work, and the others not?**

**Working hypothesis** (Arditi et al. 2024, §2.3 and §5; a structural fact): a restored slice can act only through the
attention heads that read it. The answer/refuse decision at the last prompt token is read out along one direction, which
a few heads write. Per-head selection keeps the entries those heads read; shared selection starves them.

## Data

- LongMemEval oracle, Stage A's primary items (`stage_a.load_items()`: one user evidence turn, not abstention, not
  single-session-preference: 106) **minus temporal-reasoning (35) and single-session-assistant (5)**. Both are setup
  artefacts: the stored turn has no date, and an assistant item's evidence partly sits in the assistant turn.
- **Checked before writing (Stage A's files, all three models): 66 items = single-session-user 64 + knowledge-update 2;
  57 of them verbatim (gold@k exists); S-loo exists on all 66; every item has judged labels for every condition below.**
- **Folds (fixed now):** the 66 ids sorted, `random.Random(7).shuffle`, the first 33 = fold A, the rest = fold B.
- Models: `qwen25_15b`, `qwen25_7b`, `llama32_3b` (fp32, eager attention, the Stage 0 adapters, `models.setup`).
  Granite and Gemma-4 are not run.

## Setup (reused from Stage A, by import)

- The store: `stage_a.harvest(C, turn, stats=False)` (one pass over [chat head][the turn as a user turn]); asserted equal
  to Stage A's head, turn and content positions per item. The question turn starts at n_head + n_seg.
- **Selections: Stage A's saved ones** (`tf/items/<qid>.json.gz`, `sels`), never recomputed. Shared conditions = the
  physical splice (`stage_a.splice`); per-head conditions = the whole stored turn with Stage 0's per-(layer, KV head) mask
  (`headmask.HeadMask`, via `stage_a.keep_of`), exactly as Stage A.
- **Conditions:** `nomem`, `R-full`, and at **k8 (primary) and k4**: `KVzip-g`, `KVzip-ph`, `KVzip-sh`, `S-loo`,
  `S-dream`, `random` (seed 0, the selection Stage A generated with), `gold@k` (verbatim items only).
- Every measure is read in one teacher-forced pass of [question turn][gold answer] (Stage A's `measure` layout); "the
  last prompt token" = the row before the first answer token.

## Measures (per item x condition, at the last prompt token)

1. **First-token refusal mass** P(I) + P(Sorry) + P(As) (Stage A's), and the teacher-forced answer score (log P(gold)).
2. **Projection on the answer/refuse direction, per layer** (the block-l output, unit direction per layer).
   - **Primary: d_comply**, rebuilt per model with dream_bind addendum 8's recipe (`dream_bind/run.py`
     `stage_directions8`): the 66 pairs of `data/bench_v1/comply_pairs.yaml`, greedy answers (60 tokens) without memory,
     a pair "behaves" if the personal question is refused and the general one answered (`COMPLY_REFUSAL_RE`);
     d_comply(l) = mean(general) − mean(personal) of the block-l output at the last prompt token over the behaving pairs
     (fewer than 2 behave: all pairs, recorded, as addendum 8). Up = answer.
   - **Secondary: d_told(l)** = mean(R-full) − mean(nomem) of the block-l output at the last prompt token over the
     items of the *other* fold (2-fold cross-fitted: built on one half, applied to the other).
3. **Direct attribution per attention head** (Arditi §5 style direct feature attribution): head (l, h)'s slice of the
   o_proj input times its slice of o_proj, projected on the unit d_comply at the readout layer L*. Heads = query heads at
   layers l <= L*. (Toy-tested: the heads' attributions sum to the attention output's projection.)
   - **Readout layer L\*:** the layer where the d_comply projection best separates R-full from nomem, i.e. the largest
     paired standardised difference mean(p_Rfull − p_nomem) / sd(p_Rfull − p_nomem), over the items of the other fold.
     *Candidates (a detail the design left open, fixed now): layers ⌊0.2 n⌋ <= l < ⌊0.8 n⌋ (5–21 of 28; the first commit said 6–21, an arithmetic slip corrected before any job), following
     Arditi's exclusion of the last 20% of layers; the full per-layer curve is reported.* Each fold is evaluated at the
     L* chosen on the other fold.
4. **Attention from the last prompt token**, per (layer, query head): the mass on the stored turn's slots (0 for nomem),
   and **coverage**: the share of that head's R-full attention to the stored *content* positions that falls on entries
   the condition keeps for the head's KV group (R-full 1; nomem 0; shared selections: their positions in every head).
   Licence-head coverage = the mean over the item's 8 licence heads; all-head coverage = the mean over every (layer,
   query head) (heads with no attention on content, < 1e-6, are skipped).
5. **Descriptive:** per selector, kept entries per layer and per (layer, KV head) (KVzip-g's spread vs the uniform k);
   token classes of kept entries vs their base rate among content tokens: *gold span* (verbatim items), *question words*
   (a token inside a turn word of >= 3 letters, not a stop word, that also occurs in the question: our proxy for the
   subject span, which LongMemEval does not label), *punctuation*, *other*.

**Licence heads (cross-fitted):** the top 8 query heads (layers <= L*) by the mean attribution difference
(R-full − nomem) over fold A, used on fold B, and vice versa. Both folds reported.

## Pre-registered rules

Paired item bootstrap, 2000 replicates, seed 7, 95% percentile CI. Per model; the summary counts models. Within-item
correlations are Spearman over the item's conditions; an item whose series is constant is dropped from that statistic
(n reported).

- **H-A, the direction tracks the decision.** Per item, Spearman(d_comply projection at the item's L*, first-token
  refusal mass) over its k8 conditions (nomem, R-full and the 7 k8 selections; 6 on non-verbatim items). **Holds if the
  mean over items is <= −0.4 and its CI's upper bound < 0.** The same against Stage A's judged correctness is reported
  (expected positive), and with d_told beside. **If H-A fails on >= 2 of 3 models the verdict is "no single gate", and
  H-B and H-C are reported as descriptive only.**
- **H-B, a few heads.** For fold f, m_f(h) = the mean over fold f's items of attribution(R-full) − attribution(nomem);
  share_f = Σ over the licence heads chosen on the other fold of m_f(h), divided by Σ over all heads (l <= L*) of
  max(m_f(h), 0). Pooled share = the mean of the two folds' shares; the bootstrap resamples items within each fold.
  **Holds if the pooled CI's lower bound > 0.5.** Both folds' shares reported.
- **H-C, coverage explains KVzip-g.**
  - (i) KVzip-g's licence-head coverage minus each shared selector's (S-loo, S-dream, KVzip-sh), at k8, paired over
    items. **Holds if all three CIs' lower bounds > 0.**
  - (ii) Per item, over the 7 k8 selections (6 on non-verbatim items; anchors excluded, their coverage is 0 / 1 by
    definition): ρ_lic,ref = Spearman(licence coverage, refusal) and ρ_all,ref = Spearman(all-head coverage, refusal);
    d_ref = ρ_all,ref − ρ_lic,ref (positive: licence coverage tracks lower refusal better). For judged correctness
    d_cor = ρ_lic,cor − ρ_all,cor. **The paired difference of within-item Spearman correlations, item bootstrap. Holds if
    both d_ref and d_cor have CI lower bounds > 0.**
  - (iii) Descriptive, mainly for the 1.5B: KVzip-g's licence coverage per model; the licence heads' layers; the overlap
    of the two folds' licence heads (Jaccard); whether on the 1.5B KVzip-g's coverage of its licence heads is low or its
    licence heads differ (layers, attribution share).
- **Verdict per model:** "head-gated licence" if H-A, H-B, H-C(i) and H-C(ii) all hold; "no single gate" if H-A fails;
  else "in between". **Cross-model verdict:** "head-gated licence" if that holds on >= 2 of 3 models; "no single gate" if
  H-A fails on >= 2 of 3; otherwise "in between".

## Exploratory causal check (teacher-forced, reported, not decisive)

At k8, each item with its out-of-fold licence heads:
- `R-full−lic`: R-full with the licence heads' KV groups masked over the stored turn's slots (the template head, the
  sink, stays visible as in every condition). Prediction: refusal rises, the answer score falls.
- `KVzip-sh+lic`: KVzip-sh with the licence heads' KV groups given the whole stored turn (per-head mask form; checked ==
  the splice when no group is opened). Prediction: refusal falls.
- Controls `R-full−ctrl`, `KVzip-sh+ctrl`: 8 control heads matched by layer: per licence head a random query head
  (numpy `default_rng(0)` per fold) of the same layer that is not a licence head and whose KV group holds no licence head
  of that layer; if none exists, any non-licence head of the layer (counted: then the masks partly coincide).
- Masks act on whole KV groups (Stage 0's mask is per KV head): on the 1.5B one group = 6 query heads, on the 7B 7, on
  Llama 3; disclosed as a limit. Reported: paired differences in refusal, answer score and projection vs R-full /
  KVzip-sh, licence vs control.

## Checks that fail the job

Per item: R-full == T-full (teacher-forced logits at every scored row <= 0.01); per-head keep-all == R-full (<= 0.01);
the KVzip-sh mask form == the splice (<= 0.01); every reused condition's answer score equals Stage A's within 0.01 nats
(the selections reproduce; Stage A also ran fp32 eager). Recorded: d_comply's behaviour counts (kept / dropped pairs);
the toy identity (attributions sum to the attention output's projection) is a test that runs first in every job.

## What it cannot tell us (written now)

Teacher-forced only: refusal mass is the first token's, judged correctness is Stage A's. Direct attribution ignores what
later layers do with a head's write; coverage uses R-full's attention as the reference, not each condition's own; the
"question words" class is a proxy for the subject; KV-group masks are coarser than query heads; 66 items, 2 folds of 33.

## Run plan

Per model one GPU job chain: smoke (pytest; `--tiny`: 4 items, 2 per fold, k8 and k4, every check) -> run (d_comply ->
anchors for every item -> fold / L* / licence heads / d_told -> all conditions + the causal check; resumable per item)
-> analyze (CPU: `report.md`, `verdict.json`) -> a cross-model `summary.md`. Qwen2.5-7B on gpu-a100, Llama-3.2-3B and
Qwen2.5-1.5B on gpu-v100 (the laptop GPU is not available to this session). Run root
`/scratch/vvjumle/seahorse_runs/dream_heads_<date>`; Stage A's labels and selections are read from
`/scratch/vvjumle/seahorse_runs/dream_compare_a_20261105_1822/<model>`.
