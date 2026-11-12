# dream_heads: results (run 2026-11-11/12)

*Run root `/scratch/vvjumle/seahorse_runs/dream_heads_20261111`; local copy `results/dream_heads_20261111/` (per-model
`report.md`, `verdict.json`, `items/`; cross-model `summary.md`). Pre-registration: `PREREG.md` (b5fd821), code c860a14.
Teacher-forced; 66 LongMemEval items (Stage A primary minus temporal-reasoning and single-session-assistant); judged
correctness reused from dream_compare Stage A. Section 2 is an exploratory follow-up, not pre-registered
(`explore_kvzip_target.py`).*

## The question

Why does KVzip-g (one global budget over layers x KV heads) keep the model answering from a sparse slice when the
shared-position selectors refuse? Hypothesis tested: a few attention heads write the answer/refuse direction at the
last prompt token, and KVzip-g keeps the entries those heads read.

## 1. Pre-registered rules

| model | H-A: refusal moves along d_comply | H-B: 8 heads carry >= half | H-C(i): KVzip-g covers those heads more | H-C(ii) | verdict |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | holds (rho -0.82 [-0.86, -0.77]) | fails (0.29) | fails (KVzip-g covers them LESS: 0.04 vs 0.18-0.42) | fails | in between |
| Qwen2.5-7B | holds (-0.59 [-0.68, -0.48]) | fails (0.30) | fails (0.17 vs 0.24-0.44) | fails | in between |
| Llama-3.2-3B | holds (-0.75 [-0.80, -0.69]) | fails (0.26) | fails (0.23 vs 0.24-0.34) | fails | in between |

**Cross-model verdict: in between** (head-gated on 0 of 3; "no single gate" on 0 of 3).

In plain words:
- **The decision is real and graded, and the model earns it without any push.** Within an item, slices that make
  the model refuse sit lower on the comply direction; the same projection tracks judged correctness (+0.58 to +0.68).
  The cross-fitted told direction gives the same picture.
- **It is not a few-head gate.** The top 8 heads carry only 26-30% of the push from no memory to the whole turn; on
  the Qwens the top heads change between folds (Jaccard 0.14 and 0.33). Causally, hiding the full turn from those
  heads raises first-token refusal only on the 1.5B (+0.18 vs +0.04 for layer-matched control heads); on the 7B and
  Llama it does nothing (+0.00, +0.03).
- **KVzip-g does not win by feeding those heads.** It gives them less than the shared selectors do (its budget in
  their KV groups is 0.23 / 0.61 / 0.92 of an even spread on 1.5B / 7B / Llama).

## 2. Exploratory follow-up: what KVzip-g's uneven budget does track

Per item, KVzip-g's k8 budget per (layer, KV group) against the whole turn's (R-full) attention mass from the last
prompt token to memory, and mass-weighted coverage (share of that attention the slice preserves, heads weighted by
how much they read memory), across the 7 k8 selections within an item.

| | 1.5B | 7B | Llama |
|---|---|---|---|
| KVzip-g budget vs R-full last-token memory attention (within-item Spearman) | +0.03 | +0.16 | **+0.49** |
| KVzip-g budget vs layer index | +0.13 | +0.25 (late-heavy) | +0.01 |
| mass-weighted coverage: KVzip-g / best other | 0.15 / 0.33 (S-dream) | 0.17 / 0.32 (S-dream) | **0.30** / 0.28 (KVzip-ph) |
| mass-weighted coverage vs judged (within-item Spearman) | +0.12 [-0.00, +0.24] | +0.01 [-0.09, +0.12] | **+0.44 [+0.35, +0.53]** |
| ... vs refusal | -0.44 | -0.04 | -0.38 |
| better than plain mean coverage (judged) | +0.05 | -0.01 | +0.25 [+0.17, +0.34] |

Reading, per model:
- **Llama: explained.** KVzip-g spends its budget where the last token actually reads the stored turn, preserves the
  most of that attention of any selector, and preserved attention (weighted by how much each head reads) predicts
  judged correctness, answer score and refusal.
- **1.5B: explained as a failure.** KVzip-g's allocation ignores what the 1.5B reads at the decision point (+0.03),
  it starves the heads that causally matter there (licence-group budget 0.23 of even; hiding them raises refusal
  +0.18), and it refuses as much as random (first-token refusal 0.66 vs 0.67).
- **7B: not explained at the last prompt token.** KVzip-g is best on judged correctness (0.59) and on the answer
  score (-27.4 vs -29.1 to -33.0 for the others), but first-token refusal is the same for every selector (0.05-0.08),
  its comply projection is worse than S-loo's and S-dream's, and nothing measured at the last prompt token predicts
  judged correctness. Its budget leans to late layers. The 7B's advantage is in content used while answering, which a
  last-prompt-token measurement cannot see.

## What changes

- **The licence is not a separate gate.** It is a graded readout (H-A) written by many heads, and it rises with how
  much of the whole turn's effect gets through. At the population level, "content" and "licence" move together;
  dream_subj's clean split (refusal moved, value did not) was on hand-picked positions of templated facts.
- **What makes a sparse slice work is preserving what the model actually reads from memory, head by head, weighted
  by how much each head reads.** KVzip-g approximates this through reconstruction attention, which matches the
  decision point well on Llama, poorly on the 1.5B, and (on the 7B) seems to matter during the answer rather than
  before it.
- **The "few licence heads" route is closed** (H-B and H-C fail on all three models).

## What it can't tell us

- Teacher-forced only; refusal = first-token mass. The 7B's failures are mostly not at the first token.
- All attention and attribution measures are at the last prompt token; the answer tokens were not probed.
- Coverage uses R-full's attention as the reference, not each slice's own.
- Causal masks act on whole KV groups (6 / 7 / 3 query heads on 1.5B / 7B / Llama).
- Section 2 is exploratory and was chosen after seeing section 1.
- 66 items, one benchmark, three models.

## Open, if pursued

- Probe the answer tokens (not only the last prompt token) on the 7B: is KVzip-g's late-layer budget what the
  answer's value fetch reads?
- A selection rule built directly on "attention mass the model will read, per head" (e.g. from a dreamed question's
  answer tokens) vs KVzip-g's reconstruction proxy, on the 1.5B where KVzip-g fails.
