# core_v1 pre-registration

*Written and committed before any core_v1 job ran (2026-10-07). It fixes the claims, the primary
comparisons, the decision rules, the exclusion rule and the statistics. `analyze.py` (`claims()`)
implements exactly these rules. Any later change to a rule must be reported as a deviation, with the
pre-registered result alongside it.*

## What is tested

Seahorse is a training-free activation memory beside a frozen LLM. An experience is stored as a shift of
the residual stream, filed under a whitened gist key, and added back (`h ← h + α·gate·M·q`) when a later
prompt matches. The write-up makes one core claim: **broad leanings transfer selectively; facts and
premises don't.** It rests on the claims below. Each is tested on **Qwen3.5-2B and Qwen3.5-9B** (both
fp32, thinking off) with one design, one config per model and fixed seeds (README.md).

## Design (fixed; see run.py and the configs)

- **Models:** Qwen/Qwen3.5-2B (V100) and Qwen/Qwen3.5-9B (A100 80GB). Both run in fp32, so no fp16 or bf16
  path is needed. Text path only; `enable_thinking=False` everywhere.
- **Layers:** block outputs 2B {20, 21, 23}. The 9B uses the same relative depths:
  `round((l+1)·32/24) − 1` gives {27, 28, 31}.
- **Write:** one entropy-weighted pooled moment per follow-up, template tail excluded. Isolated memory per
  item, delta rule.
  - Facts store the plain shift (with − without).
  - Preferences store the **opposite** (with − counter).
  - The mechanism references are `without` (= plain) and `centroid` (with − the mean of 8 same-frame
    alternatives).
- **Key / read:** pooled_w256, with a hard threshold at the q0.95 generic match per item and layer. No read
  on the template head. Two-pass gating.
- **Doses:** α ∈ {0.5, 1, 2, 3} per layer on the raw shift.
- **Controls at α = 2:**
  - `rand`: a fixed random direction with the recall's norm, through the item's gate.
  - `swap`: the partner item's shifts under this item's keys, so its own gate decides.
  - `placebo`: "By the way, I had a coffee this morning." with the item's follow-ups.
  - `gate_on`: the threshold removed.
- **Items** (items.yaml, validated):
  - 15 leanings
  - 8 dislikes
  - 6 one-of-many
  - 12 facts (xlayer_v1's)
  - 50 unrelated prompts
- **Generation:**
  - Per prompt: 1 greedy + 4 samples (T 1, top-p 0.95, top-k 20, presence 1.5), cap 120 tokens. Every
    condition uses the same random streams.
  - Seed set 2 (4 more samples) for nomem, ctx, mem@1, mem@2.
  - Yes/no: greedy, 10 tokens, with first-token margins.
  - Unrelated: greedy + 2 samples.

## Measures

**Preference answers:**
- `j_lean` (**PRIMARY**). The judge's (Qwen3.5-9B, bf16, thinking off, `judge_prompts.yaml`) direction
  of the answer: +1 if it recommends toward the user's trait and is coherent, −1 if away and coherent,
  0 otherwise (neutral or incoherent).
- Also reported:
  - `j_dir` (direction regardless of coherence)
  - lexicon `lean` and `lean_clean` (non-loop answers, bench/score.py)
  - loop (repeated-4-gram rate ≥ 0.3)
  - judge incoherent rate and self_claim
  - mean token NLL under the no-memory model
  - length

**Fact answers:**
- `use_judge_clean` (**PRIMARY**, on the 2 use prompts): the judge says the answer uses the correct fact
  AND is coherent.
- Also reported:
  - regex `ans_tgt` / `ans_tgt_clean` / `use_tgt_clean` (target present, not a loop)
  - `mention_unclean`: the target is mentioned, but not in a clean judged use
  - confab, wrong_value, self-claim (judge and regex)
  - loops, incoherence, NLL
  - Δlog P(target) and Δspecificity at the measure prefix (teacher-forced)

**Other measures:**
- **Yes/no:** balanced accuracy from the parsed greedy answer, (accY + accN)/2. Margin gains dmY, dmN vs
  nomem.
- **Selectivity:**
  - `unrel_same`: share of unrelated answers (greedy and samples) token-identical to nomem's. When the
    gate is shut everywhere, the answer is nomem's by construction.
  - The gate's open rate on unrelated prompts.
  - KL(nomem ‖ memory) along the nomem greedy answer.
  - Contamination.
- **Logit lens:** at every block, the stored shift through the final norm + output layer.
  - Facts: target rank, z and margin vs the foils.
  - Preferences: lex_gain per reference.

## Statistics

- **Cell:** one (item, prompt).
- **Group number:** the macro mean over items of the mean over the item's prompts.
- **CIs:** 95% percentile CIs from a two-level bootstrap. Resample items, then prompts within items; 2000
  replicates, seed 7.
- **Gains:** paired per cell (memory − nomem on the same prompt and the same random streams).
- **Rows:** seed set 1 (5 answers per prompt). Seed set 2 is reported as a robustness check.
- **Also reported:** every item separately, and the share of items and of prompts moved the wrong way.

## Exclusion rule (pre-registered)

A preference item whose **ctx lean − nomem lean < 0.3** on its model is **flagged**. Flagged items are
excluded from that model's primary numbers. They are listed and reported separately (also in the
"(all)" rows), never dropped silently.

- **Lean** = `j_lean`.
- If the judge is unusable (parse rate < 0.8), the rule falls back to the lexicon `lean_clean`, and this
  is reported.

## Claims and decision rules

**C1 Selectivity.** The memory fires only where relevant; unrelated answers are unchanged.
*Holds if* every gated memory condition (mem@α, rand, swap, placebo, centroid@α, without@α) has
`unrel_same ≥ 0.95` in every group, and `gate_on` has a lower `unrel_same` than mem@2, with
non-overlapping 95% CIs in every group.
*Also reported:* KL, false-fire rate, contamination.

**C2 Leanings transfer, specifically.**
*Primary:* leanings (ctx-rule kept), `j_lean` gain vs nomem.
*Holds if* the gain is > 0 with the CI excluding 0 at **both α = 1 and α = 2**, and each control (rand,
swap, placebo, at α = 2) has a gain ≤ ⅓ of mem@2's, with the paired CI of (mem@2 − control) excluding 0.
*Reported, not decisive:* the same for dislikes and one-of-many, α = 0.5 and 3, lexicon lean_clean,
coherence.

**C3 Facts don't come out as clean use.**
*Primary:* facts, `use_judge_clean` on the use prompts.
*Holds if*, at **every** α ∈ {0.5, 1, 2, 3}:
- mem@α's `use_judge_clean` ≤ ½ × ctx's, with the CI of (ctx − mem@α) excluding 0;

and at some α ≥ 2:
- the share of answers that mention the target without a clean use (`mention_unclean`: loops, garble,
  misuse) exceeds nomem's.

**C4 No premises.**
*Primary:* facts and leanings, at α = 1 and α = 2.
*Holds if* the balanced yes/no accuracy gain vs nomem has a CI that includes 0, and the margin shift is
a general Yes/No tilt rather than a premise. A tilt means the CI of (dmY + dmN)/2 includes 0, or
|dmY + dmN| < 0.25·(|dmY| + |dmN|).

**C5 Mechanism: what you subtract decides concept vs direction.**
*Primary:* dislikes (ctx-rule kept). For each reference (opposite = mem@α, centroid, without), take the
**highest α ∈ {0.5, 1, 2} whose loop rate ≤ 10%** (matched loop rate, not matched norm). If none
qualifies, use α = 0.5 and mark it unmatched.
*Holds if*:
- the opposite's `j_lean` gain vs nomem is > 0 (CI excludes 0), and
- the centroid's and without's gains are < 0 (CIs exclude 0), i.e. dislikes flip into likes.

*Reported, not decisive:*
- the same table at every α
- for one-of-many and the 4 mechanism leanings
- lex_gain per reference from the logit lens. The concept side is expected to be larger for centroid
  than for opposite on one-of-many and dislikes. For dislikes, centroid is expected to be wrong-signed.

**C6 A fact shift is a word, only late.**
*Holds if*:
- the blocks where the facts' median logit-lens target rank is ≤ 10 all lie at relative depth ≥ 0.75
  (and there is at least one), and
- every block at relative depth ≤ 0.5 has a median rank > 100.

**C7 Scale.** Each of C1–C6 is decided separately on each model. A claim "holds at both scales" only if
it holds on both. Disagreements between models are reported as findings, not explained away.

## Checks that must pass (else the run is invalid)

These are in prep/checks.json and stop the job on failure:
- exact recall of the last delta-rule write
- the plain write equals `diag_keys.unit_writes`
- KV-cached two-pass generation equals the full-sequence two-pass reader
- two-pass equals one-pass for a single layer
- core_v1's fields equal xlayer_v1's
- the rand control matches the recall's norm
- 2B only: the fact shifts and thresholds reproduce xlayer_v1 Stage A (relative < 1e-3; expected ~1e-6)
- no non-finite logits anywhere
- the smoke check that no-memory greedy answers are finite and coherent

Batched == unbatched generation is measured and reported, not required. Batch shape can change floating
point in the last bits, and all conditions of a prompt share one batch.

## Known limitations (stated in advance)

- **The judge is the larger model of the two under test.** Its labels are validated by hand on
  `audit_sample.jsonl` (about 200 answers, stratified over models, conditions and measures). Until that
  audit is done, judge numbers are provisional, and the lexicon numbers are reported beside them.
- **New items are uncalibrated.** This covers 8 new leanings, 3 new dislikes, and the new probes of the
  one-of-many and negated items. The ctx rule is the safeguard.
- **Unrelated prompts aren't regenerated when the gate is shut at every position.** Such answers are
  equal to nomem's by construction. Open rates are logged.
