# dream_refs (experiment 2) pre-registration

*Written and committed before any dream_refs job ran (2026-10-11). It fixes the claims S1–S5, the generation grid,
the doses, the measures, the statistics, the exclusion rule, the construction of every reference, null and
direction, and the decision rules. The spec is [docs/dreaming/02-scaffold.md](../../docs/dreaming/02-scaffold.md);
where this file is more specific, this file is what the code implements (`refs.py`, `run.py`, `analyze.py`). Any
later change to a rule is reported as a deviation, with the pre-registered result alongside it.*

**Lens, S1 and S3 results select nothing.** The generation grid below is fixed here, in advance. The logit lens,
the S1 cosines and the S3 sign accuracies are computed in phase A and reported; no cell of phase B is added,
dropped or re-dosed because of them.

## What is tested

Every Seahorse memory so far was written from three hand-made texts: the experience, the same message
**without** it, and its **opposite** (plus, for the centroid, eight hand-written **alternatives**). A real
conversation has only the first. Experiment 2 asks whether a model, offline ("dreaming"), can write the
references instead, and whether storing the concept and the sign separately (concept × sign) fixes the dislike
flip (core_v1 C5: with the centroid or `without` reference, every dislike is recalled as a like).

## Design (fixed)

- **Model under test:** Qwen/Qwen3.5-2B, fp32, V100, text path, thinking off. Block outputs **20, 21, 23**.
- **Everything that shapes a memory or a reply is core_v1's** (`experiments/core_v1/`, config_qwen35_2b.yaml):
  the entropy-weighted pooled write per follow-up (template tail excluded), pooled_w256 keys, the q0.95
  thresholds, two-pass gating, `h ← h + α·gate·M·q` at each of the three layers, the delta rule, the sampler, the
  caps, the batching and the generator. The config's shared keys are asserted equal to core_v1's in the tests.
- **Reused, not recomputed:** core_v1's 2B `prep/state.pt`
  (`/scratch/vvjumle/seahorse_runs/core_v1_20261007_1820/qwen35_2b/prep/`, sha256 `4a3321e7…`): the calibration
  (mu, whitening, thresholds), every item's keys, and the **hand:opposite** (`opposite`), **hand:centroid**
  (`centroid`), **without** (`plain`) and placebo shifts at layers 20/21/23. Every gate is therefore core_v1's
  gate, identical for every variant of an item.
- **New shifts** are computed with core_v1's own `build_moments` (the same code path), from pseudo-items whose
  counter / alternatives are the generated or null references.
- **Items:** the 29 preference items of core_v1 (`experiments/core_v1/items.yaml`): 15 leanings, 8 dislikes,
  6 one-of-many, with their hand-written opposite and 8 alternatives as ground truth. The 12 facts are not used.

## The dreaming pass (phase A, `dream.py`, `validate.py`)

**Dreamer:** Qwen/Qwen3.5-9B, bf16, thinking off, on a full A100. Prompts: `dream_prompts.yaml`. Greedy (T = 0)
for single-output prompts; T = 0.7 (top-p 0.8, top-k 20, no presence penalty), n = 12 for list prompts. One
random stream per prompt, seeded from the prompt (`seed_of("dream_refs/<step>", prompt, 0)`).

**The dreamer sees only each item's experience sentence**, and its own earlier outputs for that item (its frame,
value and key words). It never sees a hand-written opposite, alternative, lexicon or prompt; the examples inside
the prompts are about things no item is about. (A test asserts no hand reference appears in any prompt.)

Per item, in this order:
1. **Frame** (single): JSON `{value, frame, key_words, category}` (category from prompts v2, see Amendments):
   the value (the whole part of the sentence that is particular to this user, trait plus detail, not the detail
   alone), the sentence with the value replaced by `{X}`, the 1–3 words that name the value, and what kind of
   thing the value is. **Valid** iff exactly one slot, frame(value) reproduces the
   experience (case, whitespace and final punctuation ignored), and every key word is inside the value. The
   greedy output if valid, else the first valid of 12 T = 0.7 samples, else none.
2. **Opposite** (single): the spec's prompt (*"Rewrite the sentence so the speaker states the opposite
   preference, keeping the same topic and sentence frame. One sentence."*) plus *do not add negation words unless
   the original has them*. **Hard checks:** non-empty; not the experience; if it fits the frame, a different
   value; **no new negation** (no more negation words than the experience; negation = not, no, never, none,
   nobody, nothing, neither, nor, cannot, any *n't*). The greedy output if it passes, else the first of 12
   samples that passes, else the greedy output flagged as failing. **Soft flags** (reported, never used):
   same frame (fits the frame or shares ≥ half its fixed words), keeps the topic (a key word), negation kept, one
   sentence. The frame is soft for the opposite because a sentiment flip ("I can't stand jazz music." → "I love
   jazz music.") changes the frame, not the value.
3. **Alternatives** (list; 12 samples, each asked for 12, given the sentence, its frame and its category, and
   told to stay within the category): candidates are
   deduplicated by value and ranked by how many samples propose them, then by first appearance. **Hard checks:**
   fits the frame; value ≠ the experience's and ≠ the opposite's; the value contains no key word ("the
   original's lexicon words"), except key-word stems that are the category (present in ≥ half of all
   candidates' values, and never every key stem, so a single key word is always checked); not the experience or
   the opposite; no new negation. **Keep the 8 best-ranked
   that pass**; if fewer pass, fill with the best-ranked that fail only the frame check (flagged).
4. **Paraphrases** (list; 12 samples, each asked for 4): not the experience, no new negation, ≤ 40 words; those
   keeping every key word first, then by count. Keep 4. Used only for `gen:para` (lens and cosine).
5. **Frame template** = step 1; it drives steps 3 and 6.
6. **null_frame values** (single, JSON list of 8 fillers for the frame "from a completely different area of
   life", not the item's category): greedy list first, then the samples by count. Hard: not the value, no key word, not one of the kept
   alternatives' values, no new negation. Up to 8.

Every generated reference is written before the follow-up as a sentence ending in . ! or ? (a full stop is added
if missing, as every hand-written reference has one).

**Validation** (`validate.py`, CPU, no model): all of the above, re-derived from the raw outputs (the frame used
in dream.py is checked to match), plus `validation.json` and **`spotcheck.md`**: per item the experience, the frame,
hand vs generated opposite, hand vs generated alternatives, paraphrases, the nulls, lexical overlap with the hand
references (report only) and every flag. Flags are reported; **no item is dropped or regenerated because of a
flag.** The job stops (and the chain with it) only on structural failures: an item with no alternative, an empty
opposite, or fewer than 20 valence topics.

## The S1 null references (fixed)

Both shifts compared by S1 are `with − something`, so their cosine is inflated by the shared `with` term. Two
nulls are computed the same way:
- **null_swap** = with − the **partner item's hand-written** opposite (for the opposite comparison) / mean of
  the partner's 8 hand-written alternatives (for the centroid comparison). The partner is core_v1's fixed `swap`
  permutation (within the item's group, no item its own partner).
- **null_frame** = with − the item's (dreamer's) **frame filled with an unrelated value** (step 6): the first
  value for the opposite comparison, the mean over up to 8 for the centroid comparison. Deterministic (greedy,
  then fixed-seed samples). If the item has no valid frame, null_frame does not exist and the item is
  indeterminate in S1.

## The valence direction and the sign readers (fixed)

- **Topics:** the dreamer lists 50 "area | thing" lines per sample (12 samples). Deduplicated, ranked by count;
  a topic is **excluded** if any of its content words is a content word of any item's experience or of any
  dreamer key word, or if it contains any lexicon entry of any item. Then 40 are taken round-robin over areas
  (areas by total count), so no area dominates. If fewer than 40 survive, all are used and the shortfall is
  reported (fewer than 20 stops the chain).
- **Pairs:** each topic X in the spec's three frames: "I really like X." / "I can't stand X."; "X is my
  favourite." / "X is the worst."; "I always choose X." / "I always avoid X." (filled mechanically; the frames
  are the spec's verbatim).
- **States:** each sentence as a user turn followed by the neutral follow-up "What should I do this weekend?",
  pooled over the follow-up exactly as core_v1 (shared suffix, entropy weights of the follow-up-only run,
  template tail excluded).
- **v_ℓ** = the mean over all pairs of (like − hate) at block ℓ, normalised. Its logit lens is reported (expect
  generic positive vs negative words, no topic).
- **Valence reader (decides S3):** the sign of an item at layer ℓ = sign(⟨plain_ℓ, v_ℓ⟩), where plain_ℓ is the
  item's `without` shift (with − without, core_v1's) averaged over its 3 moments. The margin is the cosine.
- **Generated-opposite reader (reported):** sign(⟨plain_ℓ, gen:opposite_ℓ⟩), as the spec asks. Both vectors
  contain the same `with` term, so this reader is expected to say + for almost every item; it is not a
  competitor for S3.
- **True signs:** leanings +, one-of-many +, dislikes −.

## The write variants (fixed)

At layers 20, 21, 23, per moment (follow-up), pooled as core_v1:

| Variant | Shift | Generated? |
|---|---|---|
| `hand:opposite` | core_v1's opposite shift (= core_v1's mem@α) | reused |
| `hand:centroid`, `without` | core_v1's | reused; lens / S1 / S3 only |
| `gen:opposite` | with − generated opposite | yes |
| `gen:centroid` | with − mean of the 8 generated alternatives | yes |
| `gen:para` | mean over the 4 paraphrases of (with_para − without) | **lens and cosine only** |
| `cs:concept_x_sign` | s_ℓ · gen:centroid_ℓ, s_ℓ = the valence reader's sign for the item at layer ℓ (a dislike read as − has its concept subtracted) | yes |
| `cs:oracle` | the true sign · gen:centroid_ℓ | yes |

Each variant is stored as its own delta-rule memory under the item's core_v1 keys; the delta rule is linear, so
a signed memory is the sign times the concept memory (tested).

## The generation grid (fixed; phase B)

Per item (29), at layers 20/21/23, through the item's own gate:
- `nomem`, `ctx` (the experience in the prompt)
- at **α ∈ {0.5, 1, 2}** per layer: `hand:opposite`, `gen:opposite`, `gen:centroid`, `cs:concept_x_sign`,
  `cs:oracle`

That is 17 conditions. **`without` and `hand:centroid` are not generated.** S2 is defined on the generated
references only and needs neither. Reusing core_v1's generated `without@α` / `centroid@α` answers is not
legitimate for paired gains: this run regenerates `nomem` in a different batch composition (the same random
streams, but last-bit float differences are possible), and mixing cells across runs would break the pairing.
core_v1's own loop-matched numbers for those references are printed beside the tables as context only.

- **Prompts:** core_v1's: 1 related + 5 ambiguous per item; 1 greedy + 4 samples (T 1, top-p 0.95, top-k 20,
  presence 1.5), cap 120 tokens; the **same random streams as core_v1** (seed tag "core_v1"), every condition
  of a prompt in one batch (16 non-ctx conditions × 5 + 4 seed-2 conditions × 4 = 96 rows), ctx in its own.
- **Seed set 2** (4 more samples): nomem, ctx, gen:opposite@1, gen:centroid@1, cs:concept_x_sign@1.
- **Yes/no** (balanced probes, greedy, first-token margins) and the **50 unrelated prompts** (greedy + 2
  samples; gated rows are not regenerated when the gate is shut everywhere): **nomem, ctx and the α = 1 cells
  only** (the primary cells).
- Generation is split by item over V100 jobs of ≤ 4 h each.

## Measures

- **`j_lean` (PRIMARY):** the judge's direction of the answer: +1 a coherent recommendation toward the user's
  trait, −1 away, 0 neutral or incoherent. core_v1's rubric (`experiments/core_v1/judge_prompts.yaml`).
- **Judges:** the original judge (Qwen3.5-9B, bf16, thinking off; core_v1's) is decisive, as in core_v1. The
  panel (gpt-oss and Gemma-4 via Ollama, `experiments/core_v1/panel_judge.py`, the tooling in `tools/audit/`
  and core_v1's panel) re-judges every answer; the **majority of the three** (a tie goes to the original's label)
  is reported beside every decision. A claim whose verdict differs between the original judge and the majority
  is reported as **judge-sensitive** (as core_v1's C2). Llama-3.1 is not used (it failed core_v1's sanity check).
- **Also reported:** `j_dir`, lexicon `lean_clean`, loop rate (repeated-4-gram ≥ 0.3), judged incoherence,
  self-claim, NLL under the no-memory model, length; per item and per group; `unrel_same`, KL and gate false-fire
  for the α = 1 cells; balanced yes/no for the α = 1 cells; logit-lens `lex_gain` of every variant at the read
  layers.

## Statistics

- **Cell:** one (item, prompt). **Group number:** macro mean over items of the mean over the item's prompts.
- **CIs:** 95% percentile CIs from the **two-level bootstrap** (items, then prompts within items), **2000
  replicates, seed 7** (`seahorse.stats.boot_ci`).
- **Gains:** paired per cell (condition − nomem, same prompt, same random streams); seed set 1 (5 answers per
  prompt) for every primary number; seed set 2 as a robustness check.
- **S3 margin CI:** the same bootstrap over items (one value per item).
- **Loop-matched dose:** for each variant and group, the **highest α ∈ {0.5, 1, 2} whose loop rate ≤ 10%**; if
  none qualifies, α = 0.5, marked unmatched (core_v1 C5).

## Exclusion rule

A preference item whose **ctx j_lean − nomem j_lean < 0.3** in this run is **flagged** and excluded from the
primary numbers of S2 and S4; flagged items are listed and reported separately, never dropped silently. If the
original judge's parse rate is < 0.8, the rule falls back to the lexicon `lean_clean` (reported). Under the
panel judges and the majority, the item set is the original judge's. S1 and S3 use all items (they involve no
generation).

## Claims and decision rules

**S1. Generated references produce shifts close to the hand-written ones, beyond what the shared `with` state
gives for free.** Per reference type (opposite: gen:opposite vs hand:opposite; centroid: gen:centroid vs
hand:centroid), item and layer, from the moment-mean shifts: c_gen = cos(gen, hand), c_swap = cos(null_swap,
hand), c_frame = cos(null_frame, hand).
- The item **passes** at that layer iff c_gen ≥ 0.8 and c_gen − max(c_swap, c_frame) ≥ 0.2.
- The item is **indeterminate** at that layer if max(c_swap, c_frame) ≥ 0.8 (no reference could clear the
  margin) or a null is missing. Indeterminate items are reported, **not counted as failures**.
- A layer **holds** if ≥ 80% of its determinate items pass; it is indeterminate if fewer than 15 of the 29 items
  are determinate.
- A reference type **holds** if every layer (20, 21, 23) holds; **fails** if any determinate layer fails;
  otherwise it is indeterminate.
- **S1 holds** iff both reference types hold; fails if either fails; otherwise it is **indeterminate and the
  decision falls to S2** (which doesn't share the inflation). All three cosines are reported per item and layer.

**S2. C5 replicates with generated references only.** Dislikes (ctx-rule kept), each variant at its own
loop-matched α: *holds if* `gen:opposite`'s j_lean gain vs nomem is > 0 with the CI excluding 0, and
`gen:centroid`'s is < 0 with the CI excluding 0.

**S3. A general valence direction reads the sign of each preference.** *Holds if* at some layer ℓ ∈ {20, 21,
23} the valence reader's accuracy over the **23 signed items** (15 leanings + 8 dislikes) is **≥ 0.9** and the
95% CI of the mean signed margin (true sign × cos(plain_ℓ, v_ℓ)) is above 0. No per-item opposite is used. The
same table over all 29 items and for the generated-opposite reader is reported, not decisive.

**S4. Concept × sign.** `cs:concept_x_sign`, at its loop-matched α: *holds if* (a) on the dislikes (ctx-rule
kept) its j_lean gain vs nomem is > 0 with the CI excluding 0 (the flip is fixed without a hand-written
opposite), **and** (b) on the leanings its gain is ≥ the lower 95% CI bound of `gen:opposite`'s gain on the
leanings (each at its own loop-matched α): not worse than the opposite-reference memory. `cs:oracle` is read as
in the spec: if `cs:oracle` moves the dislikes the right way (CI > 0) and `cs:concept_x_sign` doesn't, the sign
reader is the weak part; if neither does, subtracting a concept does not produce avoidance and the dislike channel
needs the opposite after all. `hand:opposite` is reported beside them.

**S5. `without` needs no generation.** It is the same follow-up with the experience left out: mechanical. Stated
and reported; not a test.

**Decisions (02-scaffold.md §8):**
- **Scaffold removed:** S1 and S2 hold (an indeterminate S1 defers to S2).
- **Sign readable:** S3 holds.
- **Concept × sign works:** S4 holds.

## Phasing

- **Phase A** (submitted now): dream (A100) → validate (CPU) → prep + lens (V100), chained with afterok.
  Produces `spotcheck.md` and `report_A.md` (S1, S3, S5, the logit lens). **Then the run pauses.**
- **Phase B** (submitted only after the user has spot-checked the references and read S1/S3): gen → score
  (coherence + original judge) → panel (gpt-oss, Gemma-4) → analyze (S2, S4, every table). The grid is the one
  above whatever phase A shows. If the spot check finds the references broken, that is reported as a finding;
  any regeneration would be a new, separately reported run.

## Checks that must pass (else the run is invalid; they stop the job)

- core_v1's state holds every item with its plain / opposite / centroid / placebo shifts at layers 20/21/23.
- One item per group recomputed with core_v1's `build_moments` and the restored calibration reproduces the
  stored shifts (all four) and keys: **relative < 1e-3** (as core_v1 checked itself against xlayer_v1).
- The `without` shift recomputed inside every generated-reference pseudo-item equals the stored one
  (relative < 1e-3).
- Validation: every item has ≥ 1 alternative and a non-empty opposite; ≥ 20 valence topics; mock references are
  refused outside `--tiny`.
- No non-finite logits anywhere (core_v1's generator raises).
- Every job runs the test suite first.

Reported, not required: hand:* `lex_gain` vs core_v1's own lens file; the share of nomem / hand:opposite answers
token-identical to core_v1's (same prompts and streams, different batch composition).

## Known limitations (stated in advance)

- **The 9B dreams for the 2B.** On this run the claim is "*a* model writes the references, no human does". "The
  model writes its own references" is literally true only on a later 9B end-to-end run (the confirmation run,
  not part of this one). A 2B-as-its-own-dreamer pass (`dream.py --config` with the 2B) is optional and not
  part of this run.
- **The dreamer and the original judge are the same model** (Qwen3.5-9B). The judge sees only the trait and the
  answer, never a reference, but a shared bias can't be excluded; the cross-family panel is the check.
- **S1 measures closeness to one author's hand-written references**, not correctness; a generated opposite can
  be a valid opposite and still differ (e.g. for one-of-many items the hand opposite is another value, "I love
  Mexican food.", while the spec's prompt asks for the opposite preference).
- **Cosine inflation** by the shared `with` term is handled by the nulls and the indeterminate rule, not removed.
- **The valence direction** is built with one neutral follow-up; the items' own follow-ups differ.
- **The generated-opposite sign reader is degenerate by construction** (shared `with` term); it is reported
  because the spec asks for it.
- **Item calibration** is core_v1's; the ctx rule is the safeguard, re-applied on this run's own nomem / ctx.
- **No human labels** on the generated answers; the spot check of the references is the one place a human reads,
  and it is a check, not an input.

## Deviations from the spec, decided in advance

- **No lens-based cell selection** (spec §9 "gen only for the variants that passed the lens check", "cut cells
  using the lens results"): replaced by the fixed grid above (user decision).
- **`gen:para`** is lens and cosine only, not generated (user decision; the spec marks it optional).
- **Shift layers:** core_v1's `state.pt` holds shifts at the three read layers only (not every block), so the
  logit lens and S1/S3 are at layers 20/21/23.
- **Embedding similarity** between generated and hand references (spec §4, "report, not a rule") is not
  computed separately: the S1 shift cosines are the model-based similarity; lexical overlap is reported.
- **Fallbacks:** the spec says "take the greedy output"; here the greedy output is taken unless it fails a hard
  check, then the first passing fixed-seed sample (recorded per item).
- **The frame is given to the alternatives and null-value prompts** (the dreamer's own output, so everything is
  still produced from the experience alone).
- **Valence pairs** are the spec's frames filled mechanically with the dreamer's topics; the topic exclusion
  also removes any item lexicon entry (stricter than "none of our items' values").
- **cs:concept_x_sign** applies the sign per layer (s_ℓ read at layer ℓ).

## Amendments (dated; made before any outcome)

**2026-10-12, after the first dream job (952002) and before any S1, S3 or generation result existed.** The
validate job (952003) stopped on two items with no alternatives; prep + lens (952004) never ran. Reading the
dreamer's raw outputs showed one validator bug and two dreamer-prompt flaws. All three fixes are generic (no
item-specific rule, nothing tuned toward the hand references), and the dream, validate and prep + lens jobs are
rerun from scratch in a new run directory. The first run (`dream_refs_20261011_2331`) is kept, not used.

1. **Validator bug (key-word check on alternatives).** The check split a multi-word key phrase into words, so the
   category noun counted as a key word: for japanese_food ("Japanese food") every alternative ("I love Italian
   food.") was rejected for "food", and likewise horror ("movies"). Items whose key words include a category word
   (hates_jazz "music", hates_sweets "food", …) were silently restricted to alternatives without it. Fix: a
   key-word stem present in ≥ half of all candidates' values is the category and is not held against a candidate;
   never every key stem, so a single key word ("jazz") is always checked (`refs.category_stems`, tested).
2. **Frame left the trait outside the slot (prompts v1 → v2).** The dreamer chose frames such as "I'm a huge
   bookworm and read {X}." (value "every day"), "I'm an early riser and love getting up at {X}." ("dawn"),
   "I have celiac disease, so I can't eat {X}." ("gluten"), "… so I have to watch my {X}." ("sugar"). Their
   alternatives vary the detail only, so the trait is in every alternative and the generated centroid would
   subtract the very concept it is meant to keep. v2 asks for the whole distinctive part as the value (trait plus
   detail), with the test "a different user, with a different value, must be able to say the frame truthfully",
   and a generic example (a birdwatcher) about no item.
3. **"Same kind" read loosely (prompts v1 → v2).** For frames like "I really can't stand {X}." the alternatives
   left the category (hates_spicy → "loud noises", "sticky floors"; tight_budget → "a strict diet", "a packed
   schedule"; prefers_quiet → a mix of weather, food and cities). v2 has the dreamer name a CATEGORY in the frame
   step; the alternatives prompt is tied to it, and the null-value prompt excludes it.

The S1 rule, the nulls' construction, the grid and every decision rule are unchanged.
