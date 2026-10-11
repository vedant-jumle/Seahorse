# Experiment 2: dreaming as scaffold removal (model-generated references, and concept × sign)

*Can the model generate the counterfactuals that writing needs, so no human writes an opposite, a set of alternatives or a "without"? And does separating concept from sign fix the dislike flip? Exact process. Written 2026-10-11; revised after review (S1 null baselines; the dreamer framing).*

---

## 1. The problem

Every memory so far was written from three hand-made texts: the experience, the same message **without** it, and its **opposite** (plus, for the centroid, eight hand-written **alternatives**). A real conversation has only the first. The paper lists this as its main design limitation. The reference also decides what is stored (C5): the opposite keeps the direction, the alternatives keep the concept, and no single reference keeps both.

Dreaming can supply the references: the model, offline, paraphrases, negates and lists alternatives for a sentence it has just seen. It costs compute after the conversation, not human input. And once the concept (from alternatives) and the sign (from a general valence direction) are separate, they can be stored separately and applied as sign × gain × concept, which is the fix the paper proposes.

## 2. What is tested

**Predictions (PREREG.md):**
- **S1.** Model-generated references produce shifts close to the hand-written ones, **beyond what the shared "with" state gives for free.** Both shifts are `with − something`, so they share a term and their cosine is inflated by construction. The rule is therefore relative to two nulls, computed the same way: `null_swap` = with − *another item's* hand-written opposite (or alternatives), and `null_frame` = with − a random sentence in the same frame with an unrelated value. S1 holds if, per item and layer, cosine(gen, hand) exceeds both nulls' cosine with hand by ≥ 0.2 on ≥ 80% of the 29 items, and the absolute cosine is ≥ 0.8. Report all three cosines. **If a null itself scores ≥ 0.8 for an item, the cosine test cannot tell for that item** (no reference could clear the margin); such items are reported as "indeterminate", not as S1 failures, and the decision falls to S2 (the behavioural replication), which doesn't share the inflation.
- **S2.** C5 replicates with generated references only: on the dislikes, the generated-opposite shift moves replies the right way (CI > 0) and the generated-centroid shift moves them the wrong way (CI < 0), at the loop-matched dose.
- **S3.** A **general valence direction**, built from generated like/hate pairs over many topics, reads the sign of each preference item from its *plain* shift (with − without) with accuracy ≥ 0.9 over the 23 signed items (15 leanings + 8 dislikes). No per-item opposite is used.
- **S4 (concept × sign).** Storing the generated-centroid concept and applying it with the sign from S3 moves the dislikes the **right** way (CI > 0), i.e. it fixes the flip without a hand-written opposite, and keeps the leanings' gain within the CI of the opposite-reference memory.
- **S5.** "Without" needs no generation: leaving the experience sentence out of the same message is mechanical. Report that explicitly; it is already the case.

## 3. Items

All 29 preference items of core_v1 (15 leanings, 8 dislikes, 6 one-of-many), with their hand-written opposite and alternatives as ground truth. The 12 facts are not used here (facts use `without`, which needs no generation).

## 4. The dreaming pass (generation)

**Dreamer model.** Qwen3.5-9B, bf16, thinking off, temperature 0 for the single-output prompts and T = 0.7 with n = 12 for list prompts, deduplicated. Everything is produced from the experience sentence alone; the dreamer never sees the hand-written references. Prompts live in `experiments/dream_refs/dream_prompts.yaml`.

**Framing, stated honestly.** The 9B dreams for the 2B. So on the 2B run the claim is "*a* model generates the references, no human does"; "the model writes its own references" is literally true only on the 9B run. Both are run: the 2B for the main grid (cheap), and the 9B end-to-end for the winning variants, which is also the confirmation run core_v1 did. A 2B-as-its-own-dreamer variant is worth one pass to see whether reference quality drops with size.

For each item's experience sentence `e`:

1. **Paraphrases** (for robustness of the shift, optional): 4 restatements of `e` with the same meaning and frame.
2. **Opposite:** *"Rewrite the sentence so the speaker states the opposite preference, keeping the same topic and sentence frame. One sentence."* Take the greedy output. Validate: same frame (the dreamer is also asked to return the frame as a template with a slot, e.g. `"By the way, I'm {X}."`), different value, no negation words unless the original had none (so "I can't stand jazz" → "I love jazz", not "I don't hate jazz").
3. **Alternatives (for the centroid):** *"List 12 other things the speaker could have said in exactly this frame, each a different value of the same kind."* Keep 8 after deduplication, excluding the original value and the opposite. Validate: each fits the frame template; none repeats the original's lexicon words.
4. **Frame template** `t(X)`: the sentence with its value replaced by a slot; used to check 2 and 3 and to build the valence pairs in §5.

**Validation against the hand-written references** (no GPU):
- lexical overlap and embedding similarity between generated and hand-written opposite/alternatives (report, not a rule)
- a human spot check of 29 opposites and 29 alternative lists (one evening's work; this is the one place a human still reads, and it is a check, not an input)

## 5. The valence direction

A general like/dislike direction, built once, not per item:

1. Take 40 topics the dreamer lists (foods, genres, activities, places, weather, …), none of them our items' values.
2. For each topic X, the dreamer writes a matched pair in a neutral frame: *"I really like X."* / *"I can't stand X."*, plus the pair in two more frames (*"X is my favourite."* / *"X is the worst."*; *"I always choose X."* / *"I always avoid X."*).
3. Run each sentence as a user turn with one neutral follow-up (*"What should I do this weekend?"*). Pool the follow-up's states as in core_v1.
4. `v_ℓ` = the mean over all pairs of (like − hate) at block ℓ, normalised. Report its logit lens (expect generic positive vs negative words, no topic).
5. **Sign of an item** = `sign(⟨shift_plain, v_ℓ⟩)` where `shift_plain` is the item's with − without shift. Report accuracy against the known sign (leanings and one-of-many are +, dislikes are −), per layer, and the margin distribution.

Also test the obvious alternative: the sign from the **generated opposite** (`⟨shift_plain, shift_opp⟩`). If both work, the valence direction is the one to keep: it is one vector for all items.

## 6. The write variants

For each preference item, at the three Qwen3.5-2B layers (20, 21, 23), shifts pooled exactly as core_v1:

| Variant | Shift |
|---|---|
| `hand:opposite`, `hand:centroid`, `without` | core_v1's, reused from its saved `prep/state.pt` (no recompute) |
| `gen:opposite` | with − generated opposite |
| `gen:centroid` | with − mean over the 8 generated alternatives |
| `gen:para` | mean over the 4 paraphrases of (with_para − without); robustness only |
| `cs:concept×sign` | `s · c_ℓ`, where `c_ℓ` = the `gen:centroid` shift and `s` = the sign from §5 (so for a dislike the concept is *subtracted*) |
| `cs:oracle` | the same with the true sign; separates "the sign was wrong" from "signed concepts don't work" |

Dose: α ∈ {0.5, 1, 2}; compare at the loop-matched α as C5 did (highest α with ≤ 10% loops).

## 7. Generation and measures

Same prompts as core_v1 for the preference items (1 related + 5 ambiguous, 1 greedy + 4 samples), same random streams, plus yes/no and the 50 unrelated prompts for the primary cells only. Judges: original + gpt-oss + Gemma-4; majority reported.

Primary: judge lean gain vs `nomem`, per group (leanings, dislikes, one-of-many). Also: logit-lens `lex_gain` per variant (the cheapest check of what each generated shift contains, before any generation), loop, incoherence, self-claim, `unrel_same`.

## 8. Decision rules

- **Scaffold removed:** S1 and S2 hold. Generated references reproduce the mechanism; the paper's limitation is closed.
- **Sign readable:** S3 holds at some layer with accuracy ≥ 0.9 and the margin's CI away from 0.
- **Concept × sign works:** S4 holds: dislikes right way with `cs:concept×sign`, leanings not worse than `gen:opposite`. If `cs:oracle` works but `cs:concept×sign` doesn't, the sign reader is the weak part. If neither works, subtracting a concept does not produce avoidance (the "away" asymmetry was wrong), and the dislike channel needs the opposite after all.

## 9. Job plan

- `experiments/dream_refs/`: `dream.py` (generation with the 9B, A100, ~30 min), `run.py --stage prep | lens | gen | score | analyze` (2B on V100; prep reuses core_v1's state file), `PREREG.md`, tests for the frame validator, the sign reader and the valence construction.
- Order: dream → validate (CPU) → prep + lens (short; S1 and S3 are decided here) → gen only for the variants that passed the lens check → score → analyze.
- Budget: generation ≈ 29 items × 6 prompts × ~8 cells × 5 ≈ 7,000 answers; cut cells using the lens results to stay under 4 h.

## 10. What this gives the project

- A memory that writes from the experience sentence and the model's own offline work: no human counterfactuals.
- A single valence vector that reads sign, which the modulator can also use as its arousal term.
- If S4 holds, the first constructive fix in the project: dislikes recalled as dislikes without a hand-written opposite.
