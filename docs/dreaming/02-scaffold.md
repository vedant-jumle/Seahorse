# Experiment 2: dreaming as scaffold removal (model-generated references, and concept × sign)

*Can the model generate the counterfactuals that writing needs, so no human writes an opposite, a set of alternatives or a "without"? And does separating concept from sign fix the dislike flip? Exact process. Written 2026-10-11; revised after review (S1 null baselines; the dreamer framing). **Run 2026-10-15 to 16: see "Results" after §1. The spec text from §2 on is kept as written, with dated notes where the run changed it (final v3 prompts, the PrefEval addendum).***

---

## 1. The problem

Every memory so far was written from three hand-made texts: the experience, the same message **without** it, and its **opposite** (plus, for the centroid, eight hand-written **alternatives**). A real conversation has only the first. The paper lists this as its main design limitation. The reference also decides what is stored (C5): the opposite keeps the direction, the alternatives keep the concept, and no single reference keeps both.

Dreaming can supply the references: the model, offline, paraphrases, negates and lists alternatives for a sentence it has just seen. It costs compute after the conversation, not human input. And once the concept (from alternatives) and the sign (from a general valence direction) are separate, they can be stored separately and applied as sign × gain × concept, which is the fix the paper proposes.

## Results (2026-10-15 to 16)

*Everything below comes from the report files in `results/dream_refs_20261015_1545/` (the 29 hand-written items) and `results/dream_refs_pe_20261015_1603/` (PrefEval); both are local and untracked. Reports: [phase A](../../results/dream_refs_20261015_1545/report_A.md), [phase B](../../results/dream_refs_20261015_1545/qwen35_2b/analysis/report.md), [spot check](../../results/dream_refs_20261015_1545/spotcheck.md); PrefEval [phase A](../../results/dream_refs_pe_20261015_1603/report_A.md), [phase B](../../results/dream_refs_pe_20261015_1603/qwen35_2b/analysis/report.md), [spot check](../../results/dream_refs_pe_20261015_1603/spotcheck.md). Pre-registration: `experiments/dream_refs/PREREG.md` (Amendment 2 = the final v3 prompts; the PrefEval addendum; a 2026-10-15 note recording the approval of phase B). Dreamer: Qwen3.5-9B (bf16, thinking off); the memory runs on Qwen3.5-2B (fp32, layers 20, 21, 23). Gains are judged lean gain against no memory at the loop-matched dose, original judge unless said, with 95% intervals in brackets.*

**The short version.**
1. **A model can write its own references well enough for the paper's C5 to reproduce.** Generated-opposite memories move dislikes the right way (+0.19) almost as far as the hand-written opposite (+0.23); generated-centroid memories move them the wrong way (-0.39), as the hand-written centroid did. No human wrote a reference. S2 holds under every judge.
2. **But the references are not close copies of the hand-written ones, and the sign cannot yet be read automatically.** S1 (cosine to the hand-written shifts) and S3 (a valence direction reads the sign) both fail. So the formal decision "scaffold removed" is false, only because S1 failed; behaviourally it is removed.
3. **With the true sign, concept × sign is the best leaning memory so far** (leanings +0.27, dislikes +0.39). With the automatic sign it is not (leanings -0.09). The sign reader is the weak part.
4. **On PrefEval (200 items, no hand-written references at all), the generated opposite helps modestly under every judge** (P1 holds); concept × sign fixing the avoid subset (P2) is judge-sensitive.

### Phase A: the references and the two lens checks

The dreamer saw only each experience sentence. Validator outcome on the 29 items (`validate/validation.json`): all 29 frames valid, all 29 opposites passed, 28 of 29 had the full 8 alternatives, **one item dreaming-failed** (`has_young_kids`: fewer than 4 valid alternatives; it is excluded from the three variants that need them, as Amendment 2 says).

**The v3 prompt fix mostly did not take.** The v2 run had two defects (a category like "number of young children" that made every alternative the same trait; frames that kept a consequence clause such as "so I can't eat gluten"). v3 changed the prompts to ask for the kind of thing the trait is and to leave consequence clauses out of the frame. What the spot check shows:
- `has_young_kids` still dream-failed.
- **Consequence clauses are still kept in several opposites and frames.** Celiac: "I have celiac disease, so I can't eat gluten." → opposite "I have celiac disease, so I **can** eat gluten." (the hand-written opposite is "I eat bread and pasta every single day"). Diabetes: "…so I **love** my sugar." The alternatives for celiac are near-synonyms in the same frame ("wheat allergy", "ulcerative colitis… so I can't eat gluten").
- On PrefEval the same: "I avoid virtual reality experiences due to motion sickness issues." → the opposite "I enjoy virtual reality experiences due to motion sickness issues." (the reason clause is left in, which makes no sense).
- 12 of 29 items carry a validator flag; 21 of 29 opposites keep the topic. These are reported, never used to drop an item.
- The spot check was a user review of a summary with examples, not of every item; the known failures were acknowledged before phase B was approved (PREREG note, 2026-10-15).

**S1 fails.** The test: per item and layer, the generated shift's cosine with the hand-written one is at least 0.8 and beats both nulls (a swapped item's reference; a random same-frame sentence) by 0.2.

| Reference | Items passing (per layer) | Verdict |
|---|---|---|
| opposite | 10 of 29 (0.34), the same at layers 20, 21, 23 | fails (needs 0.80) |
| centroid | 12 to 13 of 25 determinate (0.48 to 0.52; 3 indeterminate, 1 dreaming-failed) | fails |

The pattern is clear in the per-item table: **7 of the 8 dislike opposites pass** (hates_jazz, hates_horror, hates_spicy, hates_hiking, hates_seafood, hates_beach, hates_sweets: cosine 0.82 to 1.00 with the hand-written opposite; `no_alcohol` fails at 0.07 to 0.11), as do loves_hiking, prefers_quiet and loves_cold among the leanings. The leanings with a diet or condition (vegan, lactose intolerant, celiac, diabetes) and the one-of-many items do not, probably because the dreamer's opposite is a different sentence from ours ("By the way, I'm a meat-eater." vs "By the way, I love eating meat."). Cosine is a strict test: two different but fine opposites do not have to have the same shift.

**S3 fails.** A valence direction (the mean of like minus hate over 120 generated pairs: 40 topics × 3 frames) is used to read the sign of each of the 23 signed items from its plain shift (with minus without). Rule: accuracy at least 0.9 at some layer.

| Layer | Accuracy (of 23) | Mean signed margin | Leanings read wrong (of 15) | Dislikes read right (of 8) |
|---|---|---|---|---|
| 20 | 0.435 | +0.016 [-0.039, +0.076] | 13 | 8 |
| 21 | 0.522 | +0.017 [-0.039, +0.078] | 11 | 8 |
| 23 | 0.522 | +0.023 [-0.029, +0.076] | 11 | 8 |

(Chance on 23 items is 0.5.) The direction itself is healthy (every generated pair agrees with it, mean cosine 0.75 to 0.76), and **all 8 dislikes are read correctly as negative** (cosine -0.13 to -0.24). The failures are almost all leanings: **restriction-type leanings are read as negative.** Vegetarian (-0.08), vegan (-0.14 / -0.15), lactose intolerant (-0.18), celiac (-0.15 to -0.17), diabetes (-0.12 to -0.14), tight budget (-0.10 to -0.13), has_young_kids (-0.10 to -0.12) all get a negative sign although the user *has* them. Only the clear "likes" (loves_hiking, bookworm, the one-of-many items) tend to read positive. The generated-opposite reader (sign of ⟨plain, gen:opposite⟩) is no better (0.65) and is expected to say + for everything.

**The insight: "avoid" is not "dislike", and the sign depends on the stored concept.** "I'm vegan" is a statement of restriction ("avoid meat, dairy"); a valence direction built from "I love X / I can't stand X" pairs reads restriction as negative, because restriction words (avoid, can't, no) live on the "can't stand" side. What the memory needs is the sign **relative to the stored concept**: for a vegan the concept is "animal products", and the sign is "away from it" (negative), while the leaning *is* the diet. The reader as built has no way to tell "the user loves a thing" from "the user's life is organised around not having a thing". A sign reader that knows the concept (reading the sign of the preference toward the concept, not of the sentence) is the open piece.

### Phase B: behaviour on the 29 items (loop-matched, original judge)

Dislikes = 7 items (the ctx rule flagged `hates_horror` and dropped it from primary numbers: the experience in context did not produce the expected lean). `has_young_kids` is excluded from the three variants that need alternatives (14 leanings). `α` is the highest dose with at most 10% loops.

| Group | Variant | α | Judged lean gain | Items pushed the wrong way |
|---|---|---|---|---|
| leanings | hand:opposite | 1 | +0.204 [+0.038, +0.376] | 0.33 |
| leanings | **gen:opposite** | 1 | **+0.167** [+0.022, +0.327] | 0.27 |
| leanings | gen:centroid | 1 | +0.271 [+0.098, +0.450] | 0.21 |
| leanings | cs:concept × sign (automatic sign) | 2 | **-0.093** [-0.290, +0.093] | 0.57 |
| leanings | **cs:oracle** (true sign) | 1 | **+0.271** [+0.098, +0.450] | 0.21 |
| dislikes | hand:opposite | 1 | +0.229 [+0.048, +0.438] | 0.00 |
| dislikes | **gen:opposite** | 1 | **+0.190** [+0.010, +0.395] | 0.29 |
| dislikes | **gen:centroid** | 1 | **-0.386** [-0.552, -0.214] | **1.00** |
| dislikes | cs:concept × sign (automatic) | 2 | +0.386 [+0.186, +0.590] | 0.00 |
| dislikes | **cs:oracle** | 2 | **+0.386** [+0.186, +0.590] | 0.00 |
| one of many | gen:opposite | 1 | +0.067 [-0.044, +0.178] | 0.17 |
| one of many | gen:centroid / cs:oracle | 0.5 | +0.150 [+0.022, +0.300] | 0.00 |
| one of many | cs:concept × sign | 1 | +0.033 [-0.056, +0.117] | 0.17 |

Context: the experience in the prompt gives +0.72 (leanings), +0.77 (dislikes), +0.71 (one of many). Unrelated prompts unchanged 0.984 to 0.997 for every cell at α = 1. Yes/no: no cell raises balanced accuracy beyond noise (as before).

- **S2 holds, under every judge** (original, gpt-oss, Gemma-4, majority): generated opposite > 0 and generated centroid < 0 on the dislikes. **This reproduces the paper's C5 with no human-written references.** On the dislikes the generated centroid pushes all 7 the wrong way, as the hand-written one did in core_v1 (-0.38 on the 2B).
- **S4 fails** (the rule: concept × sign with the automatic sign moves dislikes the right way and does not lose on leanings). The dislikes pass (+0.386, all 7 the right way) but only at α = 2 (at α = 1 it is +0.148 [-0.067, +0.362]), where concept × sign is clean (0 loops) and the leanings fail (-0.093; 57% of items wrong). The report's reading: **cs:oracle works and cs:concept × sign does not, so the sign reader is the weak part.** (The dislike sign is read correctly on all 8 items, so the dislike gain is the same for both; the leaning loss comes from the 11 leanings given the wrong sign, mostly restrictions and conditions.)
- **cs:oracle is the best leaning memory so far** (+0.27 at α = 1, 21% of items wrong) and the best dislike memory (+0.39 at α = 2). Note that for leanings it is the same shift as the unsigned centroid, because the true sign of a leaning is +; the gain over the opposite (+0.17 to +0.20) is within the intervals.
- **One-of-many items** (countries, genres) stay weak in every variant, as in core_v1.
- **Formal decisions** (PREREG): scaffold removed = **false** (S1 fails; S2 holds); sign readable = false; concept × sign works = false.

### PrefEval: 200 preferences with no hand-written references

*PrefEval (Zhao et al., ICLR 2025), the explicit split. 200 items stratified by topic × polarity (132 avoid, 43 mixed, 19 like, 6 other; the polarity label is lexical and imperfect, "mixed" holds many "avoid X, prefer Y"). Each question is written so that a generic answer violates the preference; the dreamer saw only the preference sentence. Primary measure: preference-following accuracy (an answer is helpful and does not violate it), PrefEval's own judge criteria, judged by the original judge, gpt-oss and Gemma-4. The follow-ups that the memory is keyed on were also dreamed, from the preference alone.*

**Phase A.** 6 of 200 items dreaming-failed (centroid/alternatives); no opposite or follow-up failures. The gate opens on 96% of the test questions (97% "open at any position"; 3.9% false fire on unrelated prompts). The automatic sign (the valence direction from the 29-item run) reads **all 132 avoid items as negative** at layers 20 and 21 (right) **but also 18 of 19 like items at layer 21 (right for 1 of 19)**: it calls nearly everything negative.

**Phase B**, accuracy gain over no memory at α = 1 (the experience in context is +0.43; no-memory accuracy 0.10 under the original judge, 0.26 gpt-oss, 0.46 Gemma-4, 0.28 majority, because the judges differ in strictness):

| Subset | Variant | original | gpt-oss | Gemma-4 | majority |
|---|---|---|---|---|---|
| all (200) | **gen:opposite** | **+0.031** [+0.011, +0.051] | +0.059 [+0.030, +0.091] | +0.097 [+0.062, +0.130] | +0.069 [+0.039, +0.099] |
| all | gen:centroid | +0.002 | **-0.082** [-0.123, -0.043] | -0.095 [-0.141, -0.049] | -0.088 [-0.130, -0.045] |
| all | cs:concept × sign | +0.031 [+0.005, +0.058] | +0.051 | +0.096 | +0.073 |
| avoid (132) | gen:opposite | +0.038 [+0.012, +0.064] | +0.047 | +0.124 | +0.065 |
| avoid | gen:centroid | +0.001 [-0.035, +0.038] | **-0.131** [-0.183, -0.078] | -0.126 [-0.186, -0.066] | -0.126 [-0.182, -0.074] |
| avoid | **cs:concept × sign** | **+0.049** [+0.018, +0.082] | +0.049 [+0.005, +0.095] | +0.126 [+0.077, +0.175] | +0.086 [+0.040, +0.135] |
| like (19) | cs:concept × sign | -0.075 [-0.200, +0.025] | 0.000 | +0.050 | -0.025 |

At α = 2 the generated opposite gains up to +0.136 [+0.090, +0.185] (gpt-oss), +0.164 (Gemma-4), +0.050 (original), with incoherence rising (original judge, all items: 36% at α = 2 against 27% at α = 1 and 23% at α = 0.5).

- **P1 holds, under every judge.** The generated opposite improves preference-following modestly: +0.03 to +0.10 at α = 1 depending on the judge, +0.05 to +0.16 at α = 2. The experience in context gives +0.43, so at α = 1 the memory closes roughly 7% (original judge) to 23% (Gemma-4) of the gap to the experience in context. It is small but robust.
- **P2 is judge-sensitive.** Rule: concept × sign gains on the avoid subset and the unsigned centroid loses. Concept × sign gains under every judge. The unsigned centroid loses under gpt-oss, Gemma-4 and the majority, but **under the original judge it is flat** (+0.001), so the rule fails there; the original judge is decisive by the pre-registration (its parse rate is 0.9995), so the headline is "P2 fails, judge-sensitive". The original judge is the strictest and has the lowest baseline (0.10), which may leave less room to see a drop.
- **Concept × sign hurts the "like" subset, as predicted:** under the original judge -0.075 [-0.20, +0.025] on the 19 like items (not significant; the intervals are wide), because the reader gives them a negative sign. That is the sign-reader problem again, at scale.
- Avoid is where it works only because nearly everything is read as negative. So the PrefEval "fix" is not evidence that the sign is understood.
- Other: the gpt-oss judge parsed 83% of its labels (the original judge 99.95%, Gemma-4 and the majority 100%). The acknowledgement and hallucination rates stay near 4% to 6% for every memory against 68% / 49% for the experience in context (the memory changes the answer without the model saying it was told).

### What it changes

- **The scaffold is removed behaviourally.** The paper's biggest design limitation (hand-written references) no longer blocks the leaning and dislike results: a model's own generated references give the same effects on our items (S2) and a small, robust gain on a benchmark nobody tuned the references to (P1). The 9B dreamed for the 2B, so "a model writes the references" is shown, but "the same model writes them for itself" is not (no result in these reports).
- **The open piece is the sign reader,** not the references. With the true sign, concept × sign gives the best leaning and dislike memories we have. The sign is not "positive vs negative sentiment of the sentence" but the sign of the preference *relative to the stored concept* (a restriction is a leaning *toward* a diet and *away* from a food). Next: a reader that works on the concept (for example one built per concept from the generated alternatives), or a sign dreamed offline from the experience and stored with it.
- **The references are not copies.** S1 failing says the dreamed reference does not reproduce the hand-written shift; it does not say the reference is wrong. Future work should not use "cosine to hand" as a target.
- **Known limitations:** the 9B is the dreamer and the judge, a cross-family panel checks the PrefEval labels; the polarity label is lexical; one model size; answers are capped at 200 tokens in every condition.

---

## 2. What is tested

**Note (2026-10-19): the spec below is kept as written. The run changed three things.** (1) The dreamer prompts are the final **v3** (Amendment 2): the category is "the kind of thing the trait is, never a number, amount or degree", and a clause giving a consequence or reason is left out of the frame. (2) An item whose opposite or alternatives fail a hard check is "dreaming failed": excluded, with a count, from the variants that need that reference (the run continues). (3) The PrefEval addendum (§11) was added after the 29-item run was designed.

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

**Note (2026-10-19): what it gave.** The first bullet came true (S2 and P1: no human counterfactuals needed for the effects we measure). The second did not: the valence vector does not read sign (S3 fails), so it cannot yet serve as an arousal term. The third is half true: concept × sign with the true sign is the first constructive fix for dislikes (+0.39), but with the automatic sign it is not (S4 fails). See Results.

**Note (2026-10-19): the job plan as run.** Dream (9B, A100) → validate → prep + lens (phase A) → **pause for the user's spot check** → gen → score → analyze. The v2 dream (`dream_refs_20261012_0016`) is kept but not used; v3 was dreamed, validated and run from scratch (`dream_refs_20261015_1545`). Phase B ran the grid fixed in the PREREG. The panel (gpt-oss, Gemma-4) re-judged the answers; the original judge decides, and the panel is reported as the sensitivity check.

## 11. The PrefEval addendum (2026-10-15, committed before any benchmark data was generated)

So that the results do not rest on our own 29 items, the same pipeline was run on **PrefEval** (explicit split; 1000 preferences, each with a question written so that a generic answer violates it; two thirds are dislikes or avoidances), on a fixed stratified sample of **200** (topic × polarity; seed 20261015). There are no hand-written references for it, so scaffold removal has to work on its own. The dreamer used the final v3 prompts unchanged and saw only the preference sentence; a new prompt dreamed 3 neutral follow-ups per preference to key the memory (the test question is never used for writing or keys). Conditions: no memory, the preference in context, and gen:opposite, gen:centroid and cs:concept × sign at α ∈ {0.5, 1, 2}, with α = 1 primary. Primary measure: PrefEval's preference-following accuracy (helpful and not violating) by its own judge criteria, original judge decisive, panel as the sensitivity check.

**Rules (α = 1).** **P1:** on all 200, gen:opposite − no memory has a 95% CI above 0. **P2:** on the avoid subset, cs:concept × sign − no memory is above 0 *and* gen:centroid − no memory is below 0 (the flip is present and the sign fixes it). Results: P1 holds under every judge; P2 is judge-sensitive (see Results).

**Note (2026-10-15):** the user's spot check of both phase-A sets was a review of a summary with examples, not of every item; the known failures were acknowledged before phase B was approved (opposites that keep a consequence clause, frames that keep it, `has_young_kids`). Nothing in the grid or the rules changed.
