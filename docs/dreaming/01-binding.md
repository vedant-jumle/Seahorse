# Experiment 1: dreaming as binding (the subject-position write)

*Can a fact be used if the model's own representation of it is written at the subject's position, rather than pushed at the answer? Exact process. Written 2026-10-11; revised the same day after review (stage 0 knockout pre-test, H2 fixed, gated reinstatement, item tiers, a `subj` arm on Qwen2.5), and again before stage 0 ran (the sufficiency safeguard in §2). **Run 2026-10-11 to 13: see "Results" below. The subject-position write (the plan in this file) is dead; the content channel turned out to be reinstating a stored slice of the model's own attention memory. The spec text from §2 on is kept as written, with dated notes where the results superseded it. Addendum 6 (binding-ID repair, G-key retrieval with capacity, the split of the 2B's recurrent state) was run on 2026-10-14 and is reported at the end of Results; a literature check is in [literature-exp1.md](literature-exp1.md). Addenda 7 (2026-10-22) and 8 (2026-10-25 to 29), the paper-2 controls, follow it.***

---

## 1. The idea in one paragraph

When the model reads *"My dog's name is Pepper. I need to take my dog to the vet tomorrow."*, the token **dog** in the second sentence carries, at early-middle layers, a representation enriched by the first sentence (Geva's subject enrichment). That representation is the model's *own* binding of "Pepper" to "my dog", made by its own circuits. We harvest it offline (the dreaming pass), store it keyed by the subject, and in a later session add it back **at the subject token** ("my dog") in early-middle layers, where the model's extraction heads can read it when a question arrives. We never compose the binding; we put the model's bound state where its circuits look for it.

The comparison arm reinstates the experience sentence as a short segment of positions attention can attend to (the InfLLM route). Both arms supply a *position*; the current push supplies none.

**The risk this experiment has to face first.** Geva's subject enrichment is about facts in the *weights*. A fact said earlier in the *chat* reaches the answer by the in-context route: a shared order tag on "dog" and "Pepper", and attention from the answer back to Pepper's position (Feng & Steinhardt; Dai; Prakash). If that's what happens here, the state at "dog" holds a **pointer** (a tag to a position that won't exist in session 2), not the fact, and H1 is orphaned before it's generated. Stage 0 settles this before anything is built.

## Results (2026-10-11 to 14)

*Everything below comes from the report files in `results/dream_bind_*` (local, untracked; each folder has a `jobs.txt`, and the numbers are from the `report.md` files named in each part). "Clean use" = the fact used correctly in a coherent answer, by majority of three judges (the original Qwen3.5-9B judge, gpt-oss, Gemma-4), on the use prompts of the 7 primary items. Brackets are 95% intervals. "ctx" = the experience sentence sitting in the same user turn (the ceiling). Pre-registration: `experiments/dream_bind/PREREG.md`, addenda 1 to 5 (addendum 6 and 6b, added 2026-10-14, are exploratory: everything is reported, nothing is decisive beyond the stated rules).*

**The short version.**
1. In a chat, the answer reads the value from the attribute's own position ("Pepper"), late in the network. The question's subject ("my dog") holds no copy of the value (it may still carry a pointer, the order-ID tag of Prakash et al.'s lookbacks, which a Pepper/Rocky patch can't see). So there is nothing at the subject to harvest and write back. The subject write fails (clean use 0.00).
2. What works is keeping a slice of the model's own attention memory (the keys and values, "K,V") of the sentence, and putting it back in the next session. Two stored positions are enough (the name and the full stop after it): clean use 0.86, against 0.90 with the sentence in context.
3. That slice carries more than its tokens. The same two tokens as plain text give 0.27.
4. On the hybrid model (Qwen3.5-2B), the attention slice alone is not enough. Its recurrent state is needed too. The state alone gives recognition without recall: the model says "yes, that is her name" but cannot say the name.
5. Two facts stored separately and put back together get mixed up (the wrong person's name 19% to 37% of the time). Encoding them together fixes it. That is a real job for the offline phase.

**Update (2026-10-19), addendum 6 added three more points:**
6. **The cheap repair does not work.** The "which pair" direction (Feng and Steinhardt) is real in the late layers (it flips which partner the model prefers in 75% of held-out contexts, against 0 for a random direction). But adding it to one separately stored slice does not reduce the swaps in generation (change in confusion -0.02 [-0.14, 0.11]). So the swap is only partly a tag effect, and consolidation still means re-reading the text.
7. **Finding the slice by key works and scales to 59 facts.** Keyed on the subject's words, the right slice is found 85% to 92% of the time at N = 10, 20 and 59, with no collapse. When the right one is used the answer is as good as the sentence in context; when a wrong one is used, the model states the wrong fact with confidence.
8. **On the 2B, recognition lives in the recurrent state, not in the convolution buffer.** Recurrent state alone: yes/no 0.86; convolution buffer alone: 0.50 (nothing); both: 0.93. The two-position slice plus the recurrent state alone, without the buffer, gives clean use 0.97 and yes/no 1.00.

### Stage 0: where does the answer read the fact? (pre-registered, both models)

Setup: *"My dog's name is Pepper. What's my dog's name?"* against the same prompt with *Rocky*. Copy the model's internal state at one position from the Pepper run into the Rocky run, one layer (or band of 5) at a time, and see what share of the Pepper-versus-Rocky preference moves. 7 primary items. Folder: `results/dream_bind_stage0_20261011_2321/`.

| Where we patch or block | Qwen2.5-1.5B | Qwen3.5-2B |
|---|---|---|
| The question's subject ("dog"), patched | **0.00** [-0.00, 0.00] at every layer | **0.00** [0.00, 0.01] at every layer |
| The attribute ("Pepper"), patched | **1.00** up to layer 18, then falls (0.46 at layer 24, 0.00 at layer 27) | **1.00** to layer 14, 0.85 to 0.91 for layers 15 to 18, then **0.06 at layer 19** and 0 after |
| Block the answer's view of the subject (knockout) | 0.00 at every window | not run (see §2: DeltaNet blocks cannot be masked) |
| Block the answer's view of the attribute | costs 0.59 [0.41, 0.81] of ctx's gain, **only in the last window (layers 23 to 27)** | not run |
| The subject mention inside the answer prefix ("Your dog"), exploratory | 0.00 | 0.03 [0.01, 0.04] |

In plain words: the name sits at its own position, the answer position pulls it in late (by layer 19 on the 2B it already has), and the "dog" token is only a label. The verdict, by the pre-registered rule, is **pointer-suspected** on both models.

*Subject alone* (the safeguard in §2): on Qwen2.5, blocking the attribute from the whole question already costs 0.45 [0.40, 0.49] of the preference (so the sanity value was not near 0, and the net gain over that baseline is what counts). Patching the subject state on top of that adds **0.00** [-0.00, 0.00] at every layer. The subject state is not sufficient even when the answer cannot look up the name. This closes the "outvoted" loophole: it is not that the subject holds the value and the name drowns it out.

### Analytic pre-pass (teacher-forced, no generation)

Folder: `results/dream_bind_stage0_20261011_2321/*/analytic/`. Gain in log-probability of the right name after *"Your dog's name is"*, against no memory.

| | Best subject-write cell | Gain | Ceiling (ctx) | Controls at the same cell |
|---|---|---|---|---|
| Qwen2.5 | H2, layers 5 to 10, α=2 | 1.56 [0.69, 2.56] nats | 11.35 | random vector 0.88, swapped item 0.75: the paired differences include 0 |
| Qwen3.5-2B | H2, layers 4 to 8, α=2 | 0.29 [-0.05, 0.66] nats | 10.65 (the old push at layers 20/21/23: 9.59) | nothing |

Qwen2.5 shows a nominal gain that a random vector nearly matches. The 2B shows none (the pre-registered rule says the `subj` grid is skipped there). Balanced yes/no accuracy gained 0.04 at best.

### Generation, Qwen2.5-1.5B (pre-registered): "orphaned; reinstatement wins"

Folder: `results/dream_bind_gen_20261012_0038/qwen25_15b/analysis/report.md`. 7 primary items, use prompts; G-match gates.

| Condition | What it is | Clean use | Balanced yes/no | Unrelated prompts unchanged |
|---|---|---|---|---|
| nomem | nothing | 0.00 | 0.46 | |
| subj1 | the subject write (H1, layers 5 to 10, α=0.5) | **0.00** | 0.50 | 1.000 |
| subj2 | the subject write (H2, layers 5 to 10, α=2) | 0.00 | 0.50 | 1.000 |
| ctrl:wrongpos / rand / swap / placebo | the same write at the wrong token, random, another item's, a coffee sentence | 0.00 each | 0.43 to 0.57 | 1.000 |
| ctx | the sentence in the prompt (ceiling) | 0.90 [0.71, 1.00] | 1.00 | |
| **R-full** | the whole experience turn's K,V put back (gated) | **0.87** [0.71, 1.00] | 1.00 | 1.000 |
| **R-min** | only the name and the full stop (2 positions) | **0.86** [0.63, 1.00] | 0.86 | 1.000 |
| **R-orphan** | stored after an unrelated conversation, used without it | **0.80** [0.56, 1.00] | 1.00 | 1.000 |
| R-swap | another item's slice under this item's gate | 0.00 | 0.50 | 1.000 |

- **Pre-registered verdicts:** B1 fails, B2 fails, B3 holds, B4 holds, B5 holds, B6 holds. "Orphaned; reinstatement wins": R-full minus subj1 is +0.87 [0.71, 1.00].
- **The gate is what keeps it selective.** Without the gate the same R-full leaves only 6.4% of unrelated answers unchanged (1050 answers); with it, 100%.
- **The direct question** ("What's my dog's name?") is harder for the 2-position slice: R-full 0.91, R-min 0.66 [0.40, 0.89], R-orphan 0.97.
- **A wrong slice gives confident wrong values.** R-swap writes the other item's value in 41% of use answers [0.11, 0.74]. A swapped slice leaks into the partner's question 28% of the time.
- **Ownership:** self-claim ("Teal is my favourite colour") is about 0.01 for R-full, the same as ctx, whereas the push produced "Teal is my favourite colour!" (failure 15). The slice sits in the user's turn, so it reads as the user's.
- Seed set 2 agrees (ctx 0.89, R-full 0.89, subj1 0.00).

### Generation, Qwen3.5-2B, the hybrid (6 attention blocks + 18 Gated DeltaNet blocks)

Folders: `results/dream_bind_gen2b_20261012_1456/` (addendum 3), `results/dream_bind_state2b_20261012_1759/` (addendum 4). Clean use on the use prompts unless said.

| Condition | What is stored | Clean use | Direct question | Balanced yes/no |
|---|---|---|---|---|
| ctx | | 1.00 | 0.94 | 1.00 |
| R-full-state | attention K,V plus the DeltaNet state, whole turn | 0.96 [0.89, 1.00] | 0.94 | 1.00 |
| R-full-kv6 | the K,V of the 6 attention blocks only | 0.50 [0.23, 0.77] | 0.23 | 0.61 |
| R-min-kv6 | 2 positions at the 6 attention blocks | 0.67 [0.41, 0.91] | 0.20 | 0.50 |
| **R-min-state** | 2 positions at the 6 attention blocks **plus** the DeltaNet state | **0.97** [0.91, 1.00] | 0.89 | 0.93 |
| **R-state-only** | the DeltaNet state, no attention entries | **0.00** | **0.00** | **0.93** [0.86, 1.00] |
| R-orphan-state | min entries and state, stored after an unrelated exchange | 0.99 [0.94, 1.00] | 0.91 | 1.00 |
| R-swap-min-state | another item's | 0.00 (wrong value 0.47) | 0.00 | 0.46 |

- R-full-state is within noise of ctx (-0.04 [-0.11, 0.00]). The unrelated prompts are unchanged 1.000 when gated; ungated, R-full-kv6 leaves 0.7% unchanged and R-min-state 0%.
- **The attention-only version (kv6) is a half measure.** It uses the name in slot-filling tasks (0.50) but refuses direct questions (0.23): *"I don't know your best friend's name! Could you tell me who they are?"* Its yes/no is 0.61.
- **Adding the DeltaNet state fixes it.** R-min-state minus R-min-kv6 is +0.30 [0.06, 0.57] on use prompts and +0.69 [0.51, 0.86] on the direct question. R-min-state minus R-full-state is 0.01 [-0.06, 0.09].
- **The state alone is recognition without recall.** On *"Is your dog called Pepper?"*-type probes it is right 93% of the time. Asked *"What's my dog's name?"* it says *"I don't know your dog's name!"* every time. The recurrent state tells the model that something was said and what it was about; the attention entries are what fetch the value.
- **Be honest about size.** The DeltaNet state is about 4.7 million numbers (18 blocks x 16 heads x 128 x 128, roughly 19 MB in fp32) and does not grow with the text. For one short fact that is a large object, not a compact one. It pays off only when it summarises a long episode. The "2 positions" in R-min-state are 2 attention entries *plus* this state.
- On the 2B the 6-attention-block slice alone is weaker than Qwen2.5's full-depth slice: R-full-kv6 minus Qwen2.5's R-full is -0.37 [-0.66, -0.09]. R-full-state is not (+0.09 [-0.06, 0.29]).

### Does the slice carry more than its tokens? (text control, addendum 5)

Folder: `results/dream_bind_add5_20261013_1255/{qwen25_15b,qwen35_2b}/`. T-min = the same name and full stop given as plain text in an earlier user turn. T-full = the whole sentence as an earlier turn.

| | Qwen2.5 clean use | Qwen2.5 direct | 2B clean use | 2B direct |
|---|---|---|---|---|
| nomem | 0.00 | 0.00 | 0.00 | 0.00 |
| **T-min** ("Pepper." as text) | **0.27** [0.06, 0.51] | **0.00** | **0.24** [0.06, 0.47] | 0.11 |
| **R-min / R-min-state** (the stored slice) | **0.87** [0.63, 1.00] | 0.66 | **0.97** | 0.89 |
| T-full / R-full(-state) | 0.84 / 0.84 | 0.91 / 0.91 | 0.97 / 0.97 | 0.97 / 0.97 |

- T-min minus R-min: -0.60 [-0.86, -0.34] (Qwen2.5), -0.73 [-0.93, -0.50] (2B). **The compression claim survives**: a stored slice carries about three times what the same tokens do as text. (The slice's keys and values were computed with the whole sentence in view, so each carries the sentence's context.)
- T-full equals R-full exactly (difference 0.00 [0.00, 0.00]), which is the earlier check that replaying the sentence's K,V is the same as putting the sentence as text in a previous turn. **So full reinstatement is not a new kind of memory; it is prompt caching used across sessions.** What is not obvious is that two positions of it are enough, and that the cache can be gated and moved.
- This rerun gave R-full 0.84 and R-min 0.87 on Qwen2.5 (the main run: 0.87 and 0.86); the differences are well inside the intervals.

### Two facts together: the binding-ID collision (addendum 5, both models, gates forced open)

Folders: `results/dream_bind_add5_20261013_1255/{qwen25_15b_collide,qwen35_2b_collide}/`. Two sentences, each stored in its own session, then both put back for one question. "Confusion" = the partner's value in the answer and the member's own value missing. Pairs: 3 different-subject pairs (dog/cat, sister/brother, best friend/boss) and 5 confusable pairs (older sister Petra / younger sister Maria; first dog Pepper / second dog Rocky; and so on).

| Condition (Qwen2.5) | Confusable: correct | Confusable: confusion | Different: correct | Different: confusion |
|---|---|---|---|---|
| nomem | 0.00 | 0.00 | 0.00 | 0.00 |
| ctx2 (both sentences in the prompt) | 0.93 | 0.01 | 0.91 | 0.03 |
| **R2-joint** (both encoded together, then cached) | 0.89 | 0.05 | 0.97 | 0.00 |
| **R2-sep-overlap** (stored separately, stored positions kept) | **0.51** | **0.35** | 0.62 | 0.28 |
| **R2-sep-shift** (separately, the second shifted after the first) | **0.54** | **0.37** | 0.62 | 0.26 |
| R2-min-joint (2 positions each, from the joint encoding) | 0.61 | 0.20 | 0.50 | 0.19 |
| R2-min-sep-overlap / shift | 0.46 / 0.46 | 0.28 / 0.32 | 0.50 / 0.56 | 0.23 / 0.20 |

On the 2B (every condition uses the joint DeltaNet state, so only the six attention blocks differ): ctx2 0.91 / 0.05, joint 0.89 / 0.05 (confusable correct / confusion), separately stored 0.57 to 0.59 / 0.28 to 0.33, min-joint **0.73 / 0.06**, min-separate 0.56 / 0.23 to 0.24. On different pairs: joint 1.00 / 0.00, separate 0.71 to 0.76 / 0.19 to 0.22, min-joint 0.93 / 0.04.

- **Separately stored slices swap partners 0.19 to 0.37 of the time** (full slices, both models), **even for different subjects** (a dog's and a cat's name; confusion 0.26 to 0.28 on Qwen2.5, 0.19 to 0.22 on the 2B).
- **It is not about position.** Keeping the stored positions (overlapping) or shifting the second slice after the first gives the same result (0.35 vs 0.37, 0.28 vs 0.33).
- **Re-encoding together removes it.** R2-joint is within noise of ctx2 on both models. "Re-encoding helps" (joint minus separate, correct, confusable): +0.38 [0.21, 0.57] (Qwen2.5 overlap), +0.29 [0.12, 0.49] (2B overlap). "IDs collide" (separate minus joint, confusion): +0.30 [0.13, 0.48] on Qwen2.5, +0.23 [0.07, 0.42] on the 2B; the shifted versions agree.
- **The two-position version is partly confused even when stored together on Qwen2.5** (min-joint confusion 0.20; so "IDs collide" for the min versions is 0.08 [-0.08, 0.27], not significant). On the 2B, min-joint works (0.73 / 0.06), and the separate min versions still collide (+0.17 [0.05, 0.31]).
- **The attention-only probes on the 2B stay weak** (joint-kv6 0.19, sep-shift-kv6 0.16 on confusable pairs): the state matters here too.
- **Interpretation (inferred, not tested):** in context the model tags each entity-attribute pair with an order-based binding ID (the "first pair", the "second pair"; Feng and Steinhardt; Dai). Slices stored in separate sessions all carry the same tag, so when two are put back, the question cannot tell them apart. Joint encoding gives them different tags. We have not yet measured the tag; that is the next experiment.
  - **Update (2026-10-19):** the tag was measured in addendum 6 (below). The direction exists in the late layers, but adding it to a separate slice did not remove the swaps. The binding-ID reading is therefore supported (the direction flips partners) but not sufficient (the cheap repair fails).
- **What the joint encoding is.** Re-reading the text of both sentences in one pass and storing the result. That is "dreaming" in its plainest form, and it costs a model pass over text we already have. It does not need an abstraction, but it does need to be done offline, for the memories that will be used together.

### What was and was not new

The stored object is a standard KV cache (plus the recurrent state on hybrids), used outside its normal rules: cut down to two positions, moved across sessions, gated by a match on the subject, and combined from separate sessions. Prior systems work does parts of this: Prompt Cache (reusing cached segments), CacheBlend (fusing separately cached segments), EPIC (position-independent caching), InfLLM (retrieving blocks of cache). We did not invent the mechanism. What this experiment adds is the explanation for each rule: which part of the cache carries the value (the attribute's own position, read late), why the subject write fails (the subject holds a pointer or nothing), what the recurrent state adds (recognition, not recall), and why combining separately stored slices fails (the binding ID). A literature check is being written up in `docs/dreaming/literature-exp1.md`.

**Update (2026-10-19): the literature check is done** ([literature-exp1.md](literature-exp1.md), 36 papers; FwPKM added in [literature.md group F](../reference/literature.md#f-trained-fast-weight-memory)). Verdicts:
- **New:** ~~recognition from the recurrent state (no paper probes a yes/no on a restored state)~~ *(no longer new, 2026-10-21: Afendulev et al., Sept 2026, show recurrent-only recognition on Qwen3.5-4B/9B/27B, and that it is gist familiarity that also accepts associated lures; see [literature-exp1.md](literature-exp1.md), "Added 2026-10-21")*, and the partner swaps with the binding-ID explanation (cache-reuse papers report only generic quality loss).
- **Partly known:** the 2-token slice. Sparse KV as memory exists (Memory3, KBLaM) but only with training. The training-free version and the comparison with the same tokens as text are ours.
- **Mostly known:** the transplant across positions (Prompt Cache, Feng and Steinhardt) and the read site (Lookbacks, Entity Tracking). That recall runs through attention in hybrids is also known (Michalak and Abreu, Bick et al., DASC on Qwen3-Next).
- **Two corrections to this file:** the subject carries "no copy of the value", not "nothing the answer needs" (it may hold an order-ID pointer; the collision result shows a pointer matters with two facts), and Prompt Cache's "scaffolding" is our joint-encoding fix, so joint encoding is not a new fix.
- **FwPKM** (a trained fast-weight memory) learns to use its memory only when attention cannot reach the context, and re-reading the context 2 to 4 times helps its recall (the trained analogue of joint re-encoding). It supports "reads need training" as a design direction.

### What happened to the spec's questions

- *Does the fact live at the subject or at its own position?* At its own position. (Stage 0.)
- *Is a harvested state portable across contexts (H1 vs H2)?* Moot: neither works as a write. The portability question moved to the slice: R-orphan (0.80 on Qwen2.5, 0.99 on the 2B with the state) says the slice survives a change of context.
- *Does position matter (wrongpos)?* Moot for the same reason, but the collision test shows that *which position tag* a slice carries matters.
- *Is the unified-vs-split channel design right?* Yes, with a mechanism: a content channel is a stored attention slice (plus state on hybrids); the steer channel stays for leanings.

### Addendum 6 (run 2026-10-14; exploratory, PREREG addenda 6 and 6b)

Folder: `results/dream_bind_add6_20261014_1608/`. Three parts, each with its own `analysis/report.md`: [A, binding-ID repair](../../results/dream_bind_add6_20261014_1608/qwen25_15b_idrepair/analysis/report.md) (Qwen2.5), [B, G-key retrieval and capacity](../../results/dream_bind_add6_20261014_1608/qwen25_15b_gkey/analysis/report.md) (Qwen2.5), [C, the 2B's state split](../../results/dream_bind_add6_20261014_1608/qwen35_2b/analysis/report.md). (The `results/` folder is local and untracked.)

#### A. Binding-ID repair: the direction is real, the cheap repair fails

Setup (Qwen2.5-1.5B): take the "which pair" direction as Feng and Steinhardt do, the mean difference between the states of the second pair and the first pair in jointly encoded contexts, at the entity and attribute positions. Check that it means what we think (swap it into held-out contexts and see whether the model prefers the partner's value), then add it to one of two separately stored slices and see whether the swaps go down. The Dai variant (a projection on the order-correlated component) is run beside it. Controls: a random direction, the reversed sign, the wrong token.

**Validation: does the direction flip the binding?** Share of held-out contexts where adding it changes which of the two values the model prefers (own minus partner):

| Layers | Feng direction, swapped at the attribute | Random direction | Dai direction |
|---|---|---|---|
| early | 0.00 | 0.00 | 0.00 |
| mid | 0.00 (0.43 when swapped at both) | 0.00 | 0.00 |
| **late (19 to 27)** | **0.75** [0.69, 0.81] | **0.00** | 0.25 [0.20, 0.31] |
| all layers | 0.64 [0.57, 0.70] | 0.00 | 0.16 |

The direction is valid in the late layers and not in early or mid ones; the pre-registered pick was therefore "late".

**Repair, teacher-forced** (16 two-fact members from addendum 5, separately stored, positions overlapping; confusion = the partner's value ranked above the own value):

| Condition | Confusion | Margin own minus partner (nats) |
|---|---|---|
| joint encoding (the fix that works) | 0.00 | 13.30 |
| separate (R2-sep-overlap) | 0.38 | 1.25 [0.29, 2.20] |
| separate + ID direction | 0.44 | 1.77 [-0.23, 4.02] |
| separate + random direction | 0.31 | 1.27 |
| separate + the reversed sign | 0.38 | 0.66 [-0.39, 1.73] |
| separate + ID on the wrong token | 0.38 | 1.24 |

- **The margin moves the right way, and only for the right sign.** The ID direction raises the margin by about 0.5 nats, the reversed sign lowers it, a random direction and the wrong token do nothing. So the direction is doing something specific.
- **But confusion does not fall** (0.38 to 0.44, with 16 members each is one or two items). The right answer's log-probability even drops (from -0.58 to -1.09).
- **The pre-registered pick** (the cell with the largest fall in confusion) was the two-position slice with the Dai direction, the only cell where confusion fell, by 0.06 (one member of 16). That is correct under the rule; an earlier reading of the sign of this pick was ours and has been corrected.
- **In generation** (that cell, 1440 answers, three judges' majority): partner confusion 0.28 [0.16, 0.42] against 0.30 [0.17, 0.44] without the direction, and 0.25 with a random direction. **Change in confusion: -0.02 [-0.14, 0.11].** Joint encoding of the same two-position slice: 0.19 [0.05, 0.37].
- **Pre-registered verdict: "repair works" does not hold.** The cheap tag consolidation fails.

What it means: the tag is a real thing in the late layers, but a simple addition at the stored positions does not put it where the lookup reads it, or the collision is only partly a tag effect (the earlier "binding-ID collision" reading is supported by the flip, not shown sufficient). Joint re-encoding stays the working consolidation.

#### B. G-key retrieval and capacity (Qwen2.5-1.5B, N = 10, 20, 59)

Setup: 59 facts (the 7 hand-picked primary items plus a 52-fact pool). The store holds N of them, each filed under a key; the question's own key (read at the last user token, at key layers 5, 10, 14, 19) must pick the right one, with the threshold at the 95th percentile of generic prompts' best match, as in core_v1. **K1** = a key built from the subject's words in the experience; **K2** = a key from the whole experience text. "Correct" = the right slice is the best match; "wrong slice" = another fact's slice wins; "miss" = nothing passes the gate.

| N | K1, correct (layers 5 / 10 / 14 / 19) | K2, correct | top-2 correct (K1) | false fire on unrelated prompts |
|---|---|---|---|---|
| 10 | 0.90 / 0.80 / 0.80 / 0.90 | 0.50 to 0.70 | 0.90 to 1.00 | 0.00 to 0.04 |
| 20 | 0.90 / 0.85 / 0.85 / 0.85 | 0.40 to 0.60 | 0.90 to 1.00 | 0.00 to 0.04 |
| 59 | **0.92 / 0.88 / 0.88 / 0.85** [CIs about +/- 0.08] | 0.59 / 0.54 / 0.61 / 0.42 | 0.88 to 0.93 | 0.02 to 0.04 |

- **Retrieval holds up to 59.** The subject-span key finds the right slice for direct questions 0.85 to 0.92 of the time with no collapse as N grows (a miss is almost never the failure: 0.00 at N = 59). The whole-text key is much worse (0.42 to 0.61 at N = 59). The 16 facts whose follow-up has no usable subject alias fall back to the whole-text key.
- **Top-2 is about 0.9**, so a second look (a verification step) could remove most of the remaining wrong picks.
- **Use prompts are weaker than direct questions** (0.67 to 0.92 correct, on only the hand-written facts, so wide intervals).
- **False fire is 0 to 4%** on core_v1's unrelated prompts.

**What the right slice does, and what the wrong one does** (teacher-forced, direct question; gain in log-probability of the right name against no memory):

| Condition | Gain (nats) | Wrong value when a wrong slice is injected |
|---|---|---|
| ctx (sentence in the prompt) | 9.48 [8.84, 10.12] | |
| **oracle** (the right slice, all 59) | **9.47** [8.84, 10.10] | |
| G-key at N = 59 | 8.32 [7.17, 9.35] | 0.86 [0.57, 1.00] (of 7 wrong picks) |
| swap (always a wrong slice) | **-2.62** [-3.41, -1.90] | 0.90 [0.81, 0.97] |

The oracle slice is as good as the sentence in context for every one of the 59 facts. A wrong slice does worse than nothing: it pushes the right answer down and the model states the wrong fact (0.86 to 0.90 of the time).

**Generation at N = 59** (clean use by the judges' majority; 820 answers):

| Prompts | no memory | ctx | oracle (right slice) | G-key |
|---|---|---|---|---|
| primary facts, use prompts | 0.00 | 0.90 [0.74, 1.00] | 0.87 [0.71, 1.00] | **0.49** [0.14, 0.80], wrong value 0.16 [0.00, 0.43] |
| primary facts, related prompts | 0.00 | 0.94 | 0.83 [0.71, 0.94] | 0.74 [0.49, 0.94], wrong value 0.00 |
| pool facts (the first 20 of the store; direct question only), related prompts | 0.00 | 0.86 [0.74, 0.96] | **0.62** [0.49, 0.75] | 0.59 [0.44, 0.74] |

- **The cost of retrieval is on the use prompts:** 0.87 with the right slice, 0.49 with the key's pick, and 16% confident wrong values. Intervals are wide (7 items).
- **The pool is the open question.** Even with the right slice (oracle), the pool facts reach only 0.62 against 0.86 in context, while the 7 hand-picked facts reach 0.83 to 0.87. So the two-position slice is weaker beyond the items it was developed on. This is unexplained, and no test was run.
  - **Correction (2026-10-21), after re-reading the report and the config:** the generation used **20** pool facts (`n_pool_gen: 20`), not 52 (52 = the non-primary facts in the teacher-forced pass), and **pool facts have only the direct question** (no use prompts). The fair comparison is therefore the primary facts' direct question, where the two-position slice is also weak: R-min 0.66 [0.40, 0.89] in the main run, 0.83 [0.71, 0.94] here; the pool's 0.62 [0.49, 0.75] sits within that range. Teacher-forced, the right slice equals the sentence in context on all 59 facts (+9.47 vs +9.48 nats), so the value is present. Four pool values are 2 tokens (Monopoly, Monet, Matilda, clarinet) and their slice keeps only the first. So the likely reading is "the 2-position slice is weaker on direct questions" (known from the main run), not "weaker on unseen facts". Still to check: R-full vs R-min on the pool's direct questions, split by token count.
- The label "related" is the report's; these are the question-style prompts for each fact.

#### C. The 2B's recurrent state: recognition is in the recurrent part, not the convolution buffer

Setup (Qwen3.5-2B, fp32, primary items): a Gated DeltaNet block keeps two things, a recurrent state (the big matrix, about 4.7 million numbers over the 18 blocks) and a small convolution buffer (the last few tokens, verbatim). Earlier runs restored both. Here each is restored alone. Teacher-forced balanced yes/no and gain in log-probability of the right name:

| Condition | What is restored | Yes/no, balanced | Gain (nats) |
|---|---|---|---|
| no memory | | 0.50 | 0.00 |
| ctx | | 1.00 | 10.65 |
| R-state-only | recurrent state + conv buffer | 0.93 [0.86, 1.00] | 4.24 |
| **R-state-only-rec** | **recurrent state only** | **0.86** [0.71, 0.96] | 3.74 |
| R-state-only-conv | conv buffer only | **0.50** [0.50, 0.50] | 0.66 [-0.28, 1.58] |
| R-min-state | 2 positions + both | 0.93 | 11.11 |
| **R-min-state-rec** | **2 positions + recurrent state only** | **1.00** | 10.79 |

- **Pre-registered rule holds:** recognition lives in the recurrent state (recurrent-only is above chance and within 0.1 of both-restored: 0.86 vs 0.93). The conv buffer alone does nothing.
- **In generation, dropping the conv buffer costs nothing:** R-min-state-rec gives clean use **0.97** [0.91, 1.00] (R-min-state: 0.97), direct question 0.97, yes/no 1.00, unrelated prompts unchanged 1.000 (1050 answers), no cross-binding leak. The state alone still gives no recall (clean use 0.00, direct question 0.00, whether or not the conv buffer is included), and now with a clean reading: **the "recognition without recall" of the earlier run is the recurrent state alone.**
- This also closes a confound the literature check raised (the state-only arm restored the 4-token buffer, which holds the last tokens of the stored turn verbatim). It was not the buffer.
- **No size claim:** the recurrent state is still about 19 MB in fp32; the buffer is small, so dropping it does not make the memory compact.
- **Note (2026-10-21): value vs verification, and a missing control.** The teacher-forced table shows a double split: the 2 attention positions alone carry the value (+10.01 nats, near ctx's 10.65) but give yes/no at chance (0.50) and refuse direct questions; the state alone gives yes/no 0.86 to 0.93 and recall 0.00, though it does carry part of the value (+3.74 to 4.24 nats). Afendulev et al. (2026) find the same kind of split within one session on Qwen3.5, and show that the state's "recognition" also accepts associated lures (gist familiarity). Our yes/no foils are unrelated names, so **an associated-lure control is needed** before calling the state's 0.93 recognition of the specific fact.

#### What is still open after addendum 6

1. **Why the two-position slice is weaker on the fact pool** (0.62 vs 0.83 to 0.87). *(2026-10-21: probably a prompt-type mismatch: the pool has direct questions only, where R-min is weak on the primary items too; see the correction under B.)*
2. **A safer pick:** top-2 retrieval is about 0.9, so verify the pick (for example score the candidates, or check the answer against a yes/no from the state) before putting a slice in.
3. **Cheap consolidation** is still open (selective recompute of the positions that carry the tag, rather than re-reading everything), now that adding the direction fails.
4. **A trained read path** (a small adapter on a frozen model) is the natural fallback for reads; FwPKM is the existing example of a trained read of a fast-weight memory.
5. Not run: the 9B, and a retrieval key other than the subject's words and the whole text (a learned key).

*(2026-10-23: items 1 and 3 and the 9B are answered by addendum 7, below.)*

### Addendum 7 (run 2026-10-22; PREREG addendum 7, rules fixed before any job)

Folder: `results/dream_bind_add7_20261022_1526/` (one run root; per part `<model>_{probe,dirs,refusal,pool,collide,s0pool}/` plus `llama32_3b/` for the text controls and stage 0; each has `analysis/report.md` or `report.md`). Models: Qwen2.5-1.5B, Qwen3.5-2B, Qwen3.5-9B (teacher-forced only), and **Llama-3.2-3B-Instruct**, the first model outside the Qwen family. Infrastructure: the 2B refusal job ran out of memory on the 10 GB slice and was rerun unchanged on a full A100 (`jobs.txt`). Clean use is the majority of three judges, as before.

**The short version.**
1. **The state's recognition is of the specific fact, not a gist** (A, 2B, pre-registered: "item recognition"). The recurrent state alone says yes to the true name 0.93 of the time and rejects names that look or sound like it, or share its theme, *more* reliably than unrelated names: false yes 0.00 (form) and 0.04 (associate) against 0.07 (unrelated); balanced accuracy against lures 0.96 [0.88, 1.00]. This is unlike Afendulev et al.'s word lists, where the state accepted associated lures.
2. **But the state does not bind.** With two facts' joint state and no attention entries (R2-state-only-J), the model says yes to both names for either person (27 of 32 yes on the true name and 27 of 32 on the partner's): balanced 0.50 [0.40, 0.60], "familiarity without conjunction". The state knows *which names were said*, not *whose* they are.
3. **The attention-only slice refuses because a "told" signal is missing** (B, pre-registered: "missing told signal" holds on the 2B and on Qwen2.5). Adding a direction built from ctx − nomem on 52 other facts lifts the direct question from 0.23 to 0.83 on the 2B (random direction 0.26, reversed 0.11) and from 0.66 to 0.94 on Qwen2.5, without lowering use prompts. On its own (no slice) the same direction gives no recall (0.00; it makes the model commit, e.g. "Your best friend's name is Alex."). Projections agree: at the last prompt token the 2B's R-min-state carries ctx's level of the told direction (5.47 vs 5.21 at block 23) while R-min-kv6 carries 0.96. The known-entity direction (Ferrando et al.) shows no such pattern.
4. **That signal is partly generic.** The own attention entries with *another* item's state give 0.71 on the direct question, with a placebo sentence's state 0.57, against 0.89 with the own state and 0.23 with none. The pre-registered verdict is "item-specific state" (both are more than 0.1 below), but most of the lift comes from any state of a turn in which the user said something ("Based on the context of our conversation, your boss's name is Gordon.").
5. **The pool "gap" was a prompt-type mismatch** (C, pre-registered: holds on Qwen2.5 and Llama). On the 52 non-primary facts with a use prompt, the 2-position slice gives 0.97 (Qwen2.5) and 0.98 (Llama) against 0.96 and 0.98 in context, and the same tokens as text 0.34 and 0.32. Only the direct question is weaker (0.65, 0.51), the same told-signal effect as on the primary items.
6. **Llama replicates the slice results** (E, pre-registered: "slice > text" holds, "collision replicates" holds). Use prompts: R-min 1.00, R-orphan 1.00, T-min 0.24 (R-min − T-min +0.76 [0.46, 0.97]), R-swap 0.00 with 0.33 wrong values; R-full equals T-full exactly. Two facts stored separately: partner confusion 0.41 [0.23, 0.61] against 0.01 encoded together (correct 0.49 vs 0.98).
7. **The read site holds on the pool and on Llama** (D). Patching the question's subject moves ≤ 0.02 of the preference on every model and fact set; patching the attribute moves 1.00. The verdicts read "in between" on Qwen2.5's pool and on Llama only because blocking the answer's view of the attribute costs less than 0.5 (0.48 and 0.05–0.06): the value reaches the answer through other positions as well. The 2B pool verdict is "pointer-suspected", as before.
8. **The 9B: no recognition from the state alone in yes/no, at the binary rule** (F, pre-registered: "the 2B split replicates" does not hold). Attention-only carries the value (+9.83 vs ctx +10.80 nats) with yes/no at chance, as on the 2B. But the 9B answers "No" to anything it doesn't know (margin about −3.9), and the state alone never moves the true name above zero (margin −1.27 vs −3.15 for unrelated names and −3.2 to −3.3 for lures). The margins show item-specific recognition (+2.6 nats on the true name, ≈+0.7 on any foil) that is too weak to overturn the 9B's "No" prior. Exploratory reading; the binary rule decides.
9. **The cheap collision fix fails its rule, for an instructive reason** (G, pre-registered: does not hold on any model). Re-encoding only the new memory with the old one's stored positions as prefix (R2-seq-min) reaches the jointly encoded two-position result on every model (confusion 0.09 vs 0.06 on the 2B, 0.20 vs 0.20 on Qwen2.5, 0.23 vs 0.17 on Llama), but the paired difference against storing separately is not significant (2B −0.15 [−0.33, 0.02]). On the pure transformers even joint two-position slices stay confused (0.17–0.20): two positions are too few to carry the binding tag. With full slices, re-encoding only the new memory is exactly joint encoding (asserted identical), so the working consolidation costs one pass over the new memory, not over all of them. The CacheBlend-style 15% recompute (R2-blend15) picks template tokens, not the value, and barely helps (reported only).

**Caveats.** Llama's yes/no probes are not informative: with the full sentence in an *earlier turn* it already says "No" to the true name (T-full acc on Yes 0.21), so every slice condition reads chance there. The reversed told vector makes the 2B refuse even yes/no questions ("I don't know who your brother is…"), which the parser counts as wrong (balanced 0.00), so that cell measures refusal, not inverted answers. The told vector also makes the model say yes to everything (yes/no 0.50 on the 2B): it licenses answering, not verification.

**What changes:**
- The hybrid story is sharper: **attention entries carry the value; the recurrent state carries both an item-level "this was said" recognition and a largely generic "the user told me something" licence to answer; neither alone binds which value belongs to whom.**
- "Recognition from the state" is not new (Afendulev et al.), but **item-specific recognition that rejects lures, the missing-told-signal mechanism of the refusal, and the conjunction failure are**.
- The 2-position slice result generalises: 52 more facts and a second model family.
- Open: verification and binding with the state (the conjunction failure), the 9B's strong "No" prior, and more than two positions for joint encoding on pure transformers.

### Addendum 8 (run 2026-10-25 to 29; PREREG addendum 8, rules fixed before any job)

Folder: `results/dream_bind_add8_20261025_1812/` (per model `<model>_{collide,pp_collide,probe,probepool,dirs8,refusal8,comply}/analysis/report.md`). Models: Qwen2.5-1.5B, Qwen3.5-2B, Llama-3.2-3B, all fp32. Parts 1 (is the collision a tag?), 2 (addendum 7 at the scale of the 52 pool facts and 34 pool pairs), 3 (told vs a generic "comply" direction) and 6 (a yes/no wording Llama can use). Infrastructure only: three code fixes before the final numbers (a comply guard that over-included facts, e548aef; a report crash on pool facts without form lures, f7e0f42; the refusal8 arms not generated, now a fail-loud check, 312e878) and one judge job rerun after Ollama lost the GPU (30ef343). No rule or data changed. **One verdict in the reports is wrong and corrected here:** the 2B's told-vs-comply rule prints "told is specific" because the CI lower bound is 4e-18 (floating-point zero); by the rule's order the verdict is **"generic comply"** (see 5).

**The short version.**
1. **The swaps come from storage order: a separately stored slice is always "the first pair"** (1a). Harvesting member B after a dummy fact ("My gardener's name is Brian."), then dropping the dummy's tokens, so B is stored as the second pair, removes most swaps. Partner confusion, confusable pairs, separately stored → B stored second (joint encoding in brackets):

   | | 8 addendum-5 pairs | 34 pool pairs |
   |---|---|---|
   | Qwen2.5 | 0.37 → 0.09 (0.05) | 0.22 → 0.01 (0.00) |
   | 2B | 0.33 → 0.15 (0.05) | 0.16 → 0.01 (0.00) |
   | Llama | 0.41 → 0.09 (0.01) | 0.20 → 0.01 (0.00) |

   Correct use rises to 0.76–0.98. The kept tokens are identical to the separately stored slice's except at most one (asserted), so only what the earlier fact did to them differs.
2. **But "both stored second" does not reliably bring the swaps back, and any earlier sentence helps somewhat.** The prediction was bothtagged ≈ separate (both carry "pair 2") and placebo ≈ separate. Bothtagged: 0.21 / 0.19 / 0.16 (addendum-5 pairs) and 0.20 / 0.09 / 0.03 (pool), Qwen2.5 / 2B / Llama; placebo ("By the way, I had a coffee this morning." before B): 0.26 / 0.19 / 0.31 and 0.07 / 0.05 / 0.13. The pre-registered rule "the tag causes the swaps" therefore holds on Qwen2.5 (both pair sets) and on the 2B's addendum-5 pairs, and fails on the 2B's pool pairs and on Llama, every time on the bothtagged clause. Reading: being stored second, with a different fact in front, is most of the fix; a shared tag recreating the swaps is supported only on Qwen2.5; part of the effect is generic (any earlier content).
3. **The binding sits at the subject's positions, and is written only by joint encoding** (1b, pre-registered: holds on both pure transformers, both pair sets). Adding the stored sentence's subject span ("younger sister", "friend from work") to the 2 positions, from the joint harvest, gives joint encoding's result: confusion 0.03 vs 0.20 for 2 positions on Qwen2.5, 0.01 vs 0.17 on Llama (pool: 0.00 vs 0.10 and 0.00 vs 0.03); correct 0.93 and 0.90 vs 0.60 and 0.52. The same positions from separate harvests do not help (0.34 and 0.29). This answers addendum 7's "two positions are too few": the missing part is the entity side of the pair. On the 2B the joint state already disambiguates 2 positions (0.06), so there is no room (reported).
4. **Addendum 7 holds at scale.** (2a) On 38 pool person / pet facts the 2B's state-only memory does item recognition (pre-registered): false yes to form lures 0.01 vs unrelated 0.03 (difference −0.01 [−0.05, 0.03]), balanced accuracy against form lures 0.94 [0.90, 0.97]. (2c) "IDs collide" and "re-encoding helps" hold on the 34 pool pairs on all three models (sep − joint confusion 0.22, 0.16, 0.20, all CIs above 0). (2b) The told direction built on pool half A only lifts the direct question on held-out half B: 2B 0.12 → 0.76, Qwen2.5 0.72 → 0.89, Llama 0.62 → 0.88, where :rand and :neg do not (no-memory + told makes up a name: wrong value 0.54 on the 2B). But **the "missing told signal" rule fails on the 2B and Qwen2.5**, because told now lowers use prompts on held-out facts (−0.14 [−0.25, −0.02] and −0.18 [−0.27, −0.10]; Llama −0.02, holds). Own attention entries + another fact's state: 0.65, + a placebo state 0.60, against 0.88 with the own state: "item-specific state" again.
5. **Most of the told effect is a generic "answer personal questions" direction** (3). d_comply = general − personal question ("What's the most common blood type?" − "What's my blood type?"), no memory, 62–66 behaving pairs; cos(told, comply) 0.4–0.65 in the middle and late blocks. On half B, slice + comply gives 0.67 (2B), 0.80 (Qwen2.5), 0.73 (Llama) against slice + told 0.76, 0.89, 0.88; comply:rand does nothing, and comply does not lower use prompts. Told − comply: 2B +0.09 [0.00, 0.18] → **generic comply** (corrected, see above; the 2B is the headline model), Qwen2.5 +0.09 [0.03, 0.15] and Llama +0.15 [0.07, 0.25] → told is specific. On the 7 primary items: generic comply on the 2B and Qwen2.5, in between on Llama. Reading: the attention-only slice holds the value, and what blocks direct answers is mostly the model's generic refusal for personal questions; a memory-specific "told" signal adds a little on the pure transformers.
6. **A two-choice question makes Llama's verification usable, and splits the 2B's two memories cleanly** (6). "Is my dog called Pepper or Max? Answer with the name only." (teacher-forced, both orders): Llama ctx 0.98, T-full 0.93 (rule holds), R-min 0.95 vs T-min 0.68, so slice > text holds for verification as well. The "Based on what I told you earlier, …" wording fails (it lowers Llama's ctx to 0.64). On the 2B the two wordings dissociate the parts: **attention entries only (R-min-kv6) yes/no 0.50 but two-choice 0.91; state only yes/no 0.93 but two-choice 0.48** (primary items; pool 0.54 / 0.83 and 0.86 / 0.56). The state recognises a name it is shown; the attention entries select the right name.

**Answers read.** Slice + told / + comply give the stored name ("Your ferret's name is **Shadow**."; "The name of your coworker is **Chloe**. Since you mentioned her name directly in your question…"); no-memory + told invents one ("Your dentist's name is **Dr. Sarah Jenkins**."); own entries + swapped state partly refuses ("I don't know your cousin's name!"). Collide answers under R2-sep-tagged and R2-minsubj-joint use the member's own name in the task.

**What changes:**
- The collision story now has a mechanism at both ends: separately stored slices all carry the first-pair tag (storage order), and the entity side of the tag is at the subject's positions, written only when the facts are encoded together. A storage-time fix without re-encoding (store the new memory as the k-th fact) is suggested but tested only for two facts.
- The refusal story weakens: from "a missing told signal" to "a mostly generic response gate"; the told-specific part is small and model-dependent.
- The hybrid split is sharper: familiarity in the state, selection in the attention entries (a double dissociation, not only "recognition without recall").
- Open: more than two stored facts in storage order; the use-prompt cost of told on held-out facts; the 9B in generation.

*(2026-10-30: a panel bug left some answers without gpt-oss / Gemma-4 labels, so their majority fell back to the original judge alone (2B refusal8 2939/3120 labelled, Qwen2.5 1957/2340, Llama 1880/2340). The panel output is shared by all models in a run root and skips prompts already judged, so an answer identical to another model's kept only the other model's keys. Fixed in a82244b (alias records, `panel.py --relink-only`); every addendum-8 report was re-analysed with full coverage. **No verdict changed**; clean-use numbers moved by at most 0.02. The numbers above are the re-analysed ones where they differ.)*

### Addendum 9 (run 2026-10-30; PREREG addendum 9, rules fixed before any job)

Folder: `results/dream_bind_add9_20261030_0107/` (`<model>_g9_collide/analysis/report.md`). Question: does storage order fix the collision for more than two stored facts? 30 groups, 114 members (`data/bench_v1/groups9.yaml`, new name lists): N = 3 confusable (12 groups: "eldest / middle / youngest {sister, brother, …}", "first / second / third {pet}"), N = 3 different-subject (6), N = 5 confusable (8 pet groups, "first … fifth"), N = 5 different-subject (4). Conditions: ctxN, RN-joint, RN-sep-shift (each stored alone: all "first"), **RN-sep-ordered** (member k harvested after dummy facts 1..k−1, dummies dropped: stored as the k-th fact), RN-sep-ordered-rev (the same slices placed in reverse order), RN-sep-placebo (k−1 neutral sentences instead of dummy facts). Gates forced open; the 2B with the joint DeltaNet state. Judged by the three-judge majority (full panel coverage after the relink fix; the rule uses regex partner confusion, which the fix does not touch).

**The short version.**
1. **Storage order removes most swaps at N = 3 and about half at N = 5.** Partner confusion, confusable groups, stored alone → stored as the k-th fact (joint):

   | | N = 3 | N = 5 |
   |---|---|---|
   | Qwen2.5 | 0.47 → 0.09 (0.01) **holds** | 0.54 → 0.22 (0.03) does not hold |
   | 2B | 0.38 → 0.04 (0.01) **holds** | 0.55 → 0.23 (0.03) does not hold |
   | Llama | 0.41 → 0.09 (0.00) does not hold | 0.51 → 0.30 (0.00) does not hold |

   Every drop has a CI well below 0; the failures are all on the second clause (the ordered CI does not reach joint's; Llama at N = 3 by 0.04 vs 0.01). Correct use rises by 0.27 to 0.49.
2. **Different-subject groups: storage order is enough at both sizes**, on all three models (ordered 0.00–0.03, joint 0.01; rule holds in all 6 cells).
3. **The stored order is what counts, not the placement.** Placing the ordered slices in reverse changes confusion by −0.02 to +0.03 everywhere. The tag a slice carries was set when it was stored.
4. **Not just "not at the start".** k−1 neutral sentences instead of facts help less (confusable, N = 3: 0.27 / 0.20 / 0.34 vs ordered 0.09 / 0.04 / 0.09), as predicted.
5. **The residual at N = 5 is between neighbours.** It sits at positions 3–5 (Qwen2.5 k = 4: 0.70; Llama k = 3, 4: 0.59, 0.66; positions 1–2 near 0), and the wrong name is an adjacent member's (the fourth pet answered with the fifth's or third's name). Close order tags blur, as Feng & Steinhardt's "close IDs → confusion" predicts.

**Answers read.** Ordered and reversed answers name the member's own value ("Your youngest brother's name is Tony."); errors give a neighbour's name ("Your fourth cat's name is Atlas.", the fifth cat). Placebo errors look the same as stored-alone errors.

**What changes:**
- A consolidation without re-encoding exists for small sets: store the k-th memory of a set as the k-th fact (it costs one pass over the new memory and a few dummy tokens). It reaches joint encoding for different subjects at N ≤ 5 and for confusable ones at N = 3 on two of three models; for five same-kind memories, joint encoding is still needed.
- The collision mechanism now has its scaling law in miniature: tags are distinct for the first two or three positions and blur after.

---

## 2. Stage 0: does the fact live at the subject, or at its own position? (run first)

One cheap job, no generation, in the `ctx` condition (experience + question in one user turn), on the 7 primary items (§3). **Which mention:** the subject span that is blocked or patched is the one in the *question* (after the attribute), e.g. "dog" in *"What's my dog's name?"*, never the one inside the experience sentence (which precedes the attribute and cannot hold it).

1. **Attention knockout** (Geva's method), **on Qwen2.5-1.5B only.** At the answer position, block attention to the question's **subject** span, then separately to the **attribute** ("Pepper") positions, over a window of 5 layers slid through the network. Measure the drop in `log P(target)` at the measure prefix, as a share of `ctx`'s gain over no memory. *Why not on Qwen3.5-2B:* 18 of its 24 blocks are Gated DeltaNet, which fold every past token into one recurrent state. There is no "answer → Pepper" edge to block there, and the delta rule makes a token's contribution depend on the state before it, so it can't be subtracted cleanly. Pepper's information can reach the answer through those blocks with all six softmax blocks masked, and a small drop would read as "the fact sits at the subject" when the knockout simply missed the path.
2. **Patching, on both models** (it needs no edge), in the standard causal-tracing setup **between two `ctx` runs**, not into the no-memory run. *(Into the no-memory run it doesn't work: the attribute patch has no "Pepper" position to land on, and without the experience sentence the question's "dog" sits at a different absolute position, so the patched state carries the wrong position information.)* Source: the ctx run with the true value (*"My dog's name is Pepper. What's my dog's name?"*). Target: the same prompt with the **counter** value from `facts.yaml` (*Rocky*; every item has one). Check per item that both values are single tokens so the two prompts have identical token layout; if not, pick a single-token counter. Copy the question-subject's state from source into target at block ℓ (one block at a time, and in bands), and separately the attribute position's state. Measure `logit(Pepper) − logit(Rocky)` at the measure prefix, as a share of the full source-minus-target difference. On Qwen3.5-2B this carries the read-out alone.

**Read-out, with thresholds (these go into `PREREG.md` before the job runs).** Knockout shares are of `ctx`'s gain `Δ log P(target)` over no memory; patching shares are of the full source-minus-target flip in `logit(Pepper) − logit(Rocky)`. The statistic is the mean over the 7 primary items with a 95% bootstrap CI over items; "best" means the best block or band.
- **Fact lives at the subject:** patching the question-subject's state flips **≥ 50%** of the difference at its best block (CI excludes 25%), and, on Qwen2.5, blocking the subject costs **≥ 50%** while blocking the attribute costs **< 25%**. Then H1 has something to carry; proceed to §5.
- **Fact lives at its own position (pointer at the subject):** patching the subject flips **< 25%** at every block (CI excludes 50%) while patching the attribute position flips **≥ 50%**, and, on Qwen2.5, blocking the attribute costs **≥ 50%**. Then the subject write is dead on arrival; the content channel is **gated reinstatement** (§7, arm R) or a trained read, and the generation budget goes there instead.
- **Both / in between:** anything else. Proceed, but expect H1 to work only partly and R to win; report the flipped share per block as the result.
- **If the two models disagree** (e.g. Qwen2.5 says "subject", Qwen3.5-2B patching says "pointer"): each model's result decides its own grid (the 2B result decides the 2B generation grid, the Qwen2.5 result decides the Qwen2.5 arms), and the disagreement is reported as a finding about the hybrid architecture, not averaged away.

**Sufficiency safeguard** *(revised 2026-10-11, before stage 0 ran)*. The subject patch above can under-read. In the counter run, "Rocky" still sits at its own position, and the answer can still read it there. So a patched "dog" that carries "Pepper" has to *outvote* an intact Rocky route, and a small flip could mean "the subject state is outvoted", not "the subject state is empty". That would wrongly declare "pointer". Two changes guard against it:
- **An extra cell on Qwen2.5-1.5B, "subject alone".** In the target (counter) run, block attention to the attribute position ("Rocky") from every query position after the experience sentence, at all layers. Then patch the question-subject state from the source run, as above (same blocks and bands). This asks the session-2 question: *is the subject state enough on its own, when the answer can't look up the attribute?* Report the flipped share, as for every patch.
  - **Sanity check:** with the attribute blocked and no patch, the flipped share should be about 0. Report it beside the cell. If it isn't about 0 (blocking Rocky alone moves the margin), the read-out is the patch's gain over that blocked baseline, and the sanity value is reported.
  - **How it is read:** if the subject-alone share is **≥ 50%** at its best block or band (CI excludes 25%), Qwen2.5 is *not* flagged pointer-suspected, whatever the ordinary subject patch shows. It goes to "both / in between", reported as "the subject state is sufficient but outvoted by the intact attribute".
- **Rule change: stage 0 alone can only flag "pointer-suspected".** A pointer verdict skips the `subj` generation grid (§9) only if the later analytic pre-pass (§6: the subject write in session 2, teacher-forced Δ log P(target)) *also* shows no gain. If the pre-pass shows a gain, the `subj` arm runs as planned.

Also run the **analytic pre-pass** (§6) before generation; it is the second early warning.

**Note (2026-10-13): stage 0 result.** Both models came out *pointer-suspected* under the rules above (subject patch 0.00 everywhere; attribute patch 1.00; on Qwen2.5 the attribute knockout costs 0.59 [0.41, 0.81] in the last window and the subject knockout 0.00; subject-alone net gain 0.00). The analytic pre-pass then said NO GAIN on the 2B (the `subj` grid was skipped there, as pre-registered) and a nominal GAIN on Qwen2.5 matched by the random and swapped controls, so the `subj` arm ran on Qwen2.5 and produced clean use 0.00. See Results above.

## 2b. What is tested

**Predictions, written before the run (go into `PREREG.md`):**
- **B1.** Subject-position writing yields *clean use* of the fact above the push's ceiling (core_v1: ≤ 0.15) on the use prompts, with the CI above the push.
- **B2.** Balanced yes/no accuracy rises above 0.5 with the CI above 0, for the first time in the project.
- **B3.** The **wrong-position control** (same vector, added at a non-subject token) gives no gain. If it does, the effect is just another push and B1 is not about binding.
- **B4.** Cross-binding stays low: with two facts stored (dog Pepper, cat Jasper), a question about the cat does not produce Pepper more often than with no memory.
- **B5 (comparison).** Gated segment reinstatement on Qwen2.5-1.5B (all layers full attention) reaches at least the subject write's clean use; on Qwen3.5-2B (six full-attention blocks) it is weaker. Reinstatement is an **upper bound**, not a new kind of memory: K and V are fixed functions of the text (core-idea P1), so a reinstated segment is close to `ctx` at a shifted position. Its value is to show how much of `ctx` survives the move and the gate; `subj` is the arm that would be new.
- **B6.** Unrelated prompts remain token-identical to no memory (≥ 0.95), since the injection site is the subject token and the subject does not appear in them.

## 3. Items

The 12 core_v1 facts (`data/bench_v1/facts.yaml` via `experiments/core_v1/items.yaml`), in two tiers:

- **Primary (7, a clean third-person subject):** dog_name, cat_name, sister_name, brother_name, grandmother_name, best_friend_name, boss_name. The cross-binding partners (dog/cat, sister/brother, best friend/boss) are all in this tier. **Pre-registered rules are decided on these 7.**
- **Secondary (5, the subject is the user):** job, hometown, first_language, university, favourite_colour. Their entity is "I", which appears in every prompt, so injecting at it is unselective by construction. They use topic aliases instead (*"my job"*, *"my work"*, *"my hometown"*, *"my native language"*, *"my university"*) and are reported separately. **favourite_colour has no usable site** (one follow-up, and neither use prompt names the colour or the subject); it is a no-site item and is excluded from the primary numbers.

Each item gets a **subject** block in a config (`experiments/dream_bind/config.yaml`):

```yaml
dog_name:
  subject: ["my dog", "your dog", "the dog", "dog"]      # aliases, matched in the user turn and in the model's own words
  harvest_followups: ["I need to take my dog to the vet tomorrow.", "My dog has been really energetic lately."]
```

Rules for the subject list:
- The first alias is canonical; the others are what the subject looks like in the use prompts and in the model's answers ("your dog" when the model refers back to it).
- Some use prompts don't contain the subject literally (*"the place where I grew up"*, *"my native language"*, *"my work's website"*). Add the paraphrase as an alias where it's unambiguous; otherwise leave the prompt as a **no-site** prompt. No-site prompts are reported separately: they measure what happens when there is nowhere to inject (expected: nothing, which is itself informative).
- Only the follow-ups that mention the subject are harvest follow-ups (2 of the 3 per item). The third, generic follow-up is excluded from this experiment's writes.

## 4. The dreaming pass (harvest)

Run on the frozen model, offline, once per item. Three variants, all stored; the primary is **H1**.

**H1: subject delta from the follow-ups (primary).** For each harvest follow-up:
1. Run **with**: chat template + user turn = experience + follow-up. Record the residual at every block at the **last token of the subject span** in the follow-up (Geva: the last subject token is the one that aggregates the subject).
2. Run **without**: the same, with the experience removed.
3. Subject delta at block ℓ: `d_ℓ = h_ℓ(with, subj) − h_ℓ(without, subj)`.
4. One delta per follow-up; average the two (or keep both as separate memory entries under the same key; store both options, report the average as primary).

This is the same two runs core_v1 already does; the only change is *where* the state is read: the subject token, not an entropy-weighted pool over the whole follow-up.

**H2: minimal replay.** Chat template + user turn = the experience sentence followed by one short mention of the subject (*"My dog's name is Pepper. My dog is lovely."*). Record the residual at the last subject token of the **second** mention at every block, and the corresponding state from the same two sentences with the fact removed (*"I have a dog. My dog is lovely."*). Delta as above. Tests whether a representation harvested with almost no context is more portable (less orphaned) than H1. *(The subject in the experience sentence itself comes before the attribute, so under the causal mask its state cannot contain the fact; harvesting there would store nothing. That was a bug in the first version of this spec.)*

**H3: raw enriched state (no delta).** H2's "with" state alone, not a difference. Added at reduced strength. Tests whether the model's own state is a better thing to reinstate than a difference. Expect it to hijack (core-idea P1); include it because if it works it changes the design.

**Also harvested, for the reinstatement arm (R):** the residual at every block for *every token* of the experience sentence in the minimal replay (H2's run), plus the token ids. On Qwen3.5-2B also the recurrent state of the linear-attention blocks after the segment is not stored (it can't be spliced); the arm is KV-only.

**Checks (fail the job if violated):** the H1 "without" run equals core_v1's; the subject token index is found in every harvest follow-up; the delta norm is finite and within 10× the median residual norm at that block.

## 5. The store and the key

- **Store:** one entry per item: `(key_s, {d_ℓ for ℓ in write layers})`. Delta-rule write into the existing M per layer is fine, but for this experiment the store is a lookup (isolated memories), as in core_v1.
- **Key:** the whitened, pooled key of the **subject span** in the without run (the same `pooled_w256` construction, pooled over the subject tokens only). This files the memory under *the subject*, not the follow-up's situation (fixing failure 7's cause for facts).
- **Gate:** two variants, both run:
  - **G-match:** the subject alias appears in the user text (string match on the detokenised user turn). Fires at every occurrence. Primary.
  - **G-key:** the existing cosine gate with the subject key, threshold at the q0.95 generic match. Reported; tells us whether the key alone would find the subject.
- **Injection site:** the last token of each matched subject span in the prompt; variant **+1** also injects at the following token (Feng & Steinhardt: the binding ID sits on the entity token and the one after). Optionally also at subject tokens the model generates itself ("your dog"); run as variant **gen** with the match applied to generated text in the two-pass scheme.

## 6. Where and how strongly

**Write layers.** Geva: enrichment in early-middle MLPs, extraction by upper attention. For Qwen3.5-2B (24 blocks) sweep single blocks `{4, 6, 8, 10, 12, 14, 16}` and bands `{4–8}, {6–10}, {8–12}, {10–14}`. The deltas are harvested at the same block they are written at (no cross-layer transfer; xlayer_v1 showed that fades). For Qwen2.5-1.5B (28 blocks) scale by depth.

**Dose.** `h_subj ← h_subj + α · d_ℓ`, α ∈ {0.5, 1, 2}. No norm-matching across layers (same layer in and out). Because the injection touches one or two positions rather than every position, higher α may be tolerable; report incoherence per α as always.

**Analytic pre-pass (cheap, before generation).** For every (variant, layer set, α): teacher-forced `Δ log P(target)` and specificity at the measure prefix (*"Your dog's name is"*), and the balanced yes/no margins. Pick the generation grid from this: the best three layer sets per variant, at the two best α. Also run the **logit lens on the subject delta**: it should *not* rank "Pepper" top at early layers (it's an enriched subject, not the word); if it does, we have harvested a late-layer "say Pepper" again.

## 7. Conditions for generation

Per item, per prompt, same random streams across conditions:

| Condition | What |
|---|---|
| `nomem` | nothing |
| `ctx` | experience in the prompt (ceiling) |
| `push@1` | core_v1's memory at blocks 20/21/23, α=1 (the published channel, for reference) |
| `subj:H1@{layers}@{α}` | subject-position write, primary variant, from the analytic pick |
| `subj:H2@…`, `subj:H3@…` | the other harvests |
| `subj:H1+1` | also inject at the token after the subject |
| `subj:H1:gen` | also at generated subject tokens |
| `reinst@{full-attn layers}` | the experience sentence's harvested states spliced as extra K,V entries (recomputed with the frozen W_k, W_v from the stored residuals) at a fixed position before the user turn, inside a user-role span. **Gated like `subj`** (G-match / G-key): the segment is present only on prompts where the gate opens, and absent otherwise, so B6 applies to it. An ungated version is reported once as the leak baseline (expect leakage, as `ctx` itself leaks: "Vegetarian waves"). |
| `ctrl:rand` | a random vector with `d_ℓ`'s norm at the subject site |
| `ctrl:swap` | another item's subject delta at this item's subject site |
| `ctrl:wrongpos` | this item's `d_ℓ` at a non-subject token of the same prompt (the token at the same offset from the end) |
| `ctrl:placebo` | the subject delta harvested from a placebo experience (*"By the way, I had a coffee this morning."*) |

**Prompts per item:**
- the direct question
- the two use prompts (with no-site prompts flagged)
- the balanced yes/no probes (greedy, 10 tokens, first-token margins)
- **cross-binding probes** (new): for items with a partner (dog/cat, sister/brother, best friend/boss), the partner's direct question and use prompts with *both* memories stored. Measures: the target of the *wrong* item appearing (should not rise over nomem).
- 50 unrelated prompts, greedy + 2 samples

Generation settings as core_v1: 1 greedy + 4 samples, T=1, top-p 0.95, top-k 20, presence 1.5, cap 120; second seed set for `nomem`, `ctx`, `push@1` and the primary `subj` cell.

## 8. Measures

- **Primary:** judge `clean use` on the use prompts; **balanced yes/no accuracy**.
- Also: `ans_tgt_clean` (regex), mention-without-use, confabulation, self-claim (the owner failure), loop rate, incoherence, NLL, length, `unrel_same`, cross-binding leak, teacher-forced target gain and specificity, the logit lens of the stored deltas per block.
- Judges: the original Qwen3.5-9B plus gpt-oss and Gemma-4 as in the panel; report the majority.

## 9. Decision rules (into PREREG.md)

- **Holds (binding works):** B1 and B2 and B3 and B6. The subject write's clean use CI lies above `push@1`'s *and* above `ctrl:wrongpos`'s; balanced accuracy gain CI > 0; `unrel_same ≥ 0.95`.
- **Push in disguise:** B1 holds but B3 fails (wrong position works as well). Then position isn't what mattered; report as a stronger push, not binding.
- **Orphaned:** nothing beats the push at any layer or α, while the analytic pre-pass shows `Δ log P(target)` rising. The representation is there but not readable out of context. Go to full reinstatement / the trained read path.
- **Reinstatement wins:** `reinst` beats `subj` on Qwen2.5, **both arms run on the same model**. Then the content channel is attention-based and the subject write is a shortcut that doesn't hold.
- **Stage 0 says pointer:** skip the `subj` generation grid; run only `reinst` (gated) and the controls, and treat the result as the upper bound for a training-free content channel. *(Revised 2026-10-11: only if the analytic pre-pass also shows no gain from the subject write; stage 0 alone gives "pointer-suspected". See the sufficiency safeguard in §2.)*

**Note (2026-10-13): which rule fired.** On Qwen2.5 the pre-registered verdict is **"orphaned; reinstatement wins"**: B1 and B2 fail (subj clean use 0.00, yes/no 0.50), B3, B4, B5 and B6 hold, and R-full minus subj1 is +0.87 [0.71, 1.00]. The "push in disguise" rule did not apply (B1 failed). The stage-0 "pointer" branch was followed on the 2B (reinstatement and its controls only). Beyond what §7 lists, the reinstatement arm was run in three forms (R-full, R-min with 2 positions, R-orphan) and, on the hybrid, with and without the DeltaNet state; these were added in PREREG addenda 2 to 5.

## 10. Job plan

- `experiments/dream_bind/run.py --stage knockout | harvest | analytic | gen | score | analyze`, `--model qwen35_2b | qwen25_15b`.
- **Stage 0 first** (`knockout` on Qwen2.5 only; `patch` on both models; ~30 min each) and read it before submitting anything else.
- Smoke (`--tiny`: 2 items, one layer set, one α) → harvest+analytic (short) → gen (the picked grid; budget it to ≤ 4 h on a V100: 7 primary items × ~10 prompts × ~12 conditions × 5 answers ≈ 4,200 generations, about 6 h at core_v1's rate, so **cut the grid to ≤ 8 generation conditions** from the analytic pick; the 5 secondary items run only in the primary cell and the baselines) → score on the A100 (judges) → analyze.
- Qwen2.5-1.5B run with **both** `subj` and `reinst` arms on `gpu-a100-small` (28 full-attention layers; calibration from `diag_keys`), so B5 and the "reinstatement wins" rule compare like with like.
- 9B confirmation only for the winning cell.

**Note (2026-10-13): the job plan as run.** Stage 0 (both models), the analytic pre-pass (both), generation on Qwen2.5-1.5B (`dream_bind_gen_20261012_0038`), on the 2B (`dream_bind_gen2b_20261012_1456`, `dream_bind_state2b_20261012_1759`), then the text control and the two-fact collision (`dream_bind_add5_20261013_1255`). The 9B confirmation was not run: nothing worked well enough on the subject-write side to deserve it.

## 11. What we'd learn even if it fails

- Whether, in a conversation, the fact sits at the subject or only at its own position (stage 0 answers this before anything is built, and it is a result about in-context binding in its own right).
- Whether a harvested state is portable across contexts (H2 vs H1).
- Whether position matters (wrongpos control). These three answers decide the content channel's design regardless of the headline.
