# Experiment 1: dreaming as binding (the subject-position write)

*Can a fact be used if the model's own representation of it is written at the subject's position, rather than pushed at the answer? Exact process. Written 2026-10-11; revised the same day after review (stage 0 knockout pre-test, H2 fixed, gated reinstatement, item tiers, a `subj` arm on Qwen2.5), and again before stage 0 ran (the sufficiency safeguard in §2). **Run 2026-10-11 to 13: see "Results" below. The subject-position write (the plan in this file) is dead; the content channel turned out to be reinstating a stored slice of the model's own attention memory. The spec text from §2 on is kept as written, with dated notes where the results superseded it.***

---

## 1. The idea in one paragraph

When the model reads *"My dog's name is Pepper. I need to take my dog to the vet tomorrow."*, the token **dog** in the second sentence carries, at early-middle layers, a representation enriched by the first sentence (Geva's subject enrichment). That representation is the model's *own* binding of "Pepper" to "my dog", made by its own circuits. We harvest it offline (the dreaming pass), store it keyed by the subject, and in a later session add it back **at the subject token** ("my dog") in early-middle layers, where the model's extraction heads can read it when a question arrives. We never compose the binding; we put the model's bound state where its circuits look for it.

The comparison arm reinstates the experience sentence as a short segment of positions attention can attend to (the InfLLM route). Both arms supply a *position*; the current push supplies none.

**The risk this experiment has to face first.** Geva's subject enrichment is about facts in the *weights*. A fact said earlier in the *chat* reaches the answer by the in-context route: a shared order tag on "dog" and "Pepper", and attention from the answer back to Pepper's position (Feng & Steinhardt; Dai; Prakash). If that's what happens here, the state at "dog" holds a **pointer** (a tag to a position that won't exist in session 2), not the fact, and H1 is orphaned before it's generated. Stage 0 settles this before anything is built.

## Results (2026-10-11 to 13)

*Everything below comes from the report files in `results/dream_bind_*` (local, untracked; each folder has a `jobs.txt`, and the numbers are from the `report.md` files named in each part). "Clean use" = the fact used correctly in a coherent answer, by majority of three judges (the original Qwen3.5-9B judge, gpt-oss, Gemma-4), on the use prompts of the 7 primary items. Brackets are 95% intervals. "ctx" = the experience sentence sitting in the same user turn (the ceiling). Pre-registration: `experiments/dream_bind/PREREG.md`, addenda 1 to 5.*

**The short version.**
1. In a chat, the answer reads the value from the attribute's own position ("Pepper"), late in the network. The question's subject ("my dog") carries nothing the answer needs. So there is nothing at the subject to harvest and write back. The subject write fails (clean use 0.00).
2. What works is keeping a slice of the model's own attention memory (the keys and values, "K,V") of the sentence, and putting it back in the next session. Two stored positions are enough (the name and the full stop after it): clean use 0.86, against 0.90 with the sentence in context.
3. That slice carries more than its tokens. The same two tokens as plain text give 0.27.
4. On the hybrid model (Qwen3.5-2B), the attention slice alone is not enough. Its recurrent state is needed too. The state alone gives recognition without recall: the model says "yes, that is her name" but cannot say the name.
5. Two facts stored separately and put back together get mixed up (the wrong person's name 19% to 37% of the time). Encoding them together fixes it. That is a real job for the offline phase.

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
- **What the joint encoding is.** Re-reading the text of both sentences in one pass and storing the result. That is "dreaming" in its plainest form, and it costs a model pass over text we already have. It does not need an abstraction, but it does need to be done offline, for the memories that will be used together.

### What was and was not new

The stored object is a standard KV cache (plus the recurrent state on hybrids), used outside its normal rules: cut down to two positions, moved across sessions, gated by a match on the subject, and combined from separate sessions. Prior systems work does parts of this: Prompt Cache (reusing cached segments), CacheBlend (fusing separately cached segments), EPIC (position-independent caching), InfLLM (retrieving blocks of cache). We did not invent the mechanism. What this experiment adds is the explanation for each rule: which part of the cache carries the value (the attribute's own position, read late), why the subject write fails (the subject holds a pointer or nothing), what the recurrent state adds (recognition, not recall), and why combining separately stored slices fails (the binding ID). A literature check is being written up in `docs/dreaming/literature-exp1.md`.

### What happened to the spec's questions

- *Does the fact live at the subject or at its own position?* At its own position. (Stage 0.)
- *Is a harvested state portable across contexts (H1 vs H2)?* Moot: neither works as a write. The portability question moved to the slice: R-orphan (0.80 on Qwen2.5, 0.99 on the 2B with the state) says the slice survives a change of context.
- *Does position matter (wrongpos)?* Moot for the same reason, but the collision test shows that *which position tag* a slice carries matters.
- *Is the unified-vs-split channel design right?* Yes, with a mechanism: a content channel is a stored attention slice (plus state on hybrids); the steer channel stays for leanings.

### Running now (addendum 6, not yet reported)

1. **Binding-ID repair.** Measure the "which pair" direction in the models' states, and test whether adding it to one separately stored slice stops the swaps. If it works, consolidation is cheap: no re-reading of text.
2. **G-key retrieval and capacity.** Find the right slice by the whole-conversation key rather than by a string match on the subject, with N = 10, 20 and 60 facts stored.
3. **Literature check.** `docs/dreaming/literature-exp1.md`.

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
