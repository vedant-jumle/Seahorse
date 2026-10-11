# Experiment 3: dreaming as consolidation (abstraction, reconsolidation, pruning, re-filing)

*With many memories, can an offline pass turn episodes into leanings, resolve contradictions, prune, and re-file, and beat "write everything"? Design with dependencies; to be made exact once experiments 1 and 2 have run. Written 2026-10-11; §2c's ground truth corrected after review.*

---

## 1. Why this comes last, and why it is the point

Experiments 1 and 2 each fix one channel in isolation. This one is where the project's intended contribution lives: a memory that **decides for itself what to keep**, with the steer channel and the modulator each given a defined job. It needs two things that don't exist yet: a content channel that works (experiment 1) and pressure from many memories (the capacity curve). Without both, "selective" has nothing to be selective about and no baseline to beat.

The baseline is brutal and must be stated up front: **write everything, retrieve by similarity.** It is what every agent-memory system does. Consolidation can only win where write-everything fails: interference, drift of leanings, contradictions, and cost. The experiments are designed around those four.

## 2. Four processes, four tests

### 2a. Abstraction: leanings from episodes

**Idea.** The steer channel is good at one thing: a question-independent push. Today we write that push from an explicit statement (*"I'm vegetarian"*). In life, leanings come from many small episodes (five dinners ordered without meat), none of which states the trait. Dreaming replays a cluster of episodes and keeps what they share; the shared part is the push, the residual specifics stay episodic.

**Process.**
1. **Episodes.** For each of the 15 leanings, generate (with the 9B dreamer) 6 short episodes that *imply* the leaning without stating it (*"Could you suggest a lentil recipe for tonight?"*, *"I'll skip the steak place, thanks."*). Validate with the judge that none states the trait explicitly.
2. **Write each episode** as a memory moment (experiment 2's generated-reference pipeline, or plain `without`), filed under its own key.
3. **Dreaming pass.** Cluster moments by key similarity (agglomerative, cosine threshold set on the generic prompts so unrelated moments never merge). For each cluster: the **abstracted push** = the mean shift (or the first principal component of the shifts, signed by the valence direction of experiment 2); the **episodic residuals** = each shift minus the push.
4. **Store** the push in the steer channel under the cluster's key (the mean key); keep the residuals in the episodic store.
5. **Test.** The same preference prompts as core_v1. Conditions: no memory; the explicit-statement memory (core_v1's, the reference); the six episodes written individually and read by nearest key; the abstracted push alone; push + episodic residuals. **Prediction:** the abstracted push leans about as well as the explicit statement at α = 1, and better than any single episode. Measure with the judge lean, loop, incoherence; CIs over items.

### 2b. Reconsolidation: later wins, deliberately

**Idea.** Least-squares writing averages contradictions (*"my dog is Biscuit"*, later *"my dog is Max"* → a blend). Delta-rule writing makes the last write win by accident. Dreaming replays both, decides, and rewrites.

**Process.**
1. **Pairs.** For the 12 facts, a second, contradicting experience later in session 1 (*"Actually, my dog's name is Max now."* or an unmarked *"My dog Max…"*). Two orders (old→new, new→old) and two framings (explicit correction vs plain restatement).
2. **Baselines.** Delta rule (recency), least squares (blend), and the explicit-correction case in context (the ceiling).
3. **Dreaming pass.** Replay both episodes in one context with their order preserved and ask the model, offline, which holds now (*"Given these two things the user said, in order, what is the dog's name?"*). Then: keep the winner's entry, and **suppress** (not delete) the loser: store it with a negative gain and a "superseded" tag, so it still exists for inspection and can be revived if contradicted again (extinction, not erasure).
4. **Test.** The direct question and use prompts after reconsolidation. **Prediction:** the winner's clean use matches the single-fact rate from experiment 1; the loser's value appears no more often than a foil; and for the plain-restatement framing the dreamer's decision agrees with the ceiling's answer ≥ 90% of the time.

### 2c. Pruning: does the model's own surprise predict what it will need?

**Idea.** Not everything said should be kept. The modulator's hypothesis is that the model's own signals at write time predict later usefulness. This is the one crisp, falsifiable question in the neuromodulation story.

**Process.**
1. **Episodes with ground truth that depends on content.** *(The first version of this spec chose the queried subset at random. A random label cannot be predicted by anything measured at write time, so every signal would score AUC 0.5 by construction and the test would fail no matter what. The label has to come from what the experience is.)* Two labellings, both run:
   - **By construction (the floor):** a session-1 stream of ~100 experiences mixing **lasting** content (facts and preferences) with **passing remarks** (*"I had coffee this morning."*, *"Traffic was bad today."*, *"My cousin called earlier."*), validated by the judge as carrying nothing a later conversation would need. Label = lasting vs passing. Easy by design; it checks that the signals aren't blind. **One author writes both kinds.** If the dreamer writes the passing remarks, it also rewrites the lasting items in its own words (or a human writes both). Otherwise "surprise" separates the groups by *author*: text written by a Qwen model has lower perplexity under a Qwen model, and the hand-written bench sentences would score as more surprising for the wrong reason. Report the surprise signal matched for author either way.
   - **By realistic later need (the real test):** later-session conversations for the same user (several turns each, varied topics) are generated by a dreamer **from another model family** (gpt-oss-20B, already in the judge panel), and a judge from a third family labels, per experience, whether any later turn *depends* on it. Label = needed vs never needed. Held out from everything below; the dreamer never sees the signals. *Why another family:* if a Qwen dreamer decides what later conversations need and a Qwen model's surprise is tested on predicting it, a positive result can mean "Qwen agrees with Qwen".
   - **External ground truth (the cleanest):** LongMemEval and LoCoMo label which earlier sessions each question needs, written by people with no stake in our signals (listed in [next-moves §F](../history/next-moves-v0.1.md)). Run the same prediction task on one of them: signals at write time on each earlier session vs "needed by some question". If the signal predicts need there, nobody in this project wrote the label.
   The split between the two labellings is reported; an experience "lasting by construction but never needed" is the interesting middle case.
2. **Signals at write time**, per experience, all from the frozen model with no memory:
   - **novelty:** distance of the new key from the keys already stored (free)
   - **surprise:** mean next-token negative log-probability of the experience sentence under the model (one forward pass), and its peak
   - **valence/arousal:** |projection on experiment 2's valence direction|
   - **"doesn't fit":** KL between the model's next-token distribution before and after the experience on the same follow-up
3. **Prediction task** (no generation): does each signal predict "queried later" (AUC over the 100, CIs by bootstrap)? Also a logistic combination. **This is the result**, before any memory is tested.
4. **Memory test.** Write everything vs write the top-k by each signal (k = the number actually queried), in the least-squares store; read with the existing gate. Measure recall on the queried set and leakage/interference on the rest, as a **capacity curve** k = 10, 20, 40, 60, 100. **Prediction:** signal-gated writing matches write-everything on the queried set and beats it on interference; if no signal beats random selection, the surprise story is wrong and should be said so.

### 2d. Re-filing: better keys after the fact

**Idea.** A memory is filed under the situation it was written in (follow-up gist), so it fires only in similar situations (failure 7). Offline, with hindsight, it can be re-keyed under what it is *about*.

**Process.**
1. For each stored memory, the dreamer names the subject/topic (*"the user's dog"*, *"the user's diet"*) and generates 5 varied questions that should trigger it.
2. Build the new key as the pooled whitened key of those questions (experiment 1's subject key is the special case for facts).
3. **Test.** The Norway-style misses: for each memory, 10 prompts the memory should fire on that do not resemble its follow-ups (generated by the dreamer, validated by hand). Gate open rate and recall with old keys vs re-filed keys; false-fire rate on unrelated prompts must not rise.

## 3. Order within this experiment

2c first (it is a prediction task, cheap, and it decides whether the modulator idea survives), then 2a (needs experiment 2's valence direction), then 2d, then 2b (needs experiment 1's content channel to show the winner being *used*).

## 4. Decision rules (to be pre-registered when the exact configs exist)

- **Modulator alive:** at least one write-time signal predicts "queried later" with AUC CI above 0.5, and gating writes by it matches write-everything on recall while reducing interference at k ≥ 40.
- **Abstraction works:** the abstracted push's lean is within the CI of the explicit-statement push, and above the best single episode.
- **Reconsolidation works:** winner used, loser at foil rate, dreamer agrees with the ceiling ≥ 90%.
- **Re-filing works:** gate open rate on the should-fire set rises with no rise in false fires.

## 5. What this would make Seahorse

A memory with three parts and a defined job each:
- an **episodic store** of positions the model can attend to (experiment 1), written without human counterfactuals (experiment 2)
- a **steer channel** that carries only what dreaming abstracted: question-independent pushes, applied with a sign and a small gain
- a **modulator** that decides what enters dreaming (surprise, novelty, valence), what gets suppressed (reconsolidation) and what fades (pruning)

That is the hippocampus-beside-a-cortex picture the project started from, with the one correction the paper forced: the fast channel carries a *disposition*, and content goes where attention can reach it.
