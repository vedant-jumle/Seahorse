# Dreaming: the offline phase of the memory

*The plan for Seahorse after the arXiv paper. Three experiments, in order, each with its exact process in its own file. Written 2026-10-11, in the plain style of the main docs; status updated 2026-10-19. Read [where-it-fails.md](../where-it-fails.md) and [literature.md group E](../reference/literature.md#e-how-models-bind-entities-to-attributes) first.*

---

## Why "dreaming"

The paper settled what the steer channel is: **a push, not a store.** A stored shift is a fixed average of an effect the model normally recomputes for every question, so only question-independent content (a leaning) survives, and even that only in a narrow dose window. Facts come out as the word to say, premises never, dislikes flip, and the model claims the memory as its own.

Every one of those is a **binding** failure, and the binding literature says why: the model binds an attribute to an entity through *positions*. In context, entity and attribute carry a shared order tag and a chain of attention heads finds the attribute's position and copies from it (Feng & Steinhardt; Dai; Prakash). From the weights, the attribute is loaded onto the subject's own position by MLPs and extracted by attention from the question (Geva). Our push supplies content at the *answer* position, with no partner position, no tag and no owner. Nothing can bind to it.

We cannot perform binding ourselves: any vector we compose offline is still one vector at one position. **What we can do is give the model's own circuits the material they bind with.** That needs work done *outside* the conversation: replaying an experience in a controlled context, harvesting what the model's circuits produce, generating the counterfactuals that writing needs, and later merging, re-filing and pruning. Biology does this offline too. Hence *dreaming*: the phase where everything that can't be done in one forward pass at write time gets done.

## The three experiments

| # | Doc | Question | Depends on |
|---|---|---|---|
| 1 | [01-binding.md](01-binding.md) | Can a fact be **used** if its representation is written at the subject's position (or reinstated as a segment), rather than pushed at the answer? | nothing; run now |
| 2 | [02-scaffold.md](02-scaffold.md) | Can the model **generate its own references** (opposite, alternatives, valence) so that writing needs no human? Does concept × sign fix the dislike flip? | nothing; independent of 1 |
| 3 | [03-consolidation.md](03-consolidation.md) | With many memories, can an offline pass **abstract** leanings from episodes, **resolve** contradictions and **prune**, and beat "write everything"? | 1 (a content channel), the capacity curve |
| idea | [04-combining-channels.md](04-combining-channels.md) | Parked (2026-10-19): use the steer and the slice **together**. First: can a slice carry a preference, dislikes included, with no references? | 1 and 2 (both done) |

1 and 2 are independent and can run as two jobs. 3 is where the steer channel and the modulator get their defined jobs back.

**Status (2026-10-19).**
- **Experiment 1: done, with a clear answer (2026-10-11 to 14).** The subject-position write fails (clean use 0.00 on Qwen2.5, no gain on the 2B). The fact is read from the attribute's own position, late. What works is reinstating a stored slice of the model's own attention memory: 2 positions give 0.86 clean use on Qwen2.5 (the sentence in context: 0.90); on the hybrid Qwen3.5-2B, 2 positions plus the recurrent state give 0.97. Two facts stored separately collide (the wrong name 19% to 37% of the time); encoding them together fixes it. Addendum 6 (2026-10-14) followed up:
  - The binding-ID direction is real in the late layers (it flips partners in 75% of held-out contexts, against 0 for random), but adding it to a separately stored slice does not reduce the swaps (-0.02 [-0.14, 0.11]). The cheap repair fails; joint re-encoding stays the fix.
  - Finding the slice by the subject's words works up to 59 facts (0.85 to 0.92 right slice, no collapse). A wrong slice gives a confident wrong fact. Even the right slice is weaker on the 52 pool facts (0.62) than on the 7 hand-picked ones (0.83 to 0.87), which is unexplained.
  - On the 2B, recognition lives in the recurrent state, not the convolution buffer (yes/no 0.86 vs 0.50).
  - A literature check is done: recognition from the state and the swaps with their binding-ID explanation are new; the read site is known; the 2-token slice and the transplant are partly or mostly known. Details: [01-binding.md, Results](01-binding.md), [literature-exp1.md](literature-exp1.md).
- **Experiment 2: done (2026-10-15 to 16, 29 items and PrefEval).** A model's own generated references reproduce the paper's dislike result with no human-written reference (S2 holds under every judge: generated opposite +0.19, generated centroid -0.39). The references are not close copies of ours (S1 fails) and the automatic sign reader fails (S3, about chance). With the true sign, concept × sign is the best leaning and dislike memory so far (+0.27 and +0.39). On PrefEval (200 items) the generated opposite helps modestly under every judge (P1); concept × sign fixing the avoid subset (P2) is judge-sensitive. Details: [02-scaffold.md, Results](02-scaffold.md).
- **Experiment 3: design only.** It now has more to build on (see below).

## What changed (2026-10-13)

The original picture was that dreaming gives the model's binding circuits the material they bind with, for example by writing an enriched state at the subject. For a single fact that turned out to be unnecessary. The model's own attention memory of the sentence already is that material, and the useful part of it is tiny: the name's position and the full stop after it. So for facts, **dreaming reduces to keeping the right slice** (and, on hybrids, the recurrent state). There is little to compose.

The offline jobs that remain are the ones that cannot be done by keeping a slice:
1. **Binding related memories.** Slices stored in separate sessions carry the same "first pair" tag and swap partners when used together (experiment 1, collisions). Re-encoding the memories that are used together gives each its own tag. A cheap repair (adding a "which pair" direction to one slice) was tested and fails (addendum 6), so re-reading the text of the memories that are used together is still the way.
2. **Abstraction.** Turning many episodes into a leaning and storing that in the steer channel, which is the one job that channel does well (experiment 3).
3. **Selection.** Choosing what to keep. A slice for each fact is cheap but not free (and a hybrid's recurrent state is about 19 MB regardless of length), so deciding what is worth storing, for example by how surprising it was, is real work that plain retrieval over text does not do.

This also gives the project two stories under one idea: an **understanding** story (what binds a memory: steering vectors versus cache slices versus hybrids; paper 2) and a **system** story (a memory whose offline phase does what retrieval over text cannot: binding, abstraction, selection).

## What experiment 2 changed (2026-10-19)

The paper's main design limitation was that every memory was written from hand-made references (a "without", an opposite, eight alternatives). Experiment 2 had a model write them, and the effects survived:
- **The scaffold is removed, behaviourally.** Generated-opposite memories move dislikes the right way (+0.19 vs +0.23 for the hand-written opposite), generated-centroid memories move them the wrong way (-0.39), exactly the paper's C5, with no human reference. On PrefEval, a benchmark nobody tuned the references to, the generated opposite helps modestly under every judge. (The formal "scaffold removed" decision reads false only because the cosine-to-hand test, S1, failed: the generated references are different sentences, not copies.)
- **The sign reader is the open piece.** With the true sign, concept × sign is the best leaning and dislike memory so far. With the sign read automatically it is not, because a valence direction reads restriction-type leanings (vegetarian, vegan, lactose intolerant, a tight budget) as negative. "Avoid" is not "dislike": the sign has to be read relative to the stored concept.

**A self-learning memory, so far.** The model already does binding, ownership and recognition at read time (experiment 1), and it now writes its own references (experiment 2). Still not done by the model: deciding what to store, finding the right memory (retrieval), reading the sign, and consolidation.

**Honest framing.** The stored object is a standard KV cache, used outside its normal rules. Prior systems work (Prompt Cache, CacheBlend, EPIC, InfLLM) does parts of this. What the experiment adds is the explanation of which part carries the value and why combining fails.

## What each outcome would mean

- **1 works:** the first fact ever *used* by this project. Facts have a channel; the two-channel design is confirmed with a mechanism; "whose" comes for free from the turn structure.
- **1 fails:** the orphan problem is real (a harvested state is still bound to the context it was computed in), and the content channel needs full reinstatement or a trained read path, both already designed in [next-moves §A2](../history/next-moves-v0.1.md).
- **What happened (2026-10-13):** the second one, for the subject write, and the first one, for reinstatement. Facts now have a channel (a gated slice of the model's own attention memory), with "whose" coming from the user-turn segment. See "Status" and "What changed" above.
- **2 works:** the paper's biggest stated limitation (hand-written references) is removed, and concept × sign turns C5 from a failure mode into a fix. **What happened (2026-10-19):** the first half, yes (S2, P1). The second half, only with the true sign (S4 fails with the automatic one). See "What experiment 2 changed" above.
- **3 works:** the modulator has something to modulate and a baseline it beats. That's the project's intended contribution.

## Shared conventions (apply to all three)

**Models.**
- Qwen3.5-2B first: everything is calibrated, the judge pipeline exists, the fact items exist. Note that only blocks **3, 7, 11, 15, 19, 23** are full attention (the rest are Gated DeltaNet). A residual-stream write works at any block; a KV-splice reinstatement works only at those six.
- Qwen2.5-1.5B-Instruct for anything that needs full attention at every layer (the reinstatement arm of experiment 1).
- Qwen3.5-9B to confirm whatever works on 2B, as in core_v1.

**Measurement hygiene, non-negotiable** ([metrics.md](../reference/metrics.md)):
- floor (no memory) and ceiling (the experience in context) on every table
- same random streams in every condition; 1 greedy + 4 samples per prompt, second seed set for the main conditions
- the judge (direction *and* coherence) as the primary measure; word lists, loop rate and NLL beside it
- balanced yes/no, parsed from the greedy answer; both halves reported
- unrelated prompts, token-identical to no memory
- controls: random vector of matched norm, swapped item, placebo, and (new, for experiment 1) **wrong position**
- 95% bootstrap CIs over items then prompts; paired gains
- claims and decision rules written in a `PREREG.md` before the run, as core_v1 did

**Code.** Build on `experiments/core_v1/` (items, batched two-pass generation, judge, bootstrap, report). Each experiment gets `experiments/dream_<name>/` with `run.py`, `config.yaml`, `PREREG.md`, tests for the pure functions, and a `--tiny` smoke mode. Jobs on DelftBlue through `slurm/run_v100.slurm` (2B) and `slurm/core_v1.slurm` (9B, judge).

**Items.** core_v1's 41 items (`experiments/core_v1/items.yaml`, which references `data/bench_v1/`). Experiment 1 adds a **subject** field per fact; experiment 2 uses the hand-written references as ground truth.
