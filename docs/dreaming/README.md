# Dreaming: the offline phase of the memory

*The plan for Seahorse after the arXiv paper. Three experiments, in order, each with its exact process in its own file. Written 2026-10-11, in the plain style of the main docs. Read [where-it-fails.md](../where-it-fails.md) and [literature.md group E](../reference/literature.md#e-how-models-bind-entities-to-attributes) first.*

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

1 and 2 are independent and can run as two jobs. 3 is where the steer channel and the modulator get their defined jobs back.

## What each outcome would mean

- **1 works:** the first fact ever *used* by this project. Facts have a channel; the two-channel design is confirmed with a mechanism; "whose" comes for free from the turn structure.
- **1 fails:** the orphan problem is real (a harvested state is still bound to the context it was computed in), and the content channel needs full reinstatement or a trained read path, both already designed in [next-moves §A2](../history/next-moves-v0.1.md).
- **2 works:** the paper's biggest stated limitation (hand-written references) is removed, and concept × sign turns C5 from a failure mode into a fix.
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
