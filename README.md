# Seahorse

Experience-driven memory for LLMs: a small, fast memory beside a frozen model, written from
the model's own internal states and recalled, as a nudge to those states, when the present
resembles the past.

*"Hippocampus" is Greek for seahorse.* LLMs have a neocortex (their weights) and a sensory
buffer (the context / KV cache), but no hippocampus: nothing fast, persistent and selective
that forms from experience. Seahorse is an attempt to build one: a memory that is **made from
experience rather than recorded**, written automatically when something matters, and recalled
when the present resembles the past.

## How it works, in one paragraph
A forward hook sits after one late decoder block of a frozen LLM (layer 26 of 28 in
Qwen2.5-1.5B-Instruct). In session 1 the model reads a message with and without an
experience ("By the way, I'm vegetarian."). How the experience **shifted** its states is stored
in a small matrix **M**, filed under a **key**: the whitened gist of what the user was talking
about. In any later session the hook compares, at every word, the current gist with the stored
keys. Only if the match is clearly above what ordinary text produces does it add the recalled
shift back: **h ← h + α·M·q**. Several memories share one M, written by least squares, so order
doesn't matter. No training, and the base model is never modified.

## Status
- ✅ **Carries which fact and which way a preference leans** into a fresh session.
- ✅ **Selective:** answers to unrelated questions are word-for-word identical to no memory (24 of 24, even with six memories stored together), and recall on related prompts roughly doubled when this was fixed.
- ✅ **Order-free:** with least-squares writes, each of six memories in one store keeps about half its solo strength, instead of only the last one surviving.
- ❌ **Faint:** the gist comes back ("welder", "What's her name?") but the exact fact rarely does ("Biscuit" in 7 of 300 samples). Preferences are either faint (contrastive write) or crude (plain write).
- ❌ **No premises:** yes/no questions that need the memory never flip. An added vector nudges; reasoning needs something the model can attend to, which is a second, episodic channel's job.
- ❌ **Not yet self-driven:** writing needs a hand-made with/without/opposite comparison, a memory fires only in situations like the ones it was filed under, and capacity beyond six memories is untested.

Next: the capacity curve on the benchmark's 60-fact pool, then neuromodulated memory
([where-we-are.md §8](docs/where-we-are.md#8-where-it-could-go)). The v0/v0.1 status, before
the selectivity and order fixes, is in the [experiment log](docs/history/experiment-log.md): the
conclusions of §4.1 and §4.2, and [§4.5.3](docs/history/experiment-log.md#453-what-the-diagnostics-change).

## Start here
Picking up the work? Read **[HANDOVER.md](HANDOVER.md)** first: the current state, what's pending, next steps and conventions.

Then read these three in order. They are plain-language reports and cover the whole project.
1. **[Where we are](docs/where-we-are.md):** the question, prior work, the theory, how we test it, the results stage by stage, what it means, limits and next steps, with an appendix of real model outputs.
2. **[How it works](docs/how-it-works.md):** the current design end to end: calibration, whitened gist keys, shifts, the memory matrix (delta rule and least squares), the threshold, and what happens at every word.
3. **[Where it fails](docs/where-it-fails.md):** every failure so far, with real examples, the likely cause, the evidence and possible fixes, grouped into four root causes.

Then, as needed:

| | Doc | Contents |
|---|---|---|
| **Reference** | [core-idea](docs/reference/core-idea.md) | Background: the problem (records vs memories), the target properties, inspirations (Bartlett, complementary learning systems, hippocampal indexing, the CA1 comparator, neuromodulators), premises from the literature, related work, the neuromodulator side idea |
| | [original-method](docs/reference/original-method.md) | **The original v0/v0.1 design** in full: hook point, centring, the memory operator explained in depth (maths, properties, a worked example, capacity), what gets written, write variants, reading, metrics, sanity checks, limitations |
| | [code-map](docs/reference/code-map.md) | What every file does, and how to run things (locally and on DelftBlue) |
| | [metrics](docs/reference/metrics.md) | What every number in the experiment tables means, with worked examples and what each measure can't tell you |
| **History** | [experiment-log](docs/history/experiment-log.md) | The detailed results of v0, v0.1 and the diagnostics (steer dose and selectivity, write order): setup, full tables, samples, interpretation, caveats, with dated corrections to earlier claims |
| | [open-notes](docs/history/open-notes.md) | Notes N1–N9 to come back to: confounds in the relation-probe result, the online-baseline problem, the neuromodulator idea (gain vs additive, tonic state, loop stability); some marked resolved |
| | [next-moves-v0.1](docs/history/next-moves-v0.1.md) | *Archived:* the open directions after v0.1 (the premise fork, capacity, removing the scaffold, salience, training, evaluation, parked ideas), with a note on what has been done since |

The reference and history docs keep their original numbers (1–6) and section numbers, so
references such as §2.5 or §4.5 still point to the right place.

## Quick start
```bash
conda env create -f environment.yml && conda activate seahorse && pip install -e .
pytest -q tests                                   # CPU unit tests
python experiments/v0_1/run.py --out runs/v0_1    # needs a GPU with ~8GB (fp32)
```
On DelftBlue: `EXP=v0_1 sbatch slurm/run.slurm`. The code layout, what every file does and how
to run things are in [docs/reference/code-map.md](docs/reference/code-map.md).
