# Seahorse

Experience-driven memory for LLMs: a plastic operator on a frozen model's residual stream,
written from its own internal states and recalled as steering when the present resembles the
past.

*"Hippocampus" is Greek for seahorse.* LLMs have a neocortex (their weights) and a sensory
buffer (the context / KV cache), but no hippocampus: nothing fast, persistent and selective
that forms from experience. Seahorse is an attempt to build one: a memory that is **made from
experience rather than recorded**, written automatically when something matters, and recalled
when the present resembles the past.

## The mechanism in one paragraph
Attach to one mid-to-late decoder block of a frozen LLM with a forward hook. Keep a d×d
matrix **M**. When the model lives through an experience, measure how the experience
**changed** its residual states (Δ) and imprint only the part memory couldn't already predict:
**M ← M + η·g·(Δ − M k) kᵀ** (the delta rule), where k is the centred, normalised state and g
is a salience gate from the model's own confusion (next-token entropy). Later, in any session,
the hook applies **h ← h + α·M·k(h)** at every position. Situations that resemble a stored one
recall its change; unrelated ones get nothing. No training, and the base model is never
modified.

## Status (v0, v0.1 and the diagnostics, on Qwen2.5-1.5B-Instruct)
- ✅ Carries **which** fact it was (specific: targets rise by +4 nats for names and +11 for the job, while foils stay flat) and **which way** a preference points (up to ~80% of the in-context effect with a contrastive write).
- ✅ Low output leakage at moderate strength. The **confusion gate** improves recall and halves leakage. **4 salient tokens ≈ a whole moment.**
- ❌ **Not selective.** The memory isn't too weak. At α = 2 it recalls 0.6–0.9 of a stored imprint on related prompts, but also ~0.4 on unrelated ones, and about two-thirds of the steer on unrelated prompts lands on the shared chat template.
- ❌ **No premise the model can reason with.** Yes/no inferences never flip, although memory does fire on those probes. The steer changes what comes to mind, not what the model concludes. Part of the small relation gain was a template artefact.
- ❌ **Capacity is set by write order.** In a combined memory the last write survives whole, and earlier ones are lost or reversed (RNN-style recency), with crosstalk between same-topic memories at read time. The earlier "facts survive" was only the last-written fact (job).

→ The steer behaves like a **modulatory** channel. Episodic content likely needs
**reinstatement into attention**. First, recall has to become selective. See the next moves.

## Documentation
| Doc | Contents |
|---|---|
| [docs/01-core-idea.md](docs/01-core-idea.md) | The problem (records vs memories), the target properties, inspirations (Bartlett, complementary learning systems, hippocampal indexing, the CA1 comparator, neuromodulators), premises from the literature, related work, the neuromodulator side idea |
| [docs/02-method.md](docs/02-method.md) | The exact mechanism: hook point, centring, **the memory operator explained in depth** (maths, properties, a worked example, capacity), what gets written, write variants, reading, metrics, sanity checks, limitations |
| [docs/03-code-map.md](docs/03-code-map.md) | What every file does, and how to run things (locally and on DelftBlue) |
| [docs/04-experiments.md](docs/04-experiments.md) | v0 and v0.1: setup, full results tables, samples, interpretation, caveats. The diagnostics (steer dose and selectivity, write order), with corrections to earlier claims |
| [docs/05-next-moves.md](docs/05-next-moves.md) | Open directions: the premise fork (layer sweep vs an episodic channel), capacity, removing the scaffold, salience, training, evaluation, parked ideas |
| [docs/06-open-notes.md](docs/06-open-notes.md) | Notes to come back to: possible confounds in the relation-probe result, the online-baseline problem, and thoughts on the neuromodulator idea (gain vs additive, tonic state, loop stability) |
| [docs/07-where-we-are.md](docs/07-where-we-are.md) | A plain-language report on the whole project: the question, prior work, the theory, how we test it, the results stage by stage, what it means, limits, next steps, and an appendix of real model outputs |
| [docs/08-how-it-works.md](docs/08-how-it-works.md) | The current design end to end, in plain words: calibration, whitened gist keys, shifts, the memory matrix (delta rule and least squares), the threshold, and what happens at every word |
| [docs/09-where-it-fails.md](docs/09-where-it-fails.md) | Every failure so far, with real examples, the likely cause, the evidence, and possible fixes, grouped into four root causes |

## Quick start
```bash
conda env create -f environment.yml && conda activate seahorse && pip install -e .
pytest -q tests                                   # CPU unit tests
python experiments/v0_1/run.py --out runs/v0_1    # needs a GPU with ~8GB (fp32)
```
On DelftBlue: `EXP=v0_1 sbatch slurm/run.slurm` (see [docs/03-code-map.md](docs/03-code-map.md)).

## Layout
```
src/seahorse/   memory.py (the operator) · residual.py (hooks) · sessions.py · metrics.py
tests/          delta-rule properties, hook correctness
experiments/    v0/, v0_1/  (run.py + scenarios.yaml) · diag_dose/, diag_order/  (run.py)
slurm/          DelftBlue setup and job scripts
docs/           the documentation above
```
