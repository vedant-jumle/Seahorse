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

## Status (v0, v0.1 on Qwen2.5-1.5B-Instruct)
- ✅ Carries **which** fact it was (specific: targets rise by +4 nats for names and +11 for the job, while foils stay flat) and **which way** a preference points (up to ~80% of the in-context effect with a contrastive write).
- ✅ Selective, with low leakage at moderate strength. The **confusion gate** improves recall and halves leakage. **4 salient tokens ≈ a whole moment.**
- ❌ **No premise the model can reason with.** Yes/no inferences never flip. The steer changes what comes to mind, not what the model concludes.
- ❌ **Tiny capacity for dispositions:** they collide at N = 6. Facts survive.

→ The steer behaves like a **modulatory** channel. Episodic content likely needs
**reinstatement into attention**. See the next moves.

## Documentation
| Doc | Contents |
|---|---|
| [docs/01-core-idea.md](docs/01-core-idea.md) | The problem (records vs memories), the target properties, inspirations (Bartlett, complementary learning systems, hippocampal indexing, the CA1 comparator, neuromodulators), premises from the literature, related work, the neuromodulator side idea |
| [docs/02-method.md](docs/02-method.md) | The exact mechanism: hook point, centring, **the memory operator explained in depth** (maths, properties, a worked example, capacity), what gets written, write variants, reading, metrics, sanity checks, limitations |
| [docs/03-code-map.md](docs/03-code-map.md) | What every file does, and how to run things (locally and on DelftBlue) |
| [docs/04-experiments.md](docs/04-experiments.md) | v0 and v0.1: setup, full results tables, samples, interpretation, caveats |
| [docs/05-next-moves.md](docs/05-next-moves.md) | Open directions: the premise fork (layer sweep vs an episodic channel), capacity, removing the scaffold, salience, training, evaluation, parked ideas |

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
experiments/    v0/, v0_1/  (run.py + scenarios.yaml)
slurm/          DelftBlue setup and job scripts
docs/           the documentation above
```
