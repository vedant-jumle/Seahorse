# Seahorse: handover

*The live handover: current state, where everything is, what's pending, and how to pick up. Updated 2026-10-05. For the story and the reasoning, read [README.md](README.md), then the three plain-language docs it lists.*

---

## 1. Where things stand (one paragraph)

A training-free memory for a frozen LLM (Qwen2.5-1.5B-Instruct) that works, within limits. An experience in session 1 is stored as a **shift** of the model's layer-26 states, filed under a **whitened gist key**. In any later session, the shift is added back when the current gist clearly matches a stored one.

**What it achieves:**
- **Selective:** unrelated answers are word-for-word unchanged in 24/24 cases.
- **Order-free:** least-squares writes.
- **Specific at the probability level.**

**What it doesn't:**
- In text it brings back the **gist rather than the exact fact** ("welder", not "deep-sea welder"; "Biscuit" in 7/300 samples).
- Preferences come out either faint or crude.
- It never supplies a **premise** for reasoning.
- It still needs **hand-made with/without/opposite comparisons** to write.

The newest experiment, `think_v1`: memory on Qwen3.5-2B, injected during the model's thinking phase. It **has finished but hasn't been looked at yet**.

## 2. Read first
1. [README.md](README.md): the overview and reading order
2. [docs/where-we-are.md](docs/where-we-are.md): the whole story, with real outputs (appendix C–D)
3. [docs/how-it-works.md](docs/how-it-works.md): the current design end to end
4. [docs/where-it-fails.md](docs/where-it-fails.md): the 12 failures, their causes and possible fixes

## 3. The current best design
| Piece | Setting |
|---|---|
| Model / layer | Qwen2.5-1.5B-Instruct, fp32; read/write at the output of layer 26 (of 28) |
| Write unit | one moment per follow-up sentence: an entropy-weighted mean shift over its words, chat template tail excluded |
| Shift | contrastive (with − opposite) for preferences; plain (with − without) for facts |
| Key | `pooled_w256`: mean over user-text words of PCA-256-whitened (state − background), normalised. *256 is a chosen number; a Ledoit–Wolf variant avoids it* |
| Store | delta rule for one memory; **least squares (RLS, λ=0.1)** for several, which is order-free |
| Read | at every word: match = max cosine to the stored keys; **hard threshold** = the 95th percentile of the match on ordinary prompts (soft works equally well); `h ← h + 2·gate·M·q`; never on the system prompt |

Code: `experiments/diag_keys/run.py` (the definitive implementation) and `experiments/samples_v2/run.py` (generation with the hook).

## 4. Pending: think_v1 (finished, NOT reviewed)

**What it tests:** the same design ported to **Qwen3.5-2B** (a hybrid model with linear and full attention, with built-in thinking), in three stages:
1. a layer sweep (layers 4–23)
2. one layer vs several (best 1/2/3, strength split vs full)
3. **memory strength during thinking vs the answer:** 0/0, 0/2, 2/2, **4/0, 6/0** (amplified, thinking only), **4/1, 6/1** (plus a weak answer dose), and the in-context ceiling

**Measured:**
- memory content in the thinking trace
- the exact fact in the answer
- **balanced** yes/no accuracy
- confabulation, degeneration, leakage, thinking length

**Items:** 4 preferences (vegetarian, norway, loves_hiking, loves_jazz) + 4 facts (dog_name, sister_name, job, favourite_colour) from `bench_v1`.

**Jobs**, all COMPLETED on `gpu-v100`:

| Job | ID | Runtime |
|---|---|---|
| smoke | 906983 | 15 min |
| stage 1 | 906984 | 15 min |
| stage 2 | 906985 | 40 min |
| stage 3, preferences | 906986 | 2h06 |
| stage 3, facts | 906987 | 1h25 |

**Outputs:** `/scratch/vvjumle/seahorse_runs/think_v1_20261003_0004/{smoke,s1,s2,s3_disp,s3_fact}`. They aren't pulled locally yet. Pull with `rsync -a delftblue:/scratch/vvjumle/seahorse_runs/think_v1_20261003_0004 results/`, then read each stage's `report.txt` and `samples.txt`.

**Check first:**
- did stage 1 really sweep all layers? It took the same 15 min as the smoke job, which is suspicious.
- which layers were chosen, and by what rule (documented in the reports)
- whether the thinking-only amplified conditions put the memory into the thinking text and flip balanced yes/no answers, without confabulation

**The env changed for this:** the DelftBlue `seahorse` env now has **transformers 5.18** (was 4.57), and `environment.yml` pins `>=5.18,<6`. Old experiments still pass their tests (16/16). Transformers 5 decoder layers return tensors, not tuples (handled by `residual._hidden/_replace`). Linear-attention speed-up kernels are **not** installed; it uses the pure-torch path.

## 5. Results on disk (local `results/`, untracked)
| Folder | What |
|---|---|
| `v0_408770`, `v0_1_415909` | the original experiments |
| `diag_dose_577895`, `diag_order_577896` | dose/selectivity diagnostics; write-order (recency) test |
| `diag_attn_846339` | attention-based write selection (no-go: token selection doesn't matter) |
| `bench_v1_calib_850635` | benchmark calibration (no memory): noise floor, flagged items, the "No" bias |
| `diag_keys_858681`, `diag_keys_858682` | key/selectivity experiment, parts 1 and 2 (merge needs torch: `conda run -n torch python experiments/diag_keys/run.py --merge …`) |
| `samples_v2_873325` | real text from the current design (the source of where-we-are appendix C–D) |

## 6. Key findings to remember
- **Selectivity, not strength, was the problem.** The fix was the **key**: the whitened gist plus a threshold. Recall on related prompts roughly doubled, and leakage fell to about zero.
- **Recency:** step-by-step writing means the last memory wins (RNN-style). **Least squares** removes the order dependence (each of 6 memories keeps about half its strength), but contradictions now average instead of overwrite.
- **Token-level write selection doesn't matter:** random 4 tokens ≈ entropy top-4 ≈ attention top-4. The unit of memory is the **moment**. Attention finds *relevance* ("vegetarian") but not *importance*.
- **No premises:** memory fires on yes/no questions (60–80% strength) but is never used as a premise. Earlier "relation gains" were artefacts of all-"No" probes and template tokens.
- **Text vs probabilities:** facts are specific at the probability level, but the text shows the gist. The old single-phrase-pair preference metric **overstated** effects; judge preferences by rates over sampled answers.
- **Filing:** a memory's key is the gist of its *follow-up* sentences, not of the experience, so it fires only in similar situations (Norway never fired on grocery or coffee questions).

## 7. Next steps (in rough priority)
1. **Review think_v1** (section 4). If thinking-phase injection surfaces memories as thoughts and flips balanced yes/no answers, that's the route to "premises".
2. **Capacity curve** on bench_v1's 60-fact pool with the current design (gist + threshold + RLS vs delta, N = 1 → 60).
3. **Strength (α) sweep with the current design:** now that leakage is controlled by the threshold, higher α may turn the gist into the fact. Watch for degeneration on related prompts.
4. **Rewrite the flagged bench_v1 preference items** (21/48 flagged; the calibration report lists 37 prompts to rewrite).
5. **Remove the scaffold:** automate "without" by leave-one-out re-runs (choose the sentences with attention); replace the hand-made "opposite" with the model's *typical state for the topic*.
6. **Neuromodulated memory:** an endogenous modulator state (surprise, uncertainty, arousal) controlling write strength, fading, and read gain; sparse, phasic writing. The design notes are in where-we-are §8 and history/open-notes N5–N9.

## 8. Open decisions for the user
- `experiments/v0/run.py` used to point to the parent-folder `Idea.md`; it now points to `docs/reference/original-method.md`. OK?
- The parent folder `/home/vedantjumle/projects/memories/docs/` still holds the original copies of the three reports. Delete them so the repo is the single source?
- `docs/reference/code-map.md` doesn't describe the newer experiments (`diag_attn`, `diag_keys`, `bench_v1_calib`, `samples_v2`, `think_v1`, `src/seahorse/bench/`, `slurm/run_v100.slurm`), and `docs/history/experiment-log.md` stops at the diagnostics. Fill them in?
- A note for the **other session** (the episodic "ladder" work in the same repo), not yet sent because which session is unknown: relation probes must be **balanced Yes/No** and **exclude the chat template tail**, or a general "No" drift fakes gains.

## 9. Working conventions
**Cluster (DelftBlue):** `ssh delftblue`; repo at `/scratch/vvjumle/Seahorse`.
- `gpu-a100-small`: a 10GB MIG slice, ≤2 CPUs, ≤8000MB per CPU, 4h. Script: `slurm/run.slurm`.
- `gpu-v100`: 32GB, **fp32** (no bf16), ≤5333MB per CPU. Script: `slurm/run_v100.slurm`.
- The account caps jobs at **1 day**.
- Compute nodes have **no internet**: pre-download models on the login node into `/scratch/vvjumle/hf_cache`.
- Every job runs `pytest` first (about 4–6 min).
- **Maintenance windows cancel the queue.**
- No heavy compute on the login node.

**Local:** `conda run -n torch` (torch on CPU, transformers 4.46) for smoke tests and merges. It can't load Qwen3.5.

**How the user likes to work:**
- discuss ideas conceptually ("think mode"); plans only when asked
- runs go to **subagents**, which submit jobs, confirm they started, and **don't babysit them**
- docs written in **plain language** (the style of the docs/ trio), not paper style
- use the Read tool (not sed) to view files
- Seahorse commits and pushes are allowed (end commit messages with the attribution line)
- **Measurement hygiene is non-negotiable:** balanced yes/no probes, sampled rates, random baselines, a placebo/noise floor, template tokens excluded

**Two research threads share this repo:**
- **this thread:** the modulatory "plastic steer" memory and neuromodulation
- **another session:** the episodic "compression ladder" (reinstating stored moments into attention)
