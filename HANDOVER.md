# Seahorse: handover

*The live handover: the current state, where everything is, what's pending, and how to pick up. Updated 2026-10-11. For the story and the reasoning, read [README.md](README.md), then the plain-language docs it lists.*

---

## 1. Where things stand

A training-free memory for a frozen LLM.
- **Writing:** an experience in session 1 is stored as a **shift** of the model's internal state, filed under a **whitened gist key** of the conversation.
- **Reading:** in any later session, the shift is added back when the current gist clearly matches a stored one.

It was built on **Qwen2.5-1.5B-Instruct** (layer 26 of 28) and ported to **Qwen3.5-2B** (layers 20, 21, 23 of 24) and **Qwen3.5-9B** (27, 28, 31 of 32), with thinking off by default.

The latest results come from **core_v1**, a pre-registered run on both Qwen3.5 sizes, judged by a model for direction *and* coherence (§4).

**What it achieves:**
- **Selective.** About 98% of unrelated answers are unchanged, and the gate is what does it: removing it leaves only 6–10% unchanged.
- **Order-free.** Least-squares writes.
- **Leanings, modestly, at the right dose.** At α≈1 per layer, answers lean toward a stored preference significantly on both models, but by little (+0.1 to +0.2 on a −1..+1 scale) and unevenly across items. At α=2 the text turns incoherent.
- **Direction via the opposite.** Subtracting the hand-made opposite is what keeps a dislike a dislike. This is shown on 8 dislikes, on both models.

**What it doesn't:**
- **Exact facts.** A fact memory stores *a word to say*, not a statement. It exists only in the last ~15% of layers. Injected there, it loops ("Petra Petra Petra"); injected earlier, it fades or garbles ("My name is Omar"). Clean use is ≤ 15%, against 96–98% in context.
- **Premises.** Balanced yes/no never improves on any model, layer or dose.
- **Dislikes without the opposite.** Every other reference flips them into likes.
- **Single-item preferences** (a country, a genre) either flood or do nothing.
- **Ownership.** The model claims memories as its own ("Teal is my favourite colour!").
- **Self-driven writing.** It still needs hand-made with/without/opposite comparisons.

**Where the project is now (2026-10-11):**
- **The paper is on arXiv**: *"What Survives When Activation Steering Is Used as Memory"* (`writeup/main.tex`, `main.pdf`). It commits the project to one reading: **the steer channel is a push, not a store.** A stored shift is a fixed average of an effect attention recomputes per question, so only question-independent content survives.
- **The judge was validated by a cross-family panel** (gpt-oss, Gemma-4, Llama-3.1, Qwen3.6) instead of the human audit: C3 and C5 are robust; C2 is judge-sensitive; Llama-3.1 fails a sanity check and is excluded (§4).
- **The episodic channel does not exist in code.** It was always a design (`docs/history/next-moves-v0.1.md` §A2). Nothing from the "other session" is in this repo.
- **The next phase is "dreaming"**: an offline phase that gives the model's own binding circuits the material they need. Three experiments, specified in `docs/dreaming/` (§4b).

## 2. Read first
1. [README.md](README.md): the overview and reading order
2. [docs/where-we-are.md](docs/where-we-are.md): the whole story, including Qwen3.5 and what a shift stores (stages 7–8)
3. [docs/how-it-works.md](docs/how-it-works.md): the design end to end (written for Qwen2.5; the port changes only model, layers and thresholds)
4. [docs/where-it-fails.md](docs/where-it-fails.md): every failure (now 16), its cause and possible fixes
5. [docs/reference/metrics.md](docs/reference/metrics.md): **what every number in the reports means**, with traps. Read it before trusting any table.
6. [docs/reference/literature.md](docs/reference/literature.md): the steering-reliability and introspection papers, and how they line up with our results
7. [experiments/core_v1/PREREG.md](experiments/core_v1/PREREG.md) and [README.md](experiments/core_v1/README.md): the pre-registered claims behind the write-up, and how to reproduce them
8. [docs/dreaming/README.md](docs/dreaming/README.md): what comes next, and the exact reading list for building it (§4b)

## 3. The current best design
| Piece | Qwen2.5-1.5B (original) | Qwen3.5-2B (current) |
|---|---|---|
| Model / layers | fp32; read and write at the output of layer 26 of 28 | fp32, text path; layers **23, 20, 21** of 24, each with its own keys, M and threshold; thinking **off** |
| Write unit | one moment per follow-up sentence: an entropy-weighted mean shift over its words, chat template tail excluded | same |
| Shift | facts: plain (with − without); preferences: contrastive (with − opposite) | same. **Only the opposite keeps a preference's direction** (ref_v1); others lose it. |
| Key | `pooled_w256`: running mean over user words of PCA-256-whitened (state − background), normalised | same (the Ledoit–Wolf variant is clearly worse on Qwen3.5) |
| Store | delta rule for one memory; **least squares (RLS, λ=0.1)** for several | same |
| Read | at every word: max cosine to stored keys; hard threshold at the 95th percentile on ordinary prompts; `h ← h + α·gate·M·q`, α=2; never on the system prompt | same, at each of the three layers, with gates from a no-memory pass ("two-pass"). **Use α≈1 per layer:** α=2 on three layers overdoses (core_v1). |

**The 9B** (core_v1): the same design at layers 27, 28 and 31 of 32 (the same relative depth), fp32 on a full A100 80GB.

Code:
- `experiments/diag_keys/run.py`: the definitive Qwen2.5 implementation
- **`experiments/core_v1/`: the current, consolidated Qwen3.5 implementation**, on both sizes, with batched two-pass generation, the judge and the analysis
- `experiments/ref_v1/`, `experiments/think_v1/run.py` and `experiments/xlayer_v1/`: the earlier Qwen3.5 experiments

## 4. Latest: core_v1, the pre-registered run behind the write-up (done 2026-10-08)

**What it is:** one consolidated, reproducible run of the core claims on **two scales**.

| Model | Precision | Hardware | Layers |
|---|---|---|---|
| Qwen3.5-2B | fp32 | V100 | 20, 21, 23 |
| Qwen3.5-9B | fp32 | A100 80GB | 27, 28, 31: the same relative depth |

- **Claims C1–C7 and their decision rules** were written *before* the run, in `experiments/core_v1/PREREG.md`. `experiments/core_v1/README.md` has the reproduction commands; `analyze.py` rebuilds every table from the saved files.
- **Items:** 15 leanings, 8 dislikes, 6 one-of-many, 12 facts, 50 unrelated prompts.
- **Doses:** α ∈ {0.5, 1, 2, 3}.
- **Controls:**
  - a random direction
  - a swapped memory
  - a placebo experience ("I had a coffee this morning")
  - the gate removed
- **Primary measures** come from a **judge model** (Qwen3.5-9B, bf16): direction *and* coherence.
  - **Leanings:** `j_lean` = +1 for a coherent recommendation toward the trait, −1 away, 0 otherwise.
  - **Facts:** `use_judge_clean` = the fact used correctly in a coherent answer.
- **Statistics:** 95% CIs from a two-level bootstrap (items, then prompts).
- **Results:** `results/core_v1_20261007_1820/`.
  - per-model `analysis/report.md`
  - `analysis/C7.md` and `claims_all.json`: the combined verdicts

**Verdicts (pre-registered rules):**

| Claim | 2B | 9B | In plain words |
|---|---|---|---|
| C1 Selectivity | holds | holds | About 98% of unrelated answers unchanged. With the gate removed: 6% (2B) and 10% (9B). |
| C2 Leanings transfer | **fails** | **fails** | Gain at α=1: +0.20 [0.04, 0.38] (2B), +0.11 [0.004, 0.24] (9B). At α=2 the CI includes 0, because judged incoherence jumps to 61% (2B) and 41% (9B) while word counts still rise. The rand control's paired CI also overlaps. |
| C3 Facts aren't cleanly used | holds | holds | Clean judged use ≤ 0.15, against 0.96–0.98 in context. At α ≥ 2, 85–94% of answers mention the fact without a clean use. |
| C4 No premises | **fails** (technicality) | holds | No yes/no gain anywhere; the margins are a general Yes/No tilt. On 2B, facts at α=2 *lowered* balanced accuracy (−0.15, CI excludes 0), which the rule counts as a failure. |
| C5 What you subtract decides direction (dislikes) | holds | holds | At a loop-matched α=1: opposite +0.23 / +0.25; centroid −0.38 / −0.37 (wrong way on every item); without −0.25 / −0.31 |
| C6 A fact shift is a late-layer word | holds | holds | Median target rank ≤ 10 only at relative depth ≥ 0.83 (2B) and ≥ 0.875 (9B); about 70,000 at half depth |

**What it changes:** the judge shows that **α=2 on three layers was always an overdose.** Earlier word-count wins (e.g. vegetarian at 94% in think_v1) included a lot of incoherent text. Leanings are real but modest, and only in a narrow window around α=1. They're also uneven across items:
- strong for early_riser, prefers_quiet, vegan, vegetarian
- nothing for celiac, lactose_intolerant, has_young_kids

**Judge validation (done 2026-10-10, replaces the planned hand audit):** all 34,200 answers re-judged by gpt-oss-20B, Gemma-4-12B, Llama-3.1-8B and Qwen3.6-35B-A3B (`results/core_v1_20261007_1820/panel/panel_report.md`, code `tools/audit/`).
- **C3 and C5 hold under every competent judge** and both majority votes; gpt-oss and Gemma give *larger* C5 effects (centroid −0.42 to −0.53).
- **C2 fails under every single judge but holds under majority votes**: the original judge is the strictest on coherence (62% vs 72–76%). Reported as judge-sensitive.
- **Llama-3.1-8B is excluded** by an objective check: it labels 92% of fact answers as "uses the fact", with the target string absent in 48% of those (others ≈ 1%).
- Coherence agreement is high (Fleiss κ 0.90 on facts); direction agreement moderate (κ 0.45–0.69). No human labels were collected; the paper says so.

## 4b. Next: the dreaming experiments (specified, not yet built)

The paper settled that facts, premises and ownership fail because a push at the answer position has nothing the model's **binding** circuits can use. The binding literature (`docs/reference/literature.md`, group E) says binding is positional: in context, a shared order tag at entity and attribute positions resolved by position-finding attention heads (Feng & Steinhardt; Dai; Prakash); from weights, the attribute loaded onto the subject's own position and extracted by attention (Geva). So the fix is not a better vector: it is to put stored content at a *position* the model's circuits treat as context. That work happens offline, hence "dreaming".

| # | Doc | Question | Status |
|---|---|---|---|
| 1 | `docs/dreaming/01-binding.md` | **Stage 0 first:** attention knockout in `ctx` to learn whether an in-context fact sits at the subject or only at its own position (if the latter, the subject write is dead on arrival). Then: write the model's own subject-enriched state at the subject token; compare with *gated* segment reinstatement (an upper bound by P1). 7 clean-subject facts primary, 5 user-subject secondary. | specified; **run stage 0 first** |
| 2 | `docs/dreaming/02-scaffold.md` | Let a model (the 9B dreams for the 2B; the 9B end-to-end for the literal claim) generate opposite / alternatives / a general valence direction. Does C5 replicate without human references, beyond two null references? Does concept × sign fix the dislike flip? | specified; independent of 1 |
| 3 | `docs/dreaming/03-consolidation.md` | With many memories: abstraction of leanings from episodes, reconsolidation, surprise-gated pruning vs write-everything (ground truth = *content-dependent* later need, not a random split), re-filing | design; depends on 1 and the capacity curve |

**To be fully equipped to build and run them, a new session should read, in this order:**
1. `HANDOVER.md` (this file), all of it.
2. `docs/where-it-fails.md`: the 16 failures and the root-cause section at the end. The dreaming experiments target failures 1, 2, 7, 9, 13, 15.
3. `docs/reference/literature.md`, **group E** and the theory subsection *"Why a push can't bind, and what would"*: the mechanism the experiments rely on. Then group C (Geva, ROME, MEMIT).
4. `docs/dreaming/README.md`, then `01-binding.md` and `02-scaffold.md` in full: the exact processes, predictions, controls and decision rules.
5. `docs/reference/metrics.md`: every measure, the traps, and §7b (judged measures, bootstrap, pre-registration).
6. `experiments/core_v1/README.md`, `PREREG.md`, and the code to build on: `core.py` (layer rule, conditions, batched two-pass hook and generation, judge parser), `run.py` (stages prep / gen / score), `analyze.py` (tables, CIs, claim rules), `items.yaml` + `data/bench_v1/facts.yaml` (the fact items, measure prefixes, foils, balanced yes/no probes, use prompts), `judge_prompts.yaml`.
7. `experiments/xlayer_v1/xl.py`: the two-pass gating and cross-layer injection code (experiment 1 injects at a *position*, which needs the same no-memory first pass).
8. `tools/audit/` and `results/core_v1_20261007_1820/panel/panel_report.md`: how the judge panel is run and read.
9. `docs/how-it-works.md` §2–§7 if the key / whitening / threshold machinery is unfamiliar.
10. The papers, if the mechanism needs checking: `../papers/Dissecting Factual recall.pdf`, `Binding IDs in LLMs.pdf`, `Entity Tracking Circuit.pdf`, `Representational Analysis of Binding.pdf`.

**Facts a builder needs that aren't obvious from the code:**
- Qwen3.5-2B has full attention only at blocks **3, 7, 11, 15, 19, 23** (`layer_types` in the model config); the rest are Gated DeltaNet with no KV cache. A residual-stream write works at any block; a KV splice only at those six. Use **Qwen2.5-1.5B** (28 full-attention layers, already calibrated in `diag_keys`) for the reinstatement arm.
- core_v1's `prep/state.pt` holds every item's with/without/opposite/centroid shifts, keys and thresholds at the three late layers; experiment 2 reuses it. Experiment 1 needs new harvests at early-middle layers (the subject token), so it reads states at all blocks.
- The fact use prompts don't all contain the subject literally (e.g. *"the place where I grew up"*); `01-binding.md` §3 says how to handle that (aliases, and "no-site" prompts reported separately).
- The dreamer for experiment 2 is the 9B judge model on an A100 via `slurm/core_v1.slurm`; generation of references takes about half an hour.
- Both experiments must have a `PREREG.md` committed before any job is submitted, as core_v1 did. The predictions and rules are already written in the docs; copy them.
- The specs were reviewed twice (2026-10-11) and revised: stage 0 knockout with thresholds (knockout on Qwen2.5 only, since Gated DeltaNet blocks can't be masked per token; patching on both), the H2 causal-mask bug, gated reinstatement, item tiers, S1 nulls, §2c ground truth (one author for both label classes; a cross-family dreamer; LongMemEval/LoCoMo as external labels). Read the revised versions, not any earlier copy.

## 5. Recently finished and reviewed

### xlayer_v1: read the fact late, inject it earlier ("mouth" vs "mind")? Negative.
- **Setup:** 12 facts on Qwen3.5-2B.
  - **Stage A:** a grid of read layer R {12,16,20,23} × inject layer W {8,…,23} × α, with no text generated.
  - **Stage B:** generation, including "use" prompts where the fact is needed but not asked for.
- **The fact exists only late.** In the logit lens of the stored shift, the name is the top word at layers 20/23 and ranked about 30,000–100,000th at 12/16. Memories read at layer 16 do nothing, wherever they're injected.
- **Earlier injection fades rather than turning into a thought.** At layer 8, about 10% of the memory's push toward the right name survives; at 16, about two-thirds; at 20, all of it. The later layers never amplify it.
- **Yes/no:** unchanged. The margin shifts are a general Yes/No tilt.
- **The best-scoring setting is broken text, not loops.** Read {20,21,23} → inject {14,15,16} scored best on "clean" hits (0.35 related, 0.27 use, vs 0.19 / 0.09 for the current design, with few repetition loops). But most of those answers are garbled or identity-confused:
  - "My name is Omar."
  - "I am an AI named Pepper"
  - "It's a very Leeds."

  56% of its answers are under 40 words (8% with no memory).
- **Lessons:**
  - A fact memory is *an intention to say a word*, not a statement.
  - "Not a loop" ≠ coherent. This led to the judge and coherence measures in core_v1.

### think_v1: the memory on Qwen3.5-2B, including injection during thinking
- **Layer sweep:** layers 19–23 carry facts. Selectivity holds at every layer. Balanced yes/no is at chance at every layer.
- **Several layers at full strength (23+20+21)** bring the fact out far more than one layer. **But mostly as loops:**
  - In short continuations, the fact appears in 91% of answers. Only 22% are clean (said once or twice); **70% are loops**.
  - The report's "84% of answers name the fact" (answer-only memory) is really **about 4 of 44 clean answers**, against 28 or more of 44 with the fact in context.
- **Amplified injection during thinking only** turns the thinking into "Pepper Pepper… (×383)", and the answer ignores it ("I don't know your dog's name!"). A weak dose during the answer recovers a little.
- **Yes/no:** balanced accuracy is at chance in every memory condition.
- **Other failures:**
  - The model claims memories as its own: "Teal is my favourite colour!"
  - The combined RLS memory gives nothing in thinking mode (not explained).
- **Design flaws:**
  - The 384-token thinking cap is hit about 95% of the time *even with no memory*, so many "answers" are leftover thinking.
  - The stage choosers rewarded overdose.
- **Preferences per item (with answer-only memory):**
  - vegetarian: 12% → 94% consistent answers
  - hiking: 24% → 39%; with memory during thinking too, 76%
  - norway: mostly loops
  - jazz: little or no effect
- **Readable raw outputs:** `results/think_v1_20261003_0004/raw_outputs.md`

### ref_v1: which reference should a preference shift subtract?
- **Setup:** 11 preferences (4 two-ended, 6 one-of-many, 1 negated: "I can't stand jazz") × 5 references × 2 doses. Every shift is rescaled to the length of the `without` shift.
- **Opposite** is the **only reference that keeps the direction**:
  - hates_jazz +0.48 and no_alcohol +0.24, while every other reference flips them (−0.4 to −1.0, i.e. toward jazz and toward beer)
  - it also has the fewest loops
  - but it **loses the concept** when both sides mention it: the jazz fan memory suggests "podcasts, audiobooks"
- **Centroid** (minus the average of 8 alternatives in the same category) **carries the concept most strongly.** In the logit lens, jazz scores +9.3 vs +2.5 for opposite. But it floods (jazz loops 91% at α=1), and it **loses the sign** ("I can't stand jazz" → "jazz up your meal with jazz jazz…").
- **Hum** (minus the global average) ≈ **the memory's own key**: cosine about 0.8 with the key, about 0.9 with the topic. At α=2 it breaks answers into fragments. Dead end.
- **Without / disclosure** sit in between. They also flip dislikes.
- **Countries** (Norway, Singapore) never get a clean lean above about 0.5 with any reference.
- **Caveat:** rescaling every shift to the same length boosts the centroid about 2.5×. A fair comparison would match strength by loop rate.

## 6. Results on disk (local `results/`, untracked)
| Folder | What |
|---|---|
| `v0_408770`, `v0_1_415909` | the original experiments |
| `diag_dose_577895`, `diag_order_577896` | dose/selectivity diagnostics; write-order (recency) test |
| `diag_attn_846339` | attention-based write selection (no-go: token selection doesn't matter) |
| `bench_v1_calib_850635` | benchmark calibration (no memory): noise floor, flagged items, the "No" bias |
| `diag_keys_858681`, `diag_keys_858682` | key/selectivity experiment, parts 1 and 2. The merge needs torch: `conda run -n torch python experiments/diag_keys/run.py --merge …` |
| `samples_v2_873325` | real text from the Qwen2.5 design (the source of where-we-are appendix C–D) |
| `think_v1_20261003_0004` | the Qwen3.5 port: layer sweep, single vs multi-layer, strength during thinking vs answer; plus `raw_outputs.md` (readable, loops collapsed) |
| `ref_v1_20261005_2347` | five references for preference shifts (`main/report.txt`, `main/vocab.txt` for the logit lens) |
| `xlayer_v1_20261007_1311` | cross-layer injection for facts (`stageA/report.txt` grid, `stageB/report.txt` generation) |
| `core_v1_20261007_1820` | **the pre-registered run on 2B + 9B**: `qwen35_2b/` and `qwen35_9b/` (prep, gen, score, analysis/report.md); `analysis/` (C7.md, claims_all.json, the audit sample); `panel/` (the cross-family judge re-labelling and its report) |

## 7. Key findings to remember
- **Selectivity, not strength, was the original problem.** The fix was the **key**: the whitened gist plus a threshold.
- **Recency:** step-by-step writing means the last memory wins. **Least squares** removes the order dependence; contradictions then average.
- **The moment is the unit of memory.** Token-level write selection doesn't matter. Attention finds *relevance*, not *importance*.
- **No premises,** on both models, at every layer, with or without thinking. Earlier "relation gains" were artefacts of all-"No" probes and template tokens.
- **Filing:** a memory's key is the gist of its *follow-up* sentences, so it fires only in similar situations.
- **A preference has parts:**
  - *what* (the concept)
  - *which way* (like or dislike)
  - *whose* (the user's, not the assistant's)

  The reference you subtract decides which part is stored: opposite keeps *which way*; centroid, without and disclosure keep *what*; hum keeps only the topic.
- **Negation is faint inside the model.** "I can't stand jazz" ≈ jazz. Lindsey 2026 finds the same: "don't think about X" still activates X.
- **Steering is a constant push at every word.** It suits whole-answer traits (a diet) and fails for one-slot content (a name, a country), which either never wins or floods.
- **Overdose signature:** loops, fragments, garbled text, identity confusion. **The useful dose window is narrow: about α=1 per layer on three layers** (core_v1). At α=2, word counts keep rising while the text turns incoherent.
- **Injected states feel like the model's own,** hence the self-attribution (Lindsey's prefill result).
- **A fact memory is a word, not a statement.** It's an intention to say the name, present only in the last ~15% of layers. Injecting it earlier fades or garbles it (xlayer_v1; core_v1 C6 on both sizes). Facts probably live where they were said, reached by attention, which is the episodic channel's job.
- **Scale didn't change the picture.** 2B and 9B agree on every claim except a technicality in C4.
- **Measurement:**
  - word counts are inflated by loops; word lists count mentions and puns
  - "not a loop" ≠ coherent
  - yes/no must be balanced
  - judge for direction *and* coherence

  See `docs/reference/metrics.md`.

## 8. Next steps and ideas (in rough priority)
1. **Experiment 1, dreaming as binding** (`docs/dreaming/01-binding.md`): the subject-position write with the model's own harvested state, plus segment reinstatement on Qwen2.5-1.5B. The first experiment that could make a fact *used*.
2. **Experiment 2, scaffold removal and concept × sign** (`docs/dreaming/02-scaffold.md`): model-generated opposite / alternatives, a general valence direction, and the signed-concept write. Independent of 1; can run in parallel.
3. **Experiment 3, consolidation** (`docs/dreaming/03-consolidation.md`): start with 2c, the surprise-predicts-need test, which decides whether the modulator idea survives. Then abstraction, re-filing, reconsolidation. Needs 1 and the capacity curve.
4. **The workshop version of the paper** (ICLR 2027 workshops, list out 29 Nov; NAACL SRW as the alternative): a 4-page cut from `writeup/main.tex`. Fix the empty Appendix D header and the float placement first.
5. **Paper debts a follow-up inherits:** a second model family (non-hybrid), human labels on a subset, isolated memories only.
6. Still open from before, now subsumed by the above: a thermostat instead of a push (dose), phasic injection during thinking, finer dose sweeps, the capacity curve, a trained read adapter as the fallback.

**Do not** spend more time improving the steer vector itself. The paper says that channel carries a push, not content; it keeps that job (the abstracted leaning of experiment 3) and nothing else.

## 9. Open decisions for the user
- `experiments/v0/run.py` used to point to the parent-folder `Idea.md`; it now points to `docs/reference/original-method.md`. OK?
- The parent folder `/home/vedantjumle/projects/memories/docs/` still holds the original copies of the three reports. Delete them so the repo is the single source?
- The episodic "ladder" work from the other session never reached this repo. The content channel is now specified here as `docs/dreaming/01-binding.md`. If that session has code, it should be merged against this spec; otherwise treat the spec as the single source.
- `docs/reference/glossary.md` sits uncommitted in the working tree (from the write-up chat). Commit or remove it.

## 10. Working conventions
**Framing:** Seahorse is an **independent side project**, not a thesis or supervised research. In the docs, "thesis" only ever means the project's central hypothesis. Never describe it as MSc or thesis work in anything written for others.

**Cluster (DelftBlue):** `ssh delftblue`; the repo is at `/scratch/vvjumle/Seahorse` (`git pull` there after pushing).
- `gpu-v100`: 32GB, **fp32** (no bf16), ≤5333MB per CPU. Script: `slurm/run_v100.slurm`. This is what the Qwen3.5 experiments use.
- `gpu-a100-small`: a 10GB MIG slice, ≤2 CPUs, ≤8000MB per CPU, 4h. Script: `slurm/run.slurm`. Used for the Qwen2.5 work.
- `gpu-a100` (full A100 80GB): used for the 9B in fp32 and for the judge (bf16). The core_v1 chains are in `slurm/core_v1.slurm` and `experiments/core_v1/submit.sh`.
- Don't run the 9B in fp16 on a V100: Qwen risks overflow.
- The account caps jobs at **1 day**.
- Compute nodes have **no internet**. Models are pre-downloaded into `/scratch/vvjumle/hf_cache`.
- Every job runs `pytest` first (about 2–5 min).
- **Maintenance windows cancel the queue.**
- No heavy compute on the login node.
- The remote `seahorse` env has transformers 5.18; `environment.yml` pins `>=5.18,<6`. Linear-attention speed-up kernels aren't installed (the pure-torch path is used).
- Rough speed: about 1,200 generations of ≤200 tokens take about 110 min on a V100.

**Local:** run all Python with `conda run -n torch` (CPU torch, transformers 4.46). It **can't load Qwen3.5**: use it for tests, import checks and tiny random-model checks. It has no pytest; put pytest in a scratch folder on `PYTHONPATH` rather than installing it into the env.

**How the user likes to work:**
- Discuss ideas conceptually ("think mode"); plans only when asked.
- Runs go to **subagents**, which build, test, commit, submit, confirm the jobs started, and **don't babysit them**. The user often stops them once jobs are queued.
- Docs and explanations in **plain language**, with real examples, not paper style. Numbers need context.
- Use the Read tool (not sed) to view files.
- Seahorse commits and pushes are allowed. End commit messages with the attribution line.
- **Measurement hygiene is non-negotiable:**
  - balanced yes/no probes
  - sampled rates
  - loops counted separately
  - random baselines
  - a placebo / noise floor
  - template tokens excluded
  - read the actual text

**Two research threads share this repo:**
- **this thread:** the modulatory "plastic steer" memory and neuromodulation
- **another session:** the episodic "compression ladder" (reinstating stored moments into attention)
- A third chat handles the public manuscript page (`tools/manuscript/`).
