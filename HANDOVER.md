# Seahorse: handover

*The live handover: the current state, where everything is, what's pending, and how to pick up. Updated 2026-10-13. For the story and the reasoning, read [README.md](README.md), then the plain-language docs it lists.*

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
- **Exact facts.** A fact memory stores *a word to say*, not a statement. It exists only in the last ~15% of layers. Injected there, it loops ("Petra Petra Petra"); injected earlier, it fades or garbles ("My name is Omar"). Clean use is ≤ 15%, against 96–98% in context. *(Update 2026-10-13: the steer channel still can't; a stored attention slice can, 0.86 to 0.97. See dream_bind below.)*
- **Premises.** Balanced yes/no never improves on any model, layer or dose. *(Through the steer channel. With a stored slice put back, balanced yes/no reaches 0.86 to 1.00.)*
- **Dislikes without the opposite.** Every other reference flips them into likes.
- **Single-item preferences** (a country, a genre) either flood or do nothing.
- **Ownership.** The model claims memories as its own ("Teal is my favourite colour!").
- **Self-driven writing.** It still needs hand-made with/without/opposite comparisons.

**Where the project is now (2026-10-13):**
- **Experiment 1 of the dreaming plan has been run (2026-10-11 to 13), and facts finally have a working channel.** It is not the one planned:
  - **The subject write is dead.** The answer reads the value from the attribute's own position ("Pepper"), late; the question's subject holds no copy of the value (patching 0.00 on both models; it may still carry an order-ID pointer, see docs/dreaming/literature-exp1.md). Writing a state at the subject gave clean use 0.00.
  - **What works is a stored slice of the model's own attention memory (K,V), put back in the next session and gated by a subject match.** Qwen2.5-1.5B: the whole turn 0.87, just the name and the full stop 0.86, stored after an unrelated chat 0.80, with the sentence in context 0.90. The same two tokens as plain text: 0.27.
  - **On the hybrid Qwen3.5-2B the attention slice needs the recurrent (DeltaNet) state too:** 2 positions plus the state 0.97, attention only 0.50 to 0.67, state only 0.00 use but 0.93 yes/no (recognition without recall). That state is about 4.7M numbers (~19 MB in fp32) whatever the text length, so it is not compact for one fact.
  - **Separately stored slices collide when used together** (wrong partner's name 19% to 37% of the time, even for different subjects). Encoding them together fixes it. This is the first concrete job for the offline phase.
  - Honest framing: the object is a standard KV cache used outside its normal rules. Prompt Cache, CacheBlend, EPIC and InfLLM do parts of it; our contribution is the explanation. Full numbers: `docs/dreaming/01-binding.md`, Results.
  - Running now (addendum 6): binding-ID repair, and G-key retrieval with capacity (N = 10/20/60). A literature check is being written (`docs/dreaming/literature-exp1.md`).
- **The paper is on arXiv**: *"What Survives When Activation Steering Is Used as Memory"* (`writeup/main.tex`, `main.pdf`). It commits the project to one reading: **the steer channel is a push, not a store.** A stored shift is a fixed average of an effect attention recomputes per question, so only question-independent content survives.
- **The judge was validated by a cross-family panel** (gpt-oss, Gemma-4, Llama-3.1, Qwen3.6) instead of the human audit: C3 and C5 are robust; C2 is judge-sensitive; Llama-3.1 fails a sanity check and is excluded (§4).
- **The episodic channel now exists as experiment code** (`experiments/dream_bind/`: harvest, gated K,V reinstatement, the DeltaNet-state variants, the collision test). It was a design before (`docs/history/next-moves-v0.1.md` §A2). Nothing from the "other session" is in this repo.
- **The phase is "dreaming"**: an offline phase that does what cannot be done in one forward pass at write time. Three experiments, specified in `docs/dreaming/` (§4b). Experiment 1 is done (above) and changed what dreaming means: for single facts it is keeping the right slice; the real offline jobs are binding related memories, abstracting leanings into the steer channel, and choosing what to keep.
- **Two stories under one idea:** (1) understanding (paper 2: what binds a memory: steering vectors, cache slices, or hybrid state); (2) a memory system whose offline phase does what retrieval over text does not (binding, abstraction, selection).

## 2. Read first
1. [README.md](README.md): the overview and reading order
2. [docs/where-we-are.md](docs/where-we-are.md): the whole story, including Qwen3.5 and what a shift stores (stages 7–8)
3. [docs/how-it-works.md](docs/how-it-works.md): the design end to end (written for Qwen2.5; the port changes only model, layers and thresholds)
4. [docs/where-it-fails.md](docs/where-it-fails.md): every failure (now 17), its cause and possible fixes
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

## 4b. The dreaming experiments (1 done, 2 and 3 specified)

The paper settled that facts, premises and ownership fail because a push at the answer position has nothing the model's **binding** circuits can use. The binding literature (`docs/reference/literature.md`, group E) says binding is positional: in context, a shared order tag at entity and attribute positions resolved by position-finding attention heads (Feng & Steinhardt; Dai; Prakash); from weights, the attribute loaded onto the subject's own position and extracted by attention (Geva). So the fix is not a better vector: it is to put stored content at a *position* the model's circuits treat as context. That work happens offline, hence "dreaming". **Experiment 1 confirmed the first half (position, not vector) and replaced the second (the content is the attribute's own K,V slice, not a state at the subject).**

| # | Doc | Question | Status |
|---|---|---|---|
| 1 | `docs/dreaming/01-binding.md` | Can a fact be used if stored at a position instead of pushed? Stage 0 (where is the fact in context?), the subject write, gated reinstatement, the hybrid's recurrent state, a text control, and the two-fact collision. | **done 2026-10-13** (details in §5). Follow-ups running: binding-ID repair, G-key retrieval + capacity |
| 2 | `docs/dreaming/02-scaffold.md` | Let a model (the 9B dreams for the 2B; the 9B end-to-end for the literal claim) generate opposite / alternatives / a general valence direction. Does C5 replicate without human references, beyond two null references? Does concept × sign fix the dislike flip? | specified; independent of 1 |
| 3 | `docs/dreaming/03-consolidation.md` | With many memories: abstraction of leanings from episodes (into the steer channel), reconsolidation, surprise-gated selection vs write-everything (ground truth = *content-dependent* later need, not a random split), re-filing. Now also: binding related memories (the collision result). | design; depends on 1 (done) and the capacity curve (running) |

**Note (2026-10-13):** the text below ("To be fully equipped...", "Facts a builder needs...") was written for building experiment 1 and is kept for experiments 2 and 3, which reuse the same pipeline. The statements about the subject write and "run stage 0 first" are historical.

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

### dream_bind (dreaming experiment 1): where does a fact live, and what can be stored? Positive, but not the planned way. (2026-10-11 to 13)
- **Code and pre-registration:** `experiments/dream_bind/` (`PREREG.md` addenda 1 to 5; `run.py` stages knockout / patch / harvest / analytic / gen / collide). Qwen2.5-1.5B locally on the RTX 4060 (fp32); Qwen3.5-2B on DelftBlue (fp32). Clean use = majority of three judges (original 9B, gpt-oss, Gemma-4).
- **Stage 0 (pre-registered): the answer reads the value from the attribute's own position, late.** Patching the question's subject flips 0.00 at every layer on both models; patching the attribute flips 1.00 until layer ~18 (Qwen2.5) or layer 14 to 18 (2B), then drops to ~0 (2B: 0.06 at layer 19). Qwen2.5 knockout: blocking the attribute costs 0.59 [0.41, 0.81] in the last window only; blocking the subject costs 0. Subject-alone net gain 0.00. Verdict: pointer-suspected, both models.
- **The subject write is dead.** Analytic pre-pass: Qwen2.5 a nominal gain (1.56 [0.69, 2.56] nats at the best cell) matched by random (0.88) and swapped (0.75) controls; 2B none (0.29 [-0.05, 0.66]). In generation on Qwen2.5 the write gave clean use 0.00 and yes/no 0.50 (B1, B2 fail; B3, B4 hold).
- **Reinstatement works (Qwen2.5, gated on a subject-name match):** R-full 0.87 clean use and 1.00 yes/no; R-min (the name's position and the full stop, K,V at every layer) 0.86 and 0.86; R-orphan (stored after an unrelated exchange) 0.80 and 1.00; ctx 0.90. Unrelated prompts unchanged 1.000 gated vs 0.064 ungated. A swapped slice writes the wrong value (41% of use answers). Self-claim about 0.01. Pre-registered verdict: **"orphaned; reinstatement wins"**.
- **Text control:** the same two tokens as plain text (T-min) give 0.27 (Qwen2.5) and 0.24 (2B) against 0.87 and 0.97 for the stored slice, and 0.00 on the direct question on Qwen2.5. A slice carries more than its tokens. The full sentence as text equals R-full exactly (so this is prompt caching used across sessions).
- **Hybrid Qwen3.5-2B:** the attention slice only (6 of 24 blocks) 0.50 full / 0.67 min, refusing direct questions (0.23). Adding the DeltaNet state: R-min-state 0.97, within noise of the full turn. State only: use 0.00 and direct question 0.00, but yes/no 0.93: recognition without recall. R-orphan-state 0.99. The state is ~4.7M numbers (~19 MB fp32), fixed size: not compact for one fact.
- **Collisions (both models, gates forced open):** two separately stored slices put back together give the wrong partner's name in 0.19 to 0.37 of answers (confusable pairs 0.28 to 0.37; different subjects 0.19 to 0.28), whether the stored positions overlap or the second is shifted after the first. Encoding both in one pass gives ctx2-level results (Qwen2.5 correct 0.89 vs ctx2 0.93, confusion 0.05 vs 0.01). The two-position version is still confused on Qwen2.5 even when jointly encoded (confusion 0.20); on the 2B, min-joint with a joint state works (0.73 correct, 0.06 confusion). Reading (inferred, not measured): binding-ID collision.
- **Not run:** the 9B, subj on the 2B (skipped by rule), and anything with a retrieval key other than the subject-name match.
- **Reports:** see §6 for folders; each has `analysis/report.md`.

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
| `dream_bind_stage0_20261011_2321` | experiment 1, stage 0 and the analytic pre-pass: `qwen25_15b/` and `qwen35_2b/` each with `report.md` (patching, knockout, subject alone), `analytic/report.md` (subject-write pre-pass), `prefix/report.md` (exploratory: the subject mention in the answer prefix) |
| `dream_bind_gen_20261012_0038` | Qwen2.5 generation: subj arms, controls, R-full / R-min / R-orphan / R-swap (`qwen25_15b/analysis/report.md`) |
| `dream_bind_gen2b_20261012_1456` | Qwen3.5-2B hybrid: R-full-state, R-full-kv6, R-min-kv6, R-orphan-kv6 (`qwen35_2b/analysis/report.md`) |
| `dream_bind_state2b_20261012_1759` | 2B DeltaNet follow-up: R-min-state, R-state-only, R-orphan-state (`qwen35_2b/analysis/report.md`) |
| `dream_bind_add5_20261013_1255` | text controls (`qwen25_15b/`, `qwen35_2b/`) and the two-fact collision (`qwen25_15b_collide/`, `qwen35_2b_collide/`), each with `analysis/report.md`; `panel/` has the judge labels |
| `dream_bind_gen_smoke_local` | local smoke output (ignore) |

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
- **A fact lives at its own position, not at the subject** (dream_bind stage 0). The answer pulls the value from "Pepper" late in the network; "my dog" is a label. So writing a state at the subject cannot work (clean use 0.00), and the content channel is the attribute's own K,V slice, gated by a subject match.
- **Two stored positions carry the fact; the same tokens as text do not.** Name + full stop as K,V: 0.86 clean use; as plain text: 0.27. Reinstating the whole turn equals prompt caching exactly. Our contribution is the explanation, not the mechanism (Prompt Cache, CacheBlend, EPIC, InfLLM do parts of it).
- **On a hybrid, the recurrent state is recognition and the attention entries are recall.** State only: yes/no 0.93, use 0.00. But the state is ~19 MB and fixed in size, so it pays off only for long episodes.
- **Separately stored facts collide when used together** (the wrong person's name 19% to 37% of the time). It is not about position; joint re-encoding fixes it. This is the first real job for the offline phase (failure 17).
- **"Dreaming" for single facts reduces to keeping the right slice.** The offline work worth doing is binding related memories, abstracting leanings into the steer channel, and selecting what to keep.
- **Scale didn't change the picture.** 2B and 9B agree on every claim except a technicality in C4.
- **Measurement:**
  - word counts are inflated by loops; word lists count mentions and puns
  - "not a loop" ≠ coherent
  - yes/no must be balanced
  - judge for direction *and* coherence

  See `docs/reference/metrics.md`.

## 8. Next steps and ideas (in rough priority)
1. **Binding-ID repair** (running, addendum 6): measure the "which pair" direction in the states, and test whether adding it to one separately stored slice removes the swaps. If it does, **cheap consolidation** (no re-reading of text) is possible; if not, joint re-encoding is the answer.
2. **G-key retrieval and capacity** (running, addendum 6): find the slice by the whole-conversation key instead of a subject-name match, with N = 10, 20, 60 facts. This also gives the capacity curve that was owed.
3. **Selection by surprise** (experiment 3, part 2c): does how surprising a fact was predict whether it is needed later? With a slice per fact cheap but not free (and the 2B's state ~19 MB), this decides what the offline phase keeps.
4. **Abstraction into the steer channel** (experiment 3): turn many episodes into a leaning and store it with the push, the one thing that channel does well.
5. **The literature check** (`docs/dreaming/literature-exp1.md`, being written): read it before claiming novelty; the mechanism overlaps Prompt Cache, CacheBlend, EPIC and InfLLM.
6. **Paper 1, v2:** add stage 0 (the fact lives at its own position) to the arXiv paper, since it directly supports the "push, not store" reading. **Workshop version** (ICLR 2027 workshops, list out 29 Nov; about 1 Feb deadline): a 4-page cut from `writeup/main.tex`; fix the empty Appendix D header and the float placement first.
7. **Paper 2 (understanding): what binds a memory?** Steering vectors versus cache slices versus hybrid state, built on experiment 1 (position vs vector; slice vs tokens; state = recognition; collisions).
8. **Experiment 2, scaffold removal and concept × sign** (`docs/dreaming/02-scaffold.md`): model-generated opposite / alternatives, a general valence direction, and the signed-concept write. Independent; can run in parallel.
9. **Paper debts a follow-up inherits:** a second model family (non-hybrid), human labels on a subset, isolated memories only. The 9B has not been tried on the reinstatement arm.
10. Still open from before: a thermostat instead of a push (dose), phasic injection during thinking, finer dose sweeps, a trained read adapter as the fallback (the user is open to it: frozen backbone plus a trained read adapter).

**Do not** spend more time improving the steer vector itself. The paper says that channel carries a push, not content; it keeps that job (the abstracted leaning of experiment 3) and nothing else. **Do not** revisit the subject-position write: stage 0 and the generation run closed it.

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
