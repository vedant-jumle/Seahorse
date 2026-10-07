# Seahorse: handover

*The live handover: the current state, where everything is, what's pending, and how to pick up. Updated 2026-10-07. For the story and the reasoning, read [README.md](README.md), then the plain-language docs it lists.*

---

## 1. Where things stand

A training-free memory for a frozen LLM.
- **Writing:** an experience in session 1 is stored as a **shift** of the model's internal state, filed under a **whitened gist key** of the conversation.
- **Reading:** in any later session, the shift is added back when the current gist clearly matches a stored one.

It was built on **Qwen2.5-1.5B-Instruct** (layer 26 of 28) and ported to **Qwen3.5-2B**, where it is read and injected at layers 20, 21 and 23 of 24, with thinking off by default.

**What it achieves:**
- **Selective.** Unrelated answers are word-for-word unchanged in every test on both models.
- **Order-free.** Least-squares writes.
- **Broad preferences come through.** Vegetarian reshapes whole menus, better than having the fact in the prompt.

**What it doesn't:**
- **Exact facts:** they come out as the gist, or as loops like "Petra Petra Petra". Only about 4 in 44 answers are clean.
- **Premises:** balanced yes/no stays at chance everywhere.
- **Dislikes:** they flip into likes unless the hand-made opposite is subtracted.
- **Single-item preferences** (a country, a genre) either flood or do nothing.
- **Ownership:** the model claims memories as its own ("Teal is my favourite colour!").
- **Writing** still needs hand-made with/without/opposite comparisons.

**Running now:** `xlayer_v1`, which reads fact memories at a late layer and injects them at a middle layer (§4).

## 2. Read first
1. [README.md](README.md): the overview and reading order
2. [docs/where-we-are.md](docs/where-we-are.md): the whole story, including Qwen3.5 and what a shift stores (stages 7–8)
3. [docs/how-it-works.md](docs/how-it-works.md): the design end to end (written for Qwen2.5; the port changes only model, layers and thresholds)
4. [docs/where-it-fails.md](docs/where-it-fails.md): every failure (now 16), its cause and possible fixes
5. [docs/reference/metrics.md](docs/reference/metrics.md): **what every number in the reports means**, with traps. Read it before trusting any table.
6. [docs/reference/literature.md](docs/reference/literature.md): the steering-reliability and introspection papers, and how they line up with our results

## 3. The current best design
| Piece | Qwen2.5-1.5B (original) | Qwen3.5-2B (current) |
|---|---|---|
| Model / layers | fp32; read and write at the output of layer 26 of 28 | fp32, text path; layers **23, 20, 21** of 24, each with its own keys, M and threshold; thinking **off** |
| Write unit | one moment per follow-up sentence: an entropy-weighted mean shift over its words, chat template tail excluded | same |
| Shift | facts: plain (with − without); preferences: contrastive (with − opposite) | same. **Only the opposite keeps a preference's direction** (ref_v1); others lose it. |
| Key | `pooled_w256`: running mean over user words of PCA-256-whitened (state − background), normalised | same (the Ledoit–Wolf variant is clearly worse on Qwen3.5) |
| Store | delta rule for one memory; **least squares (RLS, λ=0.1)** for several | same |
| Read | at every word: max cosine to stored keys; hard threshold at the 95th percentile on ordinary prompts; `h ← h + α·gate·M·q`, α=2; never on the system prompt | same, at each of the three layers |

Code:
- `experiments/diag_keys/run.py`: the definitive Qwen2.5 implementation
- `experiments/ref_v1/` and `experiments/think_v1/run.py`: the Qwen3.5 versions
- `experiments/xlayer_v1/`: cross-layer reading and injection

## 4. Pending: xlayer_v1 (queued, not yet run)

**Question:** do fact memories get *used* if they're read at a late layer but injected at a middle one?
- **The hypothesis:** late layers act as the "mouth" (they push words); about two-thirds deep is the "mind" (Lindsey 2026).
- **Setup:** Qwen3.5-2B, thinking off, 12 facts:
  - dog (Pepper), cat (Jasper), sister (Petra), brother (Hugo), grandmother (Esther), best friend (Omar), boss (Gordon)
  - job (pharmacist), favourite colour (teal), hometown (Leeds), first language (Portuguese), university (Edinburgh)
- **Stage A (no text generated):**
  - grid: read layer R {12,16,20,23} × inject layer W {8,12,16,20,23} × α {0.5,1,2,4}
  - measures: target log-prob vs foils, balanced yes/no margin, KL on ordinary prompts
  - the shift is rescaled by the ratio of typical state sizes between layers
  - gating comes from a no-memory pass, so the injection can't change its own gate
- **Stage B (text generated):**
  - the 2×2 of R,W ∈ {16,23} at two doses
  - the current design (20/21/23)
  - read {20,21,23} → inject {14,15,16}
  - the two best Stage-A cells
  - no memory, and the fact in context (ceiling)
- **The key measure is `use_tgt_clean`:** the fact used, unasked and not in a loop, on "use" prompts. For example, "Write a short birthday message to my sister" should say Petra. Also: balanced yes/no, `self_attr` ("my favourite colour is teal"), loops, unrelated unchanged.

**Jobs** (gpu-v100, an `afterok` chain, queued 2026-10-07):

| Job | ID |
|---|---|
| smoke | 921693 |
| Stage A | 921694 |
| Stage B | 921695 |

Code at commit `b03f5b3`. The build agent was stopped once the jobs were queued, so there are no runtime estimates.

**Outputs:** `/scratch/vvjumle/seahorse_runs/xlayer_v1_20261007_1311/{smoke,stageA,stageB}`. If the smoke job fails, read `/scratch/vvjumle/logs/seahorse_921693.out`. When done:
1. Run `rsync -a delftblue:/scratch/vvjumle/seahorse_runs/xlayer_v1_20261007_1311 results/`.
2. Read `report.txt` in each stage.
3. **Judge by clean hits and use prompts, not raw `ans_tgt`.**

**What would count as a yes:** "read late → inject middle" beats both same-layer cells on `use_tgt_clean` or balanced yes/no, without more loops.

## 5. Recently finished and reviewed

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
- **Overdose signature:** loops, fragments, identity confusion. The useful dose window is narrow. Pushing *away* from something seems not to flood.
- **Injected states feel like the model's own,** hence the self-attribution (Lindsey's prefill result).
- **Our layers are probably the "mouth".** We've always injected near the end. xlayer_v1 tests the "mind".
- **Measurement:** word counts are inflated by loops; word lists count mentions and puns; yes/no must be balanced. See `docs/reference/metrics.md`.

## 8. Next steps and ideas (in rough priority)
1. **Review xlayer_v1** (§4).
2. **Concept × sign.** Store *what* (centroid) and *which way* separately, and apply sign × gain × concept, so we do the binding.
   - Read the sign off a **general like/dislike direction**: average "I love X" − "I hate X" over many X. That removes the per-memory hand-made opposite.
   - Check first with the logit lens, then one generation run.
   - Simply *adding* opposite + centroid probably fails for dislikes, because a sum carries "jazz" and "hate" side by side but can't say "hate *about* jazz".
3. **A thermostat instead of a push.** *Set* the concept to a target level rather than adding to it. That should stop floods, and "away" memories become "keep it low".
4. **Fair doses:** match strength by loop rate. Run the planned α sweep with the current design.
5. **Thinking:** a burst at the start of thinking that then fades (phasic), instead of a constant push; a much larger thinking cap.
6. **Remove the scaffold:**
   - get *what* from the average of other disclosures or of already-stored memories (the latter is also a novelty signal)
   - get "without" by leave-one-out
7. **Neuromodulation:**
   - write strength from surprise (the model may already compute a "doesn't fit" signal)
   - confidence from agreement across a memory's moments
   - fading and read gain; a slow global mood state
8. **The capacity curve** on bench_v1's 60-fact pool, including several memories firing on one prompt.
9. **Measurement upgrades:** an LLM judge for preferences (mention vs recommendation); report the share of prompts moved the wrong way.
10. **Facts may belong to the episodic channel** (the other session). Also try a bigger model: using injected states as premises appeared mainly in large models.
11. **Fallback:** a small trained read adapter (frozen backbone), if training-free stalls.

## 9. Open decisions for the user
- `experiments/v0/run.py` used to point to the parent-folder `Idea.md`; it now points to `docs/reference/original-method.md`. OK?
- The parent folder `/home/vedantjumle/projects/memories/docs/` still holds the original copies of the three reports. Delete them so the repo is the single source?
- A note for the **other session** (the episodic "ladder" work): relation probes must be **balanced Yes/No** and **exclude the chat template tail**, or a general "No" drift fakes gains. It hasn't been sent; it's unknown whether that session already knows.

## 10. Working conventions
**Framing:** Seahorse is an **independent side project**, not a thesis or supervised research. In the docs, "thesis" only ever means the project's central hypothesis. Never describe it as MSc or thesis work in anything written for others.

**Cluster (DelftBlue):** `ssh delftblue`; the repo is at `/scratch/vvjumle/Seahorse` (`git pull` there after pushing).
- `gpu-v100`: 32GB, **fp32** (no bf16), ≤5333MB per CPU. Script: `slurm/run_v100.slurm`. This is what the Qwen3.5 experiments use.
- `gpu-a100-small`: a 10GB MIG slice, ≤2 CPUs, ≤8000MB per CPU, 4h. Script: `slurm/run.slurm`. Used for the Qwen2.5 work.
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
