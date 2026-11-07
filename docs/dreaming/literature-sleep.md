# Sleep, dreaming and offline consolidation against the literature

*A reading of Google's SLEEP paper and 20 other papers on sleep, dreaming, replay and offline consolidation for language models and agents, against Seahorse's dreaming phase. Prompted by the Google Research blog post "The sleep framework: A new way for LLMs to retain knowledge in continual learning" (Behrouz & Mirrokni, 27 Oct 2026). Written 2026-11-07. The PDFs are in the parent folder's `papers/sleep-consolidation/` (index: `papers/README.md`). As in [literature-select.md](literature-select.md), I read extracted text (pdftotext), not figures. Anything shown only in a figure is marked as such. Page numbers are PDF pages. How deeply each paper was read is stated at the end.*

Terms follow [literature-exp1.md](literature-exp1.md) and [literature-select.md](literature-select.md). "Slice" = cached keys and values (K,V) of a few positions, computed in the original context. "Steer" = the stored shift added to the residual stream (paper 1). "Weights" = the model's parameters, including adapters such as LoRA. "Activations" = anything computed per input and thrown away by a normal model: the KV cache, a hybrid's recurrent state, residual-stream vectors. "Training-free" = no gradient step anywhere, including on adapters. "Frozen" = the base model's weights never change.

---

## The short version

**What SLEEP is.** Behrouz, Hashemi, Javanmard, Mirrokni, "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories", arXiv 2606.03979 (v2, 10 Jul 2026; a version on OpenReview since September 2025; Google Research and Cornell). It is **weight-based continual learning**. In its "sleep", the model adds new low-rank experts to slower layers, trains them by distillation plus RL so that they take over what the faster layers learned, resets the faster layers' added experts, and then "dreams": it generates synthetic training data, keeps the samples with the most useful gradients, and fine-tunes on them with LoRA. Nothing in it is training-free, and the model is not frozen. Its targets are catastrophic forgetting, knowledge incorporation, long context and reasoning, not a per-user memory.

**The overlap with Seahorse, in one table.**

| Seahorse claim or framing | Verdict | Closest papers |
|---|---|---|
| The name "dreaming" for an offline phase | **Known; the word is now taken several times.** SLEEP's "Dreaming" is its synthetic-data stage. Auto-Dreamer and MindMemOS call text-memory consolidation "dreaming". | SLEEP, Auto-Dreamer, MindMemOS, SCM |
| "An offline phase does what one forward pass at write time can't" | **Known as a rationale.** Lee et al.: turning tokens into memory "need not be achievable in a single pass". Sleep-time compute: think about the context before the query exists. | Lee et al. 2026, Sleep-time Compute, SLEEP |
| Offline consolidation in general | **Known**, in weights (SLEEP, SSR, sleep replay), in a hybrid's recurrent state (Lee et al., trained), in the KV cache (SleepGate, Bottlenecked Transformers, both trained and within one sequence) and in text (LightMem, Auto-Dreamer, MindMemOS, Generative Agents, A-MEM, Mem0). | as listed |
| Offline consolidation on a **frozen** model's **own KV positions and recurrent state**, **training-free**, **between sessions** (keep 2–8 positions; store the k-th memory as the k-th fact; joint re-encoding against partner swaps) | **Still ours.** None of the 21 papers does it. The nearest are Lee et al. (writes the recurrent state, but trained end to end, within one sequence) and SleepGate (edits the KV cache, but trained, within one sequence, on a 793K-parameter model). | Lee et al., SleepGate, Bottlenecked Transformers |
| Experiment 2: the model writes its own references offline | **Partly known as an idea.** "Dreams = the model's own synthetic data" is SLEEP's and SEAL's usage, and SSR's self-synthesized rehearsal. They use the dreams as *training data*. Using them to *build an activation memory* (opposite, alternatives, sign) is ours. | SLEEP, SEAL, SSR |
| Experiment 3, abstraction (leanings from episodes) | **Known as an idea; ours as a method.** Abstraction during consolidation is in SLEEP (in weights) and in many text systems (reflections, region rewriting). Abstracting into a steer vector beside a frozen model is not found. | SLEEP, Generative Agents, Auto-Dreamer, LightMem |
| Experiment 3, reconsolidation (later wins; suppress, don't delete) | **Partly known.** Conflict resolution is standard in text memories (Mem0's UPDATE/DELETE, MindMemOS, LightMem, Auto-Dreamer). In a KV cache it is SleepGate's whole point, but trained and within one session. Training-free, cross-session suppression of a stored slice is not found. | Mem0, SleepGate, MindMemOS |
| Experiment 3, pruning by write-time surprise, novelty, valence | **Partly known.** Importance scores decide reflection (Generative Agents) and forgetting (SCM). EVAF gates LoRA writes by valence × surprise. The falsifiable test (do write-time signals predict later need, with labels from LongMemEval) is not found. SLEEP's "synaptic pruning" is a different thing with the same name. | Generative Agents, SCM, EVAF, Auto-Dreamer |
| Experiment 3, re-filing keys after the fact | **Partly known in text** (A-MEM's memory evolution rewrites old notes' tags and links). Not found for activation keys. | A-MEM, HippoRAG |
| "A self-learning memory system" | **Crowded framing.** SEAL ("self-adapting"), SLEEP ("self-modify"), MindMemOS ("self-evolving"), A-MEM ("agentic"). All of those train weights or rewrite text. | as listed |
| Paper 1 claims | **Unaffected.** Paper 1 has no offline phase. | none |
| Paper 2 claims | **Unaffected.** No sleep paper studies read sites, slices vs text, binding, or swaps. Two sentences need citations (see §4). | Lee et al. for the hybrid paragraph |
| Website | **Needs one citation sentence** in the dreaming bullet. No claim is contradicted. | SLEEP, Sleep-time Compute |

**Three closest papers overall:**
1. **SLEEP** (Behrouz et al. 2026) for the framing. It has the same wake/sleep split, the same "dreams are self-generated data", and the same NREM-like "consolidate, abstract, prune" language as experiment 3. Everything happens in weights, with training.
2. **Lee, McLeish, Goldstein, Fanti 2026** ("Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference") for the substrate. At sleep, a hybrid's Gated DeltaNet state is written by extra offline passes, and then the KV cache is cleared. That is the same state our Qwen3.5-2B slices carry. But the model is fine-tuned end to end for it, and the memory lives within one long sequence.
3. **SleepGate** (Xie 2026) for experiment 3's reconsolidation. It is a sleep cycle over the KV cache that tags superseded entries, gates them out and merges the rest. It is trained, tested on a toy model, and works within one session.

---

# 1. What SLEEP is, precisely

**The paper.** Ali Behrouz (Google, Cornell), Farnoosh Hashemi (Cornell), Adel Javanmard (Google, USC), Vahab Mirrokni (Google). "Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories". arXiv 2606.03979v2 [cs.LG], 10 Jul 2026, 26 pages. A footnote says "A version of this work has been publicly available from September 2025 on OpenReview" (p.1). No venue is given. No code is linked in the paper or the blog. The blog post (research.google, 27 Oct 2026, Behrouz & Mirrokni) summarises this paper and links only to it.

**It builds on Nested Learning / Hope** (Behrouz et al., NeurIPS 2025, arXiv 2512.24695, filed here). Hope's "Continuum Memory System" is a chain of MLP blocks, each updated by gradient descent at its own frequency. Attention is the fastest memory (it is rewritten every context), and pretrained MLPs are the slowest (never updated). SLEEP adds the offline step between levels.

**Key definitions, quoted.**
- The lifecycle: "a continual learner need to have different stages of activeness in learning, which we refer to as: (i) Active or Wake Time, and (ii) Sleep Time. Sleep time is not a passive state, rather it internally process the data to consolidate the memories from fast unstable modules to more stable low frequency (slow) components." (Fig. 1 caption, p.2)
- Wake and sleep: "In the active or wake time, continual learner receives new input data and process it as needed. In the sleep time, however, the model receives minimal (or none) input data and focus on processing existing knowledge to consolidate memories and self-improve." (box, p.4)
- Offline consolidation: "the model enters the sleep phase, where it stops processing new data, but uses self-generated data to consolidate the recent memories." (box, p.6)
- Dreaming: "given the current state of the model, it generates a set of dreams (i.e., synthetically self-generated data) to improve its own performance with particular focus on the acquiring more proficiency on the recently added knowledge." (p.4)
- On Letta's sleep-time compute: "Despite similarity in the name, their method is fundamentally different from our work. While this method aim to find a good summary of the interactions of the model with users in the text space, when the model is idle, our proposal is on using distillation to transfer text data knowledge into a form of parametric weight update." (App. A.3, p.23) The same distinction separates SLEEP from Seahorse.

**What sleep does, step by step** (§3.2–3.4, p.7–10).
1. **Parameter expansion.** The MLP blocks are treated as mixtures of experts. When block ℓ*−1 is due for its own weight update, a new low-rank expert (A, B with d_low ≪ d) is added to the next, slower block ℓ*. In practice the new parameters exist from the start and are masked until a sleep unlocks them (p.9). The appendix says 5 extra MLP blocks of dimension 64, with the active parameter count unchanged (p.25).
2. **Knowledge seeding** (an "upward distillation", from a smaller self to a larger self). A "compute, consolidate, update" protocol (p.7–8):
   - The teacher is the model before the update.
   - The student is the model with the prospective new weights for the faster block and the new expert in the slower block.
   - The teacher samples a dataset D. Only the new expert is trained, with a GKD-style on-policy distillation loss (a divergence between teacher and student distributions on teacher- and student-generated text). "we freeze all the parameters in the student model and only updates the expanded parameters" (p.8).
   - A "Learning to Imitate" RL term rewards the student for continuing teacher-generated "dreams" from a random prefix. The reward is a frozen reward model's semantic match plus a Levenshtein similarity (p.9).
3. **Synaptic pruning.** After consolidation, "we reset all the low-rank parameters that previously (in past sleep periods) have been added to MLP(fℓ*−1), making its capacity available for future" (p.9). This resets parameters. It does not choose which memories to keep.
4. **Dreaming** (REM-like; built on SEAL). Given a task context C, the model samples m dreams. During sampling, each MoE router also picks one random expert, "incorporat[ing] random irrelevant knowledge to the dreaming" (p.10).
   - Dreams are scored by the gradient of the SFT loss. The top-k are kept, plus b random ones for diversity.
   - For each kept dream, an isolated copy of the model is fine-tuned on it with LoRA. The dream is rewarded 1 if that copy improves on the task, and the dream generator is trained by ReST-EM, as in SEAL (p.10).

**Is it training-based?** Yes, at every step: gradient updates to new low-rank experts, distillation, RL (LTI and ReST-EM), and LoRA SFT. The training configuration uses LoRA rank 64, α 128, learning rate 5e-6 and 100 steps (Table 5, p.26). The base model is not frozen. Old parameters are held fixed only *during* distillation, and the faster blocks still get their scheduled base-weight updates.

**Replay or generated data?** Generated. The model samples its own data ("dreams") both for distillation (teacher samples) and for dreaming (task-conditioned samples). The paper says consolidation "should not simply replay raw data; instead, it needs to explore and extract abstractions" (p.8). There is no stored replay buffer, KV cache, activation store, retrieval or memory module in Seahorse's sense. The "memory" is a set of parameter blocks updated at different frequencies.

**Target problem.** Continual learning without catastrophic forgetting, framed as moving "fragile short-term memories into more stable long-term knowledge" (p.2). It also covers knowledge incorporation from passages, long-context understanding, few-shot learning and math reasoning. Personalisation and per-user memory are not mentioned.

**Benchmarks, models and results.**

| Task | Model | Result | Where |
|---|---|---|---|
| Class-incremental classification (CLINC, Banking77, DBpedia) | Llama-3B, Llama3-8B with Hope + consolidation | Hope (sleep) is best against ICL, EWC and InCA | Figure only (Fig. 3) |
| Long context: MK-NIAH, LongHealth, QASPER | Hope variants | Beats ICL, DuoAttention and **Cartridges**. More consolidation stages help; a higher lowest frequency hurts. | Figure only (Fig. 4) |
| Continual translation (Manchu, Kalamang, from MTOB) | Hope-1/2/3 | ICL "largely revert[s] to pre-trained behavior" when trained sequentially. Hope-3 "nearly recovers its single-language performance". Cartridges and SFT forget in at least one language. | Figure only (Fig. 5) |
| BABILong to 10M tokens | Hope (fine-tuned on BABILong) | "almost perfect" to 10M. Titans and ARMT degrade after about 1M. "all small models, including Hope, suffer substantial performance drops without fine-tuning" (p.26). | Figure only (Fig. 6) |
| Math (AIME-24/25, HMMT-25, avg@16) | Qwen3-1.7B, Qwen3-8B | 8B: Sleep 79.2 / 69.0 / 46.1 against OPSD 76.6 / 67.4 / 45.1, GRPO 76.4 / 68.1 / 44.9, base 73.8 / 68.1 / 42.4 | Table 2, p.12 |
| SQuAD knowledge incorporation (no context at test) | SEAL's setup | Base 31.9; SEAL 46.7 (n = 1) / 43.2 (CPT, n = 200); Sleep four-level 48.9 / 46.2; **removing dreaming 35.7 / 36.2** | Table 3, p.12 |
| Few-shot ARC (8 held-out tasks) | Llama-3.2-1B | ICL 0, TTT 10, SEAL 72.5, Sleep 80 | Table 4, p.12 |

**Limits, as I read them** (the paper has no limitations section).
- Most continual-learning and long-context results are shown only in figures. No seeds, intervals or variance are reported anywhere. Margins over the strongest baseline are 0.2–2.6 points on math, 2.2–3.0 on SQuAD and 7.5 on ARC (a success rate over sampled dreams on 8 held-out tasks).
- The SQuAD ablation shows that most of the knowledge-incorporation gain comes from the SEAL-style dreaming stage: without it, 35.7 against 48.9. Consolidation alone adds little there.
- Hope is both the architecture and the method in most experiments, so the effect of sleep is hard to separate from the effect of the architecture.
- Cost: per training step, SFT is 4× cheaper. Their argument is that, for the same accuracy, SFT takes 3.6–4.8× more wall-clock time (App. B.5, p.26).
- The blog says the framework "operates entirely within a fixed model architecture". The paper's own parameter expansion only fits that if "fixed" means "pre-allocated and masked".

**Neuroscience framing.** Heavy and explicit:
- complementary learning systems (McClelland et al. 1995; Kumaran et al. 2016, cited in §2.3)
- online vs offline ("systems") consolidation (p.2–3)
- hippocampus–neocortex dialogue during slow-wave sleep (p.3)
- synaptic homeostasis (Tononi & Cirelli 2006, p.3)
- NREM as "consolidate and prune" and REM as "exploring novel connections" (p.3)
- synaptic pruning (p.9)
- neuroplasticity as the motive for parameter growth (p.7)
- anterograde amnesia as the analogy for frozen LLMs (p.2, from Nested Learning)

The authors say they borrow the terms (the blog: the framework "borrows" "sleep", "dreaming" and "self-improvement").

---

# 2. Per-paper notes

Relation tags as in literature-exp1: **same**, **supports**, **adjacent**, **contradicts**, **background**. For each paper: the setting (what is written: weights, activations/KV, or text; training-based or training-free; frozen or not), what the sleep or dream phase does, the evaluation, and what it means for Seahorse.

## A. Sleep and consolidation inside the model

**SLEEP** (Behrouz et al., arXiv 2606.03979). *Adjacent in framing; different in substrate.* Full notes in §1.
- **Setting:** weights; training-based; not frozen.
- **Sleep phase:** distil fast blocks into new low-rank experts, then dream (generate, select by gradient, LoRA-SFT, RL).
- **Evaluation:** continual learning, long context, SQuAD, ARC, AIME.
- **For us:** the closest framing, and the strongest reason to cite. It shows the big-lab version of "wake/sleep with dreaming" is weight-based. It is the "consolidation into weights", the parked future step in [core-idea.md](../reference/core-idea.md), not a competitor for a frozen-model memory. Its comparison with Cartridges (a trained KV cache) on long context is the one place it touches our substrate.

**Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference** (Lee, McLeish, Goldstein, Fanti; arXiv 2605.26099v3, 5 Jun 2026; CMU and UMD; v1 was titled "Language Models Need Sleep" and was renamed to avoid SLEEP's OpenReview title, footnote p.3; listed at ICML 2026 per a search result, not checked). *Adjacent; the closest in substrate.*
- **Setting:** an SSM–attention hybrid. "Fast weights" here means the Gated DeltaNet recurrent state, per sequence, so in our terms it is an activation, not a parameter. The model is fine-tuned end to end through the sleep (Jet-Nemotron 2B, built from Qwen2.5-1.5B, and Ouro 1.4B). Training-based; not frozen.
- **Sleep phase:** "When the model's context window becomes full during inference, the model enters a 'sleep' in which it performs multiple forward passes over the accumulated context and recursively updates its fast weights via a learned local rule … After consolidation, the context window is cleared" (p.1–2). Prediction stays a single pass.
- **The key sentence for us:** "Converting observed tokens into useful weight memory is itself a nontrivial computation, and need not be achievable in a single pass" (p.2). That is the same rationale as our "what can't be done in one forward pass at write time".
- **Evaluation:** Rule 110, Depo multi-hop retrieval and GSM-Infinite. Vanilla GDN hybrids fail as reasoning depth grows, even with enough state capacity. More sleep passes N help most on the deepest examples.
- **For us:**
  1. It agrees with our hybrid finding from another angle. A frozen hybrid's state alone gives recognition without recall (addenda 6–8). Lee et al. show that a vanilla hybrid's state cannot support deep computation over evicted context, and that it takes training plus offline loops to fix that.
  2. "Offline loops that write the recurrent state before the KV is dropped" is a natural training-free experiment for us: run the frozen 2B over the stored turn N times and keep the final state. Untested by us; Lee et al. only test it with training.
  3. It belongs in paper 2's "Hybrid states" paragraph.
- Read: §1–5 and the discussion; results by search.

**Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in LLMs — SleepGate** (Xie, arXiv 2603.14517, Mar 2026, single author). *Adjacent to experiment 3's reconsolidation.*
- **Setting:** KV cache; trained (a "dual-phase training objective"); a 4-layer, 793K-parameter transformer; within one session.
- **Sleep phase:** "sleep micro-cycles" triggered by attention entropy, with three modules:
  - "a conflict-aware temporal tagger that detects when new entries supersede old ones"
  - "a lightweight forgetting gate network trained to selectively evict or compress stale cache entries"
  - "a consolidation module that merges related surviving entries into compact summary representations" (p.1)
- **Evaluation:** the PI-LLM proactive-interference task (key–value updates, query the latest value). 99.5% at depth 5 and 97.0% at depth 10, against below 18% for full KV, sliding window, H2O and StreamingLLM. There is an O(log n) interference-horizon argument under stated assumptions.
- **For us:** the KV-side version of experiment 3's 2b, "later wins". It shows the problem (stale entries compete through attention mass) is real for KV memories. It is trained, at toy scale, within a session, and on overwritten values, not on binding between different memories. Our collision is a different interference: partner swaps from shared order tags, not stale values. Read: abstract, §1–3 by search.

**Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning** (Oomerjee, Fountas, Bou-Ammar, Wang; arXiv 2505.16950, ICLR 2026). *Adjacent.*
- **Setting:** KV cache; a frozen SFT backbone (Llama-3.2 1B/3B, Qwen3 0.6B) plus a *trained* "Cache Processor"; within one generation.
- **What it does:** "periodic, non-causal, in-place KV rewrites at newline-delimited reasoning step boundaries. The Processor consolidates recently written KV entries and reconsolidates a small, top-k attention-selected set of prior entries" (abstract). It is framed explicitly as memory consolidation and reconsolidation, through the information bottleneck.
- **Evaluation:** GSM8K, MATH, SVAMP, GSM-Hard; up to +6.6 points.
- **For us:** shows "rewrite the frozen model's own KV in place as consolidation" works when the rewriter is trained. Our joint re-encoding is the training-free version: the model itself recomputes the entries. Read: abstract, setup by search.

**Nested Learning / Hope** (Behrouz, Razaviyayn, Zhong, Mirrokni; arXiv 2512.24695, NeurIPS 2025). *Background.* It supplies SLEEP's continuum memory system and the "online consolidation" that SLEEP extends offline. It is all weights, trained. Read: abstract; the rest as summarised in SLEEP §2.2.

**Self-Adapting Language Models — SEAL** (Zweiger, Pari, Guo, Akyürek, Kim, Agrawal; arXiv 2506.10943). *Background; the base of SLEEP's dreaming.*
- **Setting:** weights; SFT on "self-edits"; RL (ReST-EM) on downstream reward.
- **For us:** the origin of "the model writes its own training data". Our experiment 2 is the activation-memory analogue: the model writes its own *references*, and nothing is trained. Read: abstract.

**Self-Synthesized Rehearsal — SSR** (Huang, Cui, Wang, Yang et al.; arXiv 2403.01244, ACL 2024). *Background.* Generative replay for LLM continual learning: the base LLM generates synthetic instances in context, the latest LLM refines their outputs, and diverse high-quality ones are kept for rehearsal. Weights; trained. **For us:** the standard citation for "generative replay with LLMs". Read: abstract.

**Continual Learning via Sparse Memory Finetuning** (Lin, Zettlemoyer, Ghosh, Yih et al. (FAIR); arXiv 2510.15103). *Background; a weight-based baseline.* Only memory-layer slots that are highly activated by new knowledge are updated. On NaturalQuestions, forgetting drops from 89% (full fine-tuning) and 71% (LoRA) to 11%. It has no offline phase. **For us:** the cleanest weight-side comparator for "store facts without forgetting". Read: abstract.

**Not Just After One: Sleep-Inspired Replay Prevents Catastrophic Forgetting After Sequential Tasks** (Bazhenov, Delanois, Krishnan; arXiv 2606.08447, 4 pages). *Background.* Sleep Replay Consolidation (Tadros et al. 2022): an unsupervised sleep-like phase after several tasks partly restores all of them. Small networks, not LLMs. **For us:** the neuroscience-faithful "sleep" line, which also changes weights. Read: abstract.

**EVAF: A Test-Retest Protocol for Selective Parametric Consolidation** (Han, arXiv 2606.29916, single author, 40 pages). *Adjacent to experiment 3's 2c in its signals.*
- **Setting:** a fast prefix state coupled to a slow LoRA store; trained; CLS-inspired.
- **What it does:** a "dual valence-×-surprise write gate" decides which experiences consolidate into LoRA.
- **For us:** the same two write-time signals as our modulator (surprise, valence), used for weights. Its evidence is thin by its own statistics: the main effect on GPT-2 is "not significant at n = 4". Read: abstract only.

## B. Sleep and consolidation outside the model (text and graphs)

**Sleep-time Compute: Beyond Inference Scaling at Test-time** (Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez (Letta, UC Berkeley); arXiv 2504.13171, Apr 2025). *Supports; the training-free text version of "sleep".*
- **Setting:** text; frozen models; training-free.
- **Sleep phase:** "During sleep-time we are given the context c but not the query q. Using just this context c, we can use the LLM to infer likely questions and reason about the context ultimately producing a more new re-represented context c′" (§3). In Letta this is an agent tool, `rethink_memory`.
- **Evaluation:** Stateful GSM-Symbolic and AIME. About 5× less test-time compute for the same accuracy; amortised over related queries, 2.5× lower cost per query. The benefit tracks how predictable the query is.
- **For us:**
  1. The origin of "anticipate the user's questions offline". Our S-dream (dreamed questions for selection, [literature-select.md](literature-select.md)) is sleep-time compute's query anticipation, applied to choosing KV positions.
  2. Its "predictability" finding fits ours: dreamed questions help little over a repeat prompt.
  3. Must be cited wherever we say "dreamed questions".
- Read: abstract, §1–3, the prompt appendix by search.

**LightMem** (Fang, Deng, Xu, Jiang et al.; arXiv 2510.18866, ICLR 2026). *Adjacent; same benchmarks.*
- **Setting:** text memory over GPT and Qwen backbones; training-free (it uses an off-the-shelf LLMLingua-2 compressor, §3). It also requires that "only entries with later timestamps are allowed to update earlier ones" (§3), which is our 2b's "later wins", in text.
- **Sleep phase:** "long-term memory with sleep-time update employs an offline procedure that decouples consolidation from online inference". The update "reorganizes, de-duplicates, and abstracts these entries, resolving inconsistencies" (p.1–2).
- **Evaluation:** LongMemEval and LoCoMo. Up to +7.7% / +29.3% QA accuracy, and far fewer tokens and API calls.
- **For us:** the published text-memory system on our benchmark (LongMemEval) with an offline phase. It is the right text baseline for any "dreaming beats write-everything" claim in experiment 3. Read: abstract, §3 by search.

**Auto-Dreamer: Learning Offline Memory Consolidation for Language Agents** (Ye, Liu, Wang, Yu, Zhao, Liu, McAuley, You; arXiv 2605.20616, May 2026). *Adjacent to experiment 3 as a whole.*
- **Setting:** text memory; the consolidator is trained by GRPO; the task agent and writer are frozen. CLS-inspired: "decouples fast per-session memory acquisition from slow cross-session consolidation".
- **Sleep phase:** "region rewriting". The consolidator reads a region of the memory bank as evidence and writes a replacement set, so that "abstraction, deduplication, contradiction resolution, and omission-based forgetting become default behaviors" (p.2).
- **Reward:** task success plus "a counterfactual utility term estimated by random memory masking, which penalizes redundant entries while rewarding load-bearing ones".
- **Evaluation:** ScienceWorld +7 points with a 12× smaller bank; it transfers to ALFWorld and WebArena.
- **For us:**
  1. The text-memory twin of experiment 3: abstraction, contradiction resolution and pruning, offline, across sessions.
  2. Its masking utility is the memory-level version of our S-loo (remove an item, measure the downstream loss).
  3. It names the "dreamer" role. Read: abstract, §1, method by search.

**Useful Memories Become Faulty When Continuously Updated by LLMs** (Zhang, Lin, Wu, Sun, Li, Li, Peng; arXiv 2605.12978v2, Aug 2026). *Contradicts naive consolidation; supports keeping episodes.*
- **Finding:** "As consolidation proceeds, memory utility first rises, then degrades, and can fall below the no-memory baseline". Even consolidating from ground-truth solutions, "GPT-5.4 fails on 54% of a set of ARC-AGI problems it had previously solved without memory". An episodic-only control is competitive.
- **Advice:** "treat raw episodes as first-class evidence and gate consolidation explicitly".
- **For us:** a direct warning for experiment 3's abstraction. Our design already keeps "episodic residuals" beside the abstracted push, and we should keep the episodes (slices) as the ground truth. Also, consolidation should be gated, not run after every session. Read: abstract, §4.1.

**SCM: Sleep-Consolidated Memory with Algorithmic Forgetting** (Shinde, arXiv 2604.20943, single author, "research preview"). *Background.*
- **What it does:** an external concept graph built by local Llama 3.2. NREM consolidation strengthens important links; "REM dreaming" random walks create new associations; "intentional value-based forgetting"; importance tags.
- **Evaluation:** an 8-test self-made suite; "perfect recall accuracy over ten-turn conversations".
- **For us:** shows the sleep vocabulary is now common for text memories; weak evidence. Read: abstract, search.

**MindMemOS** (MindMemOS Team, arXiv 2608.12428, Aug 2026). *Background.*
- **What it does:** an entity–property–time text memory in which "dreaming consolidates accumulated memories by merging redundant records and resolving conflicts".
- **Evaluation:** LoCoMo 94.03%, PersonaMem 70.63%; a FactConsolidation test; active memory down to 22.5% after dreaming.
- **For us:** another use of "dreaming" for offline text consolidation, on our benchmark family. Read: abstract, search.

**Generative Agents** (Park, O'Brien, Cai, Morris, Liang, Bernstein; arXiv 2304.03442, UIST 2023). *Background; the origin of LLM "reflection".*
- "Reflections are higher-level, more abstract thoughts generated by the agent … generated periodically … when the sum of the importance scores for the latest events perceived by the agents exceeds a threshold (150 in our implementation)" (§4.2).
- Reflection starts by generating questions about recent memories.
- **For us:** the earliest "abstract episodes into a trait offline" (experiment 3's 2a), with an importance score as the trigger (2c's modulator) and self-generated questions (our S-dream) in one design. All of it is text. Read: §4.2.

**A-MEM: Agentic Memory for LLM Agents** (Xu, Liang, Mei, Gao et al.; arXiv 2502.12110, NeurIPS 2025). *Background.* Zettelkasten-style notes with "link generation and memory evolution": "Memory evolution enables existing memories to dynamically adapt as new experiences are analyzed". **For us:** text-side precedent for re-filing (2d): old memories get new tags and links in hindsight. Read: abstract, §3 by search.

**Mem0** (Chhikara, Khant, Aryan, Singh, Yadav; arXiv 2504.19413). *Background.* Extraction, then an update step with four operations: "ADD … UPDATE for augmentation of existing memories with complementary information; DELETE for removal of memories contradicted by new information; and NOOP" (§2). This happens online, per message. **For us:** the industry default for reconsolidation (2b) in text. Unlike our plan, it deletes instead of suppressing. Read: §2 by search.

**MemoryOS** (Kang, Ji, Zhao, Bai; arXiv 2506.06326). *Background.* An OS-style hierarchy: short-term, then mid-term (FIFO on dialogue chains), then long-term personal memory (segmented pages). LoCoMo F1 +49% over its baselines on GPT-4o-mini. **For us:** the tiered "promote what lasts" pattern, in text; the consolidation is a heat-based promotion, not offline. Read: abstract.

**HippoRAG** (Gutiérrez, Shu, Gu, Yasunaga, Su; arXiv 2405.14831, NeurIPS 2024). *Background.* "inspired by the hippocampal indexing theory": an offline index (LLM-extracted knowledge graph plus Personalized PageRank) acts as the hippocampus, and the LLM as the neocortex. Multi-hop QA up to +20%. **For us:** the main "hippocampus for LLMs" citation besides Larimar and EM-LLM (already filed). Its index is text and graph; ours re-evokes the model's own states (core-idea's indexing theory). Read: abstract.

## C. Found and not filed

- **Wake-Sleep Consolidated Learning** (Sorrenti et al., arXiv 2401.08623; IEEE TNNLS per a search result, not checked). Vision continual learning with wake, NREM and REM phases. Not LLMs.
- **Memory for Large Language Models** (arXiv 2607.25380). It appeared in a search for training-free sleep methods; not opened.
- **HippoRAG 2** ("From RAG to Memory", arXiv 2502.14802). Known; not filed (HippoRAG 1 covers the citation).
- **HEMA** (arXiv 2504.16754, "hippocampus-inspired extended memory" for long conversations). Found in search; not read.
- **Letta's sleep-time agents** (blog and docs). The product version of Sleep-time Compute; the paper is the citation.
- Already in the library and relevant here: **Cartridges** (offline self-study into a trained KV cache, `context-compression/`), **Titans** and **FwPKM** (test-time training as memory, `memory-architectures/`), **Larimar**, **EM-LLM**, **MemGPT** (`memory-architectures/`), **Attention Matching** and **KVzip** (training-free offline KV compaction, `kv-cache-reuse/`).

**Not found:** no paper does a **training-free, frozen-model, between-session offline phase that edits or selects the model's own KV entries or recurrent state** to build a memory. "SleepGate" exists but is trained. "Sleep consolidation for the KV cache" exists only as trained (SleepGate, Bottlenecked Transformers) or as recurrent-state writing with training (Lee et al.). "Sleep-like unsupervised replay" exists for small networks (Bazhenov et al.), not LLMs. A paper literally titled "Language Models Need Sleep" exists twice (SLEEP's OpenReview version, and Lee et al.'s v1).

---

# 3. Overlap analysis

Strict reading: "known" only where a paper we read does it; "partly known" where the idea is published but the substrate, setting or test differs in a way that matters; "still ours" where nothing we read does it.

### 3.1 The "dreaming" name and framing
- **Known, and the word is not ours.** "Dreaming" now names:
  - SLEEP's synthetic-data stage (REM-like)
  - MindMemOS's offline merge and conflict resolution
  - Auto-Dreamer's learned consolidator
  - SCM's REM random walks
  - Earlier: Ha & Schmidhuber's world models and Dreamer in RL, cited by SLEEP §2.3
- **Our usage is wider than SLEEP's.** Our "dreaming" is the whole offline phase, which is SLEEP's "sleep". SLEEP's "dreaming" is one stage. Where our offline phase generates things (experiment 2's references, S-dream's questions), it is "dreaming" in exactly SLEEP's sense: self-generated data, used offline. So the name is *consistent* with the literature, not misleading.
- **The rationale is partly known.** "An offline phase does what one forward pass at write time can't" (our README) is close to Lee et al.'s "need not be achievable in a single pass" and to Sleep-time Compute's premise. Ours adds *why* one pass can't: binding is by position and order tag, so related memories need to be read together.
- **Action:** keep the name, define it once, and cite SLEEP, Sleep-time Compute and Lee et al. at the first use. Don't claim the metaphor.

### 3.2 The offline-consolidation idea
- **Known in general**, on every substrate except ours:
  - weights: SLEEP, SSR, sleep replay, EVAF
  - a hybrid's recurrent state, trained: Lee et al.
  - KV, trained, within one sequence: SleepGate, Bottlenecked Transformers
  - text: Sleep-time Compute, LightMem, Auto-Dreamer, MindMemOS, Generative Agents, A-MEM
- **The CLS framing is old and shared.** SLEEP, Auto-Dreamer, EVAF and HippoRAG all invoke it, as our core-idea doc does.
- **Still ours:** the offline phase operating, **training-free**, on a **frozen** model's **own KV positions and recurrent state**, **between sessions**. That covers:
  - choosing the 2–8 positions to keep
  - storing the k-th memory as the k-th fact
  - re-encoding memories that are used together, so they stop swapping partners
  - the reason it is needed (shared first-pair order tags)

  The conceptual overlap with SLEEP is real ("replay recent experience offline to consolidate it"). The difference is also real:

  | | SLEEP | Seahorse |
  |---|---|---|
  | What is written | Low-rank experts inside the model's weights | Nothing in the model; a store of the model's own K,V positions, recurrent state and a steer vector beside it |
  | How | Distillation, RL, LoRA SFT | Forward passes only |
  | Unit of memory | None: knowledge diffuses into shared parameters | Discrete items, keyed, gated, inspectable, deletable per user |
  | Forgetting risk | Managed by capacity growth | None to the base model; interference only among stored items (the swaps) |
  | Target | Capabilities and knowledge for everyone | One user's facts and preferences across sessions |
  | Cost per sleep | Training runs (SFT 4× cheaper per step, by their own count) | A few forward passes per memory |

### 3.3 Experiment 3 (design only)
- **2a Abstraction (episodes → leaning → steer vector).** *Known as an idea, ours as a method.* Abstraction during consolidation is SLEEP's NREM story (in weights), Generative Agents' reflections, Auto-Dreamer's region rewriting and LightMem's abstraction (in text). Abstracting into a question-independent push beside a frozen model, with episodic residuals kept, is not found. The "Useful Memories Become Faulty" result is the main design risk: keep the episodes, and gate the abstraction.
- **2b Reconsolidation (later wins; suppress the loser with a negative gain, don't delete).** *Partly known.*
  - Text: contradiction resolution is in Mem0 (it deletes), MindMemOS, LightMem and Auto-Dreamer.
  - KV, trained, within one session: superseded-entry tagging and gating is SleepGate.
  - Still ours: extinction-style suppression of a stored slice, training-free and cross-session, with the dreamer deciding by replaying both episodes in order. SleepGate's PI-LLM task is a ready-made test for 2b.
- **2c Pruning by write-time surprise, novelty, valence.** *Partly known.*
  - Importance as the trigger: Generative Agents (LLM-rated importance) and SCM (importance tags, value-based forgetting).
  - Valence × surprise gating of what consolidates: EVAF (into LoRA, weak statistics).
  - Learned usefulness: Auto-Dreamer (masking utility).
  - SLEEP's "synaptic pruning" resets parameters and is not about what to keep.
  - Still ours: the falsifiable prediction test (does a frozen model's own surprise or novelty at write time predict later need, by AUC, with labels from LongMemEval or LoCoMo written by people with no stake in our signals). None of these papers tests that.
- **2d Re-filing keys.** *Partly known in text* (A-MEM's memory evolution; HippoRAG's offline index). Re-keying an activation memory under dreamer-generated questions is not found.
- **Overall:** experiment 3's *list of jobs* (abstract, resolve, prune, re-file) is now standard for offline text consolidation (Auto-Dreamer names almost the same four). What remains ours is doing them on activation memory with a frozen model, and the specific tests.

### 3.4 The "self-learning memory system" framing
- **Crowded.** SEAL ("self-adapting"), SLEEP ("self-modify", "recursive self-improvement"), MindMemOS ("self-evolving") and A-MEM ("agentic") all claim it, for weights or text.
- **What is distinctive in ours:** the frozen model does its own bookkeeping. It writes its own references (experiment 2), binds and owns at read time (experiment 1), and selects positions from its own answers (S-loo), with no weight change.
- **Action:** in outward writing, say "a memory the frozen model maintains itself" or "training-free self-maintained memory", not "self-learning", which reads as SEAL/SLEEP-style weight updates.

### 3.5 Paper 1 and paper 2
- **Paper 1** (arXiv 2610.14958): **unaffected.** It tests steering as memory and has no offline phase. Its design-limitation line about hand-written references points to follow-up work, which is fine.
- **Paper 2** (submitted, awaiting arXiv): **no claim is affected.** No sleep or consolidation paper studies where a fact is read, slices vs text, the hybrid's recognition/selection split, or storage-order swaps. Two places need citations in a v2:
  1. *Conclusion*, "is work that an offline consolidation phase can do and plain retrieval over text does not". This is still true as written ("plain retrieval"), but offline consolidation is published for weights and text (SLEEP, LightMem, Auto-Dreamer), so it should be cited, and the sentence should say ours is the cache-level, training-free version.
  2. *Hybrid states* paragraph: add Lee et al. (offline passes write a hybrid's GDN state before the KV is dropped, and need training; vanilla hybrids' states don't support deep computation over evicted context). It sits well beside Afendulev et al.

### 3.6 The website
- **The dreaming bullets** (`tools/manuscript/page.html`, the "direction" item, and the design list: 'an offline "dreaming" phase for what one forward pass at write time can't do …'). **Partly known; nothing contradicted.**
- **Action:** add one sentence and two links (SLEEP, Sleep-time Compute) saying how ours differs: frozen model, training-free, on cached positions. Following the project's rule, push only with the user's go-ahead.

---

# 4. What to cite, and where

**Paper 2, related work** (v2 or camera-ready). A new short paragraph after "Choosing what to keep" (suggested text; bib keys to add):

> *Offline consolidation.* Sleep-inspired methods consolidate recent experience offline. In weights, SLEEP distils fast parameters into new low-rank experts and trains on self-generated "dreams" \citep{behrouz2026sleep}. In a hybrid's recurrent state, offline recurrent passes write the state before the cache is cleared, with end-to-end training \citep{lee2026sleep}. In text, sleep-time compute rewrites the context before the query arrives \citep{lin2025sleeptime}, and agent memories are merged and rewritten between sessions \citep{fang2026lightmem,ye2026autodreamer}. Learned sleep cycles over the KV cache gate out superseded entries within a session \citep{xie2026sleepgate}. Our consolidation is training-free and operates on a frozen model's stored cache entries across sessions.

- **Hybrid states:** add `lee2026sleep` beside `afendulev2026hybrid`.
- **Conclusion:** append "as sleep-inspired methods do for weights and text \citep{behrouz2026sleep,fang2026lightmem}; here at the level of cached positions, without training".
- Bib entries: Behrouz, Hashemi, Javanmard, Mirrokni, arXiv:2606.03979 (2026); Lee, McLeish, Goldstein, Fanti, arXiv:2605.26099 (2026); Lin et al., arXiv:2504.13171 (2025); Fang et al., ICLR 2026, arXiv:2510.18866; Ye et al., arXiv:2605.20616 (2026); Xie, arXiv:2603.14517 (2026).

**Website, dreaming section** (one sentence after the dreaming bullet; with go-ahead):

> The idea of an offline "sleep" for language models is not new: Google's SLEEP trains new weights on the model's own "dreams" ([arXiv:2606.03979](https://arxiv.org/abs/2606.03979)), and Letta's sleep-time compute rewrites the context as text before the question arrives ([arXiv:2504.13171](https://arxiv.org/abs/2504.13171)). Here the model stays frozen, and the offline phase works on its own cached positions.

**Docs:**
- [README.md](README.md), "Why dreaming": add a line pointing here, and say the name follows SLEEP's usage for self-generated data.
- [03-consolidation.md](03-consolidation.md): a prior-work line under each of 2a–2d:
  - 2a: Generative Agents, Auto-Dreamer, and the "faulty memories" warning
  - 2b: SleepGate, Mem0, and the PI-LLM task as a test
  - 2c: EVAF, Generative Agents' importance, Auto-Dreamer's masking utility
  - 2d: A-MEM
  - and LightMem as the text baseline on LongMemEval
- [literature-select.md](literature-select.md): add Sleep-time Compute as the origin of anticipated queries, beside self-study.
- [core-idea.md](../reference/core-idea.md), CLS section: "Consolidation into weights is a parked future step" can now cite SLEEP as that step done by others.
- HANDOVER §1: one line under the framing bullet.

---

# 5. What this means for direction

**It supports the goal; it does not threaten it.** SLEEP is a strong signal that offline consolidation is where large labs think continual learning is going. It sits at the opposite end from Seahorse: it changes the shared weights, needs training at every sleep, has no unit of memory you can inspect or delete, and is evaluated on capabilities, not on one user's facts. For per-user memory in a served model, weight updates are costly (a training run, or an adapter, per user per sleep) and hard to undo, which is exactly where a frozen-model activation memory is useful.

The neuroscience is also on our side. In CLS terms, SLEEP is building the cortex's slow learning, and Seahorse is building the hippocampus beside a frozen cortex. Those are complements, not competitors. A combined system would keep episodes as slices (fast, exact, deletable), and could later distil only the lasting, gated ones into weights by a SLEEP-like step. "Useful Memories Become Faulty" argues for keeping that episodic store as the evidence.

**Useful consequences:**
1. A weight-based baseline. On our fact items, compare a stored slice with SEAL-style LoRA self-edits or sparse memory fine-tuning on the same sentence: use, interference on unrelated prompts, cost, and deletability.
2. Lee et al. suggests a training-free test on the 2B. Does looping the frozen hybrid over the stored turn before keeping its state turn recognition into recall?
3. LightMem and Auto-Dreamer are the text baselines experiment 3 must beat on LongMemEval.

**Positioning (one paragraph, for paper 2's v2 or the website).** Offline consolidation for language models is now an active line of work. SLEEP distils fast parameters into new experts and fine-tunes on the model's own dreams. Lee et al. train hybrids to write their recurrent state in offline passes. Text-memory systems such as LightMem and Auto-Dreamer rewrite and merge notes between sessions. Seahorse asks what an offline phase can do when nothing may be trained: the model stays frozen, and the memory is a few of its own cached positions (and, on hybrids, its recurrent state). In that setting, the offline phase is not abstraction into weights but bookkeeping the forward pass cannot do. It chooses which positions carry a memory, and it stores or re-encodes memories that are used together in an order that gives each its own binding tag, so separately stored facts stop swapping partners. These results are about a frozen model's activation memory and are complementary to weight-based sleep: a slice store is the fast, deletable episodic layer that weight consolidation could later draw on.

---

# Not read, partly read, or off-topic

- **Read in full or nearly (text):** SLEEP (main text and appendices A–B).
- **Read: abstract and key sections:**
  - Lee et al. (§1–5, discussion)
  - SleepGate (abstract, §1–2, module list by search)
  - Sleep-time Compute (abstract, §1–3)
  - Auto-Dreamer (abstract, §1)
  - Useful Memories Become Faulty (abstract, §4.1)
  - Generative Agents (§4.2)
- **Read: abstract plus text search only:** LightMem, Bottlenecked Transformers, SCM, MindMemOS, EVAF, A-MEM, Mem0.
- **Abstract only:** Nested Learning, SEAL, SSR, Sparse Memory Finetuning, Bazhenov et al., MemoryOS, HippoRAG.
- **Found by search, not read or filed:** Wake-Sleep Consolidated Learning, Memory for Large Language Models (2607.25380), HippoRAG 2, HEMA, Letta's blog.
- **Method caveat:** all reading was of extracted PDF text. SLEEP's continual-learning, long-context, translation and BABILong results are figures (Figs. 3–6), which I did not inspect; their captions and prose were read. The ICML 2026 listing of Lee et al. comes from a search result and was not checked.
