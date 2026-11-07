# Consolidating memories into weights: what the literature says happens to the things we measure

*Literature cluster C for the trained side. Question: if Seahorse distils stored memories (K,V slices) into weights, what happens to use, partner swaps, the licence gap, ownership, interference, deletion and cost? Written 2026-11-07. Read from extracted PDF text (`pdftotext -layout`); results shown only in figures are marked "figure only". Page numbers are the paper's printed page numbers. This note builds on [../dreaming/literature-sleep.md](../dreaming/literature-sleep.md), which covered SLEEP in full and most other papers at abstract level; it does not repeat that note's framing or overlap analysis.*

Terms as in the dreaming notes. "Slice" = cached K,V of a few positions of a past turn. "Teacher-forced" (TF) = scoring the gold answer's tokens; "in generation" = the model writes its own answer, judged afterwards. "Licence gap" = the model has the content (high TF) but refuses or hedges in generation. "Candidate 4" = SLEEP-style consolidation: teacher = frozen model + answering slice, student = same model + per-user LoRA without the slice, data = dreamed questions plus generic prompts, slices deleted afterwards.

---

## 1. The short version

**Takeaways.**

1. **Use context distillation, not plain SFT, and never train on the raw memory text alone.** Same synthetic data, KL-to-the-context-conditioned-teacher beats next-token loss by 8.3–8.6 chrF on MTOB and 3.7 accuracy points on LongHealth (Cartridges, App. A.2). Training on the raw passage barely moves knowledge: SEAL 33.5 vs 32.7 base on no-context SQuAD (Table 2); restatements are what work (rewrite prompt 49.4 → 55.6 after RL, Table 10).
2. **Distillation copies the teacher, including its habits.** SLEEP's own finding: after KL distillation "the student model has not learned to use it and so weakly mimics the sampling and performance of the teacher" (p.8–9), so they add an RL imitation term (removing it: AIME-24 79.2 → 76.8, Table 1). For us the teacher is the slice-conditioned model, so its refusals (0.70–0.79) and its partner swaps (19–41%) are part of what gets distilled unless we filter or fix them first. **Nobody distils from a restored KV slice**; every teacher in this cluster has the source as text in context.
3. **LoRA forgets the most of the weight routes on small, narrow data.** Learning 1K TriviaQA facts in sequence: NaturalQuestions F1 drops 89% (full FT), 71% (LoRA), 11% (sparse memory slots) at the same learning (SMF abstract). Cartridges: a LoRA-parameterised memory drops MMLU 60.0 → 45.3 as it grows to 1.06 GB; a trained KV prefix of the same size drops only to 54.3 (App. A.1). Interference has to be designed in (KL-to-base on generic prompts, small rank, few steps), not hoped for.
4. **Weights store associations directionally and as gist, not as bound specifics.** ROME/MEMIT: "Tim Cook is CEO of Apple" and "The CEO of Apple is Tim Cook" are separate edits (MEMIT §6, ROME §3.7). EVAF's LoRA consolidation of 30 persona facts lowers recognition CE (GPT-2: 8.21 → 5.75) but in free generation the persona-*discriminating* tokens (which persona has which city/instrument) appear **zero** times; "parametric encodes typology, retrieval delivers verbatim specifics" (abstract, §4.11). That is the weight-side analogue of our "recognition without recall" and a warning for binding.
5. **Sequential updates erode earlier ones; joint (batched) updates do not.** SEAL's sequential self-edits degrade earlier passages (Fig. 6, figure only). Sequential ROME collapses from n = 32 edits; batched MEMIT holds 98.9 efficacy / 88.6 paraphrase at 10,000 edits (MEMIT §5.2, Table 2). The weight-side version of our joint re-encoding: retrain one adapter over all of a user's memories, don't stack incremental ones.
6. **Bleed into similar subjects is the weight-side form of a partner swap, and it is measurable.** CounterFact's neighbourhood score (does an edit leak to similar subjects?): GPT-J unedited 83.5, MEMIT at 10k edits 73.7, fine-tuning 46.9 (and generation fluency collapses, 622 → 294) (MEMIT Table 2). Single-edit fine-tuning on GPT-J: 10.3, i.e. ~90% of neighbours wrongly changed (ROME Table 4).
7. **Deletion and ownership are not studied anywhere in this cluster.** No paper deletes one memory from an adapter, and none asks whether a learned fact is treated as the user's or as world knowledge. ROME's edits apply in every context (Pierre Curie becomes a physician in all prompts, Fig. 6), which suggests a consolidated user fact becomes "true about the world" for that adapter. Whole-user deletion = drop the adapter; per-memory deletion needs a retrain from a source we were planning to delete.
8. **Cost: a per-user LoRA is two to three orders of magnitude larger than our slices.** For Qwen2.5-1.5B, a rank-8 LoRA on all projections is ~9.2M parameters (~18 MB bf16); a 2-position slice is ~29K values (~57 KB): about 320× (my arithmetic, §3.4). Training costs in the papers: 30–45 s per LoRA fit + eval on a 7B (SEAL §5), ~30 min on 8×H100 per corpus (Cartridges §6), ~4 s per ROME edit on GPT-J (MEMIT App. B).

**Table.** "Gen." = evaluated in generation; "TF" = teacher-forced / probability comparison.

| Paper | Trained / frozen | Loss | Data | Interference / forgetting (numbers) | Gen. vs TF | Binding / similar facts | Deletion | Cost |
|---|---|---|---|---|---|---|---|---|
| SLEEP | New low-rank experts; rest frozen during distillation | GKD-style KL on teacher + student samples, plus RL "Learning to Imitate"; dreaming = LoRA SFT + ReST-EM | Self-generated "dreams", gradient-selected top-k + b random | SQuAD w/o dreaming 35.7 vs 48.9; continual-learning results figure only | Gen. (SQuAD, AIME) | Not measured | "Synaptic pruning" resets old experts wholesale | SFT 4× cheaper per step; 3.6–4.8× slower to equal accuracy |
| SEAL | LoRA r 64 (single passage); full FT (CPT) | Next-token SFT on self-edits; RL (ReST-EM) on the edit generator | Model's own implications / rewrites / QA of a passage | Sequential edits degrade earlier ones (figure only) | Gen., GPT-4.1 judge | Not measured | None | 30–45 s per inner loop (7B); RL round ~6 h on 2×H100 |
| Sparse Memory FT | Top-t memory-layer slots (t = 500 facts, 10K docs); all else frozen | Next-token on paraphrases / augmentations | Paraphrased TriviaQA statements; Active-Reading augmentations | NQ F1 −89% FT, −71% LoRA, −11% SMF | Gen. (F1) | "Core set" of 100–500 slots per fact | None (slots are localised; untested) | Same steps as baselines; SGD |
| SSR | LoRA r 8 on q,v | Next-token | Base model makes inputs (ICL); latest model writes outputs | BWT −0.79 (SSR) vs −3.02 (10% real replay) vs −42.4 (none), Llama-2-7b | Gen. (ROUGE-L) | n/a | n/a | Small |
| EVAF | LoRA r 32 on attention; gated | CE + L2 anchor + replay | Persona-fact stream, gated by valence × surprise | Not the focus; long streams drift | Both: CE improves, free generation of specifics = 0 | Cross-persona specifics never produced | n/a | Few inner steps per trigger |
| Auto-Dreamer | Text consolidator (Qwen3-14B, GRPO); agent frozen | GRPO: task reward + 0.5 × masking utility | Agent trajectories → text memory | Over-abstraction loses locations (one ALFWorld family) | Gen. (task success) | n/a | Omission-based forgetting in text | 8×H100, 200 steps |
| Useful Memories Become Faulty | Nothing (text consolidation) | n/a | Trajectories / ground-truth solutions | ARC-AGI accuracy 100% → 54% after consolidation | Gen. | Updates on one task overwrite another | Episodic-only control as good | n/a |
| ROME | One MLP matrix, rank-one | Closed form; v* from NLL + KL-to-base ("essence") | One counterfactual statement | Neighbourhood 75.4 vs 78.1 unedited (GPT-2 XL); FT 40.4 | TF (probability) + generation consistency | Directional; "Space Needle" needs two edits | Not tested | Seconds per edit |
| MEMIT | Several MLP layers, batched | Closed form, least squares | 10,000 facts in one batch | At 10k: NS 73.7 vs 83.5 unedited; FT-W 46.9 + fluency collapse | TF + generation | Conflicting (s,r) edits excluded; mixed relations ≈ average | Not tested | 7.44 h / 10k (unbatched, GPT-J) |
| Cartridges | Trainable KV prefix (or LoRA, ablation); model frozen | Token-level KL to the model with the chunk in context ("context distillation") | Self-study: model quizzes itself about each chunk, 5 seed-prompt types | LoRA: MMLU 60.0 → 45.3 with size; prefix: → 54.3 | Gen. (LongHealth acc., MTOB chrF) + perplexity (QASPER) | Independently trained cartridges compose by concatenation | Delete the file | ~30 min on 8×H100 per corpus (Llama-8B) |

---

## 2. Per-paper notes

### SLEEP (Behrouz, Hashemi, Javanmard, Mirrokni; arXiv 2606.03979v2)

Framing and results are in the dreaming note §1. Here only the distillation mechanics, the LTI loss, dream selection and the ablations.

- **What is trained, what is frozen.** During consolidation, only a newly added low-rank expert (A ∈ ℝ^{d×d_low}, B ∈ ℝ^{d_low×d}) in the slower MLP block: "we freeze all the parameters in the student model and only updates the expanded parameters. This ensures that the transferred knowledge does not interfere with the old knowledge" (p.8). Teacher = the model before the update; student = the model with the faster block's prospective base-weight update applied, its old experts reset, and the new expert added (p.7–8). In practice: 5 extra MLP blocks of dimension 64, active parameter count unchanged (App. B, p.25).
- **Distillation loss.** GKD form: L = (1−λ) E_{(x,y)∼D}[F(teacher‖student)(y|x)] + λ E_{x∼D} E_{y∼student}[F(teacher‖student)(y|x)], where F is a divergence between output distributions and λ is the fraction of on-policy student samples; no gradient through the student's sampling (p.8). D is sampled from the teacher. Which divergence (forward/reverse KL, JSD) is not stated in the text.
- **Why they add RL ("Learning to Imitate").** Quote that matters for our licence gap: "despite having access to the knowledge, the student model has not learned to use it and so weakly mimics the sampling and performance of the teacher" (p.8–9). LTI samples a random prefix of each teacher "dream" and rewards the student's continuation by γ·r_sem + (1−γ)·r_abs. r_sem is 0/1 from a frozen reward model judging semantic equivalence; r_abs = 1 − Levenshtein/ max length if below a threshold z₀, else 0 (Eq. 3–4, p.9). The combined objective is (1−α)·E[reward] − α·KL (p.9).
- **Dream selection.** Sample m dreams from the model with the task context C in context; during sampling each MoE router also picks one random expert ("incorporates random irrelevant knowledge to the dreaming", p.10). Score each dream by the gradient of the SFT loss (importance g = ∇θ L_SFT(dream)); keep top-k plus b random for diversity. Each kept dream LoRA-fine-tunes an isolated copy; reward 1 if the copy improves on the task; ReST-EM as in SEAL (Eq. 5, p.10). ARC: 60 dreams sampled, 45 rejected per task (App. B.3).
- **Ablations.**
  - Consolidation on math (Table 1, Qwen3-8B, AIME-24/25/HMMT-25): Sleep 79.2/69.0/46.1; − imitation learning 76.8/67.9/45.0; − semantic reward 78.9/69.2/44.5; − expansion 78.2/67.9/44.9; plain on-policy self-distillation (OPSD) 76.6/67.4/45.1; OPSD + expansion 77.9/68.2/45.9. Effects are 0.2–2.6 points, no variance reported.
  - Dreaming on SQuAD (Table 3, single passage / CPT n = 200): Sleep four-level 48.9/46.2; − gradient selection 47.1/45.2; − random expert 48.0/44.7; **− dreaming 35.7/36.2**; fine-tuned with no dreaming 33.4/32.0; SEAL 46.7/43.2. So for knowledge incorporation almost all of the gain is the SEAL-style self-generated data, not the consolidation.
- **Forgetting.** Continual-learning and translation results are figure only (Figs. 3, 5). The translation text says Cartridges and SFT "faced catastrophic forgetting in at least one of the languages, performing even weaker than ICL" (p.12).
- **Generation vs TF.** All reported metrics are generation (avg@16, no-context QA, ChRF). No TF probe, no multi-hop, no binding, no attribution.
- **Deletion.** "Synaptic pruning" resets all experts previously added to the faster block once they are distilled (p.9). It is wholesale capacity recycling, not per-memory deletion.
- **Cost.** Training config: LR 5e-6, batch 32, 100 steps, LoRA r 64, α 128 (Table 5). "With the same number of steps, SFT is 4x more efficient than our method"; to reach equal accuracy SFT needs 4.3×, 3.6×, 4.8× wall-clock (App. B.5).
- **Pointers SLEEP gives that matter here (App. A.4, not read):** SDFT (Shenfeld et al. 2026: demonstration-conditioned teacher, less forgetting than SFT), On-Policy Context Distillation (Ye et al. 2026: teacher conditioned on transient in-context information "so that the student internalises that information into its parameters"), OEL (Ye et al. 2026: context distillation at deployment), and **Kim et al. 2026**: conditioning the teacher on rich privileged information "can suppress the model's epistemic verbalisation (its expression of uncertainty during reasoning) … with drops of up to 40%" out of distribution. The last one is a direct warning for candidate 4 (see §3.2). Iterated self-distillation can also leak privileged information and collapse (He et al. 2026; Wang et al. 2026).
- **Read:** §3.2–4.2 and App. A.4–B.5 again in full for this note; the rest per the dreaming note.

### SEAL — Self-Adapting Language Models (Zweiger, Pari, Guo, Akyürek, Kim, Agrawal; arXiv 2506.10943v2, NeurIPS 2025)

- **Trained / frozen.** Inner loop: a LoRA (single passage; r ∈ {32, 64}, all MLP and attention projections) or full fine-tuning (CPT with 200 or 2,067 passages). Outer loop: the model that *writes* self-edits is trained by ReST-EM (filtered behaviour cloning: keep the best self-edit per context, SFT on it). GRPO and PPO "unstable" (§3.1).
- **Loss.** Inner: "the standard causal language-modeling loss over each sequence" of the self-edit (§3.2). No distillation, no KL to base. Outer: binary reward (did the update improve accuracy on the passage's questions).
- **Data (knowledge incorporation).** Qwen2.5-7B base; SQuAD v1.1 passages; the model is prompted "produce a list of implications derived directly or indirectly from the content", the output is split by newlines into training documents (App. B.3). Evaluation: 974 questions over 200 passages, **no passage in context**, generation graded yes/no by GPT-4.1 (App. B.4).
- **Main numbers (Table 2).** Single passage (LoRA): base 32.7; train on passage 33.5; passage + own implications 39.7; passage + GPT-4.1 implications 46.3; SEAL 47.0. CPT n = 200: 32.7 / 36.0 / 50.6 / 59.4 / 58.2. CPT n = 2,067: 29.0 / 31.2 / 43.4 / 49.2 / 46.4. Accuracy falls as passages grow from 200 to 2,067 at a fixed recipe.
- **Format matters more than RL (Table 10).** Base prompt → after 2 RL rounds: implications 39.7 → 47.0; implications-long 49.3 → 54.4; rewrite 49.4 → 55.6; self-QA 37.3 → 48.7; no prompt 13.8 → 18.9. A better prompt alone gives as much as RL. Chain-of-thought at evaluation ("pull out" from weights) did not help (37.8).
- **Hypernetwork comparison (App. B.8, Table 7).** Generative Adapter (a hypernetwork that writes a LoRA from the context): 66.8 single passage, but 28.0 in CPT over 200 passages (SEAL 58.2). Amortised adapter generation is good per item and does not compose.
- **Forgetting.** Sequential self-edits on a stream of passages: "performance on earlier tasks gradually declines as the number of edits increases … Still, it can perform multiple updates without complete collapse" (§5, Fig. 6). The matrix is figure only; App. Table 5 gives only standard errors (~0.02–0.04). No measurement on unrelated tasks.
- **Generation vs TF; reasoning.** All generation; single-hop SQuAD questions. No yes/no premises, no multi-hop, no paraphrase-vs-train-question split.
- **Binding, ownership, deletion.** Not studied.
- **Cost.** "each self-edit evaluation takes approximately 30–45 seconds" (§5); one ReST-EM round = 50 passages × 5 self-edits × 3 seeds = 750 inner loops ≈ 6 h on 2×H100 (App. B.5). A GPT-4.1 rubric reward instead of the inner loop: 45.6 vs 47.0, 5 min vs 6 h (App. B.10).
- **For candidate 4.** The data recipe matters more than the optimiser: diverse restatements (rewrite, long implications) of each memory, never the raw memory alone. SEAL's limitation note proposes letting the model write its own evaluation questions while the content is in context, which is our dreamed-question set.
- **Read:** in full (main text, App. B; App. A on ARC skimmed).

### Continual Learning via Sparse Memory Finetuning (Lin, Zettlemoyer, Ghosh, Yih, Markosyan, Berges, Oğuz; arXiv 2510.15103, FAIR)

- **Trained / frozen.** A 1.3B model whose layer-12 FFN (of 22) is replaced by a memory layer: 1M slots, k = 32 accesses per token, 4 heads, value dim 1024 (§5). Only the values of the top-t slots per batch are trained; everything else is frozen. Implemented as a gradient mask (§4).
- **Slot selection (TF-IDF).** For each batch, count slot accesses c(i); score = c(i)/Σc · log((|B|+1)/(#background batches touching i + 1)), with the background = 1,000 random DCLM batches, precomputed and stored (§4). TF-only ranking learns as much but forgets more; the gap widens at t = 50 (Fig. 6). Background = the learning set itself (TriviaQA) → more forgetting; background = NaturalQuestions ≈ DCLM (Fig. 7).
- **Loss / data.** Next-token. Facts: 1K TriviaQA questions rephrased as statements, each paraphrased N times to fill a batch of 64 (padding masked so paraphrases don't see each other) (§5.1). Documents: 1,824 Wikipedia chunks for 100 SimpleQA questions, each chunk expanded with Active Reading augmentations (§5.2). t = 500 (facts), 10,000 (documents).
- **Forgetting numbers.** Abstract: "NaturalQuestions F1 drops by 89% after full finetuning on new facts and 71% with LoRA, sparse memory finetuning yields only an 11% drop with the same level of new knowledge acquisition." The curves (Figs. 3–5) are figure only. App. Table 2 (SGD baselines, numbers): LoRA lr 2e-1, r = α = 128 reaches TQA F1 0.58 with NQ F1 0.09 and GSM8K NLL 2.2; full FT lr 2e-1 reaches 0.65 with NQ 0.01, NLL 3.5; unadapted ≈ 0.36 / 0.14 / 1.5. Sparse memory with SGD: TQA > 0.7, NQ < 0.15, NLL < 1.5 (caption). The Pareto sweep (Fig. 5) has LoRA ranks 32–256, α ½–4× rank, LR 5e-6 – 2e-4: forgetting grows with learning capacity for LoRA and FT.
- **Where facts live.** The "core set" (slots shared by all paraphrases and by the question) is ~100–500 slots per fact; 25–100 trainable slots suffice to answer; trainable slots align with entity boundaries (Table 1, §6).
- **Generation vs TF.** Generation (F1 on TriviaQA questions, NQ, SimpleQA); GSM8K as NLL (TF). Questions differ from training statements (paraphrase generalisation is built into the setup). No multi-hop, no binding, no attribution.
- **Personalisation.** Named as a target: "learning a user preference from a single message" is the hard small-data case (§5.1).
- **Deletion.** Not tested. Localised slots make per-fact rollback plausible (restore the touched slot values), but nothing is measured.
- **For candidate 4.** Needs a memory-layer model, so not a drop-in for Qwen. Its lessons transfer: (1) the forgetting of LoRA in a narrow small-data stream is large; (2) choosing *which* parameters to update with an IDF against generic text is cheap and effective; the LoRA analogue is a KL-to-base anchor on generic prompts. (3) Paraphrase augmentation is part of the method.
- **Read:** in full.

### Self-Synthesized Rehearsal — SSR (Huang, Cui, Wang, Yang, Liao, Song, Yao, Su; arXiv 2403.01244v2, ACL 2024)

- **Trained / frozen.** LoRA r 8 on query and value projections, Adam 2e-4, 3 epochs (§5.1). Llama-2-7b, Llama-2-7b-chat, Alpaca-7b.
- **Data.** For each earlier stage: the *base* model generates synthetic inputs and outputs from K = 2 demonstrations (ICL); the *latest* fine-tuned model rewrites the outputs ("synthetic output refinement"); K-means (20 clusters, SimCSE embeddings) picks instances near centroids; these are rehearsed alongside the new stage's data (§4). Loss: next-token.
- **Numbers (Table 1, 5 SuperNI tasks, averages over 3 orders).** Llama-2-7b: multi-task upper bound AR 53.05; no rehearsal AR 19.03, BWT −42.39; 1% real replay 49.67 / −4.02; 10% real replay 50.32 / −3.02; SSR 52.16 / −0.79. Alpaca-7b general abilities (Table 3): AlpacaEval 21.08 → 8.82 with no rehearsal, 19.68 with SSR; MMLU 41.0 → 34.1 / 37.1.
- **Key finding for us.** Rehearsing with synthetic inputs and the current model's *own* outputs beats rehearsing with real data in some orders: "synthetic instances, with lower model perplexity, embody the LLM's real-time acquired knowledge" (§5.5, Figs. 6–7). Removing output refinement hurts BWT significantly (Fig. 4, figure only).
- **For candidate 4.** The interference set should be generic prompts *with the frozen model's own answers as targets* (or a KL to the frozen model on them), not human-written answers. That is SSR's recipe and also ROME's "essence" KL term.
- **Not relevant here:** binding, ownership, deletion, knowledge use.
- **Read:** §1–5.5 in full; appendix not read.

### Nested Learning / Hope (Behrouz, Razaviyayn, Zhong, Mirrokni; arXiv 2512.24695) — skim

- Continuum Memory System = a chain of MLP blocks updated at different frequencies (§7.1). SLEEP's consolidation is the offline step between levels.
- Continual translation of two unseen languages (§9.1): ICL "faces dramatic performance drop and almost rely on its capabilities achieved in its pre-training"; Hope-3 "almost recovers" single-language performance. Figure/table not parsed.
- Own caveat: "catastrophic forgetting is not 'solved' in general … a natural consequence of compression, where the limited capacity of the network forces the model to forget" (§10).
- No per-item memory, binding, deletion or attribution. **For candidate 4:** background only. **Read:** abstract, §7.1, §9.1, conclusion by search.

### EVAF — A Test-Retest Protocol for Selective Parametric Consolidation (Han; arXiv 2606.29916v1, single author)

- **Trained / frozen.** Base model frozen; a fast "working-memory" prefix state plus a slow LoRA store (r 32 on attention projections). Consolidation loss: CE on the buffered sample + λ‖θ − θ₀‖² (an EWC-like L2 anchor) + experience replay (§3.2, Eq. 3.3). A dual valence × surprise gate decides what enters the buffer (§3.3). GPT-2 124M, TinyLlama-1.1B-Chat, Qwen3-1.7B-Base, Mistral-7B-Base.
- **Persona-fact task (§4.10).** 30 facts (5 personas × 6 templates) interleaved with 30 or 150 filler turns; then per-token CE on each fact under "About {persona}: ". Mean CE (lower is better), 4 seeds:

  | Stream, model | Frozen | EVAF (V9) | naive every-step LoRA (B1b) | RAG (B3b) |
  |---|---|---|---|---|
  | 60-turn, GPT-2 | 8.21 ± 0.55 | 5.75 ± 0.58 | 6.51 ± 0.12 | 2.18 |
  | 60-turn, TinyLlama | 12.10 ± 1.71 | 11.66 ± 2.13 | 6.96 ± 1.40 | 4.39 |
  | 180-turn, GPT-2 | 8.62 ± 2.09 | 7.06 ± 0.96 | 6.63 ± 1.15 | 2.18 |
  | 180-turn, TinyLlama | 10.28 ± 1.01 | 9.29 ± 0.63 | 7.75 ± 1.66 | 4.39 |

  Retrieval (the fact in context) beats every weight method by a wide margin.
- **The finding that matters: CE improves, specific recall in generation does not (§4.11).** Free generation from "About {persona}: {persona} " (10 × 40 tokens, T = 0.8). Broad keyword hit rate rises (GPT-2 60-turn: frozen 1%, EVAF 54% ± 44%, naive LoRA 81% ± 14%). But for the 3 persona-*discriminating* tokens per persona (e.g. Mandarin/Reykjavik/cello for Alice vs Portuguese/Marrakech/oboe for Bob), "the multi-seed mean discriminating-token count is zero — neither parametric method generates the specific discriminating tokens". Only RAG does (13.5% target vs 2% other persona). The author's summary: "parametric encodes typology, retrieval delivers verbatim specifics" (abstract).
- **Forgetting.** A 600-turn stress test worsens both methods (end-of-stream drift, §4.10 item 6). Scaling to larger models "triggers V8-style catastrophic forgetting" without retuned anchors (§1).
- **Statistics.** EVAF vs frozen CE on GPT-2 60-turn: p < 0.001 (paired t); productive recall: p = 0.095; "not significant at n = 4" (abstract, §4.10.2).
- **For candidate 4.** (1) The TF-vs-generation split exists on the weight side too, but shifted: weights give gist and familiarity, not bound specifics. Our slice result was the reverse problem (content present, licence missing). A consolidated LoRA could fail on both. (2) Free generation from a content-free prompt is a cheap leak/interference probe: does the adapter push user facts into unrelated answers? (3) Small model, single author, n = 4: weak evidence, but the zero-discriminating-token result is a clean negative worth testing on our items.
- **Read:** abstract, §3.2–3.3, §4.10–4.11 in full; the rest by search.

### Auto-Dreamer (Ye, Liu, Wang, Yu, Zhao, Liu, McAuley, You; arXiv 2605.20616) — the masking utility in detail

- **Setting.** Text memory. A frozen writer appends typed entries per session; a consolidator (Qwen3-14B, trained by GRPO) rewrites a "region" of the bank into a replacement set S. The task agent (Qwen3.5-9B) is frozen (§4).
- **Masking utility (Eq. 2–3).** Reward r_g = U_V(S_g) + α·r_cf(S_g; V), where U_V is mean return on evaluation tasks with the bank = S_g only, and r_cf = U_V(S_g) − E_{S̃∼q_ρ}[U_V(S̃)], with q_ρ masking a fixed fraction ρ of entries uniformly at random, estimated by Monte Carlo. Settings (Table 6): α ("MC-dropout weight") 0.5, drop fraction ρ 0.5, sample fraction 0.25, min N 1. "Masking load-bearing entries lowers performance, masking redundant entries has little effect because duplicate information remains, and masking harmful entries can improve performance, making r_cf negative" (§4.3).
- **How it differs from our S-loo.** It is a *set-level* random-50% ablation used as an RL reward for the writer of the set; credit to individual entries is implicit through GRPO. S-loo is a per-item leave-one-out on the frozen model's own answer, used directly as a selection score, no training. Same idea (an item's worth = the loss when it is removed), different granularity and use.
- **Ablation.** Without r_cf the bank keeps growing; with it, it grows then shrinks at no measurable cost in success (Fig. 3c–d, figure only). Untrained region rewriting already gives most of the compression (bank 6–11× smaller); GRPO adds +9.7 SR on ScienceWorld, +5.7 on WebArena, +1.0 on ALFWorld (Table 2).
- **Failure mode (Pattern 3, §5.5).** The trained consolidator replaces concrete locations ("alarmclock on desk-2, desklamp on desk-1") with generic procedures ("Key considerations for interacting with light sources"); the agent then loops on the wrong desk. That one family drives most of ALFWorld's small gain (+1.0 → +4.9 without it).
- **For candidate 4.** (1) A masking utility is the natural *gate* for what to consolidate: only memories whose removal hurts the dreamed questions are worth distilling. (2) Pattern 3 is the text version of EVAF's result: consolidation trades specifics for gist.
- **Read:** §4–5.5 and App. B.3 in full; §1–3 skimmed.

### Useful Memories Become Faulty When Continuously Updated by LLMs (Zhang, Lin, Wu, Sun, Li, Li, Peng; arXiv 2605.12978v2)

- **Setting.** Text consolidation by LLM agents (no weights). Included because its failure mechanisms carry over.
- **Numbers.** GPT-5.4 solves a set of ARC-AGI problems at 100% with no memory; after consolidating from ground-truth solutions to those problems, accuracy falls to 54% (§1, Table 19 "100% → 54%"; the abstract's wording "fails on 54%" conflicts with the body's "fails on 46%"). ScienceWorld utility rises then falls below the no-memory baseline (Fig. 1a, figure only).
- **Order dependence.** "the same set of trajectories can yield qualitatively different memories depending on the order and grouping of updates" (Static-All vs Stream, §1); "updates on one task overwrite memory of another".
- **Mechanisms (§6).** Misgrouping episodes that don't share structure; abstraction "can strip the applicability conditions of a lesson"; overfitting when the stream is narrow.
- **Control.** An episodic-only store (raw trajectories) is competitive with the consolidators; agents allowed to choose keep raw episodes and double the accuracy of forced consolidation.
- **For candidate 4.** Strong argument against deleting the slices: keep them as the evidence and gate consolidation. Also: "applicability conditions" ↔ ownership (whose fact, when it applies) is exactly what a distilled adapter may lose.
- **Read:** abstract, §1, search for numbers.

### Sleep-time Compute (Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez; arXiv 2504.13171) — skim

- Text, frozen, training-free. ~5× less test-time compute at equal accuracy; 2.5× lower cost per query when 10 related queries share a context; the benefit tracks how predictable the query is from the context (§5.3–5.4).
- **For candidate 4.** The dreamed questions define what is distilled. Predictable questions are covered; the unpredictable ones (composition, indirect use) will not be in the training set and are where a distilled adapter is least likely to work.
- **Read:** abstract, results by search.

### ROME — Locating and Editing Factual Associations in GPT (Meng, Bau, Andonian, Belinkov; arXiv 2202.05262, NeurIPS 2022)

- **Trained / frozen.** No training loop: one rank-one update to one mid-layer MLP output matrix (layer 18 in GPT-2 XL). k* = average MLP input at the last subject token over 50 random prefixes; v* = argmin of −log P(o* | prefix + prompt) + KL(P(·|"{subject} is a") ‖ unedited), the KL term "controlling essence drift" (Eq. 4).
- **Numbers (CounterFact, Table 4).** GPT-2 XL, 7,500 records: ROME efficacy 100.0, paraphrase 96.4, neighbourhood 75.4 (unedited 78.1), fluency GE 621.9 (626.6), score 89.2. Fine-tuning: efficacy 100, paraphrase 87.9, **neighbourhood 40.4**. GPT-J, 2,000 records: ROME 99.9 / 99.1 / 78.9 (unedited 83.0); FT 100 / 96.6 / **10.3**, fluency 387.8 (621.8). zsRE (Table 1): ROME efficacy 99.8, paraphrase 88.1.
- **Edit location.** Edits generalise and stay specific only at mid layers at the last subject token; "editing the late-layer attention modules leads to regurgitation" (§3.4).
- **Generation.** ROME makes generations consistent with the new fact in all contexts ("Pierre Curie often collaborated with a fellow physician", Fig. 6). Human raters: 1.8× more consistent, 1.3× less fluent than FT+L (§3.6).
- **Limits (§3.7).** "Associations edited by ROME are directional … 'The iconic landmark in Seattle is the Space Needle' is stored separately from 'The Space Needle is the iconic landmark in Seattle'." Also: "the model will guess plausible new facts that have no basis in evidence".
- **Binding, ownership, deletion.** Neighbourhood score = bleed to similar subjects (our swap analogue). No attribution, no deletion test (a rank-one update can be subtracted algebraically, but this is not evaluated).
- **For candidate 4.** (1) The KL-to-base "essence" term is the template for our interference regulariser. (2) Fine-tuning leaks to similar subjects heavily; for users with several similar facts (sister's name, dog's name), measure bleed. (3) Directionality: train both directions and perspectives.
- **Read:** §3 in full (3.1–3.7), App. tables by search.

### MEMIT — Mass-Editing Memory in a Transformer (Meng, Sen Sharma, Andonian, Belinkov, Bau; ICLR 2023)

- **Trained / frozen.** Closed-form least-squares updates spread over several mid-layer MLPs; all edits inserted in one batch.
- **Scaling (Table 1, Table 2).** zsRE, 10,000 real facts, GPT-J: MEMIT efficacy 96.7, paraphrase 89.7, specificity 26.6 (unedited 27.0); sequential ROME 21.0 / 19.6 / 0.9; fine-tuning (FT-W) 69.6 / 64.8 / 24.1. CounterFact, 10,000 counterfactuals, GPT-J: MEMIT efficacy 98.9, paraphrase 88.6, neighbourhood 73.7 (unedited 83.5), fluency 619.9 (622.4); FT-W 99.4 / 77.0 / 46.9 with fluency 293.9 ("complete generation failure"); ROME 50.2 / 50.4 / 50.2. "ROME performs well up to n = 10 but degrades starting at n = 32" (§5.2.2).
- **Similar facts together (§5.4, Fig. 7).** Mixing two relations with similar subjects or similar objects: performance "close to the average of the performance of each relation without mixing", up to 700 facts. Some relations are harder (plays sport, owned by company) and show a generalisation–specificity trade-off (§5.3).
- **Conflicts.** Edits that give the same (subject, relation) different objects are excluded by construction (Eq. 5). "Later wins" is not handled.
- **Directionality.** Same limit as ROME: "Tim Cook is CEO of Apple" vs "The CEO of Apple is Tim Cook" (§6).
- **Cost (App. B).** 10,000 edits on GPT-J: MEND 98 s; FT ~29 min; MEMIT 7.44 h (unbatched z-optimisation, "embarrassingly parallel"); ROME 12.29 h (~4 s per edit).
- **For candidate 4.** Batched/joint insertion is what scales; sequential insertion collapses. MEMIT is a training-free weight route and a fair comparator for simple (subject, relation, object) user facts, but cannot express preferences or "the user told me" framing.
- **Read:** §3, §5.2–7 in full; App. B runtime by search.

### Cartridges (Eyuboglu, Ehrlich, Arora et al.; arXiv 2506.06266v3) — context distillation, self-study, LoRA comparison only

- **What is trained.** A KV prefix Z (p virtual tokens per layer), model frozen; initialised from the KV of the corpus's first p tokens; the first (attention-sink) token is frozen, else MMLU dips below 30% during training (§3.2, App. A.1).
- **Self-study data (Alg. 1).** Chunk the corpus (512–4,096 tokens); two copies of the same model, both with the chunk in context, hold a k-turn conversation; A's first message is seeded by one of 5 generic seed-prompt types (structuring, summarisation, question, use cases, creative). k = 1 in all main experiments. Tens of thousands of synthetic conversations, one epoch (§4.1, Fig. 5).
- **Loss (Eq. 3).** Σ_i KL( F(·| chunk ⊕ x[:i]) ‖ F_Z(·| x[:i]) ): token-level KL from the model *with the chunk in context* to the model *with the cartridge*, over the synthetic sequence, using the teacher's top output probabilities collected during generation (App. A.2).
- **Distillation vs next-token, same synthetic data.** MTOB (2,048-token cartridge): 24.9 → 33.2 chrF (8.3; §5.3 says 8.6, 24.9 → 33.5); LongHealth 47.6 → 51.3 (+3.7). "for a fixed amount of SELF-STUDY compute, context-distillation is considerably more effective" (App. A.2). Next-token on the *raw corpus* "memorize[s] with near perfect perplexity" but does not generalise beyond regurgitation (§4, Fig. 3).
- **Seed prompts.** 5 types vs 1: MTOB 24.1 → 32.0; LongHealth 45.8 → 51.3 (App.) / 43.6 → 48.4 (main text); no gain on QASPER, which is mostly factual (App. A.2).
- **KV prefix vs LoRA (App. A.1, memory-matched).** Out of domain (MMLU): LoRA 60.0 → 45.3 as size grows to 1.06 GB (rank 1632); prefix 54.7 → 54.3 from 0.15 to 0.96 GB. In domain (LongHealth): best prefix 55.6% (0.96 GB), best LoRA 47.25% (0.26 GB), LoRA at 0.96 GB 41.3%. MTOB: prefix +4.5 chrF at ~0.6 GB. "It isn't clear why prefix-tuning is so much more robust than LoRA" (App. A.1).
- **Composition (§5.4).** Two independently trained cartridges, concatenated without retraining, answer questions needing both 10-Ks better than either alone or truncated ICL (Fig. 7, perplexity, figure only).
- **Serving.** A cartridge loads into ordinary prefix-cache slots; LoRA "require[s] custom infrastructure to serve efficiently to multiple users" (§3.3).
- **Cost.** "training an ICL-quality CARTRIDGE takes ∼30 minutes on a single 8×H100 node (for LLAMA-8B)" (§6), for 100k–484k-token corpora.
- **Prior context-distillation work it cites (not read):** Snell et al. 2022 "Learning by distilling context"; Kujanpää et al. "Knowledge injection via prompt distillation" (teacher with privileged text writes QA, LoRA r 1024 student trained by full-distribution distillation; SQuAD); Caccia et al. 2025 "deep context distillation" knowledge modules; Bhargava et al. "Prompt baking". LIFT (Mao et al.) found synthetic QA "offer minimal benefit" with only 10 examples (App. B.1.3).
- **For candidate 4.** (1) The loss to use. (2) Their teacher is text in context, ours would be a restored slice. (3) The cleanest evidence that, at equal memory, a trained KV object interferes far less with unrelated prompts than a LoRA. That argues for testing a "trained slice" (cartridge initialised from our slice) as a parallel arm to the LoRA.
- **Read:** §1–4, §5.3–5.4, §6, App. A.1–A.2, App. B.1 (as scoped; the rest is covered by another agent).

---

## 3. Design notes for candidate 4

### 3.1 The loss: context distillation vs SFT on dreamed QA

**Recommendation: token-level KL distillation from the slice-conditioned teacher, on dreamed QA whose teacher answers have been filtered, plus a KL-to-frozen-base term on generic prompts.** Run SFT on the same data as the baseline arm.

- **Evidence for distillation over SFT.** Cartridges, same data: +8.3–8.6 chrF (MTOB), +3.7 (LongHealth) (App. A.2). Kujanpää's prompt distillation (cited by Cartridges) reports the same with a LoRA student. SLEEP uses GKD-style distillation. SEAL uses plain SFT and gets 47.0 on SQuAD; no paper compares the two on short personal facts.
- **Why distillation should matter more for us than for them.** Our memories are one or two facts; an SFT target is a single answer string, which teaches the surface form. The teacher's full distribution over the answer carries the slice's graded beliefs (alternatives, confidence). It also carries everything else (§3.2), so the targets must be chosen.
- **Proposed loss.**
  L = Σ_{(q,a) ∈ dreamed, filtered} Σ_t KL( P_teacher(·| slice, q, a_<t) ‖ P_student(·| q, a_<t) )
    + β Σ_{g ∈ generic} Σ_t KL( P_frozen(·| g, y_<t) ‖ P_student(·| g, y_<t) )
  with a the teacher's own sampled answer (sequence-level KD) or the gold answer string, and y the frozen model's own continuation of generic prompt g (SSR's self-rehearsal; ROME's essence term). Top-k logits (k ≈ 20–50) are enough, as Cartridges stores "top output probabilities".
- **Off-policy first, on-policy second.** Start off-policy (teacher-generated answers): cheap, stable. On-policy reverse KL on student rollouts (GKD, OPSD) is the next step if the student answers correctly when forced but not when free (SLEEP's "weakly mimics" problem). SLEEP's LTI term (semantic + Levenshtein reward) is the heavy version; its measured gain is 0.3–2.4 points on math.
- **Data recipe.** From SEAL and Cartridges: several formats per memory (direct question, rewrite, implication, indirect use, yes/no premise, refusal-bait), diverse seed prompts, both directions and both perspectives (§3.5). Never the raw turn text alone (SEAL 33.5 vs 32.7).
- **Gate what gets consolidated.** Auto-Dreamer's masking utility or our S-loo applied per memory: consolidate only memories whose removal from the slice store measurably hurts answers on dreamed questions. "Useful Memories Become Faulty" says to gate and not fire consolidation after every session.

### 3.2 Expected failure modes

1. **The licence gap is copied, or flipped.** A teacher that refuses 70–79% of the time in generation produces refusal-heavy samples; a KL on those makes the student refuse. If we filter to answering samples (rejection sampling, as SEAL's ReST-EM and SSR's refinement do) or use the gold answer as the teacher-forced target, the student learns to answer. Kim et al. 2026 (cited in SLEEP App. A.4) warn of the opposite: teachers with privileged context suppress uncertainty, with OOD drops up to 40%. Expect a student that answers confidently even for facts it was never told. Measure: refusal rate on *unasked* user facts (it should stay high) beside use on asked ones. ROME's limitation is the same risk ("guess plausible new facts").
2. **Partner swaps are copied from the teacher.** If the teacher holds two separately stored slices that swap 19–41% of the time, its answers carry the swaps. Teacher slices must be the jointly re-encoded or storage-order-fixed version, or one slice per training question.
3. **New swaps from the weights' own binding.** In weights, binding is keyed on the subject and is directional (ROME, MEMIT). Similar subjects bleed (MEMIT neighbourhood 83.5 → 73.7 at 10k edits; fine-tuning far worse). EVAF: zero persona-discriminating tokens in free generation, so a LoRA may learn "this user likes a colour and has a pet" without binding which value goes with which. Measure swaps with the same multi-fact items as experiment 1.
4. **Gist instead of specifics.** EVAF and Auto-Dreamer's Pattern 3 both show consolidation keeping the type and dropping the value. TF probes will look good (EVAF: CE 8.21 → 5.75) while generation of the exact value fails. Always score in generation with a judge, as we do now.
5. **Directionality / reversal.** Trained on "What is my favourite colour? → Teal", the student may not answer "Which colour do I like best?", "Is teal my favourite?" or "Suggest a shirt for me". ROME/MEMIT state it; the reversal curse (below) generalises it.
6. **Ownership drift.** ROME edits hold in every context; a consolidated "teal is the favourite colour" may be asserted as the *model's* preference, the steer channel's failure ("Teal is my favourite colour!"). Train answers framed as the user's ("You told me your favourite colour is teal") and test both "what's my…" and "what's your…".
7. **Interference and leak.** LoRA in a narrow stream: NQ −71% (SMF); MMLU −14.7 points at large rank (Cartridges). Leak: EVAF's content-free prompts gain persona keywords (1% → 54–81%), so user facts appear in unrelated answers. Measure both: a generic-prompt KL/accuracy set, and content-free probes for unprompted mentions.
8. **Sequential sleeps erode earlier memories.** SEAL Fig. 6; sequential ROME collapses at n ≥ 32. Retrain each user's adapter from all their memories at every sleep (MEMIT-style batch), or keep the adapter fixed and add new memories as slices until the next full retrain.
9. **Order and schedule sensitivity.** "Useful Memories": the same episodes give different memories under Static-All vs Stream. MEMIT's weight decay is "highly sensitive to the number of edits". Fix the schedule and report it.
10. **Hyperparameter fragility at small scale.** SEAL tunes LR, epochs and rank per setting (Tables 3–4); SMF found AdamW interacts badly with sparse updates; EVAF needed scale-specific anchors. Budget a sweep.

### 3.3 Distilling from a context vs from a stored KV slice

- **Nobody in this cluster distils from a restored KV cache.** Every teacher has the source as text in context: Cartridges (chunk in context), SEAL (passage in context while writing self-edits), SLEEP (pre-update model), the OPSD/SDFT/OEL line cited by SLEEP (demonstrations, documents, system prompts), Kujanpää, Snell. Cartridges goes the *other* way: text-in-context teacher → trained KV student, initialised from real tokens' KV (random-vector init: 29.9% vs real-token init: 51.3–55.3% on LongHealth, §5.3).
- **What changes when the teacher is a slice.**
  - The teacher sees only 2–8 positions, not the turn. Its distribution is narrower and carries slice artefacts: refusal without the "told" signal, swaps from shared order tags, the hybrid's recognition-without-recall on Qwen3.5-2B. A text teacher has none of these. So the slice teacher is *worse* than a text teacher as a source, and the comparison we must run is **slice teacher vs text teacher (the original turn in context)** on the same student and data. If the text teacher wins clearly, keeping slices for distillation has no advantage over keeping text.
  - The teacher's answers are well defined only where the slice works. Distil only on questions where the teacher answers correctly (filter by a judge or by gold), the analogue of SEAL's reward filter.
  - The slice's strength (0.86 use vs 0.27 for the same tokens as text) suggests the slice teacher might give *sharper* answer distributions than a weak text cue. That is the hypothesis worth testing: does the student inherit the slice's advantage over text?
- **A middle arm Cartridges suggests.** Train the slice itself (a cartridge initialised from our 2–8 positions, model frozen, same KL), instead of a LoRA. It keeps the memory as a discrete, deletable, composable object and avoids LoRA's interference (Cartridges App. A.1). Its cost is a few thousand trainable values per memory, not millions.

### 3.4 Per-user LoRA cost and composability

- **Size (Qwen2.5-1.5B: 28 layers, hidden 1,536, MLP 8,960, 2 KV heads × 128).** LoRA on all seven projections costs r × 41,216 parameters per layer → r × 1.15M in total. Rank 8: 9.2M parameters, ~18 MB in bf16. Rank 64 (SEAL/SLEEP setting): 74M, ~148 MB. q,v only at rank 8 (SSR setting): 1.1M, ~2.2 MB. A 2-position slice: 2 × 2 × 256 × 28 = 28,672 values, ~57 KB. So a rank-8 all-projection LoRA ≈ 320× a slice; q,v rank 8 ≈ 38×. My arithmetic from the public config, not from a paper.
- **Training time (estimates from the papers, not measured).** SEAL: 30–45 s per LoRA fit + evaluation for one passage on a 7B (H100). On 1.5B with a few hundred dreamed QA × top-k teacher logits, expect minutes per user per sleep on one GPU, plus one teacher forward pass per training sequence with the slice restored. Cartridges at corpus scale: ~30 min on 8×H100. SLEEP-style RL terms multiply this by ~4.
- **Serving.** Many users' LoRAs need multi-adapter serving (S-LoRA/Punica-type kernels; Cartridges §3.3). Slices and cartridges use the existing prefix cache.
- **Composability.**
  - *Several memories of one user:* train jointly in one adapter (MEMIT batch result; SEAL CPT aggregates 5 self-edits per passage). Do not train one LoRA per memory and sum them: no paper here tests LoRA summation for facts; SEAL's hypernetwork adapters (one per passage) fall from 66.8 to 28.0 when 200 passages must coexist (App. B.8).
  - *Several users:* adapters stay separate (one per user); merging users' adapters is never wanted.
  - *Adapter + slices at once:* untested anywhere. Needed if new memories arrive between sleeps. Watch for conflicts when a slice updates a fact the adapter holds (MEMIT excludes conflicts; nothing handles "later wins" in weights).
  - *KV objects compose:* Cartridges concatenates independently trained cartridges with no retraining (§5.4). Per-memory trained slices keep this property; LoRAs do not have a tested equivalent here.
- **Deletion.** Whole user: delete the adapter, exact. One memory: retrain the adapter without it, which needs the other memories' sources, so **do not delete the slices** after consolidation (keep them cold, as the evidence; "Useful Memories": raw episodes as first-class evidence). Approximate unlearning (gradient ascent, negative KL) is not covered by this cluster and is known to be leaky; avoid relying on it. For a "deletable" claim, test that after retraining without memory m, the adapter's answers on m match the frozen model's.

### 3.5 Generalisation to paraphrases and other uses (reversal curse and similar)

- **Within this cluster.**
  - Paraphrase of the same question works once training data is diverse: ROME paraphrase 96.4–99.1; MEMIT 88.6 at 10k; SMF trains on N paraphrases per fact and answers the original question.
  - Other uses are weak or untested: Cartridges' next-token cartridges "cannot generalize" from memorisation to synthesis, structuring or reasoning queries; self-study with diverse seeds is needed (Fig. 3). SEAL's chain-of-thought at evaluation did not help (37.8).
  - Reverse direction: not learned (ROME §3.7, MEMIT §6).
  - Specifics in free use: not produced (EVAF R2).
- **Outside the library (from memory, not checked; verify before citing).**
  - *The Reversal Curse* (Berglund et al. 2023, arXiv 2309.12288): models fine-tuned on "A is B" do not answer "B is ?" at above-chance rates.
  - *Physics of Language Models 3.1/3.2* (Allen-Zhu & Li 2023, arXiv 2309.14316, 2309.14402): facts seen in one phrasing are memorised but not extractable for QA unless the training data contains paraphrase/permutation augmentation; inverse search and comparisons fail without chain-of-thought.
  - Context-distillation papers (Snell 2022, Kujanpää 2024) report gains in-distribution; whether distilled facts support multi-hop use is, as far as I know, untested.
- **Implications for the dreamed-question set.** Include: both directions ("my colour → teal" and "is teal mine? / whose favourite is teal?"), both perspectives ("my" / "your"), indirect uses ("suggest a shirt"), yes/no premises (true and false), composition with a second memory ("my sister and I both like…"), and negatives (facts never told). Hold out a type of question from training to measure generalisation rather than coverage. Sleep-time compute: unpredictable questions benefit least, so expect the held-out types to lag.

---

## 4. What is still open / ours

- **Distilling from a restored KV slice.** Not found. Every teacher in the literature is text in context. Whether a 2–8-position slice is a usable teacher, and whether the student inherits the slice's advantage over text (0.86 vs 0.27), is ours to test, with a text-teacher control.
- **Whether our measured failure modes survive consolidation.** No paper measures partner swaps between separately learned personal facts, a licence gap (TF vs generation, refusal), or ownership (user's fact vs model's own vs world knowledge) in weights. ROME/MEMIT's neighbourhood score and EVAF's discriminating-token test are the closest probes; we can use our existing items and judges directly.
- **Per-memory deletion from an adapter.** Untested everywhere here. Ours: retrain-without-m and check the answers on m return to the frozen model's.
- **Slice vs LoRA vs trained slice on the same facts.** A three-way comparison on use (in generation), swaps, refusal, interference (generic-prompt KL plus unprompted-mention probes), deletion and cost is not in the literature. Cartridges did prefix vs LoRA for long documents, not for single personal facts.
- **Teacher artefacts as a design variable.** Filtering refusals, using jointly re-encoded slices, or adding the "told" signal before distillation: what the teacher should look like is specific to our substrate.
- **Hybrid models.** Distilling a Qwen3.5-2B teacher whose memory includes the DeltaNet state (recognition without recall) has no precedent.
- **Known, so not claimable:** consolidation of context into weights by distillation (Cartridges, Kujanpää, Snell, SDFT, OPSD line); self-generated training data (SEAL, SLEEP, SSR); KL-to-base anchors (ROME, GKD); joint batch insertion scaling better than sequential (MEMIT); LoRA forgetting more than sparse or prefix parameterisations (SMF, Cartridges); masking-based memory utility (Auto-Dreamer); "keep raw episodes, gate consolidation" (Useful Memories).
