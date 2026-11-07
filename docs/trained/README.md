# The trained side: what the literature says (synthesis)

*Written 2026-11-07. This page summarises four literature notes, about 60 papers in all. Agents read the papers closely and wrote the notes. This page is the cross-cutting picture. The numbers and quotes are in the notes:*
- [lit-A-trained-kv.md](lit-A-trained-kv.md): trained KV and context memories (Cartridges, Attention Matching, KVLink, Memory3, KBLaM, Compression-Aware Abstention, …)
- [lit-B-read-paths.md](lit-B-read-paths.md): trained read paths and gates beside a frozen model (FwPKM, Larimar, TransMem, Pepper, ReFT, S0, Ferrando, …)
- [lit-C-weight-consolidation.md](lit-C-weight-consolidation.md): consolidation into weights (SLEEP, SEAL, Sparse Memory FT, ROME/MEMIT, EVAF, …), plus design notes for slices → weights
- [lit-D-new-search.md](lit-D-new-search.md): new papers, filed in `papers/trained-memory/` (TRIM-KV, Gating Mechanism, Knowledge Modules, Doc-to-LoRA, Rethinking LoRA Memory, Trust-Align, Sufficient Context, Substrate Asymmetry, …)

**Framing.** A frozen backbone. Per-user memory is kept as data (slices, keys, state), and small parts are trained once. An optional per-user consolidation step can come later.

---

## Where each candidate stands

| # | Candidate | Closest prior work | Status | What stays ours |
|---|---|---|---|---|
| 1 | Amortised S-loo selector | TRIM-KV (a learned query-free retention gate, frozen backbone, LongMemEval); Gating Mechanism (why gates trained on the LM loss collapse, and the fix); ForesightKV; Attention Matching (the closed-form baseline to beat) | **Recipe known, target open** | A leave-one-out / answer-utility target; 2–8 shared positions; reuse in another session |
| 2 | Licence gate or adapter | Compression-Aware Abstention (answer/abstain LoRA over an evicted cache); Trust-Align (DPO answer > refuse); TransMem (frozen backbone, evidence-only KL teacher); ReFT and Pepper (tiny adapters); Ferrando (mechanism) | **Partly known for text and evicted evidence** | Restored cross-session slices; content measured apart from licence (teacher-forced vs generation, refusal mass); wrong-slice negatives |
| 3 | Learned key plus verification | KBLaM (learned key, refusal for irrelevant questions); M+ (co-trained low-dimensional retriever); WISE and MEMOIR ("is this mine?" routers); Larimar's scope detector (an external-embedding 1-NN beats a trained classifier) | **Keys partly known; verification open** | Abstaining on a relevant-looking wrong slice; the hybrid state as a lure-robust verifier (balanced 0.96) |
| 4 | Consolidation (slices → weights) | Cartridges (context distillation into a **trained KV prefix**); Knowledge Modules/DCD, Doc-to-LoRA, PRAG, DPPM/RPMem (per-document or per-user LoRA, always with a **text** teacher); Rethinking LoRA Memory | **Done with text teachers** | A KV-slice teacher; whether swaps, licence and ownership carry over; a three-way slice vs trained slice vs LoRA comparison; deleting one memory |
| 5 | *(new)* Trained binding / slot signal | KVLink (link tokens indexed by slot number, but the whole model is fine-tuned) | **Open on a frozen backbone** | Compare with our storage-order fix (clean at N = 3, half the swaps at N = 5) |

## Seven cross-cutting lessons

1. **The licence gap is ours, and probably about attention.**
   - Text RAG with sufficient context abstains only 1.6–11% of the time (Sufficient Context). Our gold slices are refused 0.70–0.79 of the time, so this is specific to the slice format.
   - Three results point at attention, not missing content:
     - Ferrando: an entity feature at the subject decides whether the extraction heads attend to it, and it switches refusal.
     - Attention Matching: without a mass bias β, compacted caches "systematically underestimate" a block's contribution, and fail with "the text does not mention…".
     - KBLaM: early-layer memory tokens act as "instructions"; without them the model never refuses.
   - Cheapest test, no training needed: a β / mass gain on restored slices, and restoring only at some layers. Then measure refusal and judged correctness.
2. **Every trained licence piece needs negatives, or it learns "always answer".**
   - Pepper's adapter names a topic for a zero vector.
   - TransMem dropped LoCoMo's unanswerable questions.
   - TokenMem trains compliance with counterfactual passages (69–84%).
   - A per-user LoRA abstains on facts it was never given only about 9% of the time (Substrate Asymmetry).
   - Going too far the other way also fails: refusal training cuts the use of evidence that is present (Do RALMs Know: 0.738 → 0.591).
   - So use a two-term or DPO loss with three kinds of negative: no slice, wrong slice, lure slice. Measure resilience as well as accuracy.
3. **Distil against a teacher's distribution; don't use cross-entropy on answers.**
   - Cartridges: +8.6 chrF. TransMem: 54.1 vs 48.8 F1. xRAG: cross-entropy "tends to cause an overreliance on context". Doc-to-LoRA: 0.819 vs 0.763.
   - Gating Mechanism: a learned gate trained on the LM loss collapses to a no-op (RULER 4.33 vs 70.01). KL to the dense model with a fixed gate mass rescues it.
4. **The student inherits the teacher's habits.**
   - SLEEP: the student "weakly mimics" the teacher.
   - Knowledge Modules: the student reaches 70–85% of the teacher.
   - A gold-slice teacher refuses 0.70–0.79 of the time and carries partner swaps. So filter its answers, use jointly encoded slices, or use a full-turn text teacher. Compare slice teacher vs text teacher directly; that comparison is ours.
5. **Prefer a trained KV object or a tiny residual intervention to LoRA, for both interference and few-shot stability.**
   - Cartridges: at the same size, MMLU falls 60.0 → 54.3 with a trained prefix and 60.0 → 45.3 with LoRA.
   - S0: with about 48 examples, LoRA rank 64 gives −15.5 pp, while a learned GDN state gives +19 pp on the 2B.
   - ReFT: a rank-1 intervention holds 256 random pairs; 32K parameters learn a refusal persona from 5 examples.
   - User as Engram: a per-user LoRA costs +1.78 bits per byte on unrelated text.
   - Doc-to-LoRA: an always-on adapter drags unrelated questions below the base model.
   - The clean exception: Rethinking LoRA Memory finds a decode-time LoRA helps only when little KV is left. Our 2–8 positions are exactly that extreme.
6. **Weights keep gist and lose specifics; they bind one way round and keep the last write.**
   - EVAF: zero persona-discriminating tokens in free generation.
   - Auto-Dreamer: concrete locations become generic procedures.
   - ROME/MEMIT: edits are directional.
   - Engram: users who share a template collide, and the last write wins.
   - MEMIT: batched (joint) updates scale; sequential ones collapse from n = 32.
   - The weight-side twin of our "joint encoding fixes swaps" is to retrain one adapter per user over all their memories.
7. **A trained gate ignores memory when there is another route to the fact** (FwPKM's gate near 0 under full attention; Lee et al.'s layers underused with a sliding window; Memory3's memories useless without a warm-up). Train any read piece with the slice as the only source.

## Corrections to earlier internal notes

- "Writes can be untrained; reads need training" (core-idea P5) is weaker than it reads. Every trained memory in the literature trains the write *format*. Our native slices are read untrained (0.86 clean use). What we lack is the **licence**, not the format.
- Larimar's decoder is fine-tuned jointly, not frozen. The quote "not trained to perform memory-conditioned generation" is about its baseline, not Larimar.
- `papers/memory-architectures/MemoryLLM.pdf` is Jaiswal et al. (Apple), not the memory-pool MemoryLLM. The index is fixed.

## Not read yet, worth reading next

- RPMem (2609.23466): sessions → latent memory → recurrent gate → LoRA. The most relevant per-user consolidation paper we have not read.
- ForesightKV (2602.03203): a ranker distilled from an oracle eviction score.
- "Learned Structure in Cartridges", keys as routers (2508.17032).
- δ-mem (2605.12357): a tiny delta-rule state giving low-rank attention corrections on a frozen backbone.
- The Reversal Curse (2309.12288) and Physics of LMs 3.1/3.2, cited from memory in lit-C; verify before citing.
