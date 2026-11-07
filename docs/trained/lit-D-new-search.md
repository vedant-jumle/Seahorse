# Literature D: new search for the trained side (2026-11-07)

Scope: papers *not* already in `papers/`, on trained parts around a frozen backbone. Twenty PDFs went into `papers/trained-memory/`. The index is in `papers/README.md` under `## trained-memory`. Every arXiv ID below was checked against its abstract page. Nothing here comes only from search snippets except where marked *(unverified)*.

Our reference numbers:
- **Slices:** 2 positions give 0.86 clean use; the same tokens as text give 0.27.
- **Licence gap:** gold slices score 0.99 teacher-forced, but 0.17–0.49 judged in generation, with refusal at 0.70–0.79.
- **Retrieval:** subject-word keys are right 0.85–0.92 of the time.
- **Selection:** S-loo is the best selector; attention-based selectors pick punctuation.

---

## 1. Short version

**What matters most**

1. **Learned, query-free KV retention already exists, and it has a known failure mode.**
   - **TRIM-KV** (2512.03324, ICLR 2026)
     - Scores every token once, when it is written, with a small per-head gate. The backbone is frozen.
     - Trained by KL to the full-cache model plus a capacity penalty.
     - On LongMemEval_S (Qwen3-4B) it keeps 44.8 at a 32K budget, against 27.6 for SnapKV and 49.4 for the full cache.
     - Some heads keep only full stops.
   - **The Gating Mechanism of KV Cache Eviction** (2611.07605, two days old)
     - Shows that a learned eviction gate trained on the LM loss collapses to a constant: RULER 4.33 vs 70.01 dense.
     - The fixes that work: a fixed gate mass, KL to the dense model, and per-head budgets set by attention mass.
   - Neither works at "2 positions per fact for reuse in another session". Neither trains on a leave-one-out target. **Candidate 1 is partly known: the recipe exists, our target does not.**

2. **Distilling a document into a LoRA is well trodden, but only from text teachers.**
   - The approaches so far:
     - per-item gradient training: PRAG, Knowledge Modules/DCD
     - amortised hypernetworks: DyPRAG, Doc-to-LoRA, SHINE
     - per-user versions: DPPM and RPMem; DPPM reaches 54.2 on PersonaMem-v2, against 33.0 for the full context
   - **Rethinking LoRA Memory** (2606.05698, Zuo et al.) is the key result for us:
     - A document LoRA adds about nothing while the KV cache is intact.
     - It adds +13–21 ROUGE-L when no KV is left.
     - It works best as a *decode-time* adapter on top of KV built by the base model.
     - That is "slices plus a consolidation LoRA" in all but name. Nobody uses a *KV-slice* teacher.

3. **Consolidation without negatives makes up facts.**
   - **Substrate Asymmetry** (2606.11712): a per-user LoRA almost never abstains on facts the user never gave. Absence TPR is about 9%, against 99% for RAG. Its numbers are internally inconsistent, but the direction is clear.
   - **Doc-to-LoRA** Table 8: an always-on adapter drags unrelated questions below the base model, 0.096 vs 0.201.
   - **User as Engram** (2606.19172): a per-user LoRA raises loss on unrelated text (+1.78 bits per byte). Users whose facts share a template collide, and the last write wins, much like our storage-order swaps.

4. **Fixing over-refusal by training has precedent, and so does making it worse.**
   - **Trust-Align** (2409.11242): DPO with "answer preferred over refusal" on answerable cases lifts LLaMA-3-8B's answerable recall from 2.13 to 68.20. The model had been refusing almost everything under a refusal prompt, which is our regime.
   - **Do RALMs know…** (2509.01476): refusal training *lowers* use of evidence that is present. Context utilisation drops from 0.738 to 0.591 for ICFT(n).
   - **Sufficient Context** (2411.06037): small "I don't know" fine-tuning almost removed abstention and raised hallucination (Musique: 59.4 → 74.8% hallucination).
   - None of them works on KV evidence. Our licence gap, where the model holds the content but refuses, is not reported anywhere. Plain RAG abstention when context is sufficient is only 1.6–11% (Sufficient Context, Table 4a), so a refusal rate of 0.70–0.79 is specific to the slice format.

5. **Learned keys over activations exist; verification on top of them does not.**
   - **M+** (2502.00592): a co-trained low-dimensional key/query retriever over latent memory, one lookup per layer. It recovers about 30% of ground-truth tokens against 3% for random. Labels mean "same document", and it never abstains.
   - **WISE** and **MEMOIR** have activation-based "is this mine?" detectors for edits:
     - WISE uses an activation margin. Dropping the router takes locality from 1.00 to 0.72 at 1000 edits.
     - MEMOIR uses sparse-mask overlap.
     - These are the closest thing to verification.
   - **Storage-Retrieval Gap** (2608.25489): retrieving per-entity LoRAs by similarity is at chance.

**Candidate map**

| # | Candidate | Closest papers | Verdict |
|---|---|---|---|
| 1 | Amortised S-loo selector | TRIM-KV; Gating Mechanism (IndexMem++); ForesightKV; Locret; DMS; AttriBoT (fast LOO, not learned); in library: LORE-KV, KVzip | **Partly known.** Write-time, frozen-backbone learned scorers trained by KL to the dense model work and are tested on LongMemEval. Untested: a LOO/answer-utility target, budgets of 2 positions per fact, and scoring for transplant into a new session. |
| 2 | Licence-aware read gate or adapter | Trust-Align; Do RALMs know (ΔU gate); Sufficient Context (selective generation); KM's Knowledge Extractor; Engram's shared reasoning LoRA; SPARK; in library: Compression-Aware Abstention | **Partly known.** Recipes exist for text evidence: DPO answer-vs-refuse pairs, and uncertainty with vs without context. Two warnings: refusal training can cut use of present evidence, and IDK fine-tuning on small LoRAs is unstable. **Open:** gating on KV-slice evidence, and the "content present but refused" case. |
| 3 | Learned retrieval key with verification or abstention | M+ co-trained retriever; WISE router; MEMOIR mask overlap; Storage-Retrieval Gap; Cartridges keys as routers (not read); in library: KBLaM, MemArt | **Key: partly known. Verification and abstention: open.** No paper scores "this slice answers this question" and abstains on a near-miss. M+ labels are weak ("same document"). |
| 4 | Slices consolidated into a per-user LoRA by distillation | KM/DCD; Doc-to-LoRA; PRAG/DyPRAG; DPPM/RPMem; Rethinking LoRA Memory; DiSC; Self-Study Reconsidered; Substrate Asymmetry; User as Engram | **Done already for text teachers.** Per-document and per-user, amortised and not, on PersonaMem. **Open:** a KV-slice teacher, and how to avoid passing on the teacher's refusals. Expect 70–85% of the teacher, no abstention on absent facts unless negatives are added, and damage to unrelated text. |

---

## 2. Per-paper notes

The papers marked **[full]** were read through method, results and the appendices that carry results. **[skim]** means the abstract plus the main results table only. Section 4 says who read what.

### Candidate 1: learned selection and eviction

**TRIM-KV: "Cache What Lasts" (Bui et al., ICLR 2026, 2512.03324)** [full]
- **Method:** a retention gate per layer (MLP d→512→number of KV heads) gives each token one score β per head when it is created.
  - During training, attention logits get a bias (t−i)·log β_i, so a token's influence decays with age. This is soft eviction.
  - At inference, when the cache passes its budget M, the token with the lowest β^(t−i) is evicted.
- **Frozen vs trained:** the backbone is frozen; only the gates are trained. The parameter count is not stated; the reader's estimate is about 47M for Qwen3-4B.
- **Loss:** forward KL to the frozen model, plus next-token prediction, plus a capacity hinge (Eqs. 4–6).
  - Ablation at AIME24, budget 4096 (Table 5): without the capacity loss the score falls from 75.8 to 42.9. Removing KL or next-token prediction costs about 3 points each.
- **Cost:** math gates were trained on 564M tokens on 4×H100. LongMemEval gates used 10k examples of 32K–128K tokens at M=1024. GPU hours are not stated.
- **LongMemEval_S, Qwen3-4B-Instruct, query-agnostic** (the dialogue is compressed before the question; Table 8):

  | KV budget | Full cache | TRIM-KV | SnapKV |
  |---|---|---|---|
  | 131,072 (full) | 49.4 | — | — |
  | 32,768 | — | 44.8 | 27.6 |
  | 16,384 | — | 38.6 | 18.2 |
  | 8,192 | — | 34.0 | 15.8 |
  | 4,096 | — | 26.8 | 13.8 |

  - Knowledge-update at 32K: 69.2, above the full cache's 68.0.
  - The judge is Qwen3-4B itself.
  - Tables 3 and 8 swap the SnapKV and StreamingLLM numbers.
- **What it keeps** (Sec. 5.1.2, Fig. 5, Fig. 18): on average, content-word pieces and the `<|im_start|>` sink. Whitespace and punctuation get low scores. Individual heads specialise: some keep only full stops ("gist" heads), others numbers or a recency window. This is one example, examined by eye.
- **No** refusal results and **no** interference results.
- **For us:**
  - It is the closest existing learned query-free scorer, and it uses the KL + capacity recipe.
  - Its period-keeping heads match our "value + full stop" finding. Attention selectors may be right per head but wrong once averaged across heads.
  - Where it falls short as a stand-in for S-loo:
    - Scores are per head, while we pick shared positions across all layers.
    - Exponential decay builds in a preference for recent tokens.
    - The target is continuing the same sequence, not answering in a new session.
  - Use the released gates as a cheap baseline against S-loo.

**The Gating Mechanism of KV Cache Eviction (Gu et al., 5 Nov 2026, 2611.07605)** [full main text + App. E–G, J]
- **Method:** a small MLP reads h_j and outputs one score per KV head, *added inside the softmax*.
  - Failure: trained on the LM loss, the gate learns a near-constant score. A constant cancels in the softmax and gives back the frozen model exactly, so the frozen model's loss prefers this no-op. RULER pilot: 4.33 vs 70.01 dense (Table 1).
  - Fixes, step by step (pilot score after each):
    - always keep sinks and a local window, and make all other entries share a fixed gate mass β (256 of about 8064 entries): 44.86
    - replace the LM loss with reverse KL to the dense model, because the LM loss charges the gate for errors the frozen model already makes: 58.86
    - set per-head budgets by retained attention mass, because raw scores cannot be compared across heads: 67.60
  - Evicted entries are folded into centroid K/V slots *inside* the same softmax. These match the evicted entries' contribution at cosine 0.85–0.96; learned memories read outside the softmax reach only −0.09 to 0.44.
- **Size and training:** the gate is 7.5M parameters on Qwen3-0.6B. Bigger scorers, or scorers that see the past, change results by under ±1.2 points. Unfreezing the backbone hurts: 67.60 → 62.83 (Table 3). Training is 300 + 300 steps; GPU hours for 8B are not stated.
- **Results:** RULER 8K at 75% compression on Qwen3-8B: 90.14 vs full 93.86, TrimKV 80.26, SnapKV 35.13 (Table 5). Multi-key retrieval MK3: 73.0 vs TrimKV 4.4 (Table 6).
- **What it keeps** (App. J): all 39 facts used later in SWE traces are kept, but so are all 111 facts that are never used. It is not selective about need. There is no breakdown by token type.
- **For us:** this is the training recipe for an amortised S-loo.
  - Fix the kept mass.
  - Distil from the full-context (or LOO) model, not from the LM loss.
  - Expect scores to be comparable only within a head.
  - It also suggests reading memory as K/V inside attention, not through an adapter.
  - Weakness: the pilots use 0.6B with 2 seeds.

### Candidate 4: content into weights

**Knowledge Modules / Deep Context Distillation (Caccia et al., COLM 2025, 2503.08727)** [full]
- **Method:** one LoRA per document (r=16, every linear layer, 1500 steps).
  - The teacher is the same model with a 2048-token chunk in context; the student sees no document.
  - Loss: KL on logits plus normalised L1 on every layer's hidden states.
  - Distillation inputs are the next chunk (DDCD) or synthetic summaries and QA (SDCD; 8–16 summaries per chunk).
  - An optional Knowledge Extractor (a second LoRA trained across many modules) is mixed in with learned weights per layer.
- **Llama-3.1-8B, Table 1** (NarrativeQA in ROUGE-L, QuALITY in accuracy):

  | Setting | NarrativeQA | QuALITY |
  |---|---|---|
  | Base, closed book | 12.5 | 27.8 |
  | KM trained by next-token prediction | 17.9 | 27.0 |
  | KM, DDCD | 24.5 | 31.1 |
  | KM, SDCD | 27.3 | 39.0 |
  | Document in context | 38.0 | 46.0 |
  | RAG + KM + Knowledge Extractor | 40.8 | 66.3 |

- **Ablations:** logits plus hidden states 23.1, logits only 21.7, hidden states only 18.8 (Table 4). Sharpening the teacher towards one-hot slides results back to the next-token baseline (Table 6).
- **Unrelated questions:** a random module costs MMLU 1 point (63.9 → 62.9, n=400).
- **Not tested:** refusal, composing several modules, colliding facts. Routing is assumed to be correct (oracle).
- **For us:** the per-item recipe for candidate 4, with hidden-state matching for per-layer signal.
  - **Warning:** the student inherits the teacher's behaviour. A slice-in-cache teacher that refuses 70–79% of the time will put refusal into the LoRA. Distil from teacher-forced answer distributions, or from a text-context teacher.
  - The Knowledge Extractor is a trained read adapter that works across modules it never saw, which is relevant to candidate 2.

**Doc-to-LoRA (Charakorn et al., Sakana, ICML 2026, 2602.15902)** [full]
- **Method:** a Perceiver-style hypernetwork (309M parameters) reads the frozen model's per-layer activations of a document and outputs a rank-8 LoRA on `down_proj`.
  - It is meta-trained by KL to the model with the document in context. Training data is 3.2M contexts, with queries written by gemma-3-12b and top-16 logits from the base model.
  - Long documents get one adapter per 8K chunk, concatenated along the rank dimension.
- **Cost:** meta-training is about 5 days on 8×H200. Writing one adapter takes under 1 s; per-document context distillation takes 40–631 s.
- **Results** (gemma-2-2b, as a fraction of in-context performance):
  - SQuAD 0.825, DROP 0.655, ROPES 0.906.
  - 2WikiMultihopQA: D2L 0.857, CD on the real test question 0.901, CD on 25 generated queries 0.745.
  - Long documents (MultiFieldQA, QASPER): only 0.48–0.54.
  - KL beats next-token prediction (0.819 vs 0.763); rank 16 beats rank 8 (0.896 vs 0.814).
  - Needle-in-a-haystack is near-perfect up to 40K tokens.
- **Table 8 warning:** when the internalised document is unrelated to the question, D2L scores 0.096 / 0.126 vs base 0.201 / 0.175. The adapter assumes every question is about its document.
- **App. D:** the same hypernetwork can output a 20-position KV prefix instead of a LoRA. Keys generated *pre-norm and pre-RoPE* hold up much longer than keys output directly.
- **Not tested:** reversal, paraphrase, composing users.
- **For us:**
  - Amortised candidate 4 exists, but the meta-training cost is high for a side project.
  - Add irrelevant queries to the distillation data.
  - App. D supports storing slice keys before RoPE.

**PRAG: Parametric RAG (Su et al. 2025, 2501.15915)** [skim: method, Table 1, cost]
- One LoRA per document: r=2 on the FFN, about 4.7 MB at 16 bit. Trained on document rewrites plus generated QA pairs. At query time the LoRAs of the top-3 retrieved documents are merged.
- **Qwen2.5-1.5B, Table 1** (F1):

  | Benchmark | P-RAG | Standard RAG |
  |---|---|---|
  | 2WQA total | 0.3025 | 0.2431 |
  | HotpotQA total | 0.2165 | 0.2073 |
  | PopQA | 0.1885 | 0.0999 |

- On LLaMA-8B, combining P-RAG with in-context RAG is best: 2WQA 0.4258.
- No refusal or interference results; the merged LoRAs are not studied for collisions.

**DyPRAG (Tan et al. 2025, 2503.23895)** [skim: Sec. 3–4, Table 1]
- A small hypernetwork ("parameter translator", 4.04M parameters, 7.71 MB) maps a document embedding to PRAG's LoRA. This replaces 18.66 GB of per-document LoRAs on Qwen2.5-1.5B. It is trained on 4,800 (document, PRAG-LoRA) alignment pairs.
- **Qwen2.5-1.5B average F1** (Table 1):

  | Method | Avg F1 |
  |---|---|
  | Vanilla | 25.79 |
  | Standard RAG | 23.17 |
  | PRAG | 27.14 |
  | DyPRAG | 25.56 |
  | DyPRAG-Combine | 27.60 |

  At this scale the gains are within noise of vanilla.
- **LLaMA3-8B:** DyPRAG-Combine 43.69 vs PRAG 41.00.
- Claims fewer RAG hallucinations on RAGTruth (not read).

**Rethinking LoRA Memory Through the Lens of KV Cache Compression (Zuo et al., June 2026, 2606.05698)** [full]
- **Setup:**
  - Per-document DoRA (r=16, MLP only), trained on 4 self-generated QA pairs per 64-word chunk, with loss on answer tokens only.
  - The document's KV is then compressed with Compactor, a query-agnostic selector (attention score + leverage score), at retention ρ from 0 to 100%.
  - Models: Qwen3-4B and Llama-3.1-8B. Data: NarrativeQA and LongHealth, 20 documents each, scored by ROUGE-L only.
- **Results** (figures only):
  - The LoRA gains about 0 when the KV is mostly intact, and is even negative on NarrativeQA.
  - With no context left it gains about 13 (NarrativeQA) and 15–21 (LongHealth).
  - At 99% compression it recovers about half of that.
  - **Best placement:** build the KV with the base model and turn the LoRA on *only for decoding*: +7–11 and +13–17 at 99–99.5% compression (Fig. 3).
  - **Training data** (Fig. 4): QA-only 0.19–0.26; context+QA 0.11–0.13 (the adapter learns to read the prompt); raw next-token prediction about base.
  - MLP-only LoRA is about equal to all-linear.
- **Not tested:** interference (the right adapter is always supplied), refusal.
- **For us:** strong support for a hybrid where the base model builds the slices and a consolidation LoRA works at decode time. That is the regime where a LoRA helps, since 2 positions is extreme compression.
  - The student must not see the slice during distillation.
  - Compactor is a query-agnostic selector to test against S-loo.
  - Weak evidence base: n=20 documents, ROUGE-L only.

**DiSC (Padmanabhan, Gul, Goyal, Feb 2026, 2602.16093)** [skim: Tables 2–3]
- Context distillation with split contexts: teacher and student see different segments of one example, and the loss is KL on the shared tokens. No generation step is needed.
- **Qwen2.5-7B on the KUP update task** (Table 3):

  | Method | KUP | IFEval | Math |
  |---|---|---|---|
  | Before update (Mpost) | 15.46 | 80.70 | 49.85 |
  | Plain fine-tuning | 40.90 | 65.83 | 24.47 |
  | DiSC | 44.18 | 77.70 | 44.34 |

- **For us:** if per-user consolidation ever touches a shared adapter, distillation-style losses keep instruction-following. That matters, because the licence gap is partly instruction behaviour.

**Self-Study Reconsidered (Alimaskina et al., June 2026, 2606.32002)** [skim: Sec. 3, Fig. 1–2]
- **Coverage:** self-generated QA covers 75–94% of a document, but saturates early and keeps returning to the same salient spans.
  - Pairwise Jaccard overlap between prompt seeds is 0.55–0.68 on Cartridges and 0.39–0.46 on longer corpora.
  - Only point-lookup prompts select evidence reliably.
- **Injection:** the answering model follows instruction-like spans in the text. Filtering them cuts compliance from 88% to 13%.
- **For us:** our dreamed questions, which feed S-loo and would feed distillation, are a biased sampler. Tie each question to a fixed target span; this is what the paper recommends. Check for prompt injection in user sessions before distilling.

**User as Engram (Li, June 2026, 2606.19172)** [full; single author]
- **Method:** small self-pretrained Engram models (178M–1.22B) with a hash-addressed N-gram memory table read at late layers.
  - A user's fact is written by overwriting about 16 rows its trigger hashes to. Values come from a closed-form solve, a few gradient steps, or a joint optimisation over all of one user's rows.
  - One shared rank-16 LoRA (11.8 MB) carries the reasoning skill.
- **Storage:** about 88 KB per user at 100 facts (inconsistent elsewhere: 60 KB; "per row"), vs 14.2 MB for a rank-64 per-user LoRA.
- **Damage to unrelated text:** per-user LoRA +1.784 bits per byte (17 of 20 users worse) vs Engram +0.00005.
- **Recall:** direct recall is 100% vs 99% for per-user LoRA. Indirect recall is 44% vs 6%.
  - This depends on a weak base. On instruction-tuned bases, per-user LoRA *improves* indirect recall by +0.09 to +0.22.
- **Capacity:** top-5 recall stays above 90% up to about 300 facts.
- **Paraphrase:** 50% top-1 when the fact is written once. That is measured on 4 facts × 5 paraphrases; the multi-trigger results leak the test wording.
- **Chaining two facts:** 12.9% when real chaining is needed.
- **Interference:**
  - Fact sets with different trigger templates stack (66 → 63 → 64%).
  - Users sharing templates collide, and the earlier one falls to 8% ("last write wins"). This mirrors our storage-order swaps.
  - Joint optimisation of a user's facts reduces within-user interference, mirroring our joint-encoding fix.
- **For us:**
  - The layered design (external per-user content + one shared trained reader) is the architecture we are considering.
  - Its evaluation habit is worth copying: report bits per byte on unrelated text with memory loaded.
  - The method itself needs an Engram-pretrained base, so it does not port to Qwen.

**Substrate Asymmetry in User-Side Memory (Deng, June 2026, 2606.11712)** [full; single author]
- **Setup:** 50 synthetic personas, about 12 QA pairs each. One LoRA per user on q/k/v/o (r=128, 20 epochs, at most 10 minutes per user on an L40S), against BGE top-5 RAG. Qwen3-4B, replicated on Llama-3.1-8B.
- **Presence and absence true-positive rates** (App. B.2, Qwen3-4B):

  | Config | Presence TPR | Absence TPR |
  |---|---|---|
  | RAG | 35.3% | 99.0% |
  | Per-user LoRA | 56.3% | 8.7% |
  | Per-user LoRA + "say no if not relevant" clause | 42.0% | 53.3% |

  - On Llama the per-user LoRA's absence TPR is 3.0%.
  - Zeroing the LoRA's q_proj at layers 21–35 raises absence from 40.3 to 73.3 and drops presence from 42.0 to 22.3.
- **Numbers disagree inside the paper** (Sec. 4.2 vs App. B.2). The router's win likely comes from question wording, since absence probes are phrased differently.
- **For us:** consolidation trained on positives only has no "I was not told this" state. Add negative and absent-fact examples to the distillation, or keep slices and keys as the absence detector. When we evaluate this ourselves, phrase present and absent probes the same way.

**DPPM: Dual-Path Parametric Memory (Chen et al., Oct 2026, 2610.11776)** [skim: Tables 1–3, App. B]
- **Method:** the interaction history is fed through the *frozen* released D2L compiler. Two small trained paths sit on top:
  - an evidence path that pools segment representations
  - a delta path, a gated recurrent associative state with a learned read key
  - together they give a history-conditioned LoRA
- **Size and training:** about 527k trainable parameters; per-dataset cross-entropy training for 5 epochs; 5 seeds.
- **Results** (Table 1, Qwen3-4B, accuracy %):

  | Method | PersonaMem-v2 | PrefEval |
  |---|---|---|
  | DPPM | 54.22 | 86.79 |
  | RPMem | 45.73 | 82.47 |
  | Full context | 32.96 | 45.31 |
  | Mem0 | 30.46 | 75.49 |
  | Direct D2L | 24.86 | 30.06 |
  | Gold annotated state | 61.08 | 93.52 |

  - A parameter-matched RPMem gets 45.49 (Table 3).
  - The "Others" category (not attributing another person's info to the user) rises from 13.68% to 50.23%.
- **For us:** per-user consolidation into a LoRA is done on PersonaMem. The interesting part is that a *raw* D2L adapter is worse than no context; it needs a trained integration layer. LongMemEval and LoCoMo are cited but not run.

**Memory Decoder (Cao et al. 2025, 2508.09874)** [skim: abstract]
- A small decoder is pretrained to imitate a kNN-LM retriever's next-token distribution over a domain corpus, then interpolated with any model that shares the tokenizer. Perplexity drops by 6.17 on average across biomed, finance and law.
- Domain-level, not per-user. Low relevance, except as another "imitate the retriever with parameters" design.

### Candidate 2: answering vs abstaining

**Sufficient Context (Joren et al., Google, ICLR 2025, 2411.06037)** [full main text + App. A.2, B]
- **Definition:** context is "sufficient" if (question, context) admits some plausible answer; no gold answer is needed.
- **Autorater accuracy** on 115 hard cases (Table 1): Gemini 1.5 Pro 1-shot 93.0%; FLAMe 87.8%; TRUE-NLI 82.6%; "contains the gold answer" 80.9%.
- **Abstention on human-labelled data:**
  - Sufficient context (Table 4a): 1.6–11.1%, with hallucination 3.2–25.4%.
  - Insufficient context (Table 4b): 50–73%.
  - With insufficient context, models are still correct 35–62% of the time.
- **Selective generation:** logistic regression on [sufficiency label, self-rated P(True)] gives >10% accuracy gain for Gemma on HotpotQA at high selectivity, and none on Musique.
- **Fine-tuning** (App. A.2): Mistral-7B, LoRA r=4, 2k examples (Table 3, Musique):

  | Training data | Correct | Abstain | Hallucinate |
  |---|---|---|---|
  | Vanilla RAG | 28.8% | 11.8% | 59.4% |
  | Fine-tuned, no IDK | 31.4% | 0% | 68.6% |
  | Fine-tuned, 20% IDK | 23% | 2.2% | 74.8% |

  The authors call training "not smooth".
- **For us:** text RAG with sufficient context almost never abstains, so our refusal rate of 0.70–0.79 is format-specific.
  - The selective-generation gate can only *add* abstentions; it cannot turn a refusal into an answer.
  - A useful signal to borrow: score the candidate answer with refusal suppressed. Our 0.99 teacher-forced figure says that signal exists.

**Do RALMs Know When They Don't Know? (Zhou et al., AAAI 2026, 2509.01476)** [full main text; appendix not in the PDF text]
- **Over-refusal with all-irrelevant context** (0p10n, Table 2):
  - vanilla 0.355
  - R-tuning 0.678
  - ICFT(n) 0.270, with the best overall accuracy, 0.620
- **With evidence present** (1p9n, Table 2): vanilla over-refuses 0, R-tuning 0.066, ICFT(n) 0.171.
- **Context utilisation with evidence present** (Table 3): vanilla 0.738, falling to 0.591 for ICFT(n). Every refusal-trained variant uses evidence less than vanilla.
- **Two-stage ΔU gate** (uncertainty with vs without context; Table 4, 0p10n): vanilla over-refusal 0.770 → 0.282, accuracy 0.437 → 0.523. It is never tested with evidence present.
- **For us:** the clearest warning for candidate 2. Training "refuse when unsupported" pushes the model further into *our* failure.
  - ΔU (with vs without the slice) is a training-free licence signal that maps directly onto slice / no-slice and leave-one-out contrasts.
  - Missing from the text: model sizes, trained-part details, LoRA rank.

**Trust-Align (Song et al., ICLR 2025, 2409.11242)** [full main text + App. F, H]
- **Data:** about 70k (question, 5 documents) pairs from ASQA, QAMPARI and ELI5, labelled answerable or not by TRUE-NLI.
  - Preferred responses: GPT-4 answers with citations, or a fixed refusal when unanswerable.
  - Rejected responses: the model's own worst hallucinations, giving 19k pairs, including 8,786 "unwarranted refusal" cases.
  - Training: full-parameter SFT then DPO.
- **Results:** beats baselines in 26/27 configurations. DPO beats SFT in 26/27.
- **Refusal detail** (ASQA, Table 19):

  | Model | Variant | Recall on answerable | Recall on unanswerable |
  |---|---|---|---|
  | Qwen-2.5-1.5B | ICL | 99.18 | 2.66 |
  | Qwen-2.5-1.5B | DPO | 81.15 | 42.90 |
  | LLaMA-3-8B | ICL with refusal prompt | 2.13 | — |
  | LLaMA-3-8B | DPO | 68.20 | 64.79 |
  | LLaMA-3.2-3B | ICL with refusal prompt | 1.64 | — |
  | LLaMA-3.2-3B | DPO | 89.02 | — |

  - Under a refusal prompt, LLaMA-3-8B answers only 1.48% of the time.
  - Prompting alone swings between answering nearly 100% and nearly 0%.
- **For us:** the template for candidate 2. Build pairs where the slice supports the answer (answer preferred over refusal) and pairs with a distractor slice (refusal preferred), mined from our own refusals, and train a small read adapter with DPO. Caveats:
  - Their training is full fine-tuning, not a small read adapter.
  - NLI answerability labels are noisy.
  - Their metric rewards refusal heavily.

### Candidate 3: keys and verification

**M+ (Wang et al., ICML 2025, 2502.00592)** [full]
- **Memory:** MemoryLLM's latent memory pool. Tokens dropped from the pool go to a long-term memory on CPU (up to 150k per layer).
- **Retriever:** two 2-layer MLP projectors (keys at d/20).
  - The key is computed once, when a token is dropped. The query comes from the current hidden states.
  - One dot-product lookup per layer, shared by all heads: 32 lookups instead of 1024 per-head ones.
  - Training labels: positives are memory tokens from earlier chunks of the same document; negatives are memory from before that document.
- **Retrieval accuracy:** about 30% of ground-truth tokens retrieved vs 3% random (Fig. 6). Better than attention-score retrieval.
- **Other results:** LongBench 32.78 vs plain Llama-3.1-8B 35.81 at 16k. Knowledge retention extends from under 20k to over 160k tokens (plots only).
- **Not per-user, no deletion, no abstention.**
- **Cost:** stage 1 is 4 weeks on 8×A100, with continued training of the model.
- **For us:**
  - Copy: a low-dimensional learned key stored once per item, queried from hidden states, one lookup per layer, trained contrastively, with only the projectors trained.
  - Change: supervise with "the right fact vs near-miss facts".
  - Add: a verification and abstention head. M+ always takes the top-K.

**WISE (Wang et al., NeurIPS 2024, 2405.14768)** [skim: Sec. 3, Tables 2, 6, 7]
- **Method:** a side copy of one FFN value matrix holds the edits. A routing activation, the norm of the side-vs-main difference, is trained with a margin loss so that edited inputs fire and unrelated inputs don't.
  - Edits are sharded into random subspaces and merged.
  - WISE-Retrieve keeps several side memories and picks one by top-1 activation.
- **ZsRE at T=1000** (Table 2): LLaMA-2-7B average 0.83 with locality 1.00; MEMIT 0.03; GRACE 0.67, where generalisation collapses to 0.08.
- **3K edits** (Table 6): WISE-Retrieve 0.73 vs oracle routing 0.82. The gap is retrieval.
- **Without the router** (Table 7): locality at T=1000 falls from 1.00 to 0.72.
- **For us:** a trained "is this input within my memory's scope?" margin is a cheap verification head (candidate 3). The retrieval-vs-oracle gap at 3K edits shows that routing, not storage, is the bottleneck, which also holds for us.

**MEMOIR (Wang et al., NeurIPS 2025, 2506.07899)** [skim: Sec. 3, Table 1]
- **Method:** each edit writes to a distinct sparse subset of a residual memory, chosen by a top-k mask on the prompt's averaged activation. At inference, overlap between the prompt's mask and stored masks decides whether to activate edited knowledge at all.
- **ZsRE on LLaMA-3-8B** (Table 1):

  | Method | Avg at T=100 | Avg at T=1000 |
  |---|---|---|
  | MEMOIR | 0.95 | 0.93 (Rel 0.94 / Gen 0.85 / Loc 1.00) |
  | AlphaEdit | 0.88 | 0.72 (locality 0.56) |
  | WISE | 0.74 | 0.77 |

  The paper extends to 15k edits.
- **For us:**
  - Sparse, sample-specific write locations limit interference between similar facts.
  - Mask overlap is a training-free "was I told about this?" signal. That is a licence or verification cue; it uses averaged prompt activations, as our subject-word keys do.
  - Its locality definition is about unrelated prompts, not near-miss subjects, so it does not test the swap case.

**A Storage–Retrieval Gap in Parametric KG Memory (Pulici et al., Aug 2026, 2608.25489)** [full]
- **Setup:** one LoRA per MetaQA entity (Qwen3.5-2B, r=8), 10.4 MiB each, trained on 8 QA pairs.
- **Closed-book results:** base 0.007 → own adapter 0.250 EM (+0.243). A wrong adapter equals base (0.017); n=30.
- **Retrieval:** finding the right adapter by question embedding or weight-space distance gives 0.000 gain, the same as random (Table 5).
- **Collisions:** films sharing a title produced adapters that could not reproduce their own training answers; they were filtered out.
- **For us:** similarity keys don't find parametric memories, so an entity resolver or a learned key is needed. Size comparison: a 2-position fp16 slice on Qwen2.5-1.5B is about 57 KB (reader's estimate) against MBs per LoRA. They never measure confident wrong answers from a wrong adapter, which is our main retrieval risk.

---

## 3. Found but not read

All verified on arXiv. None was read beyond the abstract.

**Parametric memory and hypernetworks**
- Generative Adapter (2411.05877)
- Text-to-LoRA (2506.06105): task-level, not facts
- SHINE (2602.06358): context→LoRA hypernetwork reusing the frozen model's own parameters
- OPPU (2402.04401) and Personalized Pieces (2406.10471): per-user PEFT on LaMP style tasks
- **RPMem (2609.23466):** sessions compiled into latent memory, merged by a trained recurrent gate, then mapped to a LoRA; PERMA 85.52 on Qwen3-8B. *The most relevant unread paper for candidate 4 across sessions; read next.*
- PLUME (2609.14168): training-free context parameterisation under updates; MUSE-bench
- δ-mem (2605.12357): an 8×8 delta-rule state producing low-rank attention corrections on a frozen backbone; LoCoMo 1.20× backbone
- Metis (2607.26760)
- Understanding Parametric Knowledge Injection in RAG (2510.12668): P-RAG does not consistently beat text RAG under answer-presence accuracy; parametric representations act on deeper FFNs
- PRAG with latent LoRA routing (2511.17044)
- Parametric Memory Decoding, zero-shot LoRA routing from one prefill (2607.04118)
- Memory Decoder at Scale (2607.27919)
- Understanding LoRA as Knowledge Memory (2603.01097)
- Prometheus Mind (2601.15324): a frozen Qwen3-4B with 11 adapters; notes that "the frozen model learns to ignore injected signals"; single author
- LongMem (2306.07174)
- Jarvis, personal KV-cache retrieval for VLMs (2510.22765)
- Learned Structure in Cartridges, "keys as shareable routers" (2508.17032): relevant to candidate 3
- PSD (2609.23449)
- Axis-LoRA (2611.07820)

**Context distillation**
- Snell et al., Learning by Distilling Context (2209.15189)
- Propagating Knowledge Updates through Distillation (2306.09306)
- Knowledge Injection via Self-Distillation (2412.14964)
- MixSD (2605.16865)
- Synthetic Continued Pretraining (2409.07431)
- Reversal Curse (2309.12288)
- No 2025–26 paper tests paraphrase or reversal generalisation of *context-distilled* facts directly. The Engram paraphrase number (50% from 4 facts) is the only one seen.

**Lifelong editing**
- GRACE (2211.11031): in Tables 1–2 above, generalisation 0.02–0.37
- MELO (2312.11795)
- AlphaEdit (2410.02355)
- Lifelong Editing requires Better Regularization (2502.01636)
- WikiBigEdit (2503.05683)
- LightEdit (2604.19089): self-reported 0.94–0.99 up to 5000 edits *(numbers from the search summary, unverified)*

**Learned eviction and attribution**
- Locret (2410.01805)
- DMS (2506.05345)
- ForesightKV (2602.03203): a supervised ranker distilled from "golden eviction" using future attention, plus GRPO; the closest thing to amortising an oracle score
- IndexMem (2605.25475)
- LKV (2605.06676)
- Attention-Gate (2410.12876)
- AttriBoT (2411.15102): approximates LOO more than 300× faster with caching, hierarchy and proxy models, but does not learn it
- No paper was found that trains a scorer on leave-one-out targets.

**Abstention and gates**
- Self-RAG (2310.11511)
- RefusalBench (2510.10390): DPO beats SFT for selective refusal *(from the search summary)*
- GRACE-RL (2601.04525)
- SPARK (2606.27786): a 0.01%-parameter policy on a frozen backbone that scales FFN parametric participation, trained with GRPO
- CK-PLUG (2503.15888)
- Context-DPO (2412.15280)

**Searched for, not found**
- Any paper storing per-user *KV slices* plus a trained part, evaluated on LongMemEval, LoCoMo or PersonaMem. TRIM-KV runs LongMemEval but within one long context.
- Any paper with a learned retrieval key over KV entries plus abstention.

---

## 4. Depth-of-reading notes

- **Read in full by helper readers** (separate agents briefed with our numbers; their notes were checked for internal consistency and kept with their caveats):
  - TRIM-KV; Gating Mechanism; Rethinking LoRA Memory; Storage–Retrieval Gap
  - Sufficient Context; Do RALMs Know; Trust-Align; Substrate Asymmetry
  - Doc-to-LoRA; Knowledge Modules; User as Engram; M+
  - Figure values that did not survive text extraction were not used.
- **Skimmed by me** (method plus main table): PRAG, DyPRAG, DPPM, DiSC, Self-Study Reconsidered, WISE, MEMOIR, Memory Decoder (abstract only).
- **Known inconsistencies:**
  - TRIM-KV: Tables 3 and 8 swap the SnapKV and StreamingLLM numbers.
  - Substrate Asymmetry: Sec. 4.2 and App. B.2 disagree on presence and absence numbers.
  - User as Engram: per-user storage given as 88 KB, 60 KB and "per row".
  - Do RALMs Know: the dataset for Table 2 is not stated in the text read.
- **Rough estimates by the readers, not from the papers:** TRIM-KV gate size (~47M), D2L per-chunk adapter size (~4.8 MB), slice size (~57 KB).
