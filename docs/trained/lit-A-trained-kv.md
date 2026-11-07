# Literature A: trained KV / context memories and compression

Notes for the trained side of Seahorse: a frozen backbone, per-user memory kept as data (K,V slices, keys, recurrent state), and small parts trained once. Thirteen papers, read on 2026-11-07. Page numbers are the paper's printed page numbers; "Fig." means the number is only in a plot.

Our findings this compares against are in the brief at the top of the task (slices, binding, content vs licence, selection, retrieval). The four candidates are:

1. an amortised S-loo selector,
2. a licence-aware read gate or adapter,
3. a learned retrieval key plus verification or abstention,
4. SLEEP-style consolidation (slices → LoRA, teacher = model with an answering slice, data = dreamed questions, slices deleted after).

---

## 1. Short version

**Takeaways for the trained side**

1. **Candidate 2 is partly done, on a different source of missing evidence.** Compression-Aware Abstention trains a 10.1M-parameter LoRA on a frozen Qwen2.5-7B to answer when answer tokens survived KV eviction and refuse when they did not. Its key fact for us: from a real compressed cache, the base model answers only 4/97 items whose evidence survived (1/69 at 80% retention, p.7). The best adapter lifts this to 24/97 and 22/69. That is our "content without licence" gap, partly fixed by training. Nobody does it for slices restored across sessions, and nobody splits teacher-forced from generated scores on the same items as we do.
2. **How much attention a memory gets is a design choice in four of these papers.** Four mechanisms set it:
   - Attention Matching: a per-entry bias β that restores a compacted block's attention mass.
   - KBLaM: a log C − log M shift that keeps a large knowledge base from swamping the prompt.
   - Memory Inception: bank evidence normalised by size.
   - TokenMem: a separate softmax, so the prompt cannot starve the memory.

   Our 2–8 kept entries carry a tiny share of the turn's attention mass. A β per kept entry, untrained, is the cheapest test of the licence problem. A learned scalar per entry or per layer is the cheapest trained gate.
3. **Distilling from the model with the context in view beats next-token loss, and next-token loss makes the model over-trust context.**
   - Cartridges: +8.6 chrF on MTOB, +3.7 accuracy on LongHealth.
   - xRAG: self-distillation matters more than NLL, and "Optimization with nll loss tends to cause an overreliance on context information" (p.9).
   - So for candidates 2 and 4, train with KL to a good teacher, not cross-entropy on answers.
4. **For candidate 4, Cartridges warns against LoRA.** At matched memory, a trained KV prefix beat LoRA in domain (best LongHealth 55.6 vs 47.25). LoRA also cut MMLU from 60.0 to 45.3 at 1.06 GB, while the prefix only dropped to 54.3 (App. A.1, p.22). A trained KV "cartridge" also composes and can be deleted per user; a LoRA does neither cleanly. Our teacher is also a problem: a model reading gold slices refuses 0.70–0.79 of the time on LongMemEval. Distilling from that teacher would bake the refusals in. The teacher should be full-turn context.
5. **Separately encoded pieces lose cross-attention. Training fixes it, and the fix in KVLink is ordinal.**
   - Prompt caching with corrected positions drops Llama-3.2-1B on NQ from 44.6 to 18.6, and Llama-3.1-8B on MuSiQue from 49.1 to 6.7 (Table 1, p.6).
   - KVLink recovers most of this with trained link tokens whose identity is the document's slot number ("for the n-th reused document, its link tokens are always link n-1, ..., link n-K", p.25). That is an ordinal signal much like our "store the k-th memory as the k-th fact" fix. However, KVLink fine-tunes the whole model.
   - None of the 13 papers measures partner swaps. Memory3 places every memory at the same positions, which is exactly the setup that gave us swaps, and it does not test for them.
6. **Selection is still attention-based everywhere; nobody amortises a leave-one-out selector (candidate 1).**
   - Memory3 keeps the top 8 of 128 tokens per head by attention received, with no causal mask or positions.
   - Attention Matching selects by highest attention or OMP on reference queries, and finds self-study adds little over repeat-prefill.
   - Gist tokens give the template for amortising: predict in one pass what per-item optimisation would produce. Here the per-item optimisation is S-loo.
7. **Trained-once readers generalise across content, but they degrade out of distribution and none verifies a wrong memory.**
   - Pieces trained once: KBLaM's adapters (synthetic KB, then Enron), xRAG's projector, TokenMem's gate, the abstention LoRA.
   - The negatives each one handles are different:
     - KBLaM: unanswerable questions, i.e. no relevant triple.
     - Abstention LoRA: evidence evicted.
     - TokenMem: none. It trains blind compliance by design.
   - No paper handles "a plausible but wrong memory was retrieved", which is our wrong-slice failure (candidate 3).
8. **Our recurrent-state findings have no counterpart.** None of the 13 papers touches DeltaNet or SSM state. Attention Matching compacts only Gemma-3's global layers and says it cannot handle MLA (p.28).

**Overlap of our candidates with these papers**

| Candidate | Closest paper(s) | Already done? | What stays ours |
|---|---|---|---|
| 1. Amortised S-loo selector | Memory3 (top-8 by attention, query-free); Attention Matching (OMP on reference queries); Gist (amortised distillation as an idea) | **No.** Every selector here is attention or attention-mass fitting. Nobody trains a predictor of leave-one-out value. | Training a one-pass predictor of S-loo's picks, and the finding that LOO beats attention (punctuation picks) at 2–8 entries |
| 2. Licence-aware read gate | **Compression-Aware Abstention** (LoRA trained on answer/abstain from eviction masks); KBLaM (trained refusal on unanswerable questions); TokenMem and xRAG (the opposite: trained compliance) | **Partly.** A trained answer-or-abstain adapter over a degraded KV cache exists, with a two-term loss that guards against both refusing and blind compliance. | Restored cross-session slices rather than evicted context; separating content from licence (TF vs generation, refusal mass); the subject-position and "told" findings; a β test before any training |
| 3. Learned key + verification | KBLaM (learned key adapter on name+property; attention as retriever; trained "cannot find" refusal); Memory3 (external BGE-M3 retrieval; Remark 2 proposes the model's own sparse queries) | **Partly** for the key, **no** for verification. KBLaM learns keys and refusals for *irrelevant* questions, but never tests a retrieved wrong-but-plausible entry. | Verifying that a retrieved slice answers this question; abstaining on a confident wrong slice |
| 4. Slices → LoRA consolidation | **Cartridges** (self-study + context distillation into a trained KV prefix, per corpus); Bottlenecked Transformers (a trained processor rewrites KV in place, within a sequence); Gist/ICAE (amortised compression) | **Mostly, in KV rather than LoRA.** Cartridges is "distil a corpus into parameters with self-generated questions", and it argues against LoRA. | Per-user, cross-session, starting from 2–8 selected positions; a teacher that refuses; deleting slices after; the binding question (does consolidation fix partner swaps?) |

---

## 2. Per-paper notes

### 2.1 Cartridges (Eyuboglu, Ehrlich, Arora et al., arXiv 2506.06266 v3, June 2025; listed as ICLR 2026 in the index)

**Read:** §1–6 in full, App. A (ablations), C (seed prompts) and D.1 (GenConvo). Skipped the theory in App. E (MQAR).

- **What it is:** a small KV cache trained offline for one corpus and loaded as a prefix at inference. "we freeze the LLM and train a smaller KV cache offline by backpropagating loss into the key and value vectors in a process essentially equivalent to prefix tuning" (p.2–3).
- **Trained / frozen:**
  - The model is fully frozen. Only the p×L trainable key and value vectors are trained.
  - The first token's K,V (the attention sink) is kept frozen. Training it caused instability, with MMLU dipping "below 30%" (App. A.1, p.22).
- **Unit of memory:** one cartridge per corpus of 100k–484k tokens, p ∈ {128 … 8192} trained tokens.
  - Per-user: yes, the intended use. It names "chat histories", "a patient's medical record" and "long-term memory in chatbots".
  - Composable: yes, cartridges are concatenated without joint training.
  - Deletable: yes, by dropping the prefix.
- **Training signal:** *self-study*.
  - The model is shown a 512–4096-token chunk plus a generic seed prompt and chats with itself, one turn in the main results. Five seed types: structuring, summarization, question, use cases, creative.
  - The loss is KL to the teacher, the same model with the chunk in context (Eq. 3, p.9). Next-token loss on the raw corpus memorises "with near perfect perplexity" and uses 107× less memory, but does not generalise (Fig. 3, p.5, 7).
- **Cost:**
  - "∼30 minutes on a single 8×H100 node (for LLaMA-8B)" per ICL-quality cartridge (p.12).
  - Attention Matching's re-run measured ∼5 H100-hours per 7k-token QuALITY article: 25–30M self-study tokens, 32,768 samples, 1 epoch, lr 2e−2. LongHealth-5 (60k tokens) took ∼15 H100-hours on 100–150M tokens (AM Table 5, p.17).
- **Models and benchmarks:** Llama-3.2-3B and Llama-3.1-8B; LongHealth (accuracy), QASPER (log-perplexity), MTOB (chrF), GenConvo (AMD 10-K), MMLU as an out-of-domain check.
- **Key numbers:**
  - Matches ICL at 38.6× less memory and 26.4× higher throughput, averaged (abstract).
  - Up to 10× memory saving on LongHealth and up to 100× on QASPER at ICL-level quality (p.10).
  - Compression baselines fail already at 2× (Fig. 4).
  - **Initialisation:** random vectors 29.9%, K,V of random text 51.3%, K,V of the corpus's first p tokens 55.3% on LongHealth (p.11).
  - **Seed prompts:** five types beat one by 7.9 chrF on MTOB (24.1 → 32.0) and 4.8 on LongHealth. No gain on QASPER (p.11).
  - **Objective:** context distillation beats next-token on the same synthetic data: MTOB 24.9 → 33.5 (p.11), LongHealth 47.6 → 51.3 (App. A.2, p.24).
  - **Prefix vs LoRA (App. A.1, p.22):**
    - LoRA at 1.06 GB (rank 1632) drops MMLU from 60.0 to 45.3. The prefix drops to 54.3 at 1.06 GB, and to 54.7 at 0.15 GB.
    - In domain: prefix best 55.6% vs LoRA best 47.25%. LoRA falls to 41.3% at 0.96 GB.
    - The authors do not know why.
  - **Composition (§5.4, p.12):**
    - Setup: cartridges for pairs of 10-Ks (AMD, Pepsi, AMEX, Boeing), 6 pairs per size, tested on questions that need both documents.
    - Composition "substantially outperforms the use of a single CARTRIDGE ... or ICL". This is measured only as log-perplexity on gold answers (Fig. 7), plus examples.
- **Refusal / binding:**
  - Neither is reported.
  - The composition test cannot reveal partner swaps, because the two documents are about different companies and are scored by perplexity.
- **For us:**
  - This is candidate 4 already done in KV form, per corpus, and it argues against LoRA as the target.
  - A per-user "slice cartridge" would be: a few trained KV entries per memory, initialised from the S-loo slice (real K,V init matters a lot), sink frozen, trained by KL to a teacher on dreamed questions. It stays composable and deletable, unlike a per-user LoRA.
  - Cost would scale with our tiny memories (a turn, not 100k tokens). That is inferred, not measured.
  - The teacher must be one that answers, i.e. full turn in context. A gold-slice teacher refuses 0.70–0.79 of the time in our LongMemEval runs.
  - Hardalov et al. 2026, cited in the Shepard note, report naive composition near chance at 20 cartridges. Composition at our N (3–59 memories) is untested here.

### 2.2 Fast KV Compaction via Attention Matching (Zweiger, Fu, Guo, Kim, arXiv 2602.16284 v2, ICML 2026)

**Read:** §1–6 in full, App. B.2–B.3, D, F.3, F.4, G. Skimmed App. C and E.

- **What it is:** replace a block's (K,V) of T entries with t < T entries (C_k, β, C_v) that match the original block's attention output **and attention mass** on reference queries.
  - Keys are a subset of the originals, chosen by highest attention or by OMP on the mass.
  - β, one scalar bias per kept key, is fitted by NNLS. "Intuitively, w_j represents how many original keys' worth of attention mass the compact key (C_k)_j accounts for" (p.3).
  - C_v is fitted by least squares.
  - The cache keeps logical length T, so later tokens get the same RoPE positions (p.3).
- **Trained / frozen:** nothing is trained by gradient. Compaction is closed-form per context. A per-model, per-head budget schedule is fitted once from sensitivity curves (§3.4).
- **Unit of memory:** a compacted chunk of the context.
  - Chunks can be compacted separately and merged. "KV-based chunking" (prefill everything, then slice) is preferred over prefilling chunks in isolation, "even when chunks are semantically independent" (p.5).
  - Per-user / deletable: not discussed.
- **Signal:** attention-output and mass matching on reference queries:
  - repeat-prefill (KVzip's prompt), context-prefill, or self-study (4 fixed prompts plus sampled answers);
  - at most 50,000 queries per KV-head;
  - on QuALITY, about 5k self-study tokens and 7k repeat-prefill tokens per context (p.5).
- **Key claims:**
  - "methods that only drop or merge tokens without biases ... systematically underestimate the compacted block's contribution during future decoding" (p.3).
  - "self-study yields the best downstream performance; context-prefill and repeat-prefill are nearly as good while being faster" (p.3).
  - In the leave-one-out ablation (Fig. 5, p.8), every component is needed. The least important are self-study and on-policy queries; the most important is non-uniform head budgets.
  - "omitting attention biases and retaining the original values for selected keys still yields a reasonable approximation" (p.8).
- **Cost (Table 3, p.8, Gemma-3-12B, 60k-token context, one H200):** context-prefill 7 s, repeat-prefill 8 s, self-study 139 s, highest-attention selection 3 s, OMP 565 s, NNLS 2.2 s, least squares 1.8 s.
- **Models / benchmarks:** Qwen3-4B, Llama-3.1-8B, Gemma-3-12B; QuALITY, LongHealth, QASPER, LongBench v2 (47-context subset), RULER.
- **Key numbers:**
  - QuALITY, Qwen3-4B (Table 10, p.28): AM-HighestAttnKeys at 1% / 2% / 5% keeps 60.4 / 65.0 / 70.1, against 71.1 for the full cache and 36.8 with no context. KVMerger gets 43.4 / 49.4 / 55.1.
  - QASPER F1, Qwen3-4B (Table 1, p.7): 0.338 at 1% vs KVzip 0.127, full 0.444, no context 0.104.
  - RULER, Llama-3.1-8B (Table 11, p.28): the fastest AM variant gets 83.3 at 5% vs KVzip 46.6, and 95.1 at 12.5% vs a full cache at 95.7.
  - Against Cartridges: AM beats it at 50× compaction, but "Cartridges outperforms our approach at more extreme (100×) compaction rates on some benchmarks (LongHealth)", which they attribute to gradient search over a wider space (p.9).
  - Online compaction (App. F.3, Table 9): AIME with a 2048-token physical cache and up to 6 compactions reaches 13/30, the same as plain decoding at 8192.
- **Refusal:** one qualitative note, App. G.1 (p.28): "at very high compaction ratios ... compacted outputs tend to lack specific information (e.g., responding 'the text does not mention...') while remaining coherent—a graceful, low-risk failure mode." They read this as lost content. Given our results, part of it may be lost licence: the content is there but the attention mass is low. This is inferred.
- **Binding:** none.
- **For us:**
  1. β is the untrained experiment to run before candidate 2: refit or just scale the mass of our 2–8 kept entries to the turn's mass, then measure refusal mass. A learned β (or a gain per layer) is the smallest possible trained gate.
  2. OMP + NNLS + least-squares values is a closed-form "compaction" baseline for candidate 1. An amortised selector has to beat this, not just attention.
  3. "Logical length T" is how they handle positions. We restore at original positions, which is consistent with this.

### 2.3 KVLink (Yang, Hou et al., arXiv 2502.16002 v4, NeurIPS 2025)

**Read:** §1–5 in full, App. A.1–A.4, A.6 (compression), A.7–A.9.

- **What it is:** reuse separately precomputed document KV caches in one prompt.
  1. Store keys without RoPE and re-rotate them to their global position (§2.2).
  2. Append K trainable "link tokens" after each document, whose KV is computed at inference. Each link token can attend to everything before it, which restores cross-document attention (§2.3).
- **Trained / frozen:** **the whole LLM is fine-tuned** together with the link tokens: "we fine-tune the LLM with this attention mechanism, jointly optimizing both the model and the newly introduced link tokens" (p.4). The setup is not a frozen backbone.
- **Unit of memory:** one document's KV.
  - Per-user / deletable: yes, by construction (one cache per document).
  - Composable: that is the purpose.
  - Link tokens are **indexed by slot**: "for the n-th reused document, its link tokens are always link_{n-1}, link_{n-2}, ..., link_{n-K}" (A.2, p.25). The link identity therefore encodes order.
- **Signal:** next-token SFT on a mixture:
  - retrieval QA with 10 separately encoded passages (TriviaQA, 2WikiMQA; GPT-4 answers as targets);
  - **multi-turn conversations whose earlier turns are separately encoded** (DaringAnteater, 92,700 samples);
  - XSum split into 100-token segments;
  - plus Tulu-3 SFT and FineWeb to preserve general ability (Table 4, p.23).
  - Retrieved documents are shuffled in training.
- **Cost:** 6,000 steps, global batch 64, 8×H100 (p.5).
- **Models / benchmarks:** Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B (Instruct); NQ (gold passage rotated through all 10 slots), 2WikiMQA, TriviaQA, HotpotQA, MuSiQue, MultiNews, Samsum; IFEval, GSM8K, MMLU and others for retained ability.
- **Key numbers (Table 1, p.6):**

  | Model, task | Original, full context | PromptCache (positions fixed) | CacheBlend (18% recompute) | BlockAttention | KVLink0 | KVLink5 |
  |---|---|---|---|---|---|---|
  | 1B NQ | 44.6 | 18.6 | 25.7 | 39.0 | 40.8 | 45.0 |
  | 1B MuSiQue | 13.8 | 1.4 | 3.7 | 14.3 | 13.5 | 19.2 |
  | 8B NQ | 71.3 | 28.9 | 55.2 | 70.8 | 71.0 | 72.5 |
  | 8B MuSiQue | 49.1 | 6.7 | 6.0 | 38.7 | 39.9 | 40.8 |

  - Fine-tuning costs general ability, e.g. Llama-3.1-8B GSM8K 85.1 → about 75 for every fine-tuned variant (Table 2, p.8).
  - Time to first token drops 85–96% for 10 × 100–500-token documents (p.7).
  - Link tokens also help on compressed caches (Table 3, p.9).
- **Refusal:** not measured. The QA system prompts used in training include "If the information is insufficient, acknowledge the gap" (Fig. 5, p.24), but refusal rates are never reported.
- **Binding:** no swap analysis. The loss is "missing cross attention" (p.2), measured as accuracy.
- **For us:**
  - Separately encoded = our separately stored slices. Even with correct positions, the untrained loss is huge (PromptCache rows).
  - Slot-indexed link tokens are a trained version of our storage-order fix. A small trained "slot-k" embedding, or a linker computed at read time over a frozen backbone, is a natural trained piece for binding. It is not one of our four candidates yet. Worth adding.
  - KVLink needed full fine-tuning; whether a frozen backbone plus trained link KV suffices is open.

### 2.4 Memory3 (Yang, Lin, Xu et al., arXiv 2407.01178, July 2024)

**Read:** §3 (design) in full, §3.6, §5, §7. Skimmed §2 (theory) for Remark 1. Skipped §4 data details.

- **What it is:** a 2.4B model pretrained from scratch to read "explicit memories".
  - Each memory is the sparse K,V of a 128-token reference chunk, taken from the model's own attention: tensor shape (memory layers, 2, KV heads, tokens, head dim) = (22, 2, 8, 8, 80) (p.13).
  - Each 64-token chunk of input or generation retrieves 5 memories (BGE-M3 + FAISS) and attends to them alongside context.
- **Trained / frozen:** everything is trained (pretraining, SFT, DPO). Writing a memory needs no training: "The memory is selected from the key-value vectors of the self-attention layers, so the write process involves no training" (p.12).
- **Unit of memory:** one 128-token reference → 8 kept tokens per KV head, in the first half of the layers. The bank has 1.1×10^8 references. Encoding is independent per reference: "a reference during encoding does not need to attend to any other texts" (p.13).
  - Composable: yes, 5 at a time.
  - Deletable: yes.
  - Per-user: not the use case; the bank is a general knowledge base.
- **Selection (§3.3, p.14–15):** the top 8 tokens per head by attention received from all tokens of the reference, computed **without causal mask or position encoding** "since the explicit memories are prepared before any inference, the selection can only depend on the reference itself". Tokens can differ across heads. Key vectors are also quantised to 8.75% of their size with "no degradation" (p.22).
- **Positions:** "parallel position encoding for all explicit memories, namely the positions of all their keys lie in the same interval of length 128" (p.14). A special BOS "⟨s⟩Reference:" marks the memory.
- **Remark 3 (p.13–14):** chunk KV can be computed from the whole document so that "the key-values ... attend to all their contexts". This is why a partial KV slice can beat the same tokens as text, as in our 0.86 vs 0.27.
- **Training notes (p.17–18, 21):**
  - "pretraining with explicit memories from the beginning would render the memories useless" — the model learns to ignore them. A memory-less warm-up was needed (3.1T tokens).
  - Continual training with memories stopped at ~120B tokens because of loss divergence.
- **Key numbers:**
  - With memory, the average across 7 benchmarks is 63.31 vs 60.80 without (+2.51; Table 16, p.23).
  - Hallucination average 48.60, the best in Table 19 (p.24).
  - Decoding is 35.2% slower with retrieval, mostly from loading memories (Remark 8, p.26).
- **Refusal / binding:** not studied. Parallel positions put every memory in the same position window, which is the configuration that gave us partner swaps, but they evaluate no multi-fact binding.
- **For us:**
  - The closest published unit to our slices: a few KV entries per memory, chosen query-free from attention received.
  - Its selector is the S-attn family we found picks punctuation.
  - "Memories are ignored unless the model is trained to use them" (the warm-up finding) is the licence problem seen from the pretraining side.
  - Remark 2 (p.13) suggests retrieving with the model's own sparse attention queries instead of an external encoder; that is a cheap baseline for candidate 3.

### 2.5 KBLaM (Wang, Isazawa, Mikaelyan, Hensman, arXiv 2410.10450, ICLR 2025)

**Read:** §1–8 in full, App. A–D. Skimmed E–H (examples).

- **What it is:** each knowledge-base triple (name, property, value) becomes one "knowledge token":
  - key = W̃_K · f("The ⟨property⟩ of ⟨name⟩");
  - value = W̃_V · f(⟨value⟩);
  - f is a frozen sentence encoder (ada-002, P = 1536). There is one linear adapter per layer, giving a K and V per layer (p.4).
  - Prompt tokens attend to all knowledge tokens through a *new query head* W̃_Q per layer, initialised from W_Q. Knowledge tokens do not attend to each other ("rectangular attention", p.5–6).
- **Trained / frozen:** the backbone (Llama-3-8B-Instruct) and the encoder are frozen. Only W̃_K, W̃_V and the per-layer W̃_Q are trained (p.6).
- **Unit of memory:** one triple = one token's K,V.
  - Per-user: possible.
  - Composable: yes.
  - Deletable: yes. "update/remove/add a triple by only modifying its corresponding single knowledge token" (p.2).
  - Order-invariant by construction: "changing the order of the knowledge tokens does not affect the output" (p.6).
- **Attention mass:** at inference a constant shift log C − log M is added to the knowledge-base scores. "This is equivalent to multiplying the summations over M by C/M, such that the total contribution of the KB part stays roughly unchanged as M scales" (Eq. 11, p.6; C = 100, the largest KB seen in training).
- **Signal:** instruction tuning (next token) on synthetic KBs:
  - 45K GPT-generated names, ~135K triples, with values deliberately uncorrelated with names;
  - 10–100 triples per sample;
  - question types: simple, multi-entity, open-ended, and **unanswerable** (target: "Sorry, I cannot find relevant information in the KB.");
  - 2 of every 20 micro-batches are unanswerable (p.7–8).
- **Cost:** 20K iterations, 400 Q&A per iteration, one 80GB A100. App. D puts the one-time cost at "around 24-48 hours on a single 80GB GPU" (p.16).
- **Benchmarks:** the held-out synthetic KB (15k triples) and an Enron-email KB (out of distribution); BERTScore, GPT-4 scoring, retrieval top-1/top-5 from layer-15 attention.
- **Key numbers:** almost all are in figures.
  - Attention at layer 15 works as a retriever with high top-1 accuracy up to 10^3–10^4 triples (Fig. 5).
  - Comparable to BM25 on exact names, better than BM25 when names are paraphrased (Fig. 10).
  - Scales to more than 10K triples on one A100. In-context learning maxes out at 200 triples (p.8).
  - Enron, out of distribution, is degraded (Fig. 6).
- **Refusal (p.9, Fig. 6c):**
  - Setup: 80 answerable + 20 unanswerable questions; refusal is the positive class.
  - "when the number of triples increases, both KBLaM and in-context learning show decreased precision, i.e. the models start to incorrectly refuse ... The recall for both methods stays constant".
  - So over-refusal grows with KB size, and KBLaM degrades more slowly than ICL.
  - **Layer finding (App. B, p.15):** adding knowledge tokens only every 3 or 10 layers keeps answer quality at K = 3 but hurts refusal precision. "When K = 10, KBLaM fails to provide any refusal answer. As such we suspect that knowledge tokens in earlier layers may serve as soft instruction prompts that guides the LLM how to use the KB." Keys and values vary little across triples in early layers (Fig. 9).
- **Binding:** avoided by design, since name+property and value sit in one token's K,V. Two-entity questions are scored (BERTScore) but not checked for swaps.
- **Limitation they state:** a fixed-length vector "may fail to precisely generate the text word by word ... problematic for cases when we need exact names or numerical values" (p.10).
- **For us:**
  - This is the clearest frozen-backbone, trained-once template: an encoder-adapter memory plus a trained read path plus trained refusal. It covers part of candidate 3 (learned key) and part of candidate 2 (refusal).
  - The early-layer finding parallels our "licence vs content" split: what tells the model it may use the memory and what carries the fact live in different places. Here, early-layer knowledge tokens act as instructions; later layers carry content. This is inferred from their Fig. 9 and our results, not tested.
  - Its unanswerable case is "no relevant triple", which is easy. Ours is "a relevant-looking wrong slice".
  - Its separate query head for memory reads is a cheap trained gate we have not tried.

### 2.6 Compression-Aware Abstention (Khodabandehlou & Krishnamachari, arXiv 2608.29934, Aug 2026)

**Read:** all of §1–7, plus App. B, C, L, M. The appendix tables that the main text cites were skimmed.

- **What it is:** teach a model to answer when the answer-bearing evidence survived KV eviction and refuse when it did not.
  - Labels come from compressor survival masks over **tight** answer spans: the verbatim sub-answer, median 3 tokens.
  - CONFIDENT if minimum span survival ≥ 0.8; ABSTAIN if < 0.3; examples in between are dropped (Eq. 2, p.2).
  - Paragraph-level spans give "label entropy 0.000, vs 0.95 for tight spans" at r = 0.5 (p.3). Labels with coarse spans carry no signal.
- **Trained / frozen:** a LoRA adapter, rank 16, on q, k, v and o projections, 10.1M parameters (0.13% of Qwen2.5-7B-Instruct); the base stays frozen (p.3).
- **Unit of memory:** not a memory system. It works on a compressed context per query.
- **Signal:** cross-entropy with two terms. L_CONF (emit the answer when evidence survives) plus L_ABS (emit a fixed refusal string when it does not). "optimizing only L_ABS yields a fixed abstention prior ... while optimizing only L_CONF approaches a preservation-style objective ... (always answer)" (p.3).
- **Data / cost:**
  - Data: MuSiQue 2-hop, ~2.6K training examples for v1 and 8,772 for the mix across KVzip, Expected Attention and SnapKV masks.
  - Training: 3 epochs on one H100.
  - Held-out base: Llama-3.1-8B-Instruct.
- **Two inference paths:**
  - "ps": the surviving tokens fed as text.
  - "cc": a real compressed cache (KVzipPress), which matches deployment.
- **Key numbers:**
  - Prompt-style, Qwen val, aggregate honest accuracy 0.32 → 0.86. Hallucinations 232 → 7 (Table 1, p.4).
  - A prompt-only abstention instruction over-abstains on 49–51% of answerable items at r = 0.80; the mix adapter does so on 6–10% (p.5).
  - **Compressed cache (Table 4, p.7):**
    - The base reaches honest accuracy 0.186 overall.
    - On items whose evidence survived it is correct on **4/97 overall and 1/69 at r = 0.80**: "the instruction-tuned base model still often fails to use it once generation proceeds from the compressed cache".
    - The mix-ps adapter scores 0.730 and the mix-cc adapter 0.821.
    - Correct answers when evidence survived go 4/97 → 24/97 and 1/69 → 22/69. At r = 0.80, honest accuracy under cc is only 0.315 vs 0.658 prompt-style (p.9).
  - **A single-compressor cc adapter** cuts over-abstention to 0.203 at r = 0.80 but gives **42/69 (61%) confident wrong answers**, many of them "sub-token-fragment outputs ('Acfairs', ...)". The mix-cc adapter gives 21/69 (App. M, p.16).
  - **Controlled deletion (§5.6):** at equal length, keeping KVzip's top half leads to answers and keeping the bottom half leads to abstention. Swapping only the answer tokens for random kept tokens also raises abstention. The behaviour follows the evidence, not the length.
  - Limitation: at r = 0.80 the adapter still hallucinates on the same 4 of 7 rare ABSTAIN items across seeds. It "has not fully learned the harder behavior of introspecting on whether the specific evidence required for the current question survived" (p.9).
- **Refusal:** this is the topic of the paper. The base failure under cc is not split into refusals and wrong answers in the parts I read (Table 6 gives that split only for the adapters: 17% base over-caution at r = 0.80 prompt-style).
- **Binding:** none.
- **For us:**
  - **This is the closest prior work to candidate 2.** Training from the label is cheap: a LoRA trained on a few thousand examples.
  - The two-term loss is the right shape against "comply blindly".
  - Three things it does not cover:
    1. Restored cross-session slices: our KV entries come from another session, not from the same prefill.
    2. The content/licence split: our gold slices close 0.99 of the gap teacher-forced but are judged correct only 0.17–0.49 in generation.
    3. The "wrong slice present" case.
  - Its CONFIDENT/ABSTAIN labelling maps directly onto our gold@k vs non-gold slices. We can build ABSTAIN-gold items from deliberately wrong retrieved slices, a third class it lacks.
  - Its cc-training lesson: train with the real slice-restored cache, not a text stand-in, and mix several selectors (S-dream, S-loo, attention). Training on one selector's masks generalised badly.

### 2.7 Gist Tokens (Mu, Li, Goodman, arXiv 2304.08467, NeurIPS 2023)

**Read:** §1–5 and §8 in full, App. B (compute) by search.

- **What it is:**
  - The model is fine-tuned with an attention mask so that tokens after k gist tokens cannot see the prompt. Prompt information must pass through the gist tokens' KV.
  - Gist KV can then be cached per prompt.
  - Framed as amortised "meta" context distillation: rather than gradient descent per task, "we use G to simply predict the gist tokens (≈ parameters) of the distilled model" (p.3).
- **Trained / frozen:** the whole LM (LLaMA-7B, FLAN-T5-XXL) is instruction-finetuned with gist masks. The authors suggest retrofitting a frozen LM by training a smaller compressor as future work (§8).
- **Unit of memory:** 1–10 gist tokens per instruction. Not per-user; not tested for composition.
- **Signal:** standard next-token instruction tuning on Alpaca+ (130k examples, 104k unique tasks), not a KL loss in practice.
- **Cost:** ~7 h on 4×A100 for LLaMA-7B; ~25 h for FLAN-T5-XXL (App. B).
- **Key numbers:**
  - One gist token is enough, and more can hurt (10 tokens for LLaMA).
  - Win rate vs the uncompressed control (Table 1): 48.6% on seen tasks, 49.7% unseen, 45.8% human-written OOD (LLaMA).
  - Up to 26× compression and 40% FLOPs saving.
- **Failures (§5.1):** exact details are lost ("Culture" instead of "Arts & Culture"), and there are runaway repetitions not seen in the control.
- **Refusal / binding:** none.
- **For us:** the template for "amortise a per-item optimisation into one forward pass", i.e. candidate 1 (predict S-loo) and an amortised version of candidate 4. It also confirms that compressed carriers lose verbatim values, which is why our value tokens should be kept as slices rather than learned.

### 2.8 In-context Autoencoder, ICAE (Ge, Hu, Wang, Chen, Wei, arXiv 2307.06945, ICLR 2024)

**Read:** §1–3 in full, App. A–D by search.

- **What it is:**
  - An encoder (LoRA on the same LLM plus learned memory-token embeddings) turns a context into k memory slots (k = 128 for 512 tokens, 4×).
  - The **frozen** LLM decodes from those slots.
  - Pretrained on the Pile with autoencoding plus continuation (λ ≈ 0.4–0.6), then fine-tuned on PwC: 240k (context, prompt, response) items with GPT-4 answers.
- **Trained / frozen:** decoder frozen; encoder LoRA (r = 128) and memory embeddings trained, about 1% extra parameters (p.4–5).
- **Unit of memory:** memory slots per context.
  - Composable: only after training on concatenations. "this did not work initially, because the model had never seen multiple span concatenation patterns during training" (p.8).
  - Deletable: yes.
- **Cost:** 8×A100; 200k pretraining updates and 30k fine-tuning updates at batch 256 (Table 8, p.12).
- **Key numbers:**
  - Autoencoding BLEU 99.1 / 99.5 / 99.8 on Llama-7b / Llama-2-7b / Llama-2-13b (Table 6, p.7).
  - Normal text restores at BLEU 99.3; random text at 0.2 (Table 3), so the slots hold language-like content, not arbitrary strings.
  - Llama-2-7b-chat with k = 128 is on par with or better than GPT-4 74.2% of the time (Table 4).
  - Pretraining matters: pretrained k = 64 ≈ non-pretrained k = 128.
- **Refusal:** none.
- **Hallucination:** non-pretrained ICAE gives confident wrong values ("up to three years" instead of 30 years; "CEO" instead of VP; Table 9, p.17).
- **Binding:** composition fails without training on concatenations (above).
- **For us:** the frozen-decoder version of gist tokens. It is the cleanest evidence that a frozen model can read a trained memory format, but only after a large alignment phase. Its concatenation failure, fixed by training, is a composition result similar to KVLink.

### 2.9 xRAG (Cheng, Wang, Ge et al., arXiv 2405.13792, NeurIPS 2024)

**Read:** §1–6 in full, App. C–D by search.

- **What it is:** a retrieval embedding of a document (SFR) is projected by a 2-layer MLP into one soft token in the LLM's input. "the projector is the only trainable component, accounting for only 0.46% of the total parameters in the Mistral-7b model and 0.07% in the Mixtral-8x7b model" (p.5).
- **Trained / frozen:** the LLM and retriever are frozen; only the projector is trained.
- **Unit of memory:** one token per document, built from an embedding that already exists for retrieval. Cheap to store; composition not tested.
- **Signal:**
  1. Paraphrase pretraining: reconstruct the document from W(E).
  2. Instruction tuning on ~1M mixed items (reading comprehension, QA, summarisation), with NLL plus **self-distillation KL from the RAG teacher** (Eq. 3, p.5).
- **Cost:** 8×A100 (App. D).
- **Key numbers (Table 1, p.6):**
  - Mistral-7b average: no retrieval 36.54, RAG 43.63, xRAG 42.46 (one token vs 175).
  - Mixtral: 42.76 / 46.22 / 46.91.
  - Multi-hop lags: HotpotQA 34.05 vs 38.79.
  - Ablation (Table 3, p.8): without self-KD the average drops 44.58 → 41.49 and "resilience" (keeping already-correct answers) 82.3% → 76.2%.
  - "Optimization with nll loss tends to cause an overreliance on context information, rendering the system more vulnerable when the retriever fails to fetch relevant documents" (p.9).
  - Plain RAG flips about a quarter of previously correct answers to wrong (resilience 75.2%, p.7).
- **Refusal:** none.
- **Binding:** none.
- **For us:** evidence for candidates 2 and 4 that KL to a good teacher keeps the model's own judgement better than NLL on answers. Its "resilience" metric is the right one for "don't comply blindly with a wrong slice".

### 2.10 TokenMem (Yu, Zheng, Lu et al., arXiv 2607.22625, 2026)

**Read:** all of §1–5.

- **What it is:**
  - A retrieved passage is encoded by the frozen LLM, giving layer-wise states m^ℓ.
  - Each layer gets a cross-attention adapter that reuses the LLM's own W_Q, W_K, W_V, gated into the residual by a zero-initialised LoRA-style gate W_β (r = 16).
  - Only the gates are trained, ~3–7M parameters (p.5–6).
  - The thesis: in a shared softmax, "the model's parametric prior competes with external evidence for a fixed attention budget" (p.4). A separate softmax gives knowledge a fixed total mass of 1.
- **Signal:** next-token prediction in two phases.
  - Phase 1: 50K news CoT QA pairs, 3 epochs.
  - Phase 2: the same plus counterfactual passages (cf_arc_easy 2,745, cf_medqa 1,146) for 40 epochs.
  - The model is trained to follow the passage even when the passage is wrong about the world.
- **Models / benchmarks:** Qwen3-4B, 8B, 14B; Llama-3.1-8B; OLMo-3-7B. News, MMLU, MedQA, and the two counterfactual sets.
- **Key numbers (Table 2, p.7):**
  - Knowledge compliance on cf_arc_easy: 69.0 / 40.2 / 83.5 (Qwen 4B / 8B / 14B) vs 20.0 / 19.4 / 19.1 for plain RAG.
  - Accuracy on correct passages trails plain RAG (Qwen3-8B News 90.2 vs 96.3).
  - Llama-3.1-8B cf_medqa compliance falls to 27.6 vs 45.4 for RAG.
  - Phase 1 alone: compliance 2.4% (Table 3).
  - Gate activation rises under conflict in 31/36 layers, peaking at layers 10–12 and 19–23 (p.8).
- **Refusal / abstention:** none. The paper argues "Faithfulness is not gullibility" because the bank is user-authorised (p.8), but never tests a wrong or irrelevant passage.
- **For us:**
  - The counter-example to candidate 2: a trained gate can make a frozen model use injected memory, and the same training makes it follow wrong memory.
  - The Phase-1 result (a channel that exists but goes unused until trained against conflict) matches our "content present, no licence".
  - Its separate-softmax argument is a third mechanism, beside β and the KBLaM shift, for the attention-mass account of our refusals.

### 2.11 AutoCompressors (Chevalier, Wettig, Ajith, Chen, arXiv 2305.14788, EMNLP 2023)

**Read:** §1–6 by section, mostly from the text. Some tables (Tables 1–2) were read partly because the layout scrambled them.

- **What it is:** the model is fine-tuned to emit κ = 50 "summary vectors" per segment. These are passed as soft prompts to all later segments ("summary accumulation"), trained with BPTT and stop-gradients after 2 steps.
- **Trained / frozen:**
  - OPT-1.3B and 2.7B: full fine-tune on 2B tokens.
  - Llama-2-7B: LoRA on 15B RedPajama tokens, one A100.
  - "we do not train with a knowledge distillation objective, since the pre-trained LM has a limited context window as a teacher" (§3; this version has no printed page numbers).
- **Unit of memory:** summary vectors per segment or passage; precomputable and cacheable.
  - Composable: "fusing summary vectors is effective despite a mismatch in training since we draw independent summary vectors from separate documents ... only ever trained to accumulate 3 sets ... yet it benefits from fusing the summary vectors of up to 10 documents" (§6.1).
- **Key numbers:**
  - 150 summary vectors stand in for a 6,144-token context in perplexity.
  - Fused summaries give perplexity gains of 1.04 / 1.67 / 2.63 / 3.74% for top-1/2/5/10 retrieved passages vs REPLUG 512-token top-2 at 2.24%, with 1.7× throughput (Table 5).
  - Storing summaries for 10B tokens takes 5 TB per domain.
- **Refusal / binding:** none. Evaluation is perplexity and classification accuracy only.
- **For us:** background. Separately made soft memories can be fused, but measured only by perplexity, as with Cartridges.

### 2.12 Memory Inception (Liu, Zhang, Greenberg et al., arXiv 2605.06225 v2, May 2026)

**Read:** §1–6. Appendices only by heading.

- **What it is:** training-free.
  - Reminder text is run through the frozen model.
  - The hidden states of its content positions are projected with the model's own W_K, W_V into "bank" slots, stored **pre-RoPE** (position-independent).
  - The slots are attached only at selected layers or heads, chosen by a calibration-time selector that scores how strongly queries align with a target bank vs a reference bank (Eq. 3, p.4).
- **Attention mass:** bank evidence is normalised by bank size "so that a larger bank does not win routing weight merely because it contains more slots" (Eq. 5, p.4). Target and reference banks get query-adaptive gains.
- **Trained / frozen:** nothing trained; a selector artefact and gains are calibrated once.
- **Unit:** a bank per reminder; banks compete through bank-level evidence; banks can be updated mid-dialogue.
- **Models / benchmarks:** Llama-3.1-8B-Instruct and Qwen3-30B-A3B; Big Five persona steering, dialogue shift, PHYSICS, HARDMath.
- **Key numbers (Table 1, p.7):**
  - Llama self-evaluation DAS: 0.775 vs 0.645 for prompting and 0.185 for CAA.
  - Qwen3 dialogue-shift alignment: 0.816 vs 0.438 (prompt) and 0.526 (CAA).
  - PHYSICS: wins 10/12 cells.
  - KV savings 6–118×.
  - Hard traits need ≥ 10 layers (Fig. 3).
- **Refusal / binding:** none. Facts are not the focus.
- **For us:**
  - The training-free, layer-selective counterpart of our steer channel and of the Memory Inception idea already in the paper index.
  - Its size-normalised bank evidence is a fourth attention-mass knob.
  - Layer-selective attachment is relevant to the KBLaM finding that early-layer memory acts as "instruction". Restoring a slice only at some layers is a cheap licence experiment we have not run.

### 2.13 Bottlenecked Transformers (Oomerjee, Fountas, Bou-Ammar, Wang, arXiv 2505.16950 v4, ICLR 2026)

**Read:** §1–6.2 (method and main results) and the ablation headings. Skipped the proofs.

- **What it is:**
  - A "Cache Processor" (one small transformer block per backbone layer, 59–134M parameters in total) is invoked at each newline during generation.
  - It rewrites in place the KV of the most recent step ("consolidation") and of the top k = 32 earlier entries by attention mass with that step ("reconsolidation").
  - The rewrite is a gated residual: k ← k + σ(g)·Δk (Eq. 12, p.7).
- **Trained / frozen:** the backbone is first SFT'd for one epoch on 128k OpenMathInstruct-2 items, then **frozen**. Only the processor is trained, by next-step cross-entropy on the rewritten cache, with BPTT truncated at step boundaries.
- **Unit:** within one sequence; no cross-session memory.
- **Key numbers (Table 1, p.7):**
  - Llama-3.2-1B SVAMP 38.0 → 44.6; GSM8K 29.80 → 32.97.
  - Llama-3.2-3B GSM8K 46.78 → 51.33.
  - Losses on Gaokao-MathQA, which is in Chinese.
- **Refusal / binding:** none.
- **For us:** a trained module that edits a frozen model's KV entries in place, with selection by attention mass. A "sleep-time slice rewriter" could be built this way: rewrite the 2–8 kept entries so that they also carry the licence or binding signal, without touching weights. It sits between candidate 2 and candidate 4. This is an inference; the paper does nothing cross-session.

---

## 3. What is still open / ours

**Not found in any of the 13 papers**

- **Content vs licence measured separately on the same items.**
  - Compression-Aware Abstention shows evidence survives but is not used (4/97). Attention Matching notes "the text does not mention..." outputs at high compaction.
  - Neither scores teacher-forced and generated answers on the same items. Neither separates refusal mass from value probability (our +1.31 nats from adding the subject's position with the value probability unchanged). Neither links the gap to attention mass or to a generic comply direction.
- **Partner swaps between separately stored memories.**
  - Composition is measured only as accuracy or perplexity loss (KVLink, Cartridges, AutoCompressors, ICAE).
  - Memory3 uses the "everything at the same positions" layout that produced our swaps, and does not measure swaps.
  - KVLink's slot-indexed link tokens are the only trained mechanism that encodes order. Whether a frozen backbone plus a small trained slot signal reproduces our storage-order fix (N = 3 clean, half the swaps at N = 5) is open.
- **An amortised leave-one-out selector (candidate 1).** All selectors here use attention or mass fitting: Memory3 top-8, Attention Matching OMP, KVzip-style repeat-prefill. Training a one-pass predictor of S-loo's picks has no precedent in this cluster. Attention Matching's OMP + β + fitted values is the strongest training-free baseline to beat.
- **Verification of a retrieved wrong-but-plausible memory (candidate 3).**
  - KBLaM trains refusal for irrelevant questions. Compression-Aware Abstention trains it for evicted evidence. TokenMem trains the opposite.
  - None tests "a relevant-looking wrong slice was restored", where our system gives a confident wrong fact.
- **Recurrent state.** Our DeltaNet finding (recognition without recall, the state does not bind) has no counterpart here.
- **Budget scale.** Our memories are 2–8 positions shared across layers. The closest is Memory3: 8 tokens per head, chosen per head. Everything else compacts 1–100× of long documents.

**Practical leads from these papers, cheapest first**

1. **β / mass correction on restored slices, untrained.** Scale each kept entry's attention mass up to the turn's mass, as in Attention Matching's β. Alternatively, try a log-shift as in KBLaM or a separate softmax as in TokenMem. Then measure refusal mass and judged correctness on LongMemEval. If refusals drop without wrong answers rising, the licence gap is largely an attention-mass effect, and candidate 2 can be a learned scalar per entry or per layer.
2. **Layer-selective restore** (Memory Inception; KBLaM's early-layer "instruction" result). Restore slices at all layers vs only late layers vs adding an early-layer "you were told this" entry, to localise the licence.
3. **A CAA-style adapter for slices.** A LoRA or a gate trained with the two-term loss on our own slices:
   - answer when gold@k is restored;
   - abstain when a non-gold or wrong slice is restored;
   - train with the real restored cache, not text, and mix selectors (S-dream, S-loo, attention).
   - Use KL to a full-turn teacher, not NLL, following xRAG and Cartridges.
   - Measure resilience (keeping correct answers) as well as accuracy.
4. **A slice cartridge instead of slice → LoRA (candidate 4).**
   - Train a few KV entries per memory, initialised from the S-loo slice, with the sink frozen, by KL to a full-turn teacher on dreamed questions.
   - It stays composable and deletable per user. Cartridges' prefix-vs-LoRA ablation (MMLU 54.3 vs 45.3 at ~1 GB) is the reason to prefer it.
   - The open question is whether such a cartridge also fixes binding when trained jointly over a user's N memories.
5. **A slot or link signal for binding.** A trained, slot-indexed link entry (KVLink-style) over a frozen backbone, tested on our swap benchmark against the storage-order fix.

**Caveats on my reading**

- KBLaM's main numbers are in figures only; the claims above are what the text states.
- In AutoCompressors Tables 1–2 and Attention Matching Fig. 5–6, the numbers did not survive text extraction. I used only what the text says about them.
- The abstention paper's split of base-model failures (refusal vs wrong) under compressed-cache decoding is not in the sections I read.
- Cartridges was read in its v3 (June 2025) arXiv version. The ICLR 2026 version may differ.
