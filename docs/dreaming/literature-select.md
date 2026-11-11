# Automatic selection (dream_select) against the literature

*A focused reading of 21 newly filed papers, plus the ones already in the library, against the dream_select results (LongMemEval single-session-user, 64 items; Qwen2.5-1.5B, Qwen2.5-7B, Llama-3.2-3B; teacher-forced and judged generation). It has two parts. The first checks five claims about "KV-cache-centric long-term memory" that another AI assistant made. The second places our selectors and findings. Written 2026-11-05. The PDFs are in the parent folder's `papers/` (index: `papers/README.md`; new folder `kv-memory-cross-session/`). Page numbers are PDF pages. As in [literature-exp1.md](literature-exp1.md), I read extracted text (pdftotext), not figures. Anything shown only in a figure is marked as such. How deeply each paper was read is stated in its note and in the last section.*

Terms follow [literature-exp1.md](literature-exp1.md). "Slice" = cached keys and values (K,V) of a few positions, computed in the original context. "Selector" = a rule that picks which positions of a stored turn to keep. "Query-agnostic" = the selector never sees the question it will later serve. "Teacher-forced" (TF) = scoring log P of a given answer. "Generation" = sampled answers, judged.

Our numbers below come from `results/dream_select_20261104_0032/report.md` (the local 1.5B run) and `results/dream_select_cluster_20261104_0129/` (`summary.md`, `<model>/report_gen.md`, `<model>/tf/report.md`). All are at k = 4 unless stated.

> **Update 2026-11-10 (dream_compare Stage A).** Most of §5's reviewer baselines were run head to head (`experiments/dream_compare/`, `results/dream_compare_a_20261105_1822/summary_a.md`; 133 LongMemEval items, budgets matched in K,V entries, judged by three models; Qwen2.5-1.5B, Qwen2.5-7B, Llama-3.2-3B, Granite-4.2-30B; Gemma-4 never ran). What changed in this doc:
> - **The real KVzip was run, and it wins.** KVzip-g (its official global top-k over all layers and KV heads, no question) is the best selector on 3 of 4 models: judged k8 0.42 / 0.41 / 0.63 on the 7B / Llama / Granite (the whole turn 0.67 / 0.59 / 0.67). On Granite its mean is above even SnapKV given the real question (0.63 vs 0.58, unpaired). Real KVzip beats our S-recon on 3 of 4 (H1), so the "S-recon is a strawman of KVzip" warning in §3 was right.
> - **S-loo is no longer the best automatic selector.** It is level with the best baseline on the Qwens and worse on Llama (−0.17) and Granite (−0.24). The "claims we can keep" line in §5 ("S-loo … is the best automatic selector we tested") and the matching sentence in §6 are withdrawn.
> - **"S-loo beats gold@k" holds on the 7B only**, at k8, now with a paired test: +0.15 [0.03, 0.27]. On the 1.5B (−0.06), Llama (−0.01) and Granite (+0.03) the paired difference is not significant. The §3 table, §4 item 3 and §6 should be read with that restriction.
> - **Content vs licence (§4 item 1) generalises:** the teacher-forced score overstates judged generation for every per-head baseline on all 4 models (H3). But it is not inevitable: KVzip-g's gap is small on the 7B (TF 0.63 vs judged 0.62) and Llama (0.73 vs 0.69), large on the 1.5B (0.59 vs 0.21).
> - **Per-head vs shared positions (§5) was tested:** per-head beats shared on 2 of 4 models (7B, Granite; H2).
> - **Baselines run:** the real KVzip (g / per-head / shared), SnapKV with dreamed questions and with the real question (the ceiling), H2O, VATP, S-dream × value norm (exploratory), Expected Attention, StreamingLLM; MemArt-style blocks in Stage B (P5). **Not run:** LORE-KV, ContextCite's surrogate, EpiCache's patched prompt, KeyDiff, Attention Matching's β (and the key-norm trivial baseline).
> - Unchanged: the claims-check verdicts, the punctuation reading, storage-order collisions as still new, and the positioning on 2–8 shared positions. Current state: [HANDOVER.md](../../HANDOVER.md) §1; catch-up: [docs/catchup/2026-11-10/catchup.md](../catchup/2026-11-10/catchup.md).

---

## The short version

**The pasted claims**

| # | Claim | What is real | Verdict |
|---|---|---|---|
| 1 | MemArt, "KVCache-Centric Memory for LLM Agents", OpenReview `YolJOZOGhI`: a "Token Warehouse" of past sessions' KV, with non-contiguous blocks retrieved for the current query, as long-term personalisation | The paper exists under that title and ID. It was an ICLR 2026 submission, and the archived forum page lists it as rejected. There is no arXiv version. It stores conversation turns as 16-token KV blocks and retrieves top-k blocks by attention from the current prompt. The store is called a "memory pool"; the phrase "Token Warehouse" does not occur. The main benchmark is LoCoMo, with PersonaMem in an appendix. | **Mostly correct** (wrong name for the store; not a published paper) |
| 2 | "Virtual tokens / soft persona prompts" (user traits as a few KV pairs), arXiv 2603.04428 | 2603.04428 is "Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices" (Shkolnikov 2026). It saves each agent's *whole* KV cache to disk in 4-bit form and reloads it. Its own words: "The contribution is speed, not accuracy." There are no virtual tokens and no persona. | **Wrong citation.** The idea as described was not found as a KV method. Trained user soft prompts exist (PERSOMA, User-LLM, PPlug; found in search, not read). |
| 3 | KEEP, "KV-Cache-Centric Memory Management", 2602.23592: "Static-Dynamic Grouping" and "Multi-hop Memory Re-computation" decouple positions from KV content so it can be reloaded across sessions | The paper exists at that ID (Yang et al., DAC 2026). It is about **embodied planning** (ALFRED), where a memory changes during one task. The two components exist, as "Static-Dynamic Memory Construction" and "Multi-hop Memory Re-computation". They regroup memory by how often it changes and recompute cross-attention between memory groups, CacheBlend-style. The text never mentions RoPE or decoupling positions. | **Partly** (the paper and component names are right; the purpose is wrong) |
| 4 | "RoPE breaks transplanted KV: injecting session-1 KV into session 2 gives gibberish" | No paper we read shows this inside the trained context window. Relative positions survive gaps and shifts (Prompt Cache), and quality loss comes from missing cross-attention (CacheBlend, KVLink). Degenerate output is reported in two settings: absolute positions far beyond the window (MemArt App. A.3, about 1M tokens) and a missing attention sink (StreamingLLM; Shepard 2026). | **Wrong as a general statement.** It holds only beyond the trained window or without a sink. Our R-orphan and shift results agree with the literature. |
| 5 | Raw KV keeps "nuance" better than text notes; injection gives "sub-millisecond realignment" | MemArt beats *summarised* text memories (Mem0, Zep) on LoCoMo. That is not a test of nuance. MemArt measures 5.5 ms for re-RoPE on a 14k context and 112–118 ms for top-k selection (Table 4, p.9). Our own R-full equals T-full exactly. | **Not supported** (nuance is not measured; the measured time is milliseconds, not sub-millisecond) |

**Our findings against the literature**

| Finding | Verdict | Closest papers |
|---|---|---|
| Dreamed questions as the selection query (S-dream) | **Known idea.** Synthetic or "future" queries for picking KV are a 2025–26 line of work. The closest test also found that self-generated questions add little over a repeat prompt, as ours do. | Attention Matching (self-study queries), Cartridges (self-study), LAQ (pseudo response), LORE-KV (sampled futures), EpiCache (patched prompts), Expected Attention (a Gaussian future query) |
| S-recon (attention while repeating the turn) | **Known** (KVzip), but our version is mean-pooled and the same for every layer and head, not KVzip's per-head maximum | KVzip; Attention Matching's "repeat-prefill" |
| S-attn (attention received in the turn) | **Known** (H2O family; the "context-prefill" query set) | H2O, TOVA, Memory3 |
| Attention selectors pick punctuation | **Known mechanism, new setting.** Sentence ends get high attention as segment summaries or sinks, and sink tokens have small value norms. | SepLLM, VATP, StreamingLLM, LLM-Microscope, Gather-and-Aggregate (already filed) |
| S-loo (leave-one-out on the model's own answers) beats attention | **Partly known.** Leave-one-out beating attention is standard in context attribution, and output-perturbation scoring is a current eviction line. An end-to-end, token-level KV leave-one-out on answers to self-generated questions is not in what we read. | ContextCite, LORE-KV, VATP, Expected Attention |
| S-loo beats gold@k on the 7B in generation (0.30 vs 0.21 at k = 4; 0.56 vs 0.41 at k = 8; not a paired test) | **Not reported elsewhere** (no paper compares against the answer span as a selection) | Compression-Aware Abstention ("evidence survives" ≠ used) |
| Content without licence: gold closes 0.99 / 0.88 / 0.67 of the gap teacher-forced, but its generation accuracy is 0.17 / 0.49 / 0.24, with refusals | **Partly known.** Compressed-cache decoding often fails to use evidence that survived (4/97 correct). Compression can also flip a refusal (KVzip Table 1). The same-item split between TF and generation, with refusal as the main failure, is ours. | Compression-Aware Abstention, KVzip Table 1, Query Visibility Audit |
| Storage-order collisions | **Still new.** The composition penalty is now documented more widely, but no paper reports value swaps or storage order. | Shepard 2026 (composition penalty, BOS sinks), CacheBlend, EPIC, MemArt (concatenates blocks without reporting swaps) |
| Cross-session use at 2–8 positions | **Partly known.** Cross-session, training-free, sparse KV reuse is now established at block level (MemArt: 2k–8k tokens in 8- or 16-token blocks). Single positions shared across all layers, at 2–8 entries, are not reported. | MemArt, AgentKVShift, EpiCache, Memory3 (trained) |

**The three closest papers overall:**
1. **MemArt** for the memory setting. It reuses KV blocks from past sessions in a frozen model, retrieved by attention, with positions re-encoded, on LoCoMo and PersonaMem. It is block-level, and its retrieval is keyed on the real question.
2. **Fast KV Compaction via Attention Matching** (Zweiger et al. 2026) for selection. It uses self-study questions plus repeat-prefill as reference queries, finds that self-study adds little, and adds a per-entry bias so that a compact cache keeps the original's attention mass.
3. **KVzip** for S-recon and the query-agnostic framing. It names "personalized conversational agents retaining user instructions and chat" as a use case (p.1), and its Table 1 is a compression-induced change in refusal.

---

# 1. The claims check

## Claim 1: MemArt

**The paper.** "KVCache-Centric Memory for LLM Agents", anonymous ICLR 2026 submission, OpenReview forum `YolJOZOGhI`, <https://openreview.net/forum?id=YolJOZOGhI>.
- **Authors** (from the reference lists of two papers that cite it, Bio-MemArt [1] and Shkolnikov [25]): Yuan Zeng, Pengfei Zuo, Min Lyu, Xingkun Yang, Huatao Wu, Yinlong Xu, Zhou Yu.
- **Status.** An archived copy of the forum page (Wayback Machine, 2026-04-08) has `venueid` "ICLR.cc/2026/Conference/Rejected_Submission". OpenReview itself blocked automated access, so the PDF filed here is the Wayback copy of `openreview.net/pdf?id=YolJOZOGhI` (17 pages, with appendices).
- **Not on arXiv** under that title or under "MemArt" (arXiv API search, 2026-11-05).

**What it does** (read in full, text only):
- **Store.** Each conversational turn's KV is kept as fixed 16-token blocks in a "memory pool", **pre-RoPE**: "we omit the rotary positional encoding (RoPE) and preserve only content-dependent KV cache" (p.6).
- **Index.** Each block's keys are compressed into an axis-aligned bounding box (min and max per dimension). The query–block score is ArkVale's upper bound on the dot product (p.5).
- **Retrieval.** Scores from every token of the *current* prompt, normalised per token (softmax or reciprocal rank) and then aggregated (max beats sum). One top-k set is used for all heads (p.5–6).
- **Reuse.** The retrieved blocks are concatenated in chronological order and re-rotated to compact positions 0…N, followed by the query (p.6–7).
- **Budgets.** Top-k = 128 blocks (Llama, Qwen3) or 256 (Qwen2.5), i.e. about 2k–4k tokens (p.7). The ablation runs 2k–8k selected tokens (Fig. 5).
- **Setting.** Training-free; "historical KV blocks from past sessions" injected before prefill (p.10). Known query: retrieval uses the real question.
- **Evaluation.** Generation on LoCoMo, scored by F1, BLEU-1, BERTScore and embedding similarity. MemArt is at 98.9–101.3% of full context and +11.8–39.4% over Mem0 and Zep (Table 2, p.8). PersonaMem multiple choice is in App. A.7 (p.17).
  - **Refusals:** only on LoCoMo's adversarial subset, with an "I don't know" prompt: 57 of 100 correct abstentions against 61 for full context (App. A.5, p.16).
- **Code:** "will be made publicly available" (p.10). The reproducibility statement points to OpenReview supplementary material. No public repository was found.

**Against the claim.** The architecture is described correctly: past turns as KV blocks, non-contiguous retrieval for the current query, long-horizon memory. Three details are wrong. "Token Warehouse" is not a term in the paper. It is a rejected submission, not a published paper. And "personalisation" appears only through the PersonaMem appendix.

**What it means for us.**
- MemArt is the closest prior work for the *memory* half of Seahorse.
- It removes the novelty of "a frozen model's own KV from an earlier session, selectively put back, training-free".
- It does not touch our token-level slices, the slice-vs-text comparison, the read site, the told/licence analysis or the collisions. It concatenates blocks from many turns and never reports swaps.
- Its retrieval (an attention upper bound on block summaries, keyed on the real question) is a direct competitor to our G-key retrieval (addendum 6B).

## Claim 2: "virtual tokens / soft persona prompts", arXiv 2603.04428

**What 2603.04428 is.** Yakov Pyotr Shkolnikov, "Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices" (arXiv 2603.04428, Feb 2026). Each agent's whole KV cache is saved to disk in 4-bit form and reloaded by character-level prefix matching. "Cross-phase context injection" extends a cached prefix with new text. Reloading speeds up time to first token 11–136×. Perplexity changes by −0.7% to +3.0% (p.13).
- It states: "The persistent cache contribution is latency, not accuracy: both re-prefill and cache reload produce the same attention state (modulo Q4 quantization error)" (p.12).
- It also says "No working memory quality metric … The contribution is speed, not accuracy" (p.13).

**Verdict.** The ID is real, but the paper does not compress user traits into a few KV pairs. **The idea as stated was not found as a KV method.**
- The nearest real work is *trained*: user embeddings as soft prompts (PERSOMA, arXiv 2408.00960; User-LLM, arXiv 2402.13598; PPlug). These were found by web search and not read or filed.
- Gist tokens and Cartridges (already filed) are the trained KV-side analogue.
- The 2603.04428 paper cites MemArt with the exact OpenReview ID [25]. This may be where the other assistant picked up the ID.

**For us:** the paper is background only. It shows that whole-cache persistence across sessions is ordinary systems practice, which agrees with our "R-full = T-full: full reinstatement is prompt caching across sessions".

## Claim 3: KEEP, arXiv 2602.23592

**The paper.** Zebin Yang, Tong Xie, Baotong Lu, Shaoshan Liu, Bo Yu, Meng Li, "KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning" (arXiv 2602.23592; the comments say DAC 2026).

**What it does** (abstract, §1, §4.1–4.2 by text search):
- **Setting.** Memory-augmented embodied planning on ALFRED. The memory (objects, states, past steps) changes during a task, so cached KV keeps going stale.
- **Static-Dynamic Memory Construction.** Memories are grouped by how often they change, so that an update invalidates and recomputes less.
- **Multi-hop Memory Re-computation.** "Importance propagation" finds the memory groups whose cross-attention matters for the query, and only those are recomputed. "Following CacheBlend … we recompute all memories in the first layer and gradually reduce the recomputation ratio" (§4.2).
- **Layer-balanced memory loading.**
- **Results.** 2.68× faster than text memory; +4.13% success and 1.90× lower time to first token against CacheBlend.

**Against the claim.** The ID, the title and the two component names are right. The stated purpose, decoupling positions from KV content for reloading across chat sessions, is not in the paper: the extracted text never mentions RoPE. It is a recomputation scheduler for one evolving episode.

**For us:** another instance of "separately cached pieces need recomputed cross-attention" (finding 5 class). It is not a cross-session memory.

## Claim 4: "RoPE breaks transplanted KV (gibberish)"

**What the literature says, by condition:**

| Condition | What happens | Source |
|---|---|---|
| Relative positions kept, absolute positions discontinuous | "output quality is not affected" | Prompt Cache (p.4, quoted in literature-exp1) |
| Positions corrected, separately encoded chunks | Quality still drops, because cross-attention is missing. The example output rambles. | CacheBlend (p.4) |
| Each chunk encoded from position 0 | Attention-sink artefacts at chunk starts; a wrong entity in the answer | EPIC (p.2, 4–5) |
| Position mismatch plus missing cross-attention | "up to a 35% relative decrease in accuracy on QA tasks" (citing prior work). The fix is re-encoding plus *trained* link tokens. | KVLink (p.2) |
| Original positions, history about 1M tokens (far beyond the window) | "catastrophic failure, resulting in degenerative, repetitive text". Fixed by storing pre-RoPE and re-encoding. One case study (Fig. 8, not viewed). | MemArt App. A.3 (p.14) |
| First tokens (sink) evicted | "the model collapses … even just evicting the KV of the first token". Positions are reassigned inside the rolling cache, which the authors call "crucial" when streaming beyond the window. | StreamingLLM (p.2, p.5) |
| Every BOS column deleted from composed caches | "destroys the memory outright" (accuracy 0.106) | Shepard 2026 (p.6) |
| Re-computation of some tokens (CacheBlend, EPIC, KEEP, AgentKVShift) | Repairs cross-attention, not positions | as listed |

**Against our data:**
- **R-orphan.** A slice harvested after an unrelated chat and used at its stored positions: clean use 0.80 on Qwen2.5 and 0.99 on the 2B with its state ([01-binding.md](01-binding.md)).
- **Shift does nothing.** Overlapping stored positions vs shifting the second slice after the first: 0.35 vs 0.37 confusion.
- **Our setting is safe on both counts.** All our positions are a few hundred at most, far inside every model's window, and the chat head (the sink) is always kept.

**Verdict.** "Gibberish" is not what transplanted KV does inside the trained window with a sink present. What we read reports quality loss from missing cross-attention, and that is the class our collisions belong to. Degenerate output needs absolute positions beyond the trained window (MemArt A.3) or a missing sink (StreamingLLM, Shepard). Re-encoding positions (KVLink, MemArt, StreamingLLM's rolling cache) matters for long histories and costs little (MemArt: 5.5 ms). We could adopt it as a safeguard for long-lived stores, but our results do not need it. **The claim as stated is wrong; it is right only in those two conditions.**

## Claim 5: "nuance" and "sub-millisecond realignment"

- **Nuance.**
  - MemArt's evidence is a comparison with *summarised and segmented* text memories (Mem0, Zep) and with H2O, not with the raw text of the retrieved turns. It attributes the gain to "high-fidelity retrieval" in latent space (p.2). Retrieval and content are not separated, and nuance is not measured.
  - Our own data: R-full = T-full exactly (max |Δlogit| 1.2e-04 on the 7B). A whole turn's KV is that turn's text, no more.
  - The only place where KV carries more than its text is a *partial* slice versus the same tokens as text (0.87 vs 0.27). That is because each kept entry was computed with the whole sentence in view (Memory3's Remark 3).
  - So "KV beats a summary" can be true, but it is a statement about summarising, not about KV.
- **Sub-millisecond.** Not found. MemArt's latency breakdown (Llama-3.1-8B, 14k context) gives re-RoPE 5.5 ms, top-k selection 111.7–117.5 ms and memory operations 29.9 ms (HBM) to 237.8 ms (DRAM) (Table 4, p.9).

**Verdict: not supported.**

---

# 2. Per-paper notes

Relation tags as in literature-exp1: **same**, **supports**, **adjacent**, **contradicts**, **background**. For each paper: what it does; its setting (within or across sessions; known or unknown query; trained or training-free; selection unit; budget); its evaluation (teacher-forced or perplexity vs generation; refusals); and what it means for us.

## A. KV as cross-session memory (`papers/kv-memory-cross-session/`)

**MemArt** (Zeng et al.; ICLR 2026 submission, rejected; OpenReview YolJOZOGhI). *Same setting as our memory half; adjacent to selection.* Details under claim 1.
- **Setting:** across sessions; known query; training-free; blocks of 8–16 tokens; 2k–8k tokens kept.
- **Evaluation:** generation (F1 and similar). Refusals only on the adversarial subset (57 vs 61 of 100).
- **For us:** the closest prior system, which must be cited and positioned against. Our granularity is single positions shared across layers. Our selection is query-agnostic and done at storage time, while MemArt's is query-aware and done at retrieval time. Our analyses (read site, licence, collisions) are absent from it.

**Bio-MemArt** (Qian et al., arXiv 2609.08566, Sept 2026). *Background.* MemArt plus a biometric access gate on each stored block, with LoCoMo QA for owner and non-owner probes. Useful mainly as an independent description of MemArt's pipeline: blocks with "compressed key", timestamp, and "decoupled position encoding" under a "unified position layout" (p.2–4). Read: abstract and §1–4.

**Agent Memory Below the Prompt** (Shkolnikov, arXiv 2603.04428). *Background.*
- **Setting:** whole-cache Q4 persistence per agent across restarts; exact-prefix reuse; no selection.
- **Evaluation:** time to first token, and perplexity on 512-token windows.
- **For us:** full reinstatement is engineering, as we already say. Read: abstract, §5–6.

**KEEP** (Yang et al., arXiv 2602.23592, DAC 2026). *Adjacent to finding 5.*
- **Setting:** within one embodied episode; known query (the current step); training-free; memory groups.
- **Evaluation:** ALFRED success rate.
- **For us:** recomputing cross-attention between separately cached groups is a known fix, close to our joint encoding. Read: abstract, §1, §3, §4.1–4.2 by search.

**AgentKVShift** (Pandey et al., arXiv 2607.21604, May 2026). *Adjacent to finding 5.*
- **What it does:** training-free KV reuse for retrieved agent memory units (A-MEM-style notes with summaries and tags). The reuse error of a separately cached unit "decomposes into a shared memory-level offset plus small token-wise fluctuations". It estimates the offset from a probe set and corrects every reused token. It refreshes 10–30% of the cache.
- **Setting:** across sessions (memory units retrieved per query); known query; chunk level.
- **Evaluation:** generation on LoCoMo (F1, ROUGE, BLEU) and AMA-Bench (GPT-4o judge). No refusals reported.
- **For us:** a cheap correction for separately encoded memories that could be tested against our storage-order fix. Read: abstract and results by search.

**EpiCache** (Kim, Kundu, Kim, Dixit, Cho (Apple), arXiv 2509.17396, ICML 2026). *Adjacent to S-dream; same benchmark.*
- **What it does:** training-free, query-agnostic KV management for long conversations under a fixed memory budget. It prefills block by block, clusters the history into episodes and compresses each episode separately. Importance is scored by the maximum attention from a "patched prompt" appended after each block: an auxiliary prompt that stands in for the unknown question.
- **The closest analogue of our dreamed questions:** an "Exact-Question" patched prompt (an oracle) is best, and patched prompts built from the utterances most similar to the question nearly match it (Fig. 2, p.3). It then uses each episode's representative segments as the patched prompt (p.4).
- **Evaluation:** LongMemEval (all seven question types), Realtalk and LoCoMo, scored by GPT-4o-mini. No refusal analysis found.
- **For us:** a published query-agnostic selector on our benchmark, and a natural baseline for S-dream (its representative-segment prompt vs our dreamed questions). Read: abstract, §2.3, §3.1 by search.

**What It Costs to Compose, Rebuild, and Correct Precomputed Memory** (Shepard, arXiv 2608.30647, 2026). *Adjacent to finding 5.*
- **Findings:** precomputed memories (saved KV caches and trained cartridges, Llama-3.1-8B) "degrade when assembled from separately prepared parts". About half of the composition loss comes from the extra BOS columns, and deleting every BOS column "destroys the memory outright" (p.6). It cites Hardalov et al. 2026: naive composition collapses to near chance at 20 cartridges (p.5). An update binds only if it names its question.
- **No swaps or binding found by text search.**
- **For us:** the composition penalty now has a careful measurement. It is still generic loss, not partner swaps. Read: abstract, §1, §4 by search.

## B. Query-agnostic and future-query selection (`papers/kv-cache-reuse/`)

**KVzip** (Kim, Kim, Kwon, Lee, Yun, Song, arXiv 2505.23416, NeurIPS 2025). *Same as S-recon in idea; adjacent to S-dream.*
- **Method:** the model reconstructs the context from its cache ("Repeat the previous context:", teacher-forced in one forward pass). A KV pair's importance is "the maximum attention score each KV pair receives during this forward pass", **per KV head**, maximised over grouped queries and reconstruction tokens (p.4). Eviction uses non-uniform head budgets, the top r% across all heads (p.6). A context-independent head-level variant is offered.
- **Setting:** within a session but reused across many queries; query-agnostic; training-free; KV pairs per head; ratios 0.1–1.0, near-lossless down to 0.3 (p.6–7).
- **Evaluation:** generation (greedy) on SQuAD, GSM8K, NIAH and SCBench.
- **Refusals:** in Table 1 (p.9), a 40% KVzip cache makes Llama-3.1-8B *drop* a privacy refusal ("I cannot provide personal contact information." → the phone number). The authors link this to shallow alignment.
- **Also:** reconstruction attention "notably overlaps with attention patterns from diverse tasks" (p.4).
- **For us:**
  - Our S-recon is KVzip-*like*, not KVzip. We average attention over heads, layers and reconstruction tokens, and keep the same positions for every layer and head. KVzip takes the per-head maximum and lets heads keep different positions.
  - A reviewer will want the real KVzip at matched entry budgets.
  - KVzip's Table 1 is the opposite direction of our refusals, and the same lesson: compression moves the answer/refuse gate independently of content.
- Read: §1–3 and parts of §4.

**Expected Attention** (Devoto, Jeblick, Jégou, arXiv 2510.00636). *Adjacent to S-dream.*
- **Method:** importance = (expected attention + ε) × ‖W_o v‖. Expected attention is computed in closed form from a Gaussian fit of future queries, averaged over the next T = 512 RoPE positions (p.4).
- **Setting:** query-agnostic by design: "For prefilling, we do not assume any question about the context" (p.6). Training-free; per KV pair.
- **Evaluation:** generation on LongBench, RULER, NIAH, AIME25 and MATH-500.
- **Also:** releases **KVPress**, which implements more than 20 methods (SnapKV, KVzip, TOVA, KeyDiff, …).
- **For us:** a "future query" estimated without generating any text. It weights by value norm, which is the direct fix for picking punctuation (see C). KVPress is the practical way to run baselines. Read: §1–2 and the setup.

**Lookahead Q-Cache, LAQ** (Wang et al., arXiv 2505.20334). *Adjacent to S-dream and S-loo.* The model generates a short pseudo-response under a low-budget cache, and its queries replace SnapKV's prompt-end observation window.
- **Setting:** known instruction (the question is in the prompt), within a session, training-free.
- **Evaluation:** LongBench, NIAH (+1 to 4 points).
- **For us:** "use the model's own answer as the probe" is LAQ's idea, here with attention as the score. Read: abstract, §1, §4.2 by search.

**LORE-KV, "Monte Carlo Estimation for KV Cache Eviction"** (Bilal et al., arXiv 2610.07643, Oct 2026). *Closest to S-loo.*
- **Method:** sample M short continuations from the frozen model. Score each prompt token by a **closed-form leave-one-out deletion effect on the attention output** for those continuation queries, weighted by trajectory reliability. "Eq. 5 shows why attention alone is insufficient: a token is important only when it receives non-negligible attention and its value meaningfully differs from the current attention output" (p.4). One continuation already gives about 89% of the gain.
- **Setting:** known query (the prompt includes the question; a 32-token prompt-end window and the first token are protected); within a session; training-free; per KV head; B = 128–1024 tokens.
- **Evaluation:** LongBench and RULER (generation).
- **For us:** S-loo shares the two ideas (the model's own continuations; leave-one-out instead of attention). It differs in three ways: our questions are dreamed rather than the real one, our leave-one-out is end-to-end on log P of the answer rather than local per head, and we keep one shared set of positions. LORE-KV is a cheaper approximation of S-loo to run head to head. Read: §1–3.4.

**Fast KV Compaction via Attention Matching** (Zweiger, Fu, Guo, Kim, arXiv 2602.16284, 2026). *The closest test of S-dream vs S-recon.*
- **Method:** builds a compact cache (chosen keys plus fitted values plus a per-entry scalar bias β) that matches the original's attention output **and attention mass** on reference queries. It is solved by OMP, NNLS and least squares, with no gradient training.
- **Reference queries:** "repeat-prefill" (KVzip's prompt), "context-prefill" (H2O-style) and **self-study** (the model prompted with the context and a few fixed prompts; "we then sample responses and extract the resulting query vectors", p.3).
  - "self-study yields the best downstream performance; context-prefill and repeat-prefill are nearly as good while being faster" (p.3).
  - A related-work passage in Li, McLeish, Chen, Kalra et al., "End-to-End Context Compression at Scale" (arXiv 2606.09659, p.32; not filed), summarises Attention Matching's mixed setting this way: "self-study has limited impact on the repeat-prefill prompt".
- **Also:** methods that only drop tokens "systematically underestimate the compacted block's contribution during future decoding" (p.3). The β bias corrects the lost attention mass.
- **Evaluation:** QuALITY accuracy and log-perplexity ablations; up to 50× compaction.
- **For us:**
  1. Our S-dream ≈ S-recon ≈ S-attn in teacher forcing matches their self-study ≈ repeat-prefill ≈ context-prefill.
  2. The bias term suggests a mechanism for our refusals that we have not tested. A slice of 4 out of 64 entries has much less total attention mass than the whole turn, so the memory may look "small" to the model. Restoring the mass (a β per kept entry) is a direct experiment on the licence problem. This is inferred.
- Read: §1–3.1, the ablation table, and the self-study passage.

**TaskPress** (Park & Hwang, arXiv 2608.03276). *Adjacent to S-dream.* A task description used as a "meta-query" during prefill, plus quantisation scale factors as an outlier signal; query-agnostic; LongBench. **For us:** a fixed task prompt as the probe, between S-recon (a fixed repeat prompt) and S-dream (generated questions). Read: abstract and tables by search.

**How Query Visibility Changes KV-Cache Compression Rankings** (Luo, Liang, Xuan, arXiv 2607.11942). *Background for evaluation; must cite.*
- **Findings:** under query-agnostic compression, only KeyDiff beats a best-of-3 trivial baseline consistently, and SnapKV loses to "keep the start and the recent window" (p.1). The drop from query-aware to query-agnostic follows how visible the question is to each score.
- **A hazard:** swapping sdpa for eager attention on an *uncompressed* model shifts RULER accuracy by −0.29.
- **For us:** we must include trivial baselines (start + recent window, key norm, KeyDiff) beside random and last-k, and state the attention backend. We use eager fp32 throughout and assert R-full = T-full, so the backend is fixed, but say so. Read: abstract, §1.

**SCBench** (Li et al., ICLR 2025, arXiv 2412.10319). *Background.* A shared-context, multi-request KV benchmark: "sub-O(n) memory methods suffer in multi-turn scenarios". It is the standard citation for "query-specific eviction fails when the cache is reused". Read: abstract.

## C. Why attention selectors pick punctuation

**StreamingLLM / attention sinks** (Xiao et al., ICLR 2024, arXiv 2309.17453). *Supports.* Initial tokens collect large attention "even if they are not semantically important", and removing them collapses the model (p.1–2). Substituting "\n" tokens for the four initial ones largely restores perplexity (p.4–5), so position matters more than meaning. It also has the positional point under claim 4. Read: §1–3.

**VATP, "Attention Score is not All You Need … Value Also Matters"** (Guo, Kamigaito, Watanabe, arXiv 2406.12335). *Explains our punctuation picks.*
- In Llama-2-7B, the attention-sink tokens have "massive attention scores" but value ℓ1 norms "close to 0" (Fig. 1 caption, p.2).
- The second sink "is often the token representing the first period (.) or newline token (\n)" (footnote, p.3).
- Scoring by attention × value norm beats H2O on 12 of 16 LongBench tasks.
- **For us:** attention mass on a sentence end is not evidence that the end carries content. Read: §1–3.

**SepLLM** (Chen et al., ICML 2025, arXiv 2412.12094). *Supports.* "seemingly meaningless separator tokens (i.e., punctuations) contribute disproportionately to attention scores", so segments can be condensed into their separators (abstract, p.1). The training-free variant keeps sinks, separators and neighbours. Read: abstract.

**LLM-Microscope** (Razzhigaev et al., arXiv 2502.15007). *Supports.* Punctuation, stopwords and articles carry "surprisingly high context". Removing them hurts MMLU and BABILong-4k. Read: abstract and §1.

**Already filed:** Gather-and-Aggregate (a Gather head "summarizes each segment to its last token", literature-exp1). Our dream_bind R-min keeps the full stop for this reason.

## D. Erasure and leave-one-out importance

**ContextCite** (Cohen-Wang, Shah, Georgiev, Madry, arXiv 2409.00729). *Partly the same as S-loo.*
- A sparse linear surrogate is fitted on random ablations of context sources (sentences) to predict the response's log-probability.
- The baselines include exactly our S-loo form: "a leave-one-out baseline that ablates each source individually and compute[s] the log-probability drop of the response". LOO "is an oracle for the top-k log-probability drop metric … when k = 1, but may be prohibitively expensive" (p.7).
- Attention "is not consistently reliable for context attribution" (p.7).
- The paper also prunes the context with the attributions to improve answers.
- **For us:** "leave-one-out beats attention" is expected. What differs: our ablation is of KV entries (not text), at token level, on the model's answers to its own *dreamed* questions, and is used for storage.
- **Cost:** ContextCite's surrogate (32–128 ablations) would cut S-loo's cost (one pass per position).
- Read: §1–4.

Also **LORE-KV** (B) and **Expected Attention / VATP** (value-aware scores) cover the output-perturbation side.

## E. Compressed context and refusal

**Compression-Aware Abstention** (Khodabandehlou & Krishnamachari, arXiv 2608.29934, Aug 2026). *Partly the same as "content without licence"; opposite emphasis.* Its concern is hallucination when evidence is evicted, and it trains a LoRA adapter to abstain.
- **The key observation for us:** under real compressed-cache decoding (KVPress's KVzipPress, Qwen), "when the answer-bearing evidence is labeled as surviving, the instruction-tuned base model still often fails to use it once generation proceeds from the compressed cache". 4 of 97 are correct overall, and 1 of 69 at retention 0.80 (p.7).
- Prompt-only abstention over-abstains on about half of answerable items (p.2).
- **For us:** "the evidence is kept but not used in generation" is documented. What is ours:
  1. The same items scored both ways. At k = 4, gold@k closes 0.72 / 0.43 / 0.54 of the gap teacher-forced (1.5B / 7B / Llama), but gives 0.11 / 0.21 / 0.21 judged correct.
  2. The failure is mostly refusal or "I need more information", not a wrong value.
  3. The link to a generic "answer personal questions" gate (addendum 8).
- Read: abstract, §1, §5.

**KVzip Table 1** (above): compression removes a refusal. The direction is opposite but the point is the same: the gate moves with compression, separately from content.

## F. Positions

**KVLink** (Yang, Hou et al., NeurIPS 2025, arXiv 2502.16002). *Supports the claim-4 verdict.* Stores KV without RoPE and re-encodes at the global position. Prior separate encoding lost "up to a 35% relative decrease in accuracy on QA tasks, which results from the discrepancy of position embeddings and missing cross attention" (p.2). Trained link tokens restore cross-segment attention. It ran Prompt Cache *with* its re-encoding as a baseline. Read: §1–2.2 and the baseline setup.

---

# 3. Where each selector sits

| Selector | Our definition | Closest prior | Same or different | What stays ours |
|---|---|---|---|---|
| **S-attn** | Mean attention a position receives from later content tokens in the turn; mean over heads and layers (not the sum, to avoid H2O's early-token bias) | H2O (accumulated attention), TOVA, Memory3 (attention received, per head); "context-prefill" in Attention Matching | **Same family.** The mean is our own guard against H2O's early-token bias, a bias AhaKV also describes (found in search, not read). | Nothing as a method |
| **S-recon** | Mean attention from the teacher-forced reconstruction ("Please repeat exactly what I just told you.") | **KVzip** (max per head; non-uniform head budgets); "repeat-prefill" in Attention Matching | **Same idea, weaker form** (mean-pooled; one position set for all layers and heads) | Nothing as a method. Our version is a strawman of KVzip until the real one is run. |
| **S-dream** | Attention to the turn from 3 questions the model dreams about it (question tokens + last prompt token), summed | **Self-study queries** (Cartridges; Attention Matching), LAQ's pseudo response, LORE-KV's sampled futures, EpiCache's patched prompts, Expected Attention's Gaussian future query, TaskPress's task guide; SnapKV with a substitute observation window | **Known idea.** Our form is SnapKV-like scoring with dreamed questions as the window. | The *wording* of the dreams (memory-quiz questions in the user's voice) and its use for a cross-session store. Attention Matching's finding (self-study adds little over repeat-prefill) matches ours (S-dream − S-recon not significant in teacher forcing). |
| **S-loo** | Drop each stored position from the whole slice; the drop in log P of the model's own greedy answer to each dreamed question, summed | ContextCite's LOO baseline (sentences, text ablation, real query); LORE-KV (closed-form local deletion cost on sampled continuations, real query); value-aware scores (VATP, Expected Attention, DropKV / CriticalKV, not read) | **Partly known** (the LOO principle and self-generated futures) | End-to-end token-level KV leave-one-out with **query-agnostic, dreamed** probes, chosen once at storage time and evaluated in a later session. The result that it **beats the answer span itself** in 7B generation is not in any paper we read. |
| gold / gold@k | The answer's tokens (+1) / k tokens around them (uses the label) | Compression-Aware Abstention's "evidence survival" masks; EpiCache's "Exact-Question" oracle | An oracle for *content*, not for use | Showing that the content oracle is not an oracle for generation |

---

# 4. Which findings are new, partly known or known

Strict reading: a claim is marked known only where we read it in a paper. Unread parts are named.

1. **Content vs licence: teacher forcing overstates; refusals under compressed memory.** *Partly known.*
   - **Known:**
     - Evidence that survives compression is often not used in generation (Compression-Aware Abstention, p.7).
     - Compression shifts refusal behaviour (KVzip Table 1, opposite direction).
     - Query-visibility protocols change conclusions (Luo et al.).
     - Perplexity is a weak proxy for generation quality (CacheGen; found in search, not read).
   - **Ours:**
     - The same 64 items scored both ways on three models. Gold closes 0.99 / 0.88 / 0.67 of the gap teacher-forced but gives 0.17 / 0.49 / 0.24 judged correct.
     - The main failure is refusal or asking for information, not a wrong value. Gold's refusal regex rate is 0.70 (1.5B) and 0.79 (Llama); the 7B's failures are mostly "I would need more specific information", which the regex misses (0.12).
     - The link to a generic comply gate (dream_bind addendum 8).
   - **Untested lead:** Attention Matching's mass-preserving bias as a fix.
2. **Attention selectors favour punctuation and summary tokens.** *Known mechanism; our measurement is new only as a setting.*
   - SepLLM (separators collect attention and condense segments), VATP (the period or newline is a sink with near-zero value norm), StreamingLLM (sinks), Gather-and-Aggregate (summary at a segment's last token), LLM-Microscope.
   - Our shares: 0.48–0.64 of S-dream and S-recon picks on the Qwens (0.24–0.35 on Llama), against a base rate of 0.09.
   - Don't present it as a discovery. Present it as "the known sink and summary-token bias, which costs a lot at budgets of 2–8 entries".
   - S-attn's share is low (0.04–0.12): its queries are the turn's own later tokens. S-dream's and S-recon's queries sit after the turn and read it from outside, as an aggregator reads a segment's summary at its end (Gather-and-Aggregate). This reading is inferred, not tested.
3. **S-loo beats attention; S-loo beats gold@k on the 7B.**
   - *LOO > attention: known in principle* (ContextCite; LORE-KV's Eq. 5; VATP).
   - *Beating the answer span in generation: not reported.*
     - 7B, k = 4: 0.30 [0.20, 0.40] vs 0.21 [0.12, 0.32].
     - 7B, k = 8: 0.56 vs 0.41, and vs gold's 0.49 (gold keeps the answer's own length + 1 entries).
     - These are unpaired comparisons from the tables. A paired contrast (S-loo − gold@k, verbatim items) is needed before stating it.
   - On the 1.5B and Llama, S-loo ≈ gold@k (0.10 vs 0.11; 0.15 vs 0.21 at k = 4).
   - Reading: S-loo keeps what lets the model *answer* (the answer plus whatever licenses using it), not only the value. That fits the licence story and is inferred.
4. **Storage-order collisions.** *Still new.* The literature now has more on composition penalties (Shepard 2026; Hardalov et al. 2026 cited there, not read; AgentKVShift's per-memory offset; KEEP's recomputation). None reports partner swaps, storage order or binding IDs. MemArt concatenates many past turns' blocks without any swap analysis. That is a place where our collision would bite, so it is worth saying in a paper.
5. **Cross-session use at 2–8 positions.** *Partly known.*
   - Cross-session, training-free, sparse, non-contiguous KV reuse exists at block level (MemArt: 2k–8k tokens; LoCoMo, PersonaMem).
   - Query-agnostic selection for later reuse exists (KVzip, EpiCache, Expected Attention, Attention Matching).
   - Not reported: single positions shared across all layers and heads, at 2–8 entries per memory, with the slice-vs-same-text comparison. The smallest budgets we saw are LORE-KV's 128 tokens per head (about 0.1% of a 128K prompt) and MemArt's 2k.

**Not read** (named in what we read, not filed): Cartridges' self-study details beyond literature-exp1; Hardalov et al. 2026; DapQ, GVote, AnDPro, LaProx, OBCache, DropKV, CriticalKV, KeyDiff, AhaKV; PERSOMA, User-LLM, PPlug; FINCH (Corallo & Papotti), the patched-prompt origin cited by EpiCache; ArkVale.

---

# 5. What a reviewer would demand

**Baselines to run head to head** (same 64 items, same anchors, matched by the number of stored K,V entries per layer and head, teacher-forced and judged):
1. **The real KVzip** (per-head max, non-uniform head budgets) through KVPress. Our S-recon is not KVzip.
2. **SnapKV with dreamed questions as the observation window** and **SnapKV with the real question** (a query-aware upper bound, EpiCache's "Exact-Question"). This separates "dreams are bad questions" from "attention is a bad score".
3. **Expected Attention** (query-agnostic, value-weighted) and **KeyDiff**. These are the strongest query-agnostic scores in the audit.
4. **Trivial baselines from the audit:** start + recent window (StreamingLLM) and key norm, beside random and last-k.
5. **A value-aware attention score** (attention × ‖W_o v‖, VATP-style) for S-dream and S-recon. This is the direct test of whether the punctuation problem is the whole gap.
6. **A LORE-KV-style score** (closed-form deletion effect on the dreamed-answer queries) as the cheap version of S-loo, and **ContextCite's surrogate** as a cheaper LOO. This is where the cost argument for S-loo is won or lost.
7. **EpiCache's patched prompt** (representative utterances) vs our dreamed questions, on LongMemEval.
8. **MemArt** for the retrieval stage when we test which turn to store. The code is not public (supplementary material on OpenReview only), but the method is fully specified (AABB keys, max aggregation, re-RoPE) and can be reimplemented.
9. **Attention Matching with β** on our slices, as a test of the attention-mass explanation of the refusals.

**Per-head vs shared positions.** Every method above selects per KV head, while ours keeps one position set for all layers and heads. A reviewer will ask whether our small budgets are small only because the unit is coarse. Report the KV-entry count and run at least one per-head variant.

**Claims to drop or soften:**
- That KV reuse across sessions as memory, training-free, is new (MemArt, AgentKVShift, EpiCache). The website page's "the cache can be gated and moved into a new conversation" must cite MemArt.
- That choosing positions without the query is new (KVzip, Expected Attention, EpiCache, TaskPress, Attention Matching). Paper 2's positioning line in literature-exp1 ("the eviction methods choose positions for a query in the same session") is out of date.
- That dreamed questions are a new selector (self-study, LAQ, LORE-KV). Present S-dream as an instance of synthetic-query selection, with a negative-ish result that agrees with Attention Matching.
- That attention picking sentence ends is a finding. It is the known sink and separator bias.
- "Gold is the upper bound." It is a content oracle only. S-loo beats gold@k on the 7B.
- Any "KV keeps nuance" or "sub-millisecond" wording.
- "Slices survive position changes" as a property of ours. It is Prompt Cache's, and the conditions where it fails (beyond the window, no sink) should be stated.

**Claims we can keep:**
- 2-position, training-free slices carry a fact across sessions and beat the same tokens as text.
- The read site and the subject position.
- Storage-order collisions and their fixes.
- The hybrid split.
- On selection: S-loo, an end-to-end token-level KV leave-one-out with dreamed probes, is the best automatic selector we tested. It beats the answer span in 7B generation (pending a paired test). Teacher-forced selection scores overstate generation, and the failure is licence, not content.

---

# 6. Positioning statement

Training-free KV memory across sessions is no longer an open idea. MemArt stores past turns as KV blocks and re-inserts the ones that attention from the current question selects, with positions re-encoded. Query-agnostic eviction (KVzip, Expected Attention, EpiCache) and synthetic- or future-query selection (self-study in Cartridges and Attention Matching, LAQ, LORE-KV) already cover "choose what to keep before the question exists". Seahorse's contribution is at a different grain and asks a different question. We keep two to eight of the model's own positions per memory, shared across layers, and show which ones carry the value and why. On natural LongMemEval turns, attention-based selectors, with or without dreamed questions, mostly keep sentence ends, the known sink and separator bias. A leave-one-out selector driven by the model's answers to its own dreamed questions keeps what lets the model answer, and on the 7B it beats the answer span itself in judged generation. Teacher-forced scores overstate all of these, because a compressed memory can hold the answer and still not license the model to give it. Separately stored memories also swap values according to the order in which they were stored, which no cache-reuse paper reports.

---

# Not read, partly read, or off-topic

- **Read in full or nearly (text):** MemArt (main + appendices).
- **Read: abstract and key sections:**
  - KVzip (§1–3, parts of §4, Table 1)
  - Expected Attention (§1–2, setup)
  - LORE-KV (§1–3.4)
  - Attention Matching (§1–3.1, ablations)
  - ContextCite (§1–4)
  - Compression-Aware Abstention (abstract, §1, §5)
  - KVLink (§1–2.2)
  - StreamingLLM (§1–3)
  - VATP (§1–3)
  - EpiCache (abstract, §2.3, §3.1)
  - Agent Memory Below the Prompt (abstract, §5–6)
  - Bio-MemArt (§1–4)
  - Shepard 2026 (abstract, §1, §4)
- **Read: abstract plus text search only:** KEEP, AgentKVShift, LAQ, TaskPress, Query Visibility Audit (abstract + §1), SCBench, SepLLM, LLM-Microscope.
- **Found by search, not read or filed:** PERSOMA, User-LLM, PPlug, CacheGen, AhaKV, DropKV, CriticalKV, KeyDiff, Hardalov et al. 2026, End-to-End Context Compression at Scale (only its related-work passage on self-study was read).
- **Method caveat:** all reading was of extracted PDF text. Figures (MemArt Fig. 8, EpiCache Fig. 2, the KVzip heatmaps) were not inspected. Their captions and surrounding prose were.
