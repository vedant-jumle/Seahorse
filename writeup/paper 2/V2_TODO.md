# Paper 2, next arXiv version: related-work revision after the literature check

*Written 2026-11-05. Paper: "What Binds a Memory? Cache Slices, Recurrent States, and Why Separately Stored Facts Swap" (`writeup/paper 2/main.tex` + `refs.bib`). v1 was submitted to arXiv and is awaiting approval. These changes become the next version (v2) or a replacement of the pending submission: your choice. Nothing has been uploaded.*

Source of the changes: `docs/dreaming/literature-select.md` (2026-11-05), sections "The short version", claim 1 (MemArt), §4 and §5 "Claims to drop or soften".

**Ground rules kept:** no result, number, figure or experiment changed. Only positioning text and citations.

## To review before uploading

- [ ] Read the six changed passages below.
- [ ] Check the MemArt entry (authors come from two citing papers; see "BibTeX" below).
- [ ] Decide: v2 of the approved paper, or replace the pending submission.
- [ ] Upload `writeup/paper 2/arxiv/arxiv_source_v2.zip` (git-ignored). Suggested comment if it is a v2: "v2: adds related work on KV-cache memory across sessions (MemArt) and query-agnostic KV compression; no results changed."

## Build

- `latexmk -pdf main.tex` and the anonymous build (`-jobname=main_review`, `\REVIEW` defined): both clean, no undefined references or citations.
- **Pages: 9 before, 9 after** (both builds). The main text now ends on page 7 instead of page 6 (the Conclusion moved over); references and appendix still fit in 9 pages.
- arXiv zip: `arxiv/arxiv_source_v2.zip`, same layout as paper 1's (`main.tex`, `main.bbl`, `acl.sty`, `acl_natbib.bst`, `figures/fig2–5.pdf`). Test-compiled from a clean folder: 9 pages, no undefined references.
- No paper-2 v1 source zip was found (no `writeup/paper 2/arxiv/` folder existed before). If you need the v1 source, it is `main.tex` / `refs.bib` at commit `09c4ee6`.

## Changes (before → after)

### 1. Abstract, first two sentences
- **Before:** "… by storing part of the model's own key/value cache and putting it back later. We ask which part carries the fact, and what goes wrong when several such memories are used together."
- **After:** "… by storing part of the model's own key/value cache and putting it back later, as recent KV-cache memory systems do. We ask which positions carry the fact, and what goes wrong when several such memories are used together."
- **Why:** carrying KV across sessions is prior work (MemArt); the abstract should not read as if the idea were ours.

### 2. Introduction, second paragraph
- **Before:** "… systems work reuses cached chunks for speed \citep{…}. Our question is different: which part of the cache carries a fact, what a hybrid model's recurrent state adds, and why combining separately stored memories fails."
- **After:** "… systems work reuses cached chunks for speed \citep{…}. Stored KV reused across sessions as memory is not new either: MemArt keeps past turns as 16-token KV blocks and puts back the blocks the current question attends to, without training \citep{zeng2026memart}. We do not propose a memory system. We ask which few positions of the cache carry a fact, what a hybrid model's recurrent state adds, and why combining separately stored memories fails."
- **Why:** says plainly that cross-session KV memory is prior work and what this paper adds.

### 3. Related work, "Binding in context", last sentence
- **Before:** "… show that separately cached facts share an ID, and locate the entity side at the subject's positions."
- **After:** "… show that swaps between separately cached facts come mostly from a shared order tag, and locate the entity side at the subject's positions."
- **Why:** softening to match §5 ("Storage order explains most of the swaps; a smaller part is generic") and the mixed pre-registered verdict (holds / partly / fails).

### 4. Related work, "Cache reuse" → "Cache reuse and KV memory" (rewritten)
- **Before:** "Prompt Cache reuses separately encoded modules and notes that independence between them matters; CacheBlend, EPIC, Hypic and LinearKV recompute part of each reused chunk to recover the cross-chunk attention it lacks; InfLLM retrieves cached blocks; KV-State Grafting restores full cached blocks across sessions. These works measure speed and task accuracy. We measure which part of a cached fact carries it, why separately cached facts swap, and that a storage-time order fixes it for small sets without recompute."
- **After:** "Prompt Cache reuses separately encoded modules, finds that gaps in position do not hurt, and notes that independence between modules matters; CacheBlend, EPIC, Hypic and LinearKV recompute part of each reused chunk, and KVLink trains link tokens, to recover the cross-chunk attention it lacks; InfLLM retrieves cached blocks; KV-State Grafting restores full cached blocks across sessions. The closest prior system is MemArt: training-free, it stores earlier sessions' turns as 16-token KV blocks, retrieves the top blocks by attention from the current prompt, re-encodes their positions and is evaluated on LoCoMo. These works measure speed and task accuracy, with blocks or chunks as the unit. We measure which single positions of a cached fact carry it, against the same tokens as text, and where the answer reads it. We also show why separately cached facts swap and that a storage-time order fixes this for small sets without recompute. We found no report of such swaps, although MemArt concatenates blocks from many turns."
- **Why:** MemArt as the closest prior system; position gaps tolerated (Prompt Cache) and loss from missing cross-attention (CacheBlend, KVLink); what stays ours (single positions, slice vs text, read site, swaps).

### 5. Related work, new paragraph "Choosing what to keep"
- **Before:** (none)
- **After:** "Query-agnostic compression chooses what to keep before the question is known, for reuse by later queries \citep{kim2025kvzip,kim2026epicache,devoto2025expected}; KVzip names personalised conversational agents as a use case. We propose no selector: our two positions are the value and the full stop, located from the known value, and we ask why those carry the fact."
- **Why:** the outdated contrast "eviction methods choose positions for a query in the same session" (`docs/dreaming/literature-exp1.md`, positioning line of 2026-10-21) is wrong. It was **not** in v1's `main.tex`, but v1 implied that the 2 positions were the paper's own way of choosing. This paragraph gives the accurate contrast. TaskPress and Attention Matching are not cited (also query-agnostic; add if you want the list complete).

### 6. Related work, "Trained and steered memories", last sentence extended
- **Before:** "Known-entity and refusal directions \citep{…} bear on why attention-only memories refuse."
- **After:** "… bear on why attention-only memories refuse. Under cache compression, models also often fail to use evidence that survived \citep{khodabandehlou2026abstention}, and compression can remove a refusal \citep{kim2025kvzip}; we trace most of our refusals to a generic comply gate."
- **Why:** "stored content present but not used" is partly known.

### 7. Limitations, one sentence added (after the oracle-gates / retrieval sentence)
- **Before:** (none)
- **After:** "The two positions are located from the known value, not chosen by a selector, and we do not compare with a KV memory system such as MemArt or with query-agnostic selectors."

### Not changed (checked)
- Contributions list: claims (what carries a fact, the hybrid split, the swaps and the storage-order fix) are the ones the check keeps as ours.
- Hybrid-states paragraph, Conclusion and Figure 1 caption: no novelty claim the check contradicts.

## BibTeX added (`refs.bib`, block "added for paper 2 v2")

| Key | Entry | Verified against |
|---|---|---|
| `zeng2026memart` | @misc, "KVCache-Centric Memory for LLM Agents", Zeng, Zuo, Lyu, Yang, Wu, Xu, Yu; OpenReview YolJOZOGhI; note "ICLR 2026 submission" | Title: the PDF (`papers/kv-memory-cross-session/MemArt KVCache-Centric Memory.pdf`, anonymous). **Authors: not in the PDF**; taken from the reference lists of Shkolnikov (arXiv 2603.04428, ref. 25) and Bio-MemArt (arXiv 2609.08566, ref. 1), which agree. **Year 2026** follows those two citations; the OpenReview posting date was not checked (OpenReview blocks automated access). The archived forum page lists it as a rejected submission; the entry says only "ICLR 2026 submission". |
| `kim2025kvzip` | NeurIPS 2025 | PDF (title, 6 authors, "39th Conference on Neural Information Processing Systems (NeurIPS 2025)"), arXiv 2505.23416 |
| `kim2026epicache` | ICML 2026, PMLR 306 | PDF (title, 5 authors, "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026"), arXiv 2509.17396 |
| `devoto2025expected` | arXiv 2510.00636 | PDF (title, 3 authors, v1 1 Oct 2025; no venue found) |
| `khodabandehlou2026abstention` | arXiv 2608.29934 | PDF (title, 2 authors, v1 30 Aug 2026) |
| `yang2025kvlink` | NeurIPS 2025 | PDF (title, 5 authors, "39th Conference on Neural Information Processing Systems (NeurIPS 2025)"), arXiv 2502.16002 |

## Elsewhere (not done here)
- `docs/dreaming/literature-exp1.md` line 373 still has the outdated positioning line; `literature-select.md` §5 already flags it.
- The website page's "the cache can be gated and moved into a new conversation" should cite MemArt (literature-select §5); needs your go-ahead to push.
