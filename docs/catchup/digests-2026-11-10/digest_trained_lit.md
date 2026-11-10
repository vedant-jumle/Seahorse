# Digest: trained-side literature notes, read for the TRAINING-FREE branch

Sources (all read in full, 2026-11-10): `Seahorse/docs/trained/lit-A-trained-kv.md` (13 papers), `lit-B-read-paths.md` (16), `lit-C-weight-consolidation.md` (~12), `lit-D-new-search.md` (20 + unread list), `papers/README.md`, `papers/self-readout/NOTES.md`. Linked notes followed: `docs/dreaming/literature-sleep.md` (explicit link from lit-C), `docs/dreaming/literature-exp1.md` and `docs/dreaming/literature-select.md` (referenced by lit-B / README). Not followed further: `reference/literature.md` group F (FwPKM; lit-B says it only extends it), `core-idea.md`, `01-binding.md`.
Tags: [TF-test] = a training-free experiment we could run now; [num] = a number quoted from a paper; "(inferred)" = the note's own inference, not tested.

## 0. Our reference numbers (as quoted in the notes)
- Slice: 2 positions (name + full stop), gated → 0.86 clean use vs 0.87 full ctx; the same tokens as text 0.27. R-orphan 0.80 (Qwen2.5), 0.99 (2B with state).
- Licence gap (LongMemEval ssu, 64 items): gold closes 0.99 / 0.88 / 0.67 of gap teacher-forced (1.5B / 7B / Llama-3.2-3B) but is judged correct 0.17 / 0.49 / 0.24. Refusal regex on gold 0.70 (1.5B), 0.79 (Llama); 7B fails by "I would need more specific information" (regex 0.12). At k=4 gold@k closes 0.72/0.43/0.54 TF but 0.11/0.21/0.21 judged.
- S-loo beats gold@k on 7B gen (k=4 0.30 vs 0.21; k=8 0.56 vs 0.41, gold 0.49; unpaired). 1.5B/Llama: S-loo ≈ gold@k (0.10 vs 0.11; 0.15 vs 0.21).
- Attention selectors pick punctuation: 0.48–0.64 of S-dream/S-recon picks on Qwens (0.24–0.35 Llama), base rate 0.09; S-attn 0.04–0.12.
- Subject-word keys pick the right slice 0.85–0.92 at 59 facts. Swaps: partner confusion 0.26–0.37 (also 19–41% quoted); shift 0.35 vs 0.37 (no help); storage order fixes N=3, halves swaps at N=5.
- Adding subject position: +1.31 nats answering, value probability unchanged. "Told" direction (ctx − nomem) restores answering; it is ≈ generic comply.
- Hybrid 2B: KV-only slot-filling 0.50, direct questions 0.23 (refuses); state-only yes/no 0.93, balanced 0.96 vs form/associate lures; joint 2-fact state 0.50 balanced with partner as foil. Attn-only: value 10.0 nats but yes/no at chance; state-only: yes/no 0.93 but value 4.2 nats.
- Size: 2-position slice on Qwen2.5-1.5B = 2×2×256×28 = 28,672 values ≈ 57 KB; rank-8 all-proj LoRA ≈ 9.2M params ≈ 18 MB (≈320×); q,v r8 ≈ 1.1M (≈38×).

## 1. Training-free mechanisms and test ideas (exact numbers, IDs)

### 1a. Attention mass as the licence knob (the cheapest licence test)
- **Attention Matching β** (Zweiger, Fu, Guo, Kim, arXiv 2602.16284, ICML 2026). Compact block = kept keys C_k + scalar bias β per kept key (NNLS) + fitted values C_v (least squares), matching attention output AND mass on reference queries; logical length T kept (later tokens keep RoPE positions). Quote: dropping/merging without biases "systematically underestimate[s] the compacted block's contribution during future decoding" (p.3). β_j ≈ "how many original keys' worth of attention mass" key j stands for. "omitting attention biases and retaining the original values ... still yields a reasonable approximation" (p.8). Most important component in LOO ablation (Fig.5) = **non-uniform head budgets**; least = self-study and on-policy queries.
  - [num] QuALITY Qwen3-4B (Tab.10): AM-HighestAttnKeys 1/2/5% → 60.4/65.0/70.1 vs full 71.1, no ctx 36.8; KVMerger 43.4/49.4/55.1. QASPER F1 (Tab.1) 0.338 at 1% vs KVzip 0.127, full 0.444, none 0.104. RULER Llama-3.1-8B (Tab.11) 83.3 at 5% vs KVzip 46.6; 95.1 at 12.5% vs full 95.7. Online compaction AIME 13/30 with 2048 physical cache = plain decoding at 8192. Cartridges wins at 100× on LongHealth.
  - Cost (Tab.3, Gemma-3-12B, 60k ctx, H200): context-prefill 7 s, repeat-prefill 8 s, self-study 139 s, highest-attn selection 3 s, OMP 565 s, NNLS 2.2 s, LSQ 1.8 s.
  - App. G.1: at high compaction, outputs say "the text does not mention..." — read as lost content; lit-A suggests part may be lost licence (inferred).
  - [TF-test] Scale/refit mass of our 2–8 kept entries up to the turn's mass (β per entry, or per-layer gain), measure refusal mass + judged correctness. If refusals drop without wrong answers rising → licence gap is an attention-mass effect. Also: OMP+NNLS+LSQ is the closed-form selection baseline an S-loo must beat.
- **KBLaM log C − log M shift** (Wang et al., arXiv 2410.10450, ICLR 2025): constant added to KB attention scores so "the total contribution of the KB part stays roughly unchanged as M scales" (Eq.11; C=100). [TF-test] log-shift variant of β.
- **Memory Inception size normalisation** (Liu et al., arXiv 2605.06225): bank evidence normalised by bank size "so that a larger bank does not win routing weight merely because it contains more slots" (Eq.5); query-adaptive gains on target/reference banks. Training-free.
- **TokenMem separate softmax** (Yu et al., arXiv 2607.22625): in a shared softmax "the model's parametric prior competes with external evidence for a fixed attention budget" (p.4); separate softmax gives memory mass 1. (Their gate is trained; the separate-softmax read is a training-free variant to test.)
- **Gating Mechanism of KV Eviction** (Gu et al., arXiv 2611.07605): a constant in-softmax score cancels → learned gate collapses to no-op (RULER 4.33 vs 70.01 dense). Fix 1 is training-free in spirit: keep sinks + local window and give all other entries a **fixed shared gate mass** β (256 of ~8064 entries) → 44.86; KL to dense → 58.86; **per-head budgets set by retained attention mass** → 67.60. Evicted entries folded into centroid K/V slots *inside* the softmax match their contribution at cosine 0.85–0.96; memories read outside the softmax only −0.09 to 0.44. Scores comparable only within a head.
- **SleepGate attention-logit bias** b_i = β·log r_i, β=5 (Xie, arXiv 2603.14517): down-weights entries without deleting. [TF-test] apply a verification score as a soft logit bias on a doubtful slice rather than hard abstain.
- **Relevance to our KVzip global-budget result**: three papers say per-head / non-uniform head budgets are the biggest single factor (AM Fig.5; Gating 58.86→67.60; KVzip top-r% across all heads, p.6). Every published selector is per KV head; ours shares positions across layers/heads (reviewer will ask; lit-select).

### 1b. Layer-selective restore (where the licence vs the content lives)
- **KBLaM App. B (p.15)**: knowledge tokens only every 3 or 10 layers keep answer quality at K=3 but hurt refusal precision; "When K = 10, KBLaM fails to provide any refusal answer ... knowledge tokens in earlier layers may serve as soft instruction prompts that guides the LLM how to use the KB." Early-layer K,V vary little across triples (Fig.9). (inferred parallel: early layers = licence/instruction, later = content.)
- **Memory Inception**: slots attached only at selected layers/heads by a calibration selector (query alignment target vs reference bank, Eq.3); hard traits need ≥10 layers (Fig.3); KV savings 6–118×; Llama DAS 0.775 vs prompting 0.645, CAA 0.185; Qwen3 dialogue-shift 0.816 vs 0.438 / 0.526; PHYSICS 10/12 cells.
- **TransMem** (trained, but informative): last-4 layers best; mid layers 12–16 → 51.11 vs 54.11 LoCoMo F1; gate schedule 1.618/1.332/0.975/0.610 over layers 32→35.
- **TokenMem**: gate activation rises under conflict in 31/36 layers, peaks layers 10–12 and 19–23.
- **ROME**: edits generalise only at mid layers, last subject token; late-layer attention edits → "regurgitation".
- **Substrate Asymmetry** (2606.11712): zeroing per-user LoRA's q_proj at layers 21–35 raises absence detection 40.3→73.3, drops presence 42.0→22.3 (late layers carry "answer anyway").
- [TF-test] Restore slices at all layers vs late only vs early only, and/or add an early-layer "you were told this" entry; localise the licence.

### 1c. Subject position / known-entity mechanism
- **Ferrando, Obeso, Rajamanoharan, Nanda** (arXiv 2411.14257, ICLR 2025): SAE known/unknown-entity latents at the entity's last token (+ end-of-instruction tokens). Unknown latent steering → "almost 100% refusal" on Gemma 2 2B (Fig.3, p.6); known latent "slightly reduces refusal"; projecting unknown direction out of weights → "large reduction in refusal". Effects weaker on 9B and Llama 3.1 8B. α = 400–550 ≈ 2× residual norm (fn.5) — large pushes. Mechanism (p.7): unknown latent *reduces attention to the entity* in attribute-extraction heads L18H5, L20H3; known latent increases it — the licence gates attention, not content. Uncertainty latent (Gemma 2B IT, L13, at `model` token): AUROC 73.2, F1 72 (§7) for correct vs wrong.
  - Fits ours: subject position raises answering (+1.31 nats) with value prob unchanged; "told"/comply direction at the end is a second route. Our addendum 7: Ferrando's known-entity direction did NOT explain the attention-only refusal.
  - [TF-test] diff-of-means known/unknown direction on Qwen (no SAE suite); check if absent under attention-only slice; add at subject positions vs question end. Use end-of-instruction uncertainty read-out as a verification feature (modest alone).
- **Lindsey et al. 2025 "Biology of an LLM"** (URL): default "can't answer" circuit inhibited by known-entity features. **Arditi et al.** refusal = one direction. **Gottesman & Geva** (2406.12673): probe on entity states predicts knowledge. **Bielik** (2607.07670): familiarity decodable at 1.5B, separate from reliability; models almost never refused.
- **Lookbacks** (Prakash et al. 2505.14685): subject carries order-ID pointer, not the value → say "subject carries no copy of the value", not "nothing".

### 1d. Abstention / refusal base rates (to judge our 0.70–0.79)
- **Sufficient Context** (Joren et al., arXiv 2411.06037, ICLR 2025): text RAG with sufficient context abstains only **1.6–11.1%** (Tab.4a), hallucinates 3.2–25.4%; insufficient context abstains 50–73% (4b) and is still correct 35–62%. Autorater 1-shot Gemini 1.5 Pro 93.0%; "contains gold" 80.9%. Selective generation (LR on sufficiency + P(True)): >10% gain Gemma HotpotQA, none on Musique — can only ADD abstentions. LoRA r=4 IDK fine-tune (Mistral-7B, Musique): vanilla 28.8/11.8/59.4 (correct/abstain/halluc.); no-IDK 31.4/0/68.6; 20% IDK 23/2.2/74.8. → our refusal rate is slice-format-specific. [TF-test] score candidate answer with refusal suppressed (our 0.99 TF says the signal exists).
- **Do RALMs Know...** (Zhou et al., arXiv 2509.01476): over-refusal with all-irrelevant ctx vanilla 0.355, R-tuning 0.678, ICFT(n) 0.270; with evidence present vanilla over-refuses 0, R-tuning 0.066, ICFT(n) 0.171; context utilisation 0.738 → 0.591 (ICFT(n)) — refusal training cuts use of present evidence. **ΔU gate** (uncertainty with vs without context; training-free): over-refusal 0.770→0.282, acc 0.437→0.523 (irrelevant-ctx only). [TF-test] ΔU = slice vs no-slice / LOO contrast as licence signal.
- **Trust-Align** (Song et al., arXiv 2409.11242) baselines: Qwen-2.5-1.5B ICL answerable recall 99.18 / unanswerable 2.66; LLaMA-3-8B with refusal prompt answerable recall **2.13** (answers 1.48%); LLaMA-3.2-3B 1.64. Prompting swings ~100% ↔ ~0%.
- **Compression-Aware Abstention** (Khodabandehlou & Krishnamachari, arXiv 2608.29934): real compressed cache (KVzipPress), base Qwen2.5-7B correct on **4/97** items whose evidence survived (**1/69** at r=0.80): "still often fails to use it once generation proceeds from the compressed cache". Base honest acc 0.186. Prompt-only abstention instruction over-abstains on 49–51% of answerable at r=0.80. Labels need tight spans (median 3 tokens; paragraph spans → label entropy 0.000 vs 0.95). Controlled deletion §5.6: at equal length, KVzip top half → answers, bottom half → abstains; swapping only answer tokens raises abstention — behaviour follows evidence, not length.
- **KVzip Table 1** (2505.23416): 40% cache makes Llama-3.1-8B drop a privacy refusal — compression moves the answer/refuse gate independently of content (opposite direction to ours).
- **MemArt** (OpenReview YolJOZOGhI): LoCoMo adversarial subset, "I don't know" prompt: 57/100 correct abstentions vs 61 full ctx.
- **KBLaM** refusal (Fig.6c): 80 answerable + 20 unanswerable; precision falls as KB grows (over-refusal), recall constant; KBLaM degrades slower than ICL (ICL max 200 triples).
- **Larimar**: without scope detector, neighbourhood 80.4 → 13.7 (memory floods unrelated prompts).

### 1e. Teacher-forced vs generation (scoring hygiene)
- **Tip of the Tongue** (Ruscio & Thompson, arXiv 2603.13911): 23.2–41.8% of failures readable by logit lens; 27.2–30.8% by probe (shuffled 10.0–14.9%); setting answer projection fixes first token in 83.0–92.2% (random 0.0–0.6%) but full answer correct only **30.5–38.0%** (§5.6) — "decodability, recoverability, and generation" are separate. Covers wrong selection, not refusal.
- **EVAF** (Han, 2606.29916): CE improves (GPT-2 8.21→5.75) but persona-discriminating tokens in free generation = 0; "parametric encodes typology, retrieval delivers verbatim specifics". RAG CE 2.18.
- **Query Visibility Audit** (Luo et al., 2607.11942): query-agnostic protocol reorders methods; only KeyDiff beats best-of-3 trivial baseline; SnapKV loses to start+recent window; sdpa vs eager on an UNcompressed model shifts RULER by −0.29 → state backend; run trivial baselines.
- Rule (lit-B): score any licence fix in generation with judges, never TF/first-token only.

### 1f. Training-free verification / "was I told this?" signals (candidate-3 components)
- **Larimar ESD** (Das et al., arXiv 2403.11901): external MiniLM embedding, 1-NN cosine to stored facts: EER 2.9%, F1 0.974 (3,800 EasyEdit). Trained internal classifier (ISD) was worse on specificity: neighbourhood 15.7 (ep.4)/45.7 (ep.8) vs 80.6. → baseline: question embedding vs stored subject text + threshold.
- **RF-Mem** (2603.09250): familiarity = mean top-K similarity + entropy of softmaxed scores; thresholds θ_high/θ_low/τ; uses low familiarity to search harder (+~0.02 R@5 LongMemEval), never abstains. [TF-test] apply mean+entropy to our subject-key scores.
- **MEMOIR mask overlap** (2506.07899): overlap of top-k masks of averaged prompt activations decides whether to activate edits; ZsRE LLaMA-3-8B avg 0.95 (T=100), 0.93 at T=1000 (Rel 0.94/Gen 0.85/Loc 1.00); AlphaEdit 0.72 (loc 0.56), WISE 0.77.
- **WISE router** (2405.14768, trained margin): dropping router locality 1.00→0.72 at T=1000; WISE-Retrieve 0.73 vs oracle routing 0.82 at 3K — routing is the bottleneck.
- **Recurrent state as verifier** (ours + Afendulev 2609.04434): Afendulev recurrent-only TPR 0.93–0.98 but DRM lures accepted 0.87–0.89 (vs 0.01–0.10 unrelated), recombined compounds accepted ("features, not bound items"); ours rejects form/associate lures (0.96 balanced) on names — must recheck on non-name content.
- **Ferrando uncertainty latent** AUROC 73.2 (feature, not a gate).
- **Storage-Retrieval Gap** (2608.25489): similarity retrieval of per-entity LoRAs = chance (0.000 gain) → parametric memories need explicit keys; slices + keys stay as absence detector.

### 1g. Selection (S-loo vs attention) — training-free baselines and mechanisms
- **KVzip** (2505.23416): max attention per KV head from teacher-forced "Repeat the previous context", non-uniform head budgets (top r% across heads), near-lossless to 0.3 ratio; reconstruction attention "notably overlaps" diverse task attention. Our S-recon is mean-pooled, shared positions — "a strawman of KVzip until the real one is run".
- **Expected Attention** (2510.00636): (expected attention + ε) × ‖W_o v‖, Gaussian future query over next T=512 positions; query-agnostic; releases **KVPress** (20+ methods incl. SnapKV, KVzip, TOVA, KeyDiff).
- **VATP** (2406.12335): sinks have massive attention but value ℓ1 ≈ 0; 2nd sink "often ... the first period (.) or newline"; attention×value-norm beats H2O on 12/16 LongBench. → explains punctuation picks. [TF-test] value-aware S-dream/S-recon.
- **SepLLM** (2412.12094), **LLM-Microscope** (2502.15007), **StreamingLLM** (2309.17453; evicting even first token collapses; "\n" substitute restores), **Gather-and-Aggregate** (summary at segment's last token; newline as summary token).
- **LORE-KV** (2610.07643): sampled continuations + closed-form LOO deletion effect on attention output; "a token is important only when it receives non-negligible attention and its value meaningfully differs from the current attention output" (Eq.5); one continuation ≈ 89% of gain; per head, B=128–1024 tokens, known query. Cheap approximation of S-loo.
- **ContextCite** (2409.00729): LOO log-prob drop is oracle for top-k drop at k=1; attention "not consistently reliable"; surrogate with 32–128 ablations could cut S-loo cost.
- **AttriBoT** (2411.15102): LOO approximated >300× faster (caching, hierarchy, proxies). **ForesightKV** (2602.03203): ranker distilled from "golden eviction" (closest to amortising an oracle).
- **EpiCache** (2509.17396, Apple, ICML 2026): query-agnostic, patched prompt per block; Exact-Question oracle best, similar-utterance prompts nearly match (Fig.2); LongMemEval/LoCoMo/Realtalk — baseline for S-dream.
- **LAQ** (2505.20334): pseudo-response queries replace SnapKV window (+1–4). **TaskPress** (2608.03276): task description as meta-query. **SCBench** (2412.10319): query-specific eviction fails on reuse.
- **Attention Matching** finding = ours: self-study ≈ repeat-prefill ≈ context-prefill (S-dream ≈ S-recon ≈ S-attn TF).
- **TRIM-KV** (2512.03324, trained gates but useful as released baseline): LongMemEval_S Qwen3-4B query-agnostic, budget 32K/16K/8K/4K: 44.8/38.6/34.0/26.8 vs SnapKV 27.6/18.2/15.8/13.8, full 49.4 (judge = Qwen3-4B; Tabs 3/8 swap SnapKV/StreamingLLM). Some heads keep only full stops ("gist" heads), others numbers/recency → attention selectors may be right per head, wrong averaged.
- **Rethinking LoRA Memory** (2606.05698) uses **Compactor** (attention + leverage score, query-agnostic) — another selector to test vs S-loo.
- **Self-Study Reconsidered** (2606.32002): self-generated QA covers 75–94% but saturates on salient spans; seed overlap Jaccard 0.55–0.68 (Cartridges corpora), 0.39–0.46 longer; only point-lookup prompts select evidence reliably; generators obey injected instructions 88% → 13% with filtering → dreamed questions are a biased sampler; tie each to a target span.
- **Sleep-time Compute** (2504.13171): origin of anticipated questions; ~5× less test-time compute; 2.5× cheaper/query over 10 related queries; benefit tracks query predictability.
- Reviewer list (lit-select §5): real KVzip via KVPress; SnapKV with dreamed vs real question; Expected Attention; KeyDiff; start+recent window; key norm; VATP-style value-aware; LORE-KV; ContextCite surrogate; EpiCache patched prompt; MemArt retrieval; AM with β; ≥1 per-head variant; report KV-entry counts.

### 1h. Binding / composition of separately stored pieces
- **KVLink** (2502.16002, NeurIPS 2025) Tab.1, correct positions but separate encoding (PromptCache row): 1B NQ 44.6→18.6; 1B MuSiQue 13.8→1.4; 8B NQ 71.3→28.9; 8B MuSiQue 49.1→6.7. CacheBlend (18% recompute) 25.7/3.7/55.2/6.0; BlockAttention 39.0/14.3/70.8/38.7; KVLink5 45.0/19.2/72.5/40.8 (full FT). Link tokens slot-indexed: "for the n-th reused document, its link tokens are always link n-1, ..., link n-K" (p.25) = ordinal signal ≈ our storage-order fix. Separate encoding cost "up to a 35% relative decrease" (p.2).
- **Prompt Cache** (2311.04934): discontinuous position IDs fine if relative positions kept (p.4); "scaffolding" = encode modules together = our joint encoding.
- **CacheBlend** (2405.16444): position fix not enough, missing cross-attention (p.4); recompute ~15% high-deviation tokens; "Full KV reuse does work when the cross-attention between chunks is low".
- **EPIC** (2410.15332): chunk-start attention sinks (all chunks start at pos 0); recompute first k≤32 tokens of each later chunk. **HYPIC**: 8-token seam window. **LinearKV**: random chunk's state works as well.
- **Shepard 2026** (2608.30647): ~half of composition loss from extra BOS columns; deleting every BOS column destroys memory (acc 0.106); cites Hardalov et al. 2026: naive composition near chance at 20 cartridges.
- **AgentKVShift** (2607.21604): reuse error = shared memory-level offset + small token-wise noise; estimate offset from probe set, correct every token; refresh 10–30%. [TF-test] vs storage-order fix.
- **Memory3** (2407.01178): all memories at the same 128-length position interval ("parallel position encoding") = the layout that gave us swaps; never measured.
- **Feng & Steinhardt** (2310.17191): binding ID k = pair order; position-independent; close IDs → ~50% confusion (p.7). **Dai et al.** (2409.05448): order-ID subspace at query entity token; patching gives a causal handle. **Entity Tracking** App. J: enumeration "from the start token".
- **User as Engram** (2606.19172): same-template users collide, earlier falls to 8% ("last write wins"); joint optimisation reduces it — weight-side mirror of our swaps/joint fix.
- No paper reports value swaps / storage order (still ours).

### 1i. Positions / RoPE
- Transplant "gibberish" only beyond trained window (MemArt A.3, ~1M tokens) or without a sink (StreamingLLM, Shepard). Re-RoPE cost 5.5 ms (MemArt Tab.4; top-k 111.7–117.5 ms; memory ops 29.9–237.8 ms).
- Pre-RoPE storage: MemArt, Memory Inception, KVLink; **Doc-to-LoRA App. D**: generated keys pre-norm and pre-RoPE hold up much longer than keys output directly (supports storing slice keys pre-RoPE).

### 1j. Hybrid / recurrent state (training-free leads)
- **Afendulev et al.** (2609.04434): Qwen3.5-4B KV retrieve full 1.00 / recurrent-only 0.00 / KV-only 0.89; language-following 0.97 / 0.70 / 0.01; state-swap: value from KV, language from state ≥99%. "KV-only names the requested language correctly but answers in English." State = response mode → 2B licence may need state. Their KV-only verifies list membership 0.71–0.86 vs our 0.50 (discrepancy).
- **Lee, McLeish, Goldstein, Fanti** (2605.26099): "Converting observed tokens into useful weight memory ... need not be achievable in a single pass". Trained. [TF-test] loop the frozen 2B over the stored turn N times before keeping the state: does recognition become recall? Training lesson: hard eviction so memory path is used (§6.4).
- **DASC** (2608.30386) Qwen3-Next: zero full-attn KV → NIAH 0.000; zero state → 0.990. **LatentPort** (2609.25053): same-model restore exact; KV-only NLL 3.115 ≈ empty 3.246 vs native 2.161; +GDN 2.368 (translated)/2.291 (direct); wrong donor state worse than none (3.178 vs 2.842). Conv buffer installed with state — report whether our state-only arm included the 4-token conv buffer.
- **S0 Tuning** (2604.01168): state influence decays to 0.03% KL by prompt end yet 85% of corrected solutions diverge at first generated char; probe AUC 0.93. **WriteSAE**: out-of-context targets never reach top-1 in 20 tokens ("nudges, doesn't produce"). **Gated DeltaNet** names "memory collision" in state. **Log-Linear Attention**: pure GDN can do single-needle NIAH → our state-only 0.00 recall is division of labour, not capacity.

### 1k. Why a slice carries more than its text; what compressed carriers lose
- **Memory3 Remark 3** (p.13–14): encode whole document then chunk so K,V "attend to all their contexts" — reason slice (0.86) ≫ same text (0.27). Memory3 writing needs no training; keeps top-8 of 128 tokens per head by attention received, no causal mask/positions; key quantised to 8.75% with no degradation; "pretraining with explicit memories from the beginning would render the memories useless" (memories ignored unless trained; warm-up 3.1T tokens) = licence problem from pretraining side; +2.51 avg (63.31 vs 60.80); decoding 35.2% slower. Remark 2: retrieve with the model's own sparse attention queries (cheap key baseline).
- **Michalak & Abreu** (2510.19861): "Only–Keep" (attend only to needle tokens) 85.7% vs 50.3% unmodified — in-session analogue of R-min.
- **Gist** (2304.08467): exact details lost ("Culture" for "Arts & Culture"); **KBLaM**: fixed-length vector "may fail to precisely generate ... exact names or numerical values" (p.10) — keep value tokens as native slices. **ICAE**: random text restores at BLEU 0.2 vs normal 99.3; non-pretrained ICAE gives confident wrong values.
- **Jeong** (2603.22329): learned KV prefix at 1× retained score 0.02%, KV extension 0.0%, gated branch 0.09% vs XAttn 9.02%, Hebbian 9.23% — native KV beats learned format at low capacity; a gate in front of weak memory stays closed.
- **Cartridges init** (2506.06266, p.11): random vectors 29.9%, K,V of random text 51.3%, corpus's first-p-token K,V 55.3% LongHealth — real K,V are a far better starting point. Sink K,V must stay frozen (training it dropped MMLU <30%).
- **Rethinking LoRA Memory** (2606.05698): document LoRA adds ~0 when KV intact, +13 (NarrativeQA) / +15–21 (LongHealth) ROUGE-L with no KV; best as decode-time adapter over base-model KV (+7–11 / +13–17 at 99–99.5% compression); n=20 docs.

### 1l. Consolidation-side warnings relevant even if training-free
- **Useful Memories Become Faulty** (2605.12978): GPT-5.4 ARC-AGI 100% → 54% after consolidating from ground-truth solutions; episodic-only control competitive; "treat raw episodes as first-class evidence and gate consolidation" → keep slices.
- **Auto-Dreamer** (2605.20616): masking utility r_cf (drop ρ=0.5 at random, α=0.5) = set-level cousin of S-loo; Pattern 3: abstraction drops concrete locations. Untrained region rewriting already gives 6–11× smaller bank.
- **LightMem** (2510.18866): training-free text memory with offline sleep-time update, LongMemEval/LoCoMo +7.7%/+29.3% — text baseline. **MindMemOS**: LoCoMo 94.03, PersonaMem 70.63.
- **SleepGate**: 99.0/99.5/97.0% at 2/5/10 superseded updates, 73.5/30.5/16.5% at 15/20/30 (toy 793K) — PI-LLM task as test for "later wins".
- **Pepper et al.** (2602.10352): d+1 adapter bias ≈85% of gain = "default interpretation prior"; zero vector → confident wrong answer (Table 17). Same object as our "told ≈ generic comply": any licence push without content conditioning produces confident wrong facts.
- **Kim et al. 2026** (cited in SLEEP App. A.4, not read): teachers with privileged context suppress epistemic verbalisation, OOD drops up to 40%.

## 2. Per-paper one-liners (claim — relevance)
**Trained KV / compression (lit-A)**
- Cartridges 2506.06266 — trained KV prefix per corpus by self-study + KL; 38.6× less memory, 26.4× throughput at ICL quality; prefix > LoRA (MMLU 54.3 vs 45.3 at ~1 GB; in-domain 55.6 vs 47.25) — trained-slice arm; real-K,V init.
- Attention Matching 2602.16284 — closed-form compaction with mass bias β — top training-free licence test + selection baseline.
- KVLink 2502.16002 — pre-RoPE + slot-indexed link tokens (full FT) — quantifies separate-encoding loss; ordinal fix.
- Memory3 2407.01178 — 2.4B pretrained to read 8-of-128 sparse KV memories — closest published slice unit; parallel positions.
- KBLaM 2410.10450 — 1 KV token per triple via trained adapters on frozen Llama-3-8B; log C−log M; early layers = instruction — mass + layer tests.
- Compression-Aware Abstention 2608.29934 — LoRA r16 (10.1M) answer/abstain over evicted KV; base 4/97 → 24/97, 1/69 → 22/69; single-compressor cc adapter 42/69 confident wrong — closest to licence gap.
- Gist 2304.08467 — amortised compression into 1–10 gist tokens (full FT); win 48.6/49.7/45.8% — loses verbatim values.
- ICAE 2307.06945 — trained encoder, frozen decoder memory slots (BLEU 99.1–99.8) — composition fails until trained on concatenations.
- xRAG 2405.13792 — 1 projected token per doc; self-KD > NLL (44.58 vs 41.49; resilience 82.3 vs 76.2%); NLL → over-reliance on context — "resilience" metric.
- TokenMem 2607.22625 — separate-softmax cross-attn gates; trains blind compliance (cf_arc_easy 69.0/40.2/83.5 vs RAG ~19–20; Phase-1 only 2.4%) — channel unused until trained.
- AutoCompressors 2305.14788 — summary vectors fuse (perplexity only) — background.
- Memory Inception 2605.06225 — training-free layer-selective KV banks, size-normalised — layer test + mass knob.
- Bottlenecked Transformers 2505.16950 — trained processor rewrites frozen KV in place (SVAMP 38.0→44.6 on 1B) — "slice rewriter" idea.

**Read paths / gates (lit-B)**
- FwPKM 2601.00671 — fast-weight PKM; gate ≈0 under full attention, spikes on rare entities; re-reading NIAH <10% → >70% — remove other routes; re-reading ≈ joint re-encode.
- Larimar 2403.11901 — trained KV slot + scope detector; ESD EER 2.9%; paraphrase writes 88.4→93.6 — verification baseline.
- TransMem 2607.29032 — frozen backbone + last-4-layer block, evidence-only KL teacher; LoCoMo 40.79→54.11; excludes adversarial — teacher design.
- Trained Persistent Memory (Jeong) 2603.22329 — GPT-2; learned prefix fails at 1× — native > learned.
- Pepper et al. 2602.10352 — d+1 adapter 0.04% → 82.9% recall@1; bias = prior — comply-direction warning.
- SleepGate 2603.14517 — learned per-entry keep score as logit bias (toy) — amortised S-loo template.
- Lee et al. 2605.26099 — offline loops write GDN state (trained) — training-free loop test on 2B.
- S0 Tuning 2604.01168 — learned initial GDN state +23.6 pp (4B), +19.0 (2B); LoRA r64 −15.5 — state as adaptation surface.
- Afendulev 2609.04434 — KV = recall, state = mode/familiarity — hybrid split prior.
- Ferrando 2411.14257 — entity latents gate extraction-head attention + refusal — licence mechanism.
- Tip of the Tongue 2603.13911 — decodable ≠ generated (30.5–38.0%) — scoring hygiene.
- ReFT 2404.03592 — LoReFT 0.03% params, 80.2 vs LoRA 74.7; rank-4 at L15 last token (32,772 params) learns refusal persona from 5 examples in 30 s; rank-1 stores 256 ID→token pairs (13B); bias-only loses ~10 pts on arithmetic — licence adapter surface.
- Titans 2501.00663 — MAC: memory as attention entries works when trained with them — why our frozen model lacks licence.
- RF-Mem 2603.09250 — training-free familiarity switch — verification rule.
- Tensor Memory 2605.27686 — global gate; off-topic. MemoryLLM (Apple) 2602.00398 — FFN lookups; off-topic (not the memory-pool MemoryLLM).

**Weight consolidation (lit-C)**
- SLEEP 2606.03979 — distil fast blocks to low-rank experts + dreaming; "student ... weakly mimics" teacher; SQuAD 48.9 vs 35.7 w/o dreaming — framing only.
- SEAL 2506.10943 — self-edits LoRA; raw passage 33.5 vs base 32.7; rewrite prompt 49.4→55.6 — restatements matter.
- Sparse Memory FT 2510.15103 — NQ F1 drop −89% FT, −71% LoRA, −11% SMF — LoRA forgetting.
- SSR 2403.01244 — self-synthesised rehearsal BWT −0.79 vs −3.02 (10% real) vs −42.39 — interference set recipe.
- EVAF 2606.29916 — CE improves, specifics never generated — gist not specifics.
- ROME 2202.05262 — rank-one mid-layer edit; directional; FT neighbourhood 40.4 (GPT-2 XL), 10.3 (GPT-J) — bleed = swap analogue.
- MEMIT — batched 10k edits efficacy 98.9/para 88.6/NS 73.7 (unedited 83.5); sequential ROME collapses n≥32 — joint > sequential.
- Nested Learning/Hope 2512.24695 — background. Sleep-time Compute, Auto-Dreamer, Useful Memories — see §1l.

**New search (lit-D)**
- TRIM-KV 2512.03324 — learned write-time retention gates; LongMemEval numbers §1g.
- Gating Mechanism 2611.07605 — gate collapse + fixes (fixed mass, KL, mass-based head budgets).
- Knowledge Modules/DCD 2503.08727 — per-doc LoRA KL+hidden; NarrativeQA 12.5→27.3 (SDCD) vs in-ctx 38.0; random module −1 MMLU.
- Doc-to-LoRA 2602.15902 — hypernetwork <1 s/doc; SQuAD 0.825 of ICL; unrelated question 0.096 vs base 0.201; pre-RoPE keys better.
- PRAG 2501.15915 — per-doc LoRA r2 (~4.7 MB); Qwen2.5-1.5B 2WQA 0.3025 vs RAG 0.2431. DyPRAG 2503.23895 — hypernet LoRA; 1.5B gains within noise.
- Rethinking LoRA Memory 2606.05698 — §1k. DiSC 2602.16093 — split-context distil keeps IFEval 77.70 vs FT 65.83.
- Self-Study Reconsidered 2606.32002 — §1g. User as Engram 2606.19172 — per-user LoRA +1.784 bpb on unrelated text vs Engram +0.00005; collisions.
- Substrate Asymmetry 2606.11712 — per-user LoRA absence TPR 8.7% vs RAG 99.0% (presence 56.3 vs 35.3); internally inconsistent.
- DPPM 2610.11776 — per-user D2L + trained paths; PersonaMem-v2 54.22 vs full ctx 32.96; raw D2L 24.86 (worse than ctx).
- Memory Decoder 2508.09874 — kNN-LM imitator; domain-level; low relevance.
- Sufficient Context 2411.06037, Do RALMs Know 2509.01476, Trust-Align 2409.11242 — §1d.
- M+ 2502.00592 — co-trained low-dim key/query retriever; ~30% vs 3% random; no abstention. WISE 2405.14768, MEMOIR 2506.07899, Storage-Retrieval Gap 2608.25489 — §1f.

**KV reuse / cross-session (lit-select, lit-exp1)**
- MemArt (OpenReview YolJOZOGhI, rejected ICLR 2026) — past turns as 16-token pre-RoPE KV blocks, top-k by attention from current prompt; 98.9–101.3% of full ctx on LoCoMo; +11.8–39.4% over Mem0/Zep — closest system; removes novelty of training-free cross-session KV.
- Bio-MemArt 2609.08566, Agent Memory Below the Prompt 2603.04428 ("speed, not accuracy"), KEEP 2602.23592 (embodied recompute, no RoPE) — background.
- EpiCache, AgentKVShift, Shepard, KVzip, Expected Attention, LAQ, LORE-KV, TaskPress, Query Visibility, SCBench, VATP, SepLLM, LLM-Microscope, StreamingLLM, ContextCite — §1g/§1h.
- Lookbacks 2505.14685 (ICLR 2026) — payload + address at attribute token; read late (L52–56/80) — read-site mechanism. Retrieval Head 2404.15574 — <5% heads copy-paste.
- Hybrids: Michalak & Abreu 2510.19861; Gather-and-Aggregate 2504.18574 (Zamba2 MMLU 64.3→34.9 with 9 heads ablated); Component Ablation 2603.22473; Systematic Analysis 2507.06457 (ratio controls recall, <3:1); Hymba 2411.13676 (snapshot vs fading memory); State Soup 2406.08423; Marconi; Tail-Replay 2608.30310 (5–10% replay keeps 92.8–99.9%); Mamba-2-Hybrid "fuzzy memory"; Jamba; Cortes et al. 0710.1333 (familiarity capacity ∝ synapses, recall ∝ sqrt; one Hopfield net, two read-outs; file misnamed "Bogacz").

## 3. papers/README.md folder index (papers/ in parent dir; filenames unchanged; `.move-map.tsv` maps old→new)
- **steering-and-editing/** — activation steering/editing (ActAdd, CAST, BiPO, ReFT, RepE, Refusal in LLMs = Arditi one direction, Function Vectors, Multiple Behaviours, MAT-Steer, Compositional Attribute Steering, Neural Procedural Memory, steering-vector limits/unreliability).
- **facts-binding-interp/** — factual recall + binding mech-interp (ROME, MEMIT, Binding IDs, Representational Analysis of Binding, Entity Tracking, Lookbacks, Retrieval Head, Gather-and-Aggregate, tuned lens, Linearity of Relation Decoding, LLM-Microscope Punctuation, ContextCite).
- **introspection/** — Emergent Introspective Awareness, Emotion Concepts, Anxiety in LLMs.
- **self-readout/** — can a model describe its own vectors (Patchscopes, SelfIE, Pepper et al., LatentQA, Activation Oracles, Li et al. self-explanation, InversionView, Introspection is Content-Agnostic, Latent Introspection, LMs Fail to Introspect, Looking Inward); NOTES.md covers this + next two folders.
- **linear-representations-and-sae/** — additivity, LRH, superposition, SAE failure modes, detokenisation, erasure (Toy Models, LRH, Categorical/Hierarchical geometry, Not All Features 1-D, SAE Dark Matter, Absorption, Onion RNNs, Inner Lexicon, Token Erasure, Local Composition, MP-SAE, LEACE, Amnesic Probing, Sparse Feature Circuits, AxBench).
- **composition-and-binding-operators/** — TPR/HRR/VSA and composition (HRR, VSA Survey, RNNs≈TPRs, ROLE, TP-Transformer, Baroni-Zamparelli, Mitchell-Lapata, Bilinear MLPs).
- **memory-architectures/** — trained memory modules (Larimar, Titans, TransMem, MemoryLLM [Apple FFN, off-topic], Memory3, KBLaM, Trained Persistent Memory, RMT, FwPKM, Tensor Memory, Hopfield, MemGPT, InfLLM, EM-LLM, RF-Mem).
- **kv-cache-reuse/** — reusing/transporting KV across prompts (Prompt Cache, CacheBlend, EPIC, HYPIC, LinearKV, Marconi, Tail-Replay, LatentPort, DASC, SnapKV, H2O, Memory Inception, KV-State Grafting, KVzip, Expected Attention, LAQ, LORE-KV [closest prior for S-loo], Attention Matching, Query Visibility Audit, SCBench, TaskPress, VATP, StreamingLLM, KVLink).
- **kv-memory-cross-session/** — KV as cross-session memory (MemArt [closest prior for memory half], Bio-MemArt, Agent Memory Below the Prompt, KEEP, AgentKVShift, EpiCache, Shepard composition cost); notes in `docs/dreaming/literature-select.md`.
- **linear-attention-ssm-hybrids/** — DeltaNet, Gated DeltaNet, Based, Zoology, Jamba, Hymba, Mamba-2-Hybrid, Systematic Analysis, Component Ablation, RNNs not Transformers, Repeat After Me, Just Read Twice, Some Attention, State Soup, S0 Tuning, WriteSAE, Afendulev "What Attention Recalls..." [closest for hybrid split].
- **knowledge-awareness-and-refusal/** — Ferrando, Gottesman & Geva, Bielik, Tip of the Tongue, Kadavath P(IK), Compression-Aware Abstention; URL: Lindsey Biology.
- **context-compression/** — Cartridges, Gist, ICAE, AutoCompressors, xRAG, LLMLingua (text-compression baseline), TokenMem, SepLLM.
- **sleep-consolidation/** — SLEEP, Lee et al. offline recurrence, SleepGate, Bottlenecked Transformers, Nested Learning, SEAL, SSR, Sparse Memory FT, Sleep Replay, EVAF, Sleep-time Compute, LightMem, Auto-Dreamer, Useful Memories, SCM, MindMemOS, Generative Agents, A-MEM, Mem0, MemoryOS, HippoRAG; notes `docs/dreaming/literature-sleep.md`.
- **neuroscience/** — Backpropamine, Bogacz Familiarity Capacity (actually Cortes et al. 2007), neuromodulation papers, ANML.
- **eval-and-models/** — negated prompts, LLM-as-judge, Qwen3.5 tech report, dQwen3.5.
- **trained-memory/** — 20 PDFs from lit-D (PRAG, DyPRAG, Knowledge Modules, Doc-to-LoRA, Memory Decoder, Rethinking LoRA Memory, DiSC, Self-Study Reconsidered, User as Engram, Substrate Asymmetry, DPPM, WISE, MEMOIR, TRIM-KV, Gating Mechanism, Sufficient Context, Trust-Align, Do RALMs Know, M+, Storage-Retrieval Gap) + long URL-only list (RPMem 2609.23466 = most relevant unread for cross-session consolidation; δ-mem; SPARK; Locret; DMS; ForesightKV; AttriBoT; Reversal Curse; etc.).

## 4. self-readout/NOTES.md key findings (2026-10-19; abstracts + targeted search, not end to end)
- Working claims tested: (1) residual ≈ near-additive sparse sum + MLP AND-features; (2) binding = tensor product across positions, attention unbinds; (3) self-readout more reliable than introspection, works small; (4) descriptions report dominant concept, miss negation/sign.
- **Patchscopes** (2401.06102): training-free readout of NATURAL hidden states works at 6–13B (token identity beats vocab projection from ~L10; attribute extraction beats probes 6/12); forward pass can add info; no injected/negation tests.
- **Pepper et al.** (2602.10352): untrained SelfIE on contrastive topic vectors 0.04% recall@1 (0.9% @100) at Llama-3.1-8B; d+1 scalar-affine adapter 79.4%, full-rank 82.9% (98.4% @100); bias ≈85% of gain ("interpretation prior"); adapter labels beat auto-interp 70 vs 50% at 70B; LoRA oracle ≈ same. Kharlapenko 2024: narrow valid injection scale, wrong scale → fluent ungrounded text.
- **LatentQA** (2412.08686) trained readers work 1B–72B, gradients can steer; **Activation Oracles** (2512.15674) beat white-box baselines 3/4 but uncalibrated ("frequently makes incorrect guesses"); **Li et al.** (2511.08579) privileged self-access: Qwen3-8B best explainer of itself; untrained explainers random on patching outcomes. **InversionView** (2405.17653) inverse-image readout.
- Introspection: **Mahowald & Lederman** (2603.05414) 821 concepts, 235B/405B: detection 53.9%/31.7%, identification 1.3–13.9%, 0% false positives; confabulations frequent, calm, MORE POSITIVE ("apple" top); priming helps identification only. **Latent Introspection** (2602.20031): steering applied only while building first-turn KV cache; logit-lens detection in L50–60 fades in final layers; accurate framing raises P(yes) 0.3% → 39.9% (+0.6% FP); MI 0.61 → 1.05 bits of 3.17 — concept injected into CACHED states is partly readable later but masked by late layers. **Hu et al.** (2503.07513): no privileged self-access in 21 models via metalinguistic prompts. **Looking Inward** (2410.13787): fine-tuned self-prediction at ≥70B, narrow.
- Additivity (B): LRH + causal inner product (orthogonality of causally separable concepts, non-Euclidean); hierarchy = parent ⟂ (child − parent); some features multi-dimensional (circles; Engels); SAE dark matter ~half of error linearly predictable (>90% of norm); absorption makes decomposition non-canonical; small GRUs encode by magnitude; Inner Lexicon: FFN writes whole-word vector in 85% of words, 23% never decodable; Token Erasure: constituent info erased at last token of multi-token words; Liu & Neubig: affine best but idioms fit too.
- Composition (C): TPR fits RNNs but natural-data encoders ≈ bag of words (McCoy); ROLE causal; TP-Transformer had to ADD binding; LRE linear for some relations only; Function Vectors compose sometimes; summing steering vectors at one site fails (van der Weij), works if near-orthogonal and each at its own layer (Kang et al. 2609.08410); MAT-Steer gated orthogonal.
- Decomposition (D): MP-SAE for hierarchical; LEACE closed-form erasure as causal check; Amnesic Probing — probe accuracy ≠ use; AxBench (Gemma-2-2B/9B): prompting > fine-tuning > reps for steering; DiffMean best detection; SAEs not competitive; ReFT-r1 competitive.
- Verdicts: claim 1 mostly supported with qualifications (parts fade); claim 2 untested in LLMs; claim 3 half right (natural states yes; untrained readout of difference vectors ~0% at 8B, fixed by frozen d+1 adapter); claim 4 untested, hints: weak content → positive high-frequency default; ask polarity yes/no rather than describe. Open: compound vector = parts + AND; TPR binding in LLMs; sign/negation in free descriptions.

## 5. Brief map of the trained side (for orientation only)
- Four candidates: (1) amortised S-loo selector; (2) licence-aware read gate/adapter; (3) learned key + verification/abstention; (4) SLEEP-style consolidation (slices → LoRA, teacher = slice-answering model, dreamed questions, slices deleted). Extra suggested: (5) slot-indexed link entry for binding (KVLink-style).
- (1) Partly known: TRIM-KV, Gating Mechanism (KL to dense + fixed mass + per-head budgets), SleepGate, ForesightKV; nobody trains on LOO targets or at 2 positions/fact for transplant.
- (2) Partly known: Compression-Aware Abstention (two-term loss L_CONF + L_ABS; mix selectors; train on real cache), Trust-Align (DPO answer>refuse), KBLaM refusal; warnings: Pepper bias prior, TokenMem blind compliance, Do RALMs (refusal training cuts evidence use), Sufficient Context IDK instability. Suggested shape: LoReFT rank ~4 at restored subject+value positions, KL to full-turn text teacher, negatives (no slice / wrong slice / lure → "I don't know"), S0-style state offset on 2B, bias-only ablation, judged generation.
- (3) Key partly known (KBLaM, M+, Memory3 Remark 2); verification of a plausible WRONG slice: open everywhere.
- (4) Done for text teachers (Cartridges, KM/DCD, D2L, PRAG, DPPM); nobody distils from a restored KV slice; slice teacher refuses 0.70–0.79 and swaps 19–41% → filter or use full-turn teacher; expect 70–85% of teacher, no absence detection without negatives (Substrate Asymmetry 8.7%), damage to unrelated text; prefer trained-slice/cartridge (init from S-loo slice, sink frozen) or ReFT/S0 state over LoRA (S0: LoRA r64 −15.5 pp, 8/10 seeds negative); never delete slices (Useful Memories); retrain jointly, not sequentially (MEMIT, SEAL Fig.6).
- Not claimable: distillation of context into weights, self-generated data, KL-to-base anchors, joint > sequential, LoRA forgets more, masking utility, keep-episodes. Still ours: content vs licence on same items, partner swaps/storage order, amortised LOO selector, wrong-slice verification, DeltaNet-state findings, 2–8-position budget.
