# Where Seahorse stands (catch-up, 22 Oct – 10 Nov 2026)

*Text version of the catch-up page (https://claude.ai/artifact/ShvfnyfP6U5pdjYKMgUhnf). The page has the plots; every
plotted number comes from `results/` through `extract.py`. Numbers in this text were read from the same reports.*

## The short version

The mechanism still works: a stored slice of the model's own K,V, restored in a later session, brings a fact back, and
the whole stored turn equals the turn given as text. What changed is the picture around it. We moved from hand-written
facts to real memory benchmarks and compared our selectors and our memory format against published methods. As a
*system* it mostly went against us: published KV compression (KVzip) picks positions as well as or better than our best
selector, and storing cache instead of text buys no accuracy at much higher storage. As *understanding* it went for us:
the gap between "the content is there" and "the model is willing to use it" (the licence) shows up for every per-head
selector, and one method nearly closes it. On 7 Nov the work split into a training-free understanding branch and a
trained-parts branch.

**Headline findings**
1. **dream_subj (22 Oct): the subject's position decides whether the model answers.** Adding "dog" to the 2-position
   slice: answer score +1.31 nats [1.01, 1.63], first-token refusal 0.15 → 0.02; the value's probability unchanged.
   Teacher-forced, Qwen2.5-1.5B, 45 facts.
2. **dream_select (4 Nov): automatic selection beats random, but recovers little.** S-dream > random on 3/3 models TF,
   2/3 judged. k = 4 judged correct 0.07 / 0.11 / 0.15 vs whole turn 0.62 / 0.87 / 0.77 (1.5B / 7B / Llama).
3. **Teacher forcing hides refusals.** The answer's tokens (gold) close 0.99 / 0.88 / 0.67 of the gap TF but are judged
   correct 0.17 / 0.49 / 0.24, mostly through refusals or requests for details.
4. **Literature (5 Nov): cross-session KV memory and query-free selection are prior work** (MemArt, KVzip, Expected
   Attention, EpiCache). Still ours: 2–8 positions, slice vs the same text, storage-order swaps, the hybrid split,
   content vs licence on the same items.
5. **Stage A (9 Nov): KVzip-g is the best selector on 3 of 4 models; S-loo no longer leads.** Judged k8: KVzip-g 0.42 /
   0.41 / 0.63 on 7B / Llama / Granite. S-loo level on 1.5B and 7B, worse on Llama (−0.17) and Granite (−0.24). Licence
   gap (H3) holds on 4 of 4 models for per-head baselines.
6. **Stage B (partial, 8–10 Nov): stored cache does not beat text at equal retrieval.** Text RAG ≥ KV RAG by 0.01–0.05
   (pooled, 3 models). Sparse slices lose 0.1–0.3 on LongMemEval/LoCoMo for 7–58× less storage than full cache, still
   far more than text.

**Still running or broken (checked 10 Nov ~14:30)**
- Running: Granite Stage B PersonaMem continuation 1250454 (23:11 of 23:59; 505/589 items); 1250455 queued after it.
- Waiting: Granite judge / gpt-oss / Gemma-4 / analyze 1217775–78, then Stage B summary 1217779.
- Broken: Granite BEAM OOM 4× (1217774, 1250456–58); 64/320 items (4/20 conversations) written.
- Broken: Gemma-4-26B-A4B never ran (Stage 0 smoke identity tolerance on MoE routing; diag 1250442 OOM; repair stopped).
  Missing from Stage A and B. Its Stage B judge/analyze ran on no data; outputs deleted 7 Nov.
- Done: Stage A on 4 models; Stage B analysed for Qwen2.5-1.5B, Qwen2.5-7B, Llama-3.2-3B.

## The thread through the runs

When a memory holds the answer, what makes the model use it?
1. **dream_subj**: a slice needs the subject's position to be answered from (+1.31 nats; refusal 0.15 → 0.02; value unchanged).
2. **dream_select**: automatic selectors and even the answer span keep content but not use (gold TF gap 0.99 / 0.88 / 0.67,
   judged 0.17 / 0.49 / 0.24; refusal 0.70 on the 1.5B, 0.79 on Llama).
3. **Stage A**: the licence gap is general for per-head published selectors (H3 4/4), except KVzip-g with its global
   budget (7B TF 0.63 vs judged 0.62; Llama 0.73 vs 0.69).
4. **Stage B**: end to end, KV RAG ≤ text RAG (7B pooled −0.040), sparse slices lose 0.10–0.29 vs full cache and refuse
   more (+0.05 to +0.26); full cache costs ≈ 1–7 × 10⁴ × the text bytes.

The page puts dream_compare (Stage A and B) first as "the core", then the earlier runs.

## Background

- **Seahorse**: a training-free memory for a frozen LLM. Steer channel (paper 1: a push, not a store) and slice channel
  (the model's own K,V for a few positions, restored at stored positions before a later question). On the hybrid
  Qwen3.5-2B, the recurrent (DeltaNet) state is recognition; attention entries are recall.
- **R-full** = all positions of the turn (= the turn as text, ~1e-4 in logits). **R-min** = value + next token.
  **R+subj** = R-min + the subject's last token. **T-min** = the same tokens as text.
- **Teacher-forced vs generation**: TF reads log P(gold answer); generation is judged by Qwen3.5-9B, gpt-oss, Gemma-4
  (majority, ties to the original judge).
- **Gap closed** = (condition − nomem) / (R-full − nomem), on the TF score or on judged correctness.
- **Licence**: what makes the model answer from a memory rather than refuse. Licence gap = TF gap closed − judged gap
  closed. In dream_bind first read as a missing "told" signal, then (addendum 8) mostly a generic comply direction.
- **Storage order / binding**: separately stored slices are each "the first pair" and swap partners (0.19–0.37);
  storing the k-th memory as the k-th fact fixes N = 3, halves N = 5; the binding sits at the subject's positions.

Example (dream_subj, "My sister's name is Petra."): R-min, R+subj and R-full all give +10.88 nats on "Petra" TF, but
R-min answers "I'm sorry, but I don't have any information about your sister's name…" while R+subj answers "Yes, I
remember. Your sister's name is Petra."

## What we ran

### 22–30 Oct · dream_bind addenda 7–9 (context)
- 2B state = item recognition (rejects look-alike names); a joint state of two facts does not bind.
- "Told" is mostly a generic comply direction (2B 0.67 generic vs 0.76 told).
- Storage order is the collision: new fact stored second cuts confusion 0.33–0.41 → 0.09–0.15 (3 models). N = 3:
  0.38–0.47 → 0.04–0.09; N = 5: → 0.22–0.30 (fails).
- Llama-3.2-3B replicates slice > text (1.00 vs 0.24) and the collision.
- Source: `docs/dreaming/01-binding.md`, HANDOVER §1.

### 22 Oct · dream_subj
- **Question**: does answering a direct question need the subject's position in the restored slice?
- **Why**: R-min reached 0.86 on use prompts but 0.66 on direct questions; entity tracking predicts a two-step lookup.
- **Setup**: Qwen2.5-1.5B, TF only, 45 person/pet facts; measure = log P(answer prefix).
- **Result**: R+subj − R-min +1.31 [1.01, 1.63]; R+fill +0.07, R+rel −0.21; 81% of R-min→R-full; refusal 0.15 → 0.02;
  yes/no 0.82 → 1.00; R+subj − T-subj3 +3.53 nats. Verdict: "subject position matters (two-step lookup)".

| condition | answer score (nats) | refusal mass |
|---|---|---|
| nomem | −9.06 | 0.69 |
| T-min | −7.22 | 0.77 |
| T-subj3 | −4.49 | 0.51 |
| R-min | −2.28 | 0.15 |
| R+fill | −2.20 | 0.12 |
| R+rel | −2.49 | 0.14 |
| R-subj2 | −1.77 | 0.15 |
| R+subj | −0.96 | 0.02 |
| R-full | −0.66 | 0.02 |
| ctx | −0.94 | 0.03 |

- **Means**: the subject changes the decision to answer, not the value. Limits: TF only; primary 7 alone n.s.; cannot
  separate binding-ID lookup from "subject mentioned".

### 4 Nov · dream_select (local, then DelftBlue)
- **Question**: can the model choose the positions itself, without the question?
- **Setup**: LongMemEval oracle, 64 single-session-user items, evidence turn given. Selectors S-dream (primary), S-attn,
  S-recon, S-loo, random, last-k, gold@k. Local TF on 1.5B; cluster TF + generation + 3 judges on 1.5B, 7B, Llama.
- **Result**: S-dream > random TF on all 3 (LB 0.35 / 1.49 / 2.33 nats), judged on 7B (+0.081 [0.009, 0.162]) and Llama
  (+0.116 [0.034, 0.197]), not 1.5B (+0.034) → replicates 2/3. Primary rule (gap ≥ 0.5, beats S-attn) fails everywhere.
  Punctuation = 0.48 of S-dream's picks (base rate 0.09).

| k8 (gold: answer + 1) | TF gap closed 1.5B / 7B / Llama | judged correct 1.5B / 7B / Llama |
|---|---|---|
| gold | 0.99 / 0.88 / 0.67 | 0.17 / 0.49 / 0.24 |
| S-loo | 0.59 / 0.71 / 0.74 | 0.23 / 0.56 / 0.37 |
| S-dream | 0.39 / 0.48 / 0.63 | 0.16 / 0.32 / 0.34 |
| R-full | 1 | 0.62 / 0.87 / 0.77 |

- **Means**: the threshold for the cluster follow-up was met; a TF score is not evidence of use.

### 5 Nov · literature check (`docs/dreaming/literature-select.md`)
- Known: training-free cross-session KV reuse (MemArt; ICLR 2026 submission, rejected), query-free selection (KVzip,
  Expected Attention, EpiCache, Attention Matching), S-dream ≈ self-study queries, S-recon = weak KVzip, punctuation =
  sink/separator bias. Partly known: S-loo > attention (ContextCite, LORE-KV); content vs licence (Compression-Aware
  Abstention). Ours: 2–8 single positions, slice vs the same text, storage-order swaps, hybrid split.

### 5 Nov · dream_compare planned and submitted
- Stage 0 infrastructure (adapters incl. Gemma-4 MoE, per-head masks, LongMemEval / LoCoMo / PersonaMem / BEAM adapters,
  judges, BM25). Stage A: 133 single-evidence-turn items (106 primary), 18 selections, budgets matched in K,V entries,
  rules H1–H4 at k8 judged. Stage B: paradigms P0 (none), P1 (full context), P2 (text RAG), P3 (KV RAG, same turns),
  P4loo/P4dream (8 positions per turn), P5/P5s (MemArt-style 16-token blocks; P5s matched to P4).

### 7 Nov · SLEEP check, repair, fork
- SLEEP (Behrouz et al., arXiv 2606.03979) = weight-based continual learning; no collision. "Dreaming" and offline
  consolidation are taken as words/ideas; training-free between-session consolidation on a frozen model's own KV
  positions and state stays ours. Avoid "self-learning memory".
- Stage B OOM repair (infrastructure only); Llama finished, Granite BEAM kept failing.
- Fork: this branch = training-free understanding; sibling = trained parts on a frozen backbone (`docs/trained/README.md`).

### 9 Nov · Stage A
| model | H1 | H2 | H3 | H4 (S-loo vs best) | S-loo − gold@k k8 | R-full / S-loo / best (k8 judged) |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | no | no (0/3) | passes 3/4 | indistinguishable (EA-ph) | −0.057 [−0.129, 0.017] | 0.46 / 0.16 / 0.16 |
| Qwen2.5-7B | holds (KVzip-g) | holds 3/3 | passes 4/4 | indistinguishable (KVzip-g) | +0.151 [0.034, 0.266] | 0.67 / 0.39 / 0.42 |
| Llama-3.2-3B | holds (KVzip-g) | no (1/3) | passes 4/4 | worse (−0.17) | −0.009 | 0.59 / 0.25 / 0.41 |
| Granite-4.2-30B | holds (KVzip-g) | holds 3/3 | passes 4/4 | worse (−0.24) | +0.029 | 0.67 / 0.40 / 0.63 |
| Gemma-4-26B-A4B | missing | | | | | |

- Licence gap per method (k8, TF vs judged gap closed): KVzip-g 0.59 / 0.21 (1.5B), 0.63 / 0.62 (7B), 0.73 / 0.69
  (Llama), 1.31 / 0.94 (Granite; ratio means noisy). Per-head methods sit well below the diagonal everywhere.
- H2 pairs (k8, paired): KVzip ph − sh +0.089 (7B), +0.223 (Granite); SnapKV-dream ph − sh +0.121 (7B), +0.187 (Granite);
  H2O ph − sh +0.053 (7B), +0.149 (Granite); 1.5B none hold; Llama only KVzip (+0.058).
- KVzip-g's mean is above SnapKV-oracle-ph (which sees the real question) on 7B (0.42 vs 0.34), Llama (0.41 vs 0.32) and
  Granite (0.63 vs 0.58); below on the 1.5B (0.10 vs 0.12). Unpaired.
- Literature context (not tested): Attention Matching's ablation (Fig. 5) ranks non-uniform head budgets the most
  important component; Gating Mechanism (2611.07605) RULER 44.86 (fixed shared gate mass) / 58.86 (KL to dense) → 67.60
  with per-head budgets by retained attention mass (dense 70.01); KVzip keeps top r% across all heads. KVzip-g is our only
  non-uniform-budget condition → candidate explanation. Caution: Compression-Aware Abstention, KVzip-compressed cache,
  Qwen2.5-7B correct on 4/97 items whose evidence survived.
- **Means**: our selector is not the contribution; what KVzip-g keeps that carries the licence is the open question.

### 8–10 Nov · Stage B (partial: 1.5B, 7B, Llama)
- B1 pooled P3 − P2: m3 1.5B −0.025 [−0.043, −0.007], 7B −0.040 [−0.057, −0.023], Llama −0.012 (equivalent); m10
  −0.047, −0.048, −0.039 (all text better).
- 7B LongMemEval oracle (m3): P1 0.496, P2 0.492, P3 0.434, P4loo 0.21 (100 items), P4dream 0.148, P5 0.292, P5s 0.088.
  Storage per conversation: text 0.03 MB, P3 720 MB, P4 22.6 MB, P5 765 MB. TTFT: P3 0.11 s, P2 0.59 s, P1 4.2 s.
- B3: P4dream beats P5s (same budget) on most cells (7B: +0.06 LME, +0.047 LoCoMo, +0.061 BEAM). B5: P4 refuses more than
  P3 (7B LME +0.126 [0.09, 0.162]).
- B5 (refusal, m3): P4dream − P3 +0.05 to +0.26 on every non-PersonaMem cell; P3 − P2 near 0. LoCoMo caveat: the
  third-party frame asks for "Not mentioned in the conversation" and the regex counts it, so LoCoMo refusal is ~1.0 even at
  P0 on the Qwens (0.999 / 0.996).
- B4 storage frontiers are text or nothing on every benchmark; full cache = 0.36–30 GB per conversation (11,000–70,000×
  the text). Cache only wins on time to first token.
- PersonaMem m1 P3 ≡ P2 by construction (whole session restored in its own context).
- **Means**: as a system, cache memory does not pay on these benchmarks; P3 < P2 is itself an understanding result
  (missing neighbours' cross-attention, same family as collisions).

## Looking ahead

**Survive**: 2-position slice > same tokens as text; value read late from its own position, subject position decides
answering; storage-order swaps and the order fix; hybrid recognition vs recall; TF overstates generation for every
per-head selector (H3 4/4); R-full = text everywhere.

**Retracted or weakened**: "S-loo is the best automatic selector" (KVzip-g ≥ S-loo everywhere); "S-loo beats gold@k"
(paired: only 7B k8); "the licence gap is a property of slices" (KVzip-g nearly closes it on 7B and Llama; large on the
1.5B); "stored cache keeps more than text" (only for tiny slices vs their tokens; B1 says text ≥ KV at turn level);
S-dream / cross-session KV reuse as new; "self-learning memory" framing.

**Open questions**: What does KVzip-g keep that carries the licence (subject span? entities? early layers?)? Is the
licence attention mass (Attention Matching β, untrained test)? Why is the 1.5B different? Is P3 < P2 the collision
effect at scale, and would ordered storage or joint re-encoding close it? Does cache memory pay anywhere (TTFT, long
histories, the hybrid's fixed-size state)?

**Pending decisions**: resume repair (Gemma-4 tolerance or drop; Granite BEAM: smaller budgets, skip P1/P5, or report
partial); let Stage B finish; refresh HANDOVER.md; paper 2 v2 consolidation paragraph + MemArt/KVzip positioning;
website SLEEP citation and stale bullets (needs go-ahead); venue (NAACL SRW mentorship by 16 Nov; ICLR workshop list
29 Nov).

## Glossary
- Slices: R-full, R-content (no template tokens), R-min, R+subj / R+fill / R+rel, R-subj2, R-orphan, T-min / T-subj3,
  R-min-state (2B: 2 attention positions + DeltaNet state), gold (answer tokens + 1), gold@k (k tokens around the
  answer), k10 (10% of content tokens, min 2).
- Selectors: S-dream (attention from 3 dreamed questions), S-attn (attention received in the turn), S-recon (attention
  while repeating), S-loo (leave-one-out on own answers to dreams), S-dream-VATP, random, last-k; KVzip-g/-ph/-sh
  (global budget / per head / shared), SnapKV-dream / -oracle (real question; ceiling), H2O, VATP (H2O × value norm),
  EA (Expected Attention), StreamingLLM (sink + recent).
- Paradigms P0–P5s; m = retrieved turns (sessions for PersonaMem). Rules H1–H4 (Stage A), B1–B5 (Stage B).

## Where things live
See `README.md` in this folder.
