# Experiment 1 against the literature

*A focused reading of 36 papers against the five findings of experiment 1 (dream_bind: 7 primary single-fact items, Qwen2.5-1.5B and Qwen3.5-2B). For each finding: is it new, partly known or known, and which papers come closest. Then one short note per paper, a check of the earlier search-agent report, and what we can and must say in a write-up. Written 2026-10-13. The PDFs are in the parent folder's `papers/` (index: `papers/README.md`). Page numbers are PDF pages. I read the PDFs as extracted text (pdftotext). Figures were not looked at as images, so anything shown only in a figure is marked as such or left out.*

Tone and terms follow [../reference/literature.md](../reference/literature.md). "Slice" = cached keys and values (K,V) of a few positions, computed in the original context. "State" = the Gated DeltaNet (GDN) recurrent state. "Recall" = producing the value ("Pepper"). "Recognition" = verifying it (balanced yes/no).

---

## The short version

| # | Our finding | Verdict | Closest papers |
|---|---|---|---|
| 1 | In-context recall reads the value late, from the attribute's own position. The subject tokens hold no copy of it. | **Known** (for transformers; we add the chat setting and patching on a GDN hybrid) | Lookbacks (Prakash et al. 2025), Entity Tracking (Prakash et al. 2024), Retrieval Head (Wu et al. 2024) |
| 2 | A 2-position slice (the name plus the full stop), gated, carries the fact into a new session (0.86 vs 0.87). The same two tokens as text do not (0.27). | **Partly known.** Sparse KV as memory exists, but only with training. Nobody restores a training-free 1–2 token slice, and nobody compares a slice with the same tokens as text. | Memory3, KBLaM, Gather-and-Aggregate, Michalak & Abreu |
| 3 | Slices harvested after unrelated content still work at other positions. | **Mostly known as a property** (cached K,V tolerate position shifts; binding is not positional). The cross-session framing is ours. | Prompt Cache, Feng & Steinhardt, Memory3, the PIC papers |
| 4 | Hybrid split: state alone = recognition without recall; attention KV alone = recall in slot-filling but refusals on direct questions; both = ceiling; with two facts, a jointly encoded state disambiguates. | **Partly known.** "Recall runs through attention, and the state alone does not recall" is well established, including on a Qwen GDN hybrid. **Recognition from the state is new**: no paper probes it. KV-only refusals and the two-fact role of the state are also new. | DASC (Qwen3-Next), Michalak & Abreu, Bick et al., LatentPort, Hymba (prediction in spirit) |
| 5 | Two facts cached in separate sessions swap values (partner confusion 0.26–0.37), even for different subjects. Shifting positions doesn't help. Joint encoding fixes it. Explanation: binding-ID / order-tag collision. | **The failure class is known; the swaps and the explanation are new.** Cache-reuse papers report only generic quality loss and blame missing cross-chunk attention or attention sinks. None reports swaps or mentions binding. The binding papers predict exactly our pattern but never test cross-context caches. | CacheBlend, Prompt Cache ("scaffolding" = joint encoding), EPIC; Feng & Steinhardt, Dai et al., Lookbacks |

**The three closest papers overall:**
1. **Lookbacks** (Prakash et al.): the mechanism for findings 1, 2 and 5. The answer is fetched from the attribute token's residual stream, and that token also carries the order IDs ("addresses") the question matches against.
2. **Feng & Steinhardt, Binding IDs**: predicts finding 5 almost point by point. Binding is by ID, not position (so shifting does nothing), and IDs that are close give near-chance confusion.
3. **DASC**: the closest measurement for finding 4. On Qwen3-Next, zeroing the attention KV kills needle retrieval and zeroing the recurrent state costs at most one point.

On the systems side, **CacheBlend** and **Prompt Cache** must be cited for finding 5. Prompt Cache's "scaffolding" is our joint-encoding fix, and it is described there in one paragraph.

---

## Q1. Recognition in the state, recall in attention (finding 4)

**The recall half is known.** Four kinds of evidence:
- **Ablation in pretrained hybrids.** In RecurrentGemma and Jamba, removing attention gives 0% needle retrieval (Michalak & Abreu).
- **Which heads do it.** Hybrids "delegate the role of Aggregate Heads to attention" (Bick et al., shown on Zamba2).
- **On a Qwen GDN hybrid.** On Qwen3-Next, removing the full-attention KV gives 0.000 on NIAH, and removing the recurrent state gives 0.990 (DASC, Table 1a, p.3). That is our "state alone, recall 0.00" and "KV carries recall" on the same model family.
- **Hybrid ratio studies.** The linear-to-full ratio "primarily controls recall" (Systematic Analysis).

**The recognition half is not in any paper.** No paper runs a verification (yes/no) probe on a restored state, or on a hybrid with attention removed. Every hybrid ablation scores generation (NIAH, QA, MMLU, NLL). The nearest statements:
- **Hymba predicts the split in spirit, not the recognition part** (§2.1, p.4): attention heads "act like snapshot memories in the human brain, storing detailed recollections of a moment or event. In contrast, the SSM heads summarize the context through a constant cache and thus function as fading memories, which gradually forget the details of past events while retaining their core or gist." This is gist vs detail, stated as design intent and not measured. It does not mention familiarity or verification.
- **Pure SSMs keep partial information without exact recall.**
  - *Mamba-2-Hybrid study (p.8):* "pure SSM-based models have fuzzy memory... while they cannot predict the phone number exactly, they do generally respond with phone numbers that are similar to the correct answer."
  - *Jamba (p.11):* pure Mamba answers "Very Good", "Very Positive" instead of the required "Positive"/"Negative". It knows the sentiment but can't copy the label.
  - Both are adjacent: information is in the state but not reproducible exactly. Neither tests recognition, and both are pure SSMs, not the state of a hybrid.
- **The theory analogue is from neuroscience, not LLMs.** Cortes et al. (the file named "Bogacz Familiarity Capacity", p.3, p.15) say familiarity can be read from the same Hebbian weights that support recall, with far larger capacity: "familiarity needs to store just a single bit (familiar versus non-familiar) per pattern, whereas to recall an event requires the retrieval of the whole pattern". They also write that "the original Hopfield model can be used to model both recollection... and familiarity". A delta-rule state is a fast-weight associative matrix (Systematic Analysis, p.4: it "behaves similar to a fast, continually trained associative memory"). So "one store, two read-outs" carries over by analogy. That mapping is ours.

**One mild tension.** Michalak & Abreu (p.5) write that in hybrids "SSM layers do not even develop the common fuzzy memory". They measured only generation (NIAH). Our state-only yes/no 0.93 says the state does hold fact-specific information, readable by verification. So this is a gap in their probe, not a contradiction of their data.

**A second tension, for the KV-only arm.** In DASC, zeroing the state left NIAH intact (0.99). Our KV-only arm used the name in slot-filling (0.50) but refused direct questions (0.23). This is consistent with DASC on slot-filling. The refusal is new. A plausible reading: NIAH prompts tell the model the answer is in the text, while a chat question needs a "something was said about this" signal first, which is the recognition role of the state. That reading is inferred.

**Verdict: adjacent prior work plus a prediction in spirit (Hymba). Nobody has shown it. Nothing contradicts it.**

## Q2. Do cache-reuse papers see value swaps? (finding 5)

**No. All three report only generic quality loss.**

| Paper | What failure they show | What they blame | What they recompute |
|---|---|---|---|
| **CacheBlend** | F1 drops as more chunks are reused (Fig. 2). The one worked example (Fig. 3, p.4) **rambles and gives no answer**, not a wrong one: "The question is asking for information about FIFA World Cups. The names of Messi and Ronaldo are well-known…" | **Missing cross-attention, explicitly not positions** (p.4): "even with the positional information preserved, a more fundamental problem is that the KV cache of non-prefix text chunk (e.g., C2) ignores the cross-attention between the chunk and the preceding text (e.g., C1)." | All tokens at layer 1, then the 10–20% "high-KV-deviation" tokens per layer, narrowed layer by layer (p.6–7). The default is 15% (p.8). |
| **EPIC** | The naive reuse example (Fig. 4, p.5) answers "1.Derek" instead of "Chrysan Company". That is a wrong entity (the query subject), not the partner's value. | **Attention sinks** (p.2): "the initial tokens of each immutable chunk disproportionately absorb attention, impeding subsequent tokens from attending to relevant context". The cause given (p.4) is that "EPIC independently compiles each chunk with position IDs starting from zero." | "LegoLink recomputes k (k ≤ 32) initial tokens of each chunk (except the first chunk), allowing these tokens to recognize their non-initial positions" (p.2). |
| **Prompt Cache** | No failure examples. A general warning (p.6): masking between modules "can enhance or degrade output quality depending on the semantic independence of the modules." | Modules encoded separately can't attend to each other ("attention masking effect"). | Nothing selective. The fix is **"scaffolding"** (p.6): "sets of prompt modules that are encoded together to share the attention span". **This is our joint-encoding fix**, offered as a memory-for-consistency trade. |
| HYPIC (hybrid) | Deviation from full recompute "concentrates heavily at the beginning of each reused segment" (p.5) | "the intra-segment attention sink" | A seam window of the first w = 8 tokens of each interior segment (p.8) |
| LinearKV (hybrid) | "a chunk prefilled in isolation lacks the cross-chunk attention a joint prefill would supply" (p.1) | missing cross-chunk attention | reuses the CacheBlend / EPIC / ProphetKV selectors at about 20% |

- **Nobody connects chunk-reuse failure to binding IDs.** No cache paper mentions binding, entity tracking or partner confusion. A search of all six texts for "swap", "confus", "bind" finds nothing relevant. Their benchmarks (multi-hop QA, summarization, NIAH) would not show a swap as such.
- **Our "not positional" result agrees with CacheBlend's explicit statement** that position correction alone is not enough. It also agrees with Feng & Steinhardt's position independence (below). Our "shift" control is the Prompt Cache / RoPE-style fix, and it fails as both predict.
- **How the explanations fit together.** EPIC and HYPIC: each separately encoded chunk "thinks it is at the start" (attention sink). Ours: each separately encoded fact thinks it is "the first pair" (order ID). Both are start-of-context artefacts of separate encoding. CacheBlend's "missing cross-attention" is the general cause. The binding ID is a specific thing that this missing cross-attention fails to set. Our explanation narrows CacheBlend's; it doesn't compete with it.
- **The binding papers predict our pattern without testing caches.**
  - Feng & Steinhardt (p.7): "When v1 is close to v0, the LM gets close to 50% accuracy, which indicates confusion." Two separately cached facts that both carry ID 0 are the extreme case of "close".
  - They also show binding ignores position (p.6): moving entities past each other gives "almost no change". So shifting the second slice should not help, and it didn't.
  - Dai et al. (p.1): the ordering ID is "the input order... of entities and attributes, no matter they are bound by a relation... or not".
  - Prakash et al. 2024 (App. J, p.24): "the model is enumerating the association of the boxes and their corresponding object from the start token".
  - In separate sessions, every fact is enumerated first.
- **Caveat on swaps across different subjects** (dog/cat). In Lookbacks the question matches order IDs, and the attribute's address holds the character's and object's OIs, not their identities. If both slices carry "first character, first object", a question about the cat can land on the dog's slot. So cross-subject swaps are what the lookback account predicts. We have not measured the tags. The explanation stays inferred.

**Verdict: the failure class (separately cached chunks lose quality) and the fix (encode together / recompute with the other chunk in view) are known. Value swaps, cross-subject swaps, and the binding-ID explanation are new.**

## Q3. Does one or two tokens' K,V carry a bound fact? (findings 2–3)

**Nobody shows a training-free 1–2 token slice carrying a fact across sessions. Nobody compares a cached slice with the same tokens as text.** The pieces around it:
- **Sparse KV as memory, with training.**
  - *Memory3 (p.14):* "we select 8 tokens out of 128 for each key-value head". Positions are chosen per head by attention received, in a model built for it with "a two-stage pretraining scheme" (p.1).
  - *KBLaM (p.2):* one "knowledge token, which has sizes identical to the KV cache of a single token" per triple. It comes from a sentence encoder through trained adapters. It admits (p.10) it "may fail to precisely generate the text word by word... problematic for cases when we need exact names". Our slice is the model's own K,V and does give the exact name.
- **Why a slice beats its own text.** Memory3's Remark 3 (p.13–14) gives the reason, as a design choice: "encode each document as one sequence, then chunk the attention key-values... This procedure allows the key-values to attend to all their contexts." Our text control (0.87 vs 0.27) is the first direct measurement of how much that buys, at 2 positions.
- **Why the full stop matters.** Gather-and-Aggregate (p.2, p.9): a Gather head "summarizes each segment to its last token", and "the newline token (\n) at the end of each answer choice acts as a summary token". The end-of-segment token as the summary site is a known pattern. That the full stop after the name carries the sentence's binding is our inference from it.
- **Attending to the needle alone is enough, in-session.** Michalak & Abreu (Table 1, p.7): letting attention see only the needle tokens during generation ("Only–Keep") scores 85.7%, against 50.3% for the unmodified model. The cache was still computed in full context, in the same session. This is the closest in-session analogue of R-min.
- **Positions can move (finding 3).**
  - *Prompt Cache (p.4):* "LLMs can operate on attention states with discontinuous position IDs. As long as the relative position of tokens is preserved, output quality is not affected."
  - *Memory3 (p.14):* puts every memory at the same "parallel" positions.
  - *Feng & Steinhardt (p.4):* the model "looks up attributes based on binding ID and not position in the context".
  - So R-orphan working at a mismatched position is expected. What is new is that the slice was cut out of an unrelated conversation and still binds to the subject.

**Verdict: partly known (sparse KV memory with training; segment summary tokens; position tolerance). The training-free 2-position slice and the slice-vs-text gap are new.**

## Q4. Restoring recurrent states across sessions (finding 4, the state arm)

**State reuse is a serving technique. As personal memory, nobody does it.**

| Paper | What is restored | Finding that touches us |
|---|---|---|
| **LatentPort** | Qwen3.5-4B state and KV installed into Qwen3.5-9B (cross-model) | **Same-model restoration is exact** (p.4: 100% top-1 agreement, zero logit difference). KV-only (translated, GDN "fresh and uninitialized", p.3) gives NLL **3.115, close to empty (3.246)**; native is 2.161. Adding the GDN package lowers it to 2.368 (p.4). A **wrong donor state is worse than none** (3.178 vs 2.842, p.6): the state is content-specific. |
| **DASC** | Compressed state checkpoints for prefix caching | Zeroing the state leaves NIAH at 0.990, but the state is "important for multi-key, aggregation, and reasoning tasks" (p.2). This is adjacent to our two-fact result, where the joint state disambiguates. |
| **LinearKV** | One cached chunk's state as the initializer, plus concatenated KV | **"what matters is reading from a single cached source, not which source"** (p.14): a random chunk's state works as well. With recompute, the state's specific content matters little for their QA tasks. That supports attention doing the retrieval. |
| HYPIC | States of separately cached segments, composed exactly | Exact composition is exact only at layer 0. Deeper, the drift is "a limitation every prior PIC system shares" (p.12). |
| Tail-Replay | The state rebuilt from a short recent suffix | GDN is "a structured, lossy compression of the input prefix" (p.1). A 5–10% replay keeps 92.8–99.9% of quality. |
| Marconi | State checkpoints, exact-prefix hits only | Systems work only. States "cannot be rolled back" (p.1). |
| **State Soup** | Mamba-2.8B states stored, retrieved, averaged | States as a retrievable library of ICL skills (p.1–4). Mixing separately processed chunks is "still worse than the model that saw both" (p.4). That is the state-side version of our "encode jointly". Pure SSM, task skills, no facts. |
| **S0 Tuning** | A *learned* initial state on Qwen3.5 (+19.0 pp HumanEval at 2B, +23.6 at 4B) | p.8: the state's effect on logits fades over the prompt, but "the signal has not vanished. It has been absorbed into the recurrent state as a low-magnitude directional bias", which flips the first generated token. This matches what our state-only arm shows: a bias that tips a yes/no, not a value. |
| WriteSAE | Rank-1 edits to the GDN cache (Qwen3.5-0.8B) | Edits shift chosen tokens' logits, but "Out-of-context targets shift rank but never reach top-1 inside 20 tokens" (p.7). The state nudges and doesn't produce. This is adjacent to "recognition without recall". |

**Verdict: restoring and composing states is established for serving (same model) and cross-model handoff. As a cross-session memory, probed by recall vs recognition, it's new.** A confound LatentPort points out also applies to us: the runtime installs recurrent and convolution state together (App. A, p.10). Our state-only arm should say whether the 4-token convolution buffer was included. It holds the last few tokens of the stored turn verbatim.

## Q5. Is "read from the attribute position, nothing at the subject" consistent? (finding 1)

**Yes, with one wording caveat.**
- **Lookbacks (p.4):** "Information from the correct state token (beer) flows directly from its residual stream to that of the final token in later layers". The answer payload appears at the final token only after layer 56 of 80 (p.6).
  - Ours: by layer 19 of 24 on the 2B, and the last window (23–27 of 28) on Qwen2.5. Both are "late".
- **Entity Tracking (p.4):** "correct object information is fetched directly from its token residual stream, instead of getting it from the query label token residual stream."
- **Retrieval Head (p.3):** retrieval heads copy-paste. The most-attended token equals the generated one.
- **Geva et al.** (literature.md §9) puts facts at the subject, but for facts in the weights. That is a different route, not a contradiction.
- **The caveat: the subject is not empty.** In Lookbacks the query's character and object tokens carry order IDs. They are "retrieved from their earlier occurrences and passed to the final token" (p.4) and act as pointers (layers 20–34, p.8). Dai et al. find the order ID at the query entity token (p.2). Our Pepper/Rocky patch swaps runs with the same order structure, so it cannot see a pointer. And with one fact in context, nothing needs disambiguating, so blocking the subject costs nothing.
  - Say "the subject carries no copy of the value". Don't say "the subject carries nothing the answer needs".
  - [01-binding.md](01-binding.md) currently says the latter ("The question's subject ('my dog') carries nothing the answer needs"). That is too strong for two-fact contexts, and the collision result itself shows a pointer matters there.
- **Why the slice works, in Lookbacks' terms.** The attribute token holds the payload and an address (copies of the character and object OIs), side by side (p.7). Our 2-position slice therefore carries both the value and its binding tag. That explains why it binds in a new session, and why two separately harvested slices with the same tag collide.

**Verdict: consistent with all three. No contradiction found.**

---

# Per-paper notes

Relation tags: **same** (shows our result), **supports**, **adjacent**, **contradicts**, **background**.

## A. Hybrids and recall

**Michalak & Abreu, "Some Attention is All You Need for Retrieval"** (arXiv 2510.19861, NeurIPS 2025 MechInterp workshop). *Supports 4 (recall half), adjacent to 2.*
- RecurrentGemma-2B/9B and Jamba-Mini. "retrieval depends exclusively on self-attention layers" (abstract, p.1). Attention ablation gives 0% NIAH, and JRT prompting doesn't rescue it.
- 15% of heads suffice, with 84% of MMLU kept.
- "Only–Keep" (attention restricted to the needle tokens at generation) scores 85.7% (Table 1, p.7).
- Claims SSMs "do not even develop the common fuzzy memory" (p.5), from generation probes only. Never tests verification.
- Not GDN, not Qwen.

**Bick, Xing & Gu, "Gather-and-Aggregate"** (arXiv 2504.18574). *Supports 4 (recall half) and 2 (why the full stop).*
- Retrieval runs through a few Gather heads, which summarize each segment into its last token (p.2, p.9), and Aggregate heads, which read the summaries.
- "pretrained hybrid models... delegate the role of Aggregate Heads to attention" (p.1). Zamba2: ablating nine attention heads drops MMLU from 64.3% to 34.9%, with knowledge tasks stable (p.12).
- SSM versions of these heads are "smoother".

**Borobia et al., "Component Ablation for Efficient Hybrid LM Architectures"** (arXiv 2603.22473). *Adjacent to 4.*
- Qwen3.5-0.8B: skipping all 18 GDN layers multiplies WikiText perplexity by about 35,200; skipping the 6 attention layers, by about 82 (Table 2, p.9).
- "the alternative sequence-processing pathway is the dominant contributor to next-token likelihood" (p.9).
- Caveat: a skip removes the whole layer including its MLP (p.6), so this is not state-vs-KV of a context. No recall or recognition probes.

**Wang, Zhu et al., "A Systematic Analysis of Hybrid Linear Attention"** (arXiv 2507.06457). *Background, supports the recall half.*
- 72 models from scratch: "While language modeling remains stable across linear-to-full attention ratios, recall significantly improves with increased full attention layers, particularly below a 3:1 ratio" (p.1). Qwen3.5 is 3:1.
- Calls the delta-rule state "a fast, continually trained associative memory" (p.4). That is the hook for the familiarity analogy.

**HYPIC** (Liu et al., arXiv 2607.01299). *Adjacent to 4 and 5.*
- Position-independent caching for hybrids, including Qwen3.5-35B/122B. Composes GDN states exactly with cached transition operators.
- Repairs attention with an 8-token seam window at each segment start, where deviation "concentrates heavily" because of the intra-segment attention sink (p.5, p.8).
- Reports quality gaps only (1.71 points from full recompute). No swaps.

**LatentPort** (Villani, arXiv 2609.25053). *Supports 4 (the state carries context beyond KV); adjacent to transplant.*
- Qwen3.5-4B → 9B handoff. KV-only NLL 3.115, empty 3.246, native 2.161. KV plus the GDN package: 2.368 translated, 2.291 direct-copied (Table 3, p.5).
- Same-model restoration is exact (p.4). A shuffled donor state is worse than empty (p.6).
- Caveats: cross-model with *translated* KV, base models, teacher-forced NLL on books. It is not "attention KV alone behaves as if nothing was said" on one model. Recurrent and convolution state can't be separated in their runtime.

**LinearKV** (Liu, Qi et al., arXiv 2608.11231). *Adjacent to 4 and 5.*
- Hybrid PIC on OLMo-Hybrid, Qwen3.6-27B (GDN) and Granite (Mamba-2): concatenated chunk KV plus one cached chunk state as initializer, with CacheBlend / EPIC / ProphetKV recompute at about 20%. Up to 92% of full quality.
- A random chunk's state works as well as the last (p.14). Reports quality only; no swaps.

**Hymba** (Dong et al., arXiv 2411.13676; the PDF shows no venue). *Adjacent; the nearest prediction of 4.*
- Abstract (p.1): "Attention heads provide high-resolution recall, while SSM heads enable efficient context summarization." The snapshot vs fading-memory quote is under Q1 (p.4).
- Removing one SSM head costs more on HellaSwag (1.1%) than one attention head (0.24%) (p.4).
- Gist, not recognition. Not measured with probes.

**State Soup** (Pióro et al., arXiv 2406.08423). *Adjacent to 4 and 5.*
- Mamba-2.8B states as stored, retrieved and averaged "task vectors" (p.1).
- Mixing states of separately processed chunks is worse than processing both in order (p.4).
- Pure SSM, ICL skills, not facts.

**Marconi** (Pan et al., arXiv 2411.19379, MLSys 2025). *Background (systems).* Prefix caching for hybrids. Recurrent states allow exact-prefix hits only (p.1). Read: abstract and introduction.

**Tail-Replay** (Liu, Qi et al., arXiv 2608.30310). *Adjacent to 4.* Rebuilds GDN state from a short recent suffix, because the state attenuates earlier inputs (p.1). Supports "the state is a lossy, recency-weighted summary". Read: first 3 pages.

**DASC** (Yu et al., arXiv 2608.30386). *Same as the recall half of 4, on a Qwen GDN hybrid.*
- Cache ablation on RULER NIAH-S1, Table 1a (p.3):

  | Zeroed | Qwen3-Next | Kimi-Linear |
  |---|---|---|
  | full-attention KV | 0.000 | 0.000 |
  | recurrent state | 0.990 | 1.000 |

- "This probe is consistent with retrieval being carried by full-attention KV despite recurrent-state removal" (p.3). Notes the state matters for multi-key, aggregation and reasoning (p.2).
- No recognition probe.

**S0 Tuning** (Young, arXiv 2604.01168). *Adjacent to 4.*
- A learned initial GDN state on frozen Qwen3.5: +19.0 pp HumanEval at 2B (p.15), +23.6 at 4B.
- The state's influence becomes "a low-magnitude directional bias" (p.8) that flips the first token. A probe reads task information from the state (AUC 0.93, p.9).
- Shows a state injected before the prompt still acts. Trained, not harvested.

**WriteSAE** (Young, arXiv 2605.12770). *Adjacent to 4.* Rank-1 atoms replace GDN writes on Qwen3.5-0.8B. Formula-chosen writes raise mid-rank tokens, but out-of-context targets never reach top-1 (p.7). A tool for the follow-up (which state directions encode "dog ↔ Pepper").

**dQwen3.5** (Xue et al., arXiv 2609.20751). **Off-topic** (adapting Qwen3.5 into a diffusion LM). Useful only for the architecture fact "three GDN layers for every standard attention layer" (p.4). S0 Tuning's table confirms Qwen3.5-2B has 18 GDN layers and a 4.7M-number state.

**Log-Linear Attention** (Guo, Yang et al., arXiv 2506.04761, ICLR 2026). *Background.* Pure GDN "already achieved perfect accuracy in several cases" of single-needle NIAH (p.9). So a delta-rule state *can* recall one key in a model trained without attention. Our state-only recall 0.00 is about this hybrid's division of labour, not a capacity limit. Read: abstract and NIAH section.

**Gated DeltaNet** (Yang, Kautz, Hatamizadeh, arXiv 2412.06464, ICLR 2025). *Background.* The architecture. Its S-NIAH case study (p.5) names "memory collision" in the state ("information becomes superimposed and indistinguishable"). That is a state-side collision, different from our KV-side one.

**Based** (Arora et al., arXiv 2402.18668). *Background.* "a key tradeoff between a model's state size and recall ability" (p.1). Says nothing about verification. Read: abstract.

**Just Read Twice** (Arora et al., arXiv 2407.05483). *Background.*
- Recall "reduces to the hardness of a problem called set disjointness" (p.1). The memory needed depends on data order (p.2).
- Set disjointness is itself a yes/no decision, and JRT shows it needs as much memory as the first set. So JRT does **not** say recognition is cheaper than recall. The earlier report overreads it.

**Zoology** (Arora et al., arXiv 2312.04927). *Background.* "82% of the gap is explained by each model's ability to recall information that is previously mentioned in-context" (p.1). Read: abstract.

**Repeat After Me** (Jelassi et al., arXiv 2402.01032). *Background.* GSSMs "are fundamentally limited by their fixed-size latent state" at copying (p.1). Read: abstract.

**RNNs are not Transformers (Yet)** (Wen et al., arXiv 2402.18510). *Background.* Theory: RNNs fail in-context retrieval, and "adding a single Transformer layer" closes the gap (p.1). Read: abstract.

**Mamba-2-Hybrid empirical study** (Waleffe et al., arXiv 2406.07887). *Adjacent to 4.* Pure SSMs have "fuzzy memory" on Phonebook: near-miss numbers, with "compressed information about each phone book entry" in the state (p.8). The hybrid fixes it. Read: Phonebook section.

**Jamba** (Lieber et al., arXiv 2403.19887). *Adjacent to 4.* Pure Mamba gives semantically right but wrongly formatted labels. The hybrid with "only 1 out of 8 layers" attention does ICL, with induction-like heads in the attention layers (p.11). Read: §6.2.

**Retrieval Head** (Wu et al., arXiv 2404.15574). *Supports 1.* Fewer than 5% of heads copy-paste needle tokens to the output. Masking them makes the model hallucinate; masking some of them gives partial retrieval (p.2). Transformers only.

## B. Cache reuse

**Prompt Cache** (Gim et al., arXiv 2311.04934, MLSys 2024). *Supports 3; same fix as 5.*
- Discontinuous position IDs are fine if relative positions are kept (p.4).
- Separately encoded modules can hurt "semantically dependent modules". The fix, "scaffolding", encodes modules together (p.6). It has a personalization demo with cached profile modules (p.10–11), judged only qualitatively.
- No swaps and no binding.

**CacheBlend** (Yao et al., arXiv 2405.16444, EuroSys 2025). *Adjacent to 5; must cite.*
- Reusing separately cached chunks loses cross-attention. The example output rambles (p.4).
- Says plainly that the position fix isn't enough (p.4).
- Recomputes about 15% high-KV-deviation tokens per layer (p.6–8). "Full KV reuse does work when the cross-attention between chunks is low" (p.4).
- No swaps, no binding. Multi-hop QA and summarization.

**EPIC** (Hu et al., arXiv 2410.15332, ICML 2025). *Adjacent to 5.*
- Blames attention sinks at each chunk's first tokens, because every chunk starts at position 0 (p.2, p.4).
- Recomputes the first k ≤ 32 tokens of each later chunk (p.2).
- The naive output is the wrong entity ("1.Derek", Fig. 4, p.5), not a partner swap.

**Memory3** (Yang et al., arXiv 2407.01178). *Adjacent to 2 and 3.*
- Explicit memories are sparse attention K,V: 8 of 128 tokens per head, half the layers, parallel positions (p.14), in a model pretrained for it.
- Remark 3 (p.13–14): encode the whole document so chunked K,V "attend to all their contexts". That is the reason our slice beats its text.

**KBLaM** (Wang et al., arXiv 2410.10450, ICLR 2025). *Adjacent to 2.*
- One trained KV "knowledge token" per (name, property, value) triple, in a frozen Llama-3 with trained adapters and a learned query head. Order-invariant (p.6).
- Weak on exact names (p.10). Our untrained 2-position slice gives exact names.

**RF-Mem** (Zhang et al., arXiv 2603.09250, ICLR 2026). *Adjacent, terminology only.* A dual-process (familiarity / recollection) retriever over text memories for personalization (p.1). Nothing about model internals. Cite at most for the vocabulary.

## C. Binding

**Prakash et al., "Language Models use Lookbacks to Track Beliefs"** (arXiv 2505.14685, ICLR 2026; the earlier report says NeurIPS 2025, but the PDF header says ICLR 2026). *Supports 1, explains 2 and 5.*
- The "lookback": a source reference is copied into an address beside a payload at the recalled token, and into a pointer at a later token. Attention dereferences the pointer by a QK match (p.2).
- The state token holds both the character and object OIs (address) and the state OI (payload) (p.7). The answer is fetched late (layers 52–56 of 80, p.6–8).
- Models: Llama-3-70B, Llama-3.1-405B, Qwen2.5-14B (App. N).
- Transformers, stories with two characters. Never tested across contexts.

**Feng & Steinhardt, "How do Language Models Bind Entities in Context?"** (arXiv 2310.17191, ICLR 2024). *Predicts 5.*
- Binding IDs are numbered by pair order: "we can identify the k-th abstract binding ID with the integer k" (p.4).
- Position independence: the model "looks up attributes based on binding ID and not position" (p.4–6).
- IDs that sit close give near-chance confusion (p.7). The heads that compare IDs are left open (p.8: "we speculate").
- Never tests IDs across contexts.

**Dai, Heinzerling & Inui, "Representational Analysis of Binding"** (arXiv 2409.05448, EMNLP 2024). *Supports 5.*
- The ordering ID is a low-rank subspace encoding input order "no matter they are bound by a relation... or not" (p.1). Found at the query entity token, in Llama-2/3, Qwen1.5 and Pythia.
- Patching along it makes "Box Z contains the stone" (p.2). A causal handle for the planned repair.

**Prakash et al. 2024, "Fine-Tuning Enhances Existing Mechanisms: Entity Tracking"** (arXiv 2402.14811, ICLR 2024). *Supports 1 and 5.*
- Value Fetcher heads fetch the object "directly from its token residual stream" (p.4). The circuit uses positional information (p.7).
- App. J (p.24): "the model is enumerating the association of the boxes and their corresponding object from the start token". In separate sessions every fact is number one.

## D. Familiarity

**"neuroscience/Bogacz Familiarity Capacity.pdf" is not Bogacz et al.** It is **Cortes, Greve, Barrett & van Rossum, "Dynamics and robustness of familiarity memory"** (arXiv 0710.1333, 2007), which builds on Bogacz & Brown (2003). *Adjacent (theory) to 4.*
- Familiarity capacity is "proportional to the number of synapses", recall "to the square root" (p.3). Later stated as N² vs N (p.15).
- One Hopfield network supports both: recollection as the attractor, familiarity as the energy transient after the probe (p.15).
- Rename the file or cite it correctly. Bogacz et al. 2001 is not in the folder.

---

# Checking the earlier search-agent report

*(`../../../Literature check recognition vs recall in hybrid LLMs.md`, same date.)* Mostly accurate. Corrections:

| Claim in the report | What the PDF says |
|---|---|
| LatentPort: adding the GDN package "brings it to 2.291 and removes about 80%" | Two numbers are mixed. 2.291 is KV + **direct-copied** GDN. The ~80% (79.7%) gap reduction is for the **translated** package, 2.368 (p.4–5). Also, the KV is *translated* across models, so "KV-only ≈ nothing said on Qwen3.5" is a loose analogy, not the same experiment. |
| JRT: "recognition (query after) can be cheaper/different from recall" | Not in the paper. JRT says memory depends on data *order*. Set disjointness is itself a yes/no decision needing Ω(min(\|A\|,\|B\|)) memory. |
| Bogacz et al. 2001: "familiarity networks... cannot do associative recall" | The PDF in the folder is Cortes et al. 2007, and it says the opposite emphasis: the same Hopfield network supports both read-outs (p.15). Nothing in it says familiarity networks cannot recall. |
| Lookbacks is NeurIPS 2025 | The PDF says ICLR 2026. |
| Hymba Fig. 1(b) "interpret from the memory aspect" | Correct. The text is in §2.1, p.3–4. |
| Michalak & Abreu; Bick et al.; DASC not mentioned | The first two are quoted correctly. **The report missed DASC's Table 1a**, the closest measurement to our split on a Qwen GDN hybrid. |
| WriteSAE as "mild tension (state can push tokens)" | Fair, but the same paper finds out-of-context targets never reach top-1 (p.7). That fits "nudges, doesn't produce". |
| Zani, Michalak & Abreu (ICLR 2025 workshop); Greve et al. 2010; Schlag et al. 2021; Weighted State Souping | Not in the folder; not checked here. |

The report's main conclusion stands: the recall half is known, and the recognition half has no precedent in the PDFs.

---

# What we can still claim

1. **Recognition without recall from a hybrid's recurrent state.** A restored GDN state alone gives balanced yes/no 0.93 and recall 0.00 on the same items. No paper probes verification on a state; all hybrid ablations score generation only. Scope it to "in this trained hybrid": pure GDN models do recall single needles (Log-Linear, Gated DeltaNet).
2. **The full dissociation in one restored memory.**
   - KV alone: slot-filling 0.50, refusals on direct questions 0.23.
   - State alone: the reverse.
   - Both: ceiling.
   - The KV-only refusal is new, and it qualifies DASC's "state is redundant for simple recall".
3. **Two cached positions carry a bound fact across sessions, training-free,** and carry about three times what the same tokens do as text (0.87 vs 0.27). Sparse KV memory exists only with training (Memory3, KBLaM). The slice-vs-text comparison has not been done before.
4. **Separately cached facts swap values, even across subjects; shifting positions doesn't help; joint encoding fixes it.** Cache-reuse papers report only generic loss. We name the specific failure and link it to binding IDs.
   - This explanation stays **inferred** until the tag is measured (addendum 6).
   - Write it as "consistent with Feng & Steinhardt's near-chance confusion for close binding IDs and their position independence". Don't write "caused by".
5. **The in-context read site in a chat on a GDN hybrid** (patching). New only as a setting. The mechanism is known.

What we cannot claim:
- that cross-session KV reuse is new (Prompt Cache, CacheBlend, EPIC)
- that joint encoding is a new fix (Prompt Cache's scaffolding, CacheBlend's recompute)
- that recall needs attention in hybrids (Michalak & Abreu, Bick, DASC)
- that the subject "carries nothing" (Lookbacks: it carries a pointer)

# What we must cite

| For | Cite |
|---|---|
| Finding 1 (read site) | Lookbacks; Entity Tracking (Prakash 2024); Retrieval Head; Geva (contrast: facts in weights) |
| Finding 2 (slice) | Memory3 (sparse KV memory; Remark 3 on context-bearing K,V); KBLaM; Gather-and-Aggregate (segment summary token); Michalak & Abreu ("Only" attention) |
| Finding 3 (transplant) | Prompt Cache (discontinuous positions); Feng & Steinhardt (position independence); EPIC / CacheBlend (position-independent caching) |
| Finding 4 (hybrid split) | DASC (Table 1a); Michalak & Abreu; Bick et al.; LatentPort; Hymba (snapshot vs fading); Systematic Analysis; Gated DeltaNet; S0 Tuning; Cortes et al. 2007 / Bogacz & Brown 2003 for the familiarity analogy |
| Finding 5 (collision) | CacheBlend; Prompt Cache (scaffolding); EPIC; HYPIC / LinearKV for hybrids; Feng & Steinhardt; Dai et al.; Lookbacks; Entity Tracking App. J |
| State as stored memory | State Soup; LatentPort; Marconi; Tail-Replay |

Also correct two lines in [01-binding.md](01-binding.md) when it is next edited:
- **The subject sentence.** "The question's subject carries nothing the answer needs" → "carries no copy of the value". (This note doesn't edit other files.)
- **The "What was and was not new" paragraph.** It should add DASC (recall in attention on Qwen3-Next) and Prompt Cache's scaffolding (joint encoding).

# Added later

- **FwPKM (Zhao & Jones, Sakana AI, arXiv 2601.00671, 2026)**, read 2026-10-14, full notes in [literature.md group F](../reference/literature.md#f-trained-fast-weight-memory). Relevant here:
  - **Finding 4:** in a GDN + attention hybrid, the learned gate ignores the memory when full attention can reach the context, and uses it when attention is restricted. That supports "recall runs through attention where attention can reach".
  - **Finding 5:** re-reading the context 2–4 times raises needle retrieval from under 10% to over 70%. That's the trained analogue of joint re-encoding.
  - **Not about cache slices, binding IDs, or recognition from a state,** so the novelty claims above stand.

# Not read, partly read, or off-topic

- **Off-topic:** dQwen3.5 (diffusion adaptation; architecture facts only). RF-Mem (terminology only).
- **Read only the abstract and key sections (background):** Based, Zoology, Repeat After Me, RNNs are not Transformers Yet, Log-Linear Attention, Marconi, Tail-Replay, Jamba (§6.2), Mamba-2-Hybrid (Phonebook section).
- **Read in full or nearly:**
  - Lookbacks, Binding IDs, Gather-and-Aggregate, Some Attention, Component Ablation
  - LatentPort, LinearKV (main text), HYPIC (main text), CacheBlend (§1–5), EPIC (§1–5), Prompt Cache (§1–3, 5.6)
  - State Soup, S0 Tuning (main text), WriteSAE (§1–4), DASC (§1–3)
  - Memory3 (§3), KBLaM (§1–4, 8), Entity Tracking (main text + App. J), Dai et al. (§1–2), Retrieval Head (§1–3), Cortes et al. (full), Systematic Analysis (§1–4), Hymba (§1–2)
- **Not in the folder, not read:** Bogacz et al. 2001; Greve et al. 2010; Zani et al. 2025; Schlag et al. 2021; Geva et al. (covered in literature.md); InfLLM (in the folder but not on this list).
- **Method caveat:** all reading was of extracted PDF text. Figure-only content (attention maps, curves) was not inspected. Table values were read from the extracted text and spot-checked against the surrounding prose.
