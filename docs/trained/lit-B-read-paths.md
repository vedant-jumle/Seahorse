# Literature B: trained read paths, gates and memory modules beside a (mostly) frozen model

*Written 2026-11-07 for the trained side of Seahorse. 16 papers from `papers/`. Each one is compared with what we found: the slice channel, binding, content vs licence, automatic selection, and retrieval. Page and table numbers refer to the PDFs. How deeply I read each paper is stated in its section. Earlier notes on FwPKM (`reference/literature.md`, group F), Afendulev and Ferrando (`dreaming/literature-exp1.md`) are not repeated, only extended.*

Candidates being weighed (numbering used throughout):
1. **Amortised S-loo selector**: a trained network that predicts which positions to slice.
2. **Licence-aware read gate or adapter**: makes the model answer from a slice it already holds.
3. **Learned retrieval key plus verification or abstention.**
4. **SLEEP-style consolidation**: slices → LoRA, with the answering-slice model as teacher and dreamed questions as data; the slices are deleted afterwards.

---

## 1. Short version

**Takeaways for the trained side**

1. **No paper trains a gate that decides "answer from memory, or say I don't know".** The nearest is Larimar's *scope detector*. It decides "memory-conditioned decoding or plain decoding", not "answer or abstain". Without it, the memory floods unrelated prompts: neighbourhood accuracy falls from 80.4 to 13.7 (Larimar-6B, Table 8, p.14). Its best version is an **external sentence encoder with 1-nearest-neighbour cosine** (EER 2.9%). A classifier trained on Larimar's own encodings was *worse* on specificity (neighbourhood 15.7–45.7 against 80.6; Table 11). Trained abstention conditioned on memory is open, and it is ours to do.
2. **A tiny trained part can fix a frozen model's read path. But most of what it learns is a prior, and the prior answers with confidence when there is nothing to read.** Pepper et al.: a d+1-parameter adapter takes self-interpretation from 0.04% to 82.9% recall@1. The bias vector alone gives "approximately 85% of the total gain" (p.4). Fed a zero vector, the adapter confidently names "the Russian composer who wrote the opera Eugene Onegin" (App. J, Table 17). For us this is the same thing as "told ≈ generic comply". A licence adapter trained only on answerable cases will learn "always answer". So it **must be trained with negatives**: no slice, the wrong slice, and lure slices, each with "I don't know" as the target.
3. **Very small residual interventions are enough for both licence and storage.** ReFT trains about 0.03% of parameters and beats LoRA on commonsense tasks (LLaMA-7B: 80.2 vs 74.7, Table 1). A rank-4 intervention at one position (layer 15, last prompt token; 32,772 parameters) learns a refusal persona from **5 examples in 30 s** (App. G.2). A rank-1 intervention with about 4K parameters holds **256 random ID→token pairs** (App. F.2). So candidate 2 (an adapter at a few chosen positions) and candidate 4 (a per-user store far smaller than a LoRA) both have direct precedent.
4. **The licence is probably wired through the subject position, and it gates attention, not content.** Ferrando et al.: known/unknown-entity directions at the entity's last token regulate how much the attribute-extraction heads attend to the entity (L18H5, L20H3; Fig. 4, p.7). They also causally switch refusal: "almost 100% refusal" when the unknown latent is pushed (Gemma 2 2B, Fig. 3, p.6). This matches our result that adding the subject's position raises answering while the value's probability stays the same. **Put the licence adapter at the slice's subject positions, not only at the question end.** On the hybrid, Afendulev et al. add that the *response mode* lives in the recurrent state ("KV-only names the requested language correctly but answers in English", Fig. 4). The 2B licence may need a state-side term.
5. **Train against a teacher distribution, not gold answers.** TransMem's evidence-conditioned self-distillation uses forward KL to the same frozen model given only the evidence and question. It beats SFT on gold answers (54.11 vs 48.82 LoCoMo F1), GRPO (53.18) and on-policy distillation (52.80) (Table 5). Adding hidden-state matching *hurts* (51.8–52.2, Table 9). This is the teacher design that candidates 2 and 4 already plan.
6. **If attention can reach the information another way, a trained memory gate learns to ignore the memory.** FwPKM's gate clusters near 0 under full attention (Finding 2, p.7–8). Lee et al. find that "giving the model access to a sliding-window KV cache can make the newly inserted Jet layers underutilized", and fix it with a hard-eviction warm-up (§6.4). So **train the read adapter with the slice as the only source of the fact**: no text copy in context.
7. **For per-user consolidation with little data, LoRA is the risky choice.** In S0 Tuning, with about 48 examples, LoRA at rank 64 makes things worse (−15.5 pp, 8 of 10 seeds negative). A learned initial GDN state gives +23.6 ± 1.7 pp (+19.0 on Qwen3.5-2B) (Tables 1–2). With a handful of dreamed questions per user, consider a learned state offset (2B) or a rank-1 ReFT (both models) as the consolidation target before LoRA. Paraphrases at write time matter: Larimar 88.4 → 93.6 with two rephrases (Table 8).
8. **Getting the first token right is not the same as answering.** Tip of the Tongue: an edit that recovers the first answer token in 82.7–92.5% of failures yields correct full answers in only 30.5–38.0% (§5.6). This is the same gap as our teacher-forced 0.99 against 0.17–0.49 judged in generation. **Any trained licence piece must be scored in generation with judges, not teacher-forced.**

**Overlap table** (✔ direct precedent, ◐ partial or adjacent, ✗ nothing; "fs" = trained from scratch, not a frozen backbone)

| Paper | Backbone | 1 Amortised selector | 2 Licence gate / adapter | 3 Key + verify / abstain | 4 Consolidation | Learns *when* to use memory? | Abstention? |
|---|---|---|---|---|---|---|---|
| FwPKM | fs (112M) | ◐ gate also weights writes | ◐ per-token scalar read gate | ◐ product keys (no verify) | ◐ re-reading 1→4 passes; retention fails | ✔ (rare entities) | ✗ |
| Larimar | decoder fine-tuned | ✗ | ◐ one trained KV slot | **✔ scope detector (memory vs no memory)** | ◐ paraphrase writes | ✔ via scope detector | ◐ writes "UNKNOWN" to block leakage |
| TransMem | **frozen** | ◐ fixed segment ends; B-invariant | **✔ trained block + gate, last 4 layers** | ✗ | ◐ teacher design | ◐ per-dim gate, layer schedule | ✗ (adversarial set excluded) |
| Trained Persistent Mem. | **frozen** GPT-2 | ◐ slot top-k write | ◐ gated branch *fails* at 1× | ✗ | ✗ | ✗ | ✗ |
| Titans | fs | ◐ surprise-weighted writes | ◐ MAC: attention over retrieved memory | ✗ | ✗ | ◐ attention decides | ✗ |
| MemoryLLM (Apple) | fs | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Tensor Memory | fs | ✗ | ◐ global scalar gate | ✗ | ✗ | ◐ global only | ✗ |
| RF-Mem | text retriever | ✗ | ✗ | ◐ training-free familiarity switch | ✗ | ◐ deeper search when unfamiliar | ✗ |
| Pepper et al. | **frozen** | ✗ | **✔ d+1 adapter fixes a read** | ✗ | ✗ | ✗ | ✗ (zero vector → confident answer) |
| SleepGate | fs (793K) | **✔ learned per-entry keep score** | ◐ attention-logit bias | ◐ supersession tagger | ◐ sleep cycle | ✔ (stale vs current) | ✗ |
| Lee et al. (sleep) | fine-tuned | ◐ query-first consolidation | ✗ | ✗ | **✔ offline passes → GDN state** | ✗ | ✗ |
| S0 Tuning | **frozen** | ✗ | ◐ first-token trajectory steer | ✗ | **✔ state as adaptation surface** | ✗ | ✗ |
| Afendulev et al. | frozen (no training) | ✗ | ◐ state = response mode | ◐ state familiarity (lures) | ✗ | n/a | n/a |
| Ferrando et al. | frozen (no training) | ✗ | **✔ mechanism: entity direction gates extraction + refusal** | ◐ uncertainty latent AUROC 73.2 | ✗ | n/a | ✔ steerable refusal |
| Tip of the Tongue | frozen (no training) | ✗ | ◐ selection margin | ◐ decodable ≠ selected | ✗ | n/a | ✗ |
| ReFT | **frozen** | ◐ positions are a hyperparameter | **✔ tiny position-wise interventions; refusal from 5 examples** | ✗ | **✔ rank-1 stores 256 pairs** | ✗ | ◐ blanket refusal persona |

**One correction to the earlier internal conclusion** ("writes can be untrained; the read path needs training"). It holds, but in a weaker sense than it sounds. In every paper here the write *format* is trained, even when nothing is optimised at write time:
- FwPKM: the slow weights producing q, v and g are trained.
- Larimar: the encoder is trained.
- Jeong: random but fixed write projections.
- TransMem: no write at all; it re-reads states that are still in context.

Our slices are native K,V, so our write is *more* untrained than any of these. That makes the read problem different: the model already reads native K,V, so the missing piece is the licence, not the format.

**Two library notes.** `MemoryLLM.pdf` is Jaiswal et al. (Apple, arXiv 2602.00398), "Plug-n-Play Interpretable Feed-Forward Memory". It is **not** the self-updatable memory-pool MemoryLLM (Wang et al. 2024) that `papers/README.md` describes. Larimar's decoder is fine-tuned jointly, not frozen (p.3).

---

## 2. Per-paper notes

### 2.1 FwPKM: Zhao & Jones (Sakana AI), "Fast-weight Product Key Memory", arXiv 2601.00671 v2
*Read: full main text (§1–7) plus App. A.5, C, D (ablations, addressing). Earlier notes: `reference/literature.md` §16. Only additions here.*

- **Trained / frozen.** Nothing is frozen. 12-layer, ~112M Qwen-Next-style models are trained from scratch on 5B tokens of LongContext64 and 5B of FineWeb-Edu, with 4K sequences. Slow weights (the projections producing q, v and gate g) are trained by the LM loss. The memory values and keys are *fast weights*, updated at train and test time by a local MSE loss (values) and a slot-usage entropy loss (keys).
- **Read site and gate.** One FwPKM layer is inserted *before* the token mixer at layers 2, 6 and 10 (or 2 and 10). It has 512² = 262,144 slots, of which the top 8 are read. Read: `o_t = g_t·v̂_t + (1−g_t)·v_t`, where g_t = σ(Linear(RMSNorm(h_t))), one scalar per token (eq. 10, 12). **The same g_t also weights the write loss** (App. A.5: g "serves as a natural importance weight for deciding which tokens should write more strongly"). Ablation: weighting the MSE by g "is important for NIAH retrieval accuracy" (App. D.1).
- **Parameters.** Total grows ~5× (112M → ~516–519M, mostly the slot tables). Active parameters stay ~113–117M (Table 1). The gate is d+1 parameters per layer.
- **When to use memory.** Learned, without supervision on the gate:
  - "these models learn to ignore FwPKM, with gating weights clustering near zero" under unrestricted full attention (p.8).
  - Restricting attention (window 512 with p = 0.9 during training, "pSWA") makes the model use it.
  - In the upper layer the gate spikes on "rare named entities and proper nouns" (p.12).
  - No abstention.
- **Numbers.**
  - NIAH: 1-iteration accuracy is "< 10%"; re-reading the haystack takes it to "> 50%" at 2 iterations and "> 70%" at 3–4, from 4K to 128K (Finding 4, p.9).
  - Effective iterative memorisation "correlates with high gating values".
  - Retention: after adapting to a new Pile domain, earlier gains "often do not persist" (Finding 7, p.12).
- **For our candidates.**
  - **(1)** The gate-weighted write is a learned salience signal trained end to end, and it lands on rare entities, not punctuation. The contrast with our attention selectors (which pick punctuation) is the point: a *trained* salience signal can learn content.
  - **(2)** A per-token scalar gate on a read is the simplest licence gate. It only learns to open when the memory is the sole route to the answer.
  - **(4)** Re-reading consolidates; cross-domain retention fails without replay.

### 2.2 Larimar: Das et al. (IBM), ICML 2024, arXiv 2403.11901
*Read: full main text plus App. B–D (scope ablations, ZsRE).*

- **Trained / frozen.** The BERT-large encoder, the prior memory M₀ (512×768) and the decoder (GPT-2 Large or GPT-J 6B) are **trained jointly**. The decoder is not frozen: "During training all the three modules – encoder (e), associative memory (M), and decoder (d) – are jointly trained" (p.3). A pretraining-data term keeps the decoder's LM ability. Training: 7.6M WikiText chunks of 64 tokens, 10 epochs (Larimar-6B, 8×A100).
- **Write.** Closed-form least squares (pseudo-inverse), with sequential updates through a recursive least-squares form. Forgetting uses α = −1 (eq. 3–4). There is no gradient at edit time.
- **Read site.** "From the Z_r (the readout vector), using learned linear projection W_M, the hidden states are transformed and broadcasted to act as a KV cache across all layers" (p.5). That is **one trained KV slot at every decoder layer**. It is the trained analogue of a 1-position slice, and the format is invented, not native.
- **Gate (the important part).** The scope detector decides whether a query reads memory or goes to the unconditioned decoder (§4):
  - **ESD** (external): MiniLM sentence embeddings, 1-NN cosine to the stored facts. "achieves a detection equal-error-rate of 2.9% and an F1 score of 0.974" on 3,800 EasyEdit samples.
  - **ISD** (internal): a binary classifier on Larimar's own encodings, trained with rephrasings as positives and neighbouring facts as negatives.
- **When to use memory / abstention.**
  - Without the scope detector the memory overrides everything. CounterFact Larimar-6B neighbourhood is 80.4 with scope and **13.7 without** (Table 8). The paper explains this as "the model reroutes the prompts from the unconstrained decoder to the memory-constrained one, where the memory influence makes it harder to cover prompts unrelated to in-memory content" (p.14).
  - ISD versus ESD (Table 11): ISD raises paraphrase (94.6 vs 87.6) but collapses neighbourhood to 15.7 (epoch 4) or 45.7 (epoch 8), against 80.6 for ESD.
  - Abstention exists only as a *write*: to block leakage, write "prompt → UNKNOWN". The rephrase-attack success is 17.6%, against 29.0% for ROME and 49.3% for MEMIT (Table 5).
- **Numbers.**
  - CounterFact single edit (Larimar-6B): edit 99.6, paraphrase 88.4, neighbourhood 80.4 (Table 2). With 1–2 rephrases written to memory, paraphrase goes 88.4 → 92.9 → 93.6. ZsRE: 70.4 → 82.2 (Table 13).
  - Batch editing: near 100% up to 512 edits (= K), 82% at 1,024 (Fig. 2a).
  - Selective forgetting: forgotten-fact recall 0.0–0.03, retained 0.993 at N = K, falling to 0.71 at N = 2K (Table 4).
  - Long-context fact recall: 0.89 → 0.86 from 64 to 256 facts (Larimar-1.3B, Table 6).
  - The often-quoted line "it did not show competitive recall performance because it was not trained to perform memory-conditioned generation" (p.8) is about the *Supersizing Transformer baseline*, not about Larimar.
- **For our candidates.**
  - **(3)** This is the closest precedent for "should this query read memory at all?" The best gate was an *external* embedding with a nearest-neighbour threshold, not a trained internal classifier. That makes a strong, cheap baseline for our verification step: an embedding of the question vs the stored subject text, plus a threshold. Our subject-word keys (0.85–0.92 right slice at 59 facts) are already this kind of thing; the missing part is the threshold or abstain branch.
  - **(4)** Writing paraphrases of the fact raises generalisation by 5–12 points, which supports dreamed questions as consolidation data.
  - **(2)** The "UNKNOWN" write is a crude licence-off signal at memory level.

### 2.3 TransMem: Lei et al., arXiv 2607.29032 (AAAI-format preprint, July 2026)
*Read: full paper including appendix.*

- **Trained / frozen.** The backbone is **frozen** (Qwen3-4B-Instruct main; also Qwen2.5-14B, Llama3.1-8B, Qwen2.5-7B, Qwen3-8B). Trained: one TransMem module in each of the **last four layers**. Each module has a single Qwen3-architecture transformer block f_m, a projection W_Δ ∈ R^{d×d} and a gate W_g ∈ R^{d×d}. **Parameter count is not reported.** If each block is full-size it is ~100M per module on Qwen3-4B (my estimate), so this is not tiny.
- **Read site.** At layer K, during answer generation only:
  - Memory = final hidden states of B segments of the context (B = 4; "the optimal value being only 5"). The block attends from the current answer token over these memory states.
  - Then `ĥ = h + g ⊙ W_Δ h_m`, with `g = 2·σ(W_g h_m / τ)`, a per-dimension gate (eq. 6–7).
  - Important: **the student also has the full context in its KV cache**. The memory states are re-read from context that is still present (except chunks beyond the window).
- **Training signal.** "Evidence-conditioned self-distillation": the teacher is the same frozen backbone given only (evidence E, question Q); the student gets the full context C. The loss is forward KL over the teacher's answer trajectory (eq. 3). Data: the HotpotQA training set only. One A100.
- **When to use memory.**
  - The gate learns a layer schedule. Mean gates at layers 32→35 are 1.618, 1.332, 0.975, 0.610, and the same on LoCoMo (App., Fig. 4): "a layer-specific correction schedule rather than a dataset-specific binary switch".
  - No abstention. LoCoMo's **adversarial (unanswerable) category is excluded** ("Following mem0... we exclude the adversarial question category").
- **Numbers.**
  - LoCoMo F1: Qwen3-4B 40.79 → 54.11; Qwen2.5-14B 42.74 → 54.32; Llama3.1-8B 22.39 → 51.64 (Table 1).
  - HotpotQA F1 (4B): 56.00 → 66.20. MemoryAgentBench: 29.54 → 40.00 (Table 2).
  - Ablation (Table 3): without the gate 53.09 (multi-hop 48.04 vs 52.04); without the transformer block 40.79, *exactly the baseline*.
  - Gate variants (Table 7): fixed g = 1 gives 53.09, 2σ(z) gives 54.11, signed 3σ(z) − 1 gives 54.01.
  - Training objective (Table 5): SFT 48.82, GRPO 53.18, OPD 52.80, ECSD 54.11.
  - Hidden-state regression to the teacher (β = 0.3–3) gives 51.79–52.18, worse than 54.11 (Table 9).
  - Number of segments B = 4–32 at inference changes almost nothing (Table 13).
  - Depth (Table 4): Last-4 is best; middle layers 12–16 give 51.11.
- **Caveat** (already in `core-idea.md`): there is no random-memory control, and B-invariance suggests much of the gain may be a learned "attend to the evidence" correction rather than memory content.
- **For our candidates.**
  - **(2)** This is the direct precedent: a trained read module beside a frozen backbone, trained against an evidence-only teacher, is enough to lift answering from information the model already has. That is our licence problem in a different costume (content present, not used).
  - The gate itself is a small part of the gain (+1.0 F1). The block does the work.
  - **(4)** Distil to the teacher's *distribution* and do not add hidden-state targets.
  - Their teacher never abstains (adversarial questions excluded), so the recipe as published would learn "always answer".

### 2.4 Trained Persistent Memory for Frozen Decoder-Only LLMs: Jeong, arXiv 2603.22329
*Read: full paper.*

- **Trained / frozen.** Frozen GPT-2 (124M). Trained: the adapter θ_Mem (read projections, gates). **Write projections are detached, "fixed random maps"** (§5). Memory bank P: 64×d at 1× capacity, persisting across turns and sessions. Training: AdamW for 10 epochs on LoCoMo, truncated backprop over 8 turns.
- **Read sites** (six methods, §4):
  - KV prefix (M.1, M.3 per-layer)
  - a parallel cross-attention inserted after each block with β initialised to 0 (M.2)
  - a Hebbian matrix recall projected to KV (M.4)
  - a **gated additive branch** `H' = H + g ⊙ XAttn(H, P)` with `g = σ(W_g[H; c] + b_g)` and a closed-at-init bias (M.5)
  - slot top-k write with KV read (M.6)
- **Parameters.** ~0.8M (Hebbian) to ~4.7M (gated branch) (Table 2).
- **Numbers** (LoCoMo retained-memory score: F1 with memory minus F1 with memory ablated; Table 4, Fig. 10):
  - At 1×: XAttn 9.02% (peak 17.8%), Hebbian 9.23%, Slot 7.08%. **KV prefix 0.02%, KV extension 0.0%, gated branch 0.09%.**
  - At 10× capacity all six reach 7.6–10.3%.
  - Knowledge gain ΔK: Slot 9.71, Hebbian 7.84, XAttn 7.34, against a baseline of 5.57.
  - Single seed, single benchmark, GPT-2 only (§8.4).
- **When to use memory.** The paper motivates training as needed "for the frozen backbone to discriminate useful memory entries from noise" (p.1, citing Jeong's earlier paper). **The content-dependent gate is one of the methods that failed** at constrained capacity. No abstention.
- **For our candidates.**
  - **(2)** Weak evidence overall (tiny model, effects of a few F1 points). The one useful warning: a learned gate in front of a weak memory signal stays closed, so a gate is not a substitute for usable content.
  - Our case is the reverse: the content is usable (0.86 clean use) and only the licence is missing. That makes a gate *more* promising for us than for Jeong.
  - A KV prefix of *learned* vectors failed at 1×, while our *native* KV slices work without training: native format beats learned format at low capacity.

### 2.5 Pepper et al., "Learning Self-Interpretation from Interpretability Artifacts", ICML 2026, arXiv 2602.10352
*Read: §1–5, App. I–J closely; the rest skimmed. Earlier notes: `papers/self-readout/NOTES.md`.*

- **Trained / frozen.** The LM is frozen (Llama-3.1-8B-Instruct, Llama-3.3-70B, Gemma-2-9B-IT, Qwen-2.5 7B–72B). Trained: an adapter f(h) that maps a stored vector into the layer-0 embedding of a placeholder token in "What is the meaning of 'TOKEN'?".
- **Site.** The input embedding at one position. One injection, read by the whole frozen model.
- **Parameters.** Identity 0; scale-only 1; **scalar affine α·h + b: d+1 = 4,097**; +low-rank up to 2.1M; full-rank 16.8M (Table 1).
- **Training signal and data.** Cross-entropy on label tokens. The pairs are (SAE decoder vector, auto-interp label) or (contrastive topic vector, synthetic description): 45–61K Goodfire features and 49,637 Wikipedia topics. About 10 GPU-hours at 70B.
- **Numbers.**
  - Topic recall@1 is 82.9% (full-rank) against 0.04% untrained (§3.2). Scalar affine: 79.4% (Table 2).
  - Validation loss 4.834 (identity) → 1.787 (scalar affine) → 1.62 (with low rank). The bias alone gives about 85% of the gain (Table 1).
  - 70B SAE labels: adapter 69.7% hit rate against 50.0% for the training labels (Table 3). Bridge entities: 91.0% against 56.4% untrained (§3.6).
  - A LoRA Activation Oracle is comparable (81.6 vs 82.9 recall@1).
- **When to use / abstention.** None. Exact quotes: the bias "encodes a 'default interpretation prior': when applied to a zero vector, it generates generic descriptions matching the training distribution" (p.9). A zero vector yields "the 1960s British rock band that pioneered psychedelic rock" (Qwen-7B, Table 17). "The scale controls how much input semantics override the prior" (App. I.4).
- **For our candidates.**
  - **(2)** This is the strongest evidence that a d+1 adapter can fix a frozen model's read of an injected vector. It is also the clearest warning: most of the gain is a learned *default*, and with no input the adapter produces a fluent, confident, wrong answer. Our "told" direction restoring answering was found to be mostly a generic comply direction. That is the same object.
  - **Design rule:** train with null and wrong inputs whose target is "I don't know", and evaluate on them. Otherwise the gate becomes our generic comply direction plus a bias, and retrieval errors turn into confident wrong facts (we already see that from a wrong slice).
  - Simpler adapters generalised better across datasets than full-rank ones (Table 2).

### 2.6 SleepGate: Xie, "Learning to Forget: Sleep-Inspired Memory Consolidation...", arXiv 2603.14517
*Read: §1, §3–4, Table 2, §8. Proof-of-concept only.*

- **Trained / frozen.** All trained from scratch: a 4-layer, 793K-parameter transformer on synthetic proactive-interference sequences. The modules:
  - a conflict-aware tagger: a semantic signature from a projected key plus local pooling, and a "superseded" flag when a later signature has cos > δ
  - a forgetting gate: a 2-layer MLP, hidden size 128, "< 0.01%" of a base model
  - a consolidation step
- **Read site / gate.** For each KV entry i, features `[k_i ‖ v_i ‖ age ‖ signature ‖ superseded flag ‖ cumulative attention ‖ global context]` → retention score r_i. That score enters attention as a **pre-softmax bias b_i = β·log r_i** with β = 5 (eq. 10–11), so entries are down-weighted, not deleted.
- **Training signal.** LM loss in the wake phase, plus retrieval accuracy after consolidation, plus a BCE "gate alignment" loss that pushes the gate toward the tagger's superseded labels (§4).
- **Numbers (Table 2, p.12).**
  - Retrieval accuracy 99.0 / 99.5 / 97.0% at 2 / 5 / 10 superseded updates. It falls to 73.5 / 30.5 / 16.5% at 15 / 20 / 30, with 62% stale retrieval at n = 30.
  - All baselines are below 18%. Full KV is 8.0% even at n = 1, which says more about the 793K model than about full KV.
  - "Whether the mechanism transfers to pre-trained models at scale ... remains to be validated" (§8.5).
- **For our candidates.**
  - **(1)** This is the only paper here with a learned per-cache-entry keep score trained on downstream retrieval. It is the template for an amortised S-loo: features of a position (its k, v, attention received, the subject and value tags), label = the S-loo outcome, output = a keep score.
  - **(2)/(3)** The attention-logit bias is a soft way to apply a verification score to a retrieved slice: down-weight a doubtful slice instead of a hard abstain. It also targets binding interference directly (superseded vs current), which is related to our separately-stored-slices collisions.
  - Toy evidence only.

### 2.7 Lee, McLeish, Goldstein, Fanti, "Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference", arXiv 2605.26099 v3
*Read: full main text (§1–6.4).*

- **Trained / frozen.** **Not frozen.** Small GDN–attention hybrids are trained from scratch for the synthetic tasks. Pretrained Jet-Nemotron 2B (a hybrid derived from Qwen2.5-1.5B) and Ouro 1.4B (looped, plus 6 inserted Jet layers, < 10% extra parameters) are fine-tuned end to end. Muon, 1–2 H100-days per run.
- **Mechanism.** When the context window is full, the model runs N extra forward passes over the window ("sleep"). These passes update only the recurrent fast weights; then the KV cache is evicted. Prediction is a single pass from the refined state. Gradients flow through the refined fast weights (Alg. 1). For Jet, the loop covers the middle 14 of 28 blocks.
- **Data.** Rule 110 automaton (depth t), Depo k-hop graphs, GSM-Infinite (2,000–3,300 tokens, 1–8 operations, no chain of thought). In GSM-Infinite **the question is placed before the context**, "allowing it to selectively consolidate information relevant to the question" (§6.3).
- **Numbers.**
  - Rule 110, t = 32: no loop ~10% exact; 2 loops ~20%; 3–4 loops > 30% (Fig. 2b).
  - GSM-Infinite, Jet with 6 loops: 6-op 0.742 → 0.812, 8-op 0.351 → 0.388. Ouro with 4 loops: 6-op 0.419 → 0.615 (Fig. 4).
  - Sliding-window eviction (L = 512): 2-op 0.596 → 0.905 (Fig. 5).
  - Gains are largest on the deepest problems.
- **When to use.** Not addressed. The relevant quote: "giving the model access to a sliding-window KV cache can make the newly inserted Jet layers underutilized ... using hard eviction for the warm-up stage is crucial" (§6.4).
- **For our candidates.**
  - **(4)** This is the closest substrate to our 2B work: consolidating evicted context into the GDN state with offline compute. It argues that consolidation is a computation that benefits from depth, not a copy.
  - It also suggests a cheaper candidate-4 target on the hybrid: a per-user *state* refined offline, rather than a LoRA. But their model is trained to read such states; Qwen3.5 is not.
  - "Question before context" is their form of query-aware selection, and **(1)**'s dreamed questions play the same role.
  - Training rule: remove alternative routes (hard eviction) so the memory path is learned.

### 2.8 S0 Tuning: Young, arXiv 2604.01168 v2
*Read: §1–5.6.*

- **Trained / frozen.** All weights frozen. Trained: one initial state S₀ per recurrent layer, injected at t = 0 and scaled by α = 0.07 on Qwen3.5.
- **Parameters.** 12.6M on Qwen3.5-4B (24 GDN layers × 32×128×128; 0.3%); 34.6M on FalconH1-7B. About 48 MB.
- **Training signal and data.** Completion-only cross-entropy on about **48 execution-verified HumanEval solutions sampled from the model itself**, with L2 on S₀.
- **Numbers.**
  - HumanEval pass@1, Qwen3.5-4B: 48.8% → 72.2% (+23.6 ± 1.7 pp, 10 seeds), against LoRA rank 24 at 61.5% (Table 1).
  - Parameter-matched LoRA rank 64: −15.5 ± 18.9 pp, 8 of 10 seeds negative; rank 48 has one seed collapsing to 4.8% (Table 2).
  - Scale (§5.4): Qwen3.5-0.8B +2.6 (n.s.), 2B **+19.0 ± 1.2**, 9B +44.0.
  - A per-step state-offset variant gives +27.1 pp.
  - Mechanism: "S₀'s direct influence decays to 0.03% KL by the end of the prompt, yet 85% of corrected solutions diverge at the first generated character" (p.2).
  - A prefix-tuning control on Qwen2.5-3B *degrades* by −13.9 pp in all nine configurations.
- **For our candidates.**
  - **(4)** In the few-example regime a per-user consolidation will face, LoRA is unstable and a state-shaped parameter is not. On Qwen3.5-2B a learned per-user state (or state offset) is a strong alternative target to LoRA.
  - **(2)** The effect works by changing the first generated token, which is where our refusal is decided. A state-side licence term on 2B is plausible: our state-only arm already gives "recognition", and Afendulev puts response mode in the state.
  - On the pure transformer (Qwen2.5-1.5B) the analogous prefix-tuning failed, so there the licence piece should be a ReFT-style intervention, not a learned prefix.

### 2.9 Afendulev et al., "What Attention Recalls and Recurrence Controls in Hybrid Language Models", arXiv 2609.04434
*Read: §4–6 plus the Appendix F table. Full notes: `dreaming/literature-exp1.md`. Additions only.*

- No training; frozen Qwen3.5-4B/9B/27B, Falcon-H1 and others.
- Key numbers (Qwen3.5-4B, App. table):
  - KV retrieve: full 1.00, recurrent-only 0.00, KV-only 0.89.
  - Language-following: full 0.97, recurrent-only 0.70, KV-only 0.01.
  - State swap: value from the KV side and language from the state side on ≥ 99% of items.
- Quote: "The KV cache lets the model report what the instruction was. To actually follow it, the recurrent state has to be carried over" (§5.2). And: "The recurrent state works as a compressed prior that shapes language, persona, and semantic field" (§6).
- **For our candidates.**
  - **(2)** On the hybrid, a licence piece that only touches attention-side slices may miss the place where "mode" lives. Our attention-only slice refuses; our state-only arm gives yes/no but not the value. A trained licence term for 2B should include a state component (S0-style) or be trained with the state present.
  - **(3)** The state is a familiarity signal. Afendulev's state accepts associated lures (DRM 0.87–0.89). Ours rejects form and associate lures (balanced 0.96, addendum 7). That makes our state usable as a *verification* read-out, a training-free candidate-3 component. This use is ours.

### 2.10 Ferrando, Obeso, Rajamanoharan, Nanda, "Do I Know This Entity?", ICLR 2025, arXiv 2411.14257
*Read: §5–7 closely; §1–4 skimmed. Earlier notes: `dreaming/literature-exp1.md`.*

- No training of the LM. Directions come from base-model SAEs (Gemma Scope) and are applied to Gemma 2 2B/9B IT (and Llama 3.1 8B).
- **Site.** The last token of the entity, plus the end-of-instruction tokens.
- **Numbers.**
  - Steering with the unknown-entity latent gives "almost 100% refusal across all entity types in Gemma 2 2B". The known-entity latent "slightly reduces refusal rates". Projecting the unknown direction out of the weights gives "a large reduction in refusal responses" (Fig. 3, p.6). 9B and Llama 3.1 8B show "less pronounced effects".
  - The steering coefficient α is 400–550, "around two times the norm of the residual stream" (fn. 5). These are large pushes.
  - Uncertainty latent (Gemma 2B IT, layer 13, at the `model` token) separates correct from wrong answers with **AUROC 73.2, F1 72** (§7, p.9).
- **Mechanism (p.7).** "Steering with the top unknown entity latent reduces the attention to entity, even in prompts with a known entity, while steering with the known entity latent increases the attention scores", in the attribute-extraction heads (L18H5, L20H3).
- **For our candidates.**
  - **(2)** The licence is not a free-floating "comply" bit. It is a feature at the subject that controls whether downstream heads *read* the subject.
  - That explains our results in one picture: adding the subject's slice position raises answering, the value's probability is unchanged under teacher forcing, and a generic "told" direction at the end can force answering (a second route, the "comply" end).
  - It recommends placing a trained licence intervention at the restored subject positions, on the residual stream from which those heads' keys are computed.
  - **(3)** An end-of-instruction uncertainty read-out at AUROC 73 is a modest, training-free verification signal: useful as a feature for a learned abstain head, not a gate on its own.

### 2.11 Ruscio & Thompson, "On the Tip of the Tongue: Why LLMs Hallucinate Answers They Can Decode", arXiv 2603.13911 v2
*Read: abstract, §1, Table 2 headline, §5.6.*

- No training (probes only). Frozen Llama-3.x, Qwen2.5/3 and Mistral checkpoints.
- **Numbers.**
  - 23.2–41.8% of failures are readable by logit lens; 27.2–30.8% by a trained probe (shuffled-label control: 10.0–14.9%).
  - Setting the answer projection to its typical success value recovers first-token selection in 83.0–92.2% of failures (0.0–0.6% for random edits).
  - **But "the full answer is correct in 30.5–38.0% of the same failure cohort"** (§5.6). They name three separate things: "decodability, recoverability, and generation".
- **For our candidates.**
  - **(2)/(3)** This is the published version of our teacher-forced vs generation gap (0.99 vs 0.17–0.49).
  - Any trained licence piece judged only at the first token or teacher-forced will look much better than it is. Score in generation with our judge panel.
  - A refusal decided at the first token is a separate failure from wrong-answer selection. Their analysis covers wrong selection, not refusal.

### 2.12 ReFT: Wu, Arora et al., "Representation Finetuning", NeurIPS 2024, arXiv 2404.03592
*Read: §1–5 and App. D.2, E, F, G closely; Tables 1–4.*

- **Trained / frozen.** The LM is frozen. Trained: interventions Φ at chosen (layer, positions).
  - LoReFT: `Φ(h) = h + Rᵀ(Wh + b − Rh)`, R ∈ R^{r×d} with orthonormal rows (eq. 2).
  - DiReFT: `h + W₂ᵀ(W₁h + b)` (eq. 3).
  - Positions: the first p and last s prompt tokens; layers from a set; weights tied or untied across positions.
- **Parameters.**
  - 0.031% of LLaMA-7B (commonsense); 0.0039% for instruction tuning ("≈1MB parameters on disk").
  - Rank-1 single-layer, single-position: **4,097 parameters** (7B).
  - Rank-4 at layer 15, last prompt token: 32,772.
- **Signal and data.** Plain cross-entropy on outputs. Hyperparameters were tuned on dev sets: positions p + s ∈ {1…11}, rank 8–64, layers.
- **Numbers.**
  - Commonsense avg (LLaMA-7B): LoReFT 80.2 vs LoRA 74.7 and DoRA 78.1 (Table 1).
  - Arithmetic with chain of thought: worse than LoRA (42.6 vs 46.9, Table 2). The authors suspect long generations dilute a prompt-position intervention.
  - Alpaca-Eval: 85.60 vs LoRA 81.48 (Table 3).
  - Memorisation: rank-1, last token, one layer recovers the first **2,048 tokens of Alice in Wonderland at 100%**, but random token sequences only up to 128 (App. F.1). It stores **256 random ID→token pairs** "with near-perfect recall in layer 20 of the 13B model" (App. F.2).
  - GOODY-2 imitation: "rank-4 LoReFT at layer 15 on top of the last prompt token (32,772 learned parameters in total)", 5 examples, 100 epochs, < 30 s (App. G.2).
  - Practical notes (App. D.2): "Intervening on multiple positions delivers significant gains"; rank < 32 is enough; tie weights.
  - The input-dependent term matters. On arithmetic, a bias-only `h + Rᵀb` gives 31.3 (7B) and 39.5 (13B), against 42.6 and 49.6 for LoReFT. Input-dependent variants match LoReFT (App. E, Table 17). Unlike Pepper's setting, a fixed direction is *not* most of the effect here.
- **When to use / abstention.** No conditional behaviour is studied. GOODY-2 is *blanket* refusal, the opposite of what we need, but it shows that refusal and compliance are trainable with this tiny a surface.
- **For our candidates.**
  - **(2)** A LoReFT at the restored slice positions (subject + value + full stop), rank about 4, at a few mid layers, trained with the TransMem-style teacher on dreamed answerable *and* unanswerable questions, is the most direct test of "a tiny adapter fixes the licence". It is input-dependent (the Wh term), so unlike a fixed "told" vector it *can* condition on whether the slice's content matches the question. Whether it does is exactly what the negatives must test.
  - **(4)** A per-user rank-1 ReFT is a consolidation target about 1,000× smaller than a typical LoRA, and it holds 256 pairs at 13B. Untested: paraphrased questions, and capacity at 1.5B/2B.

### 2.13 Titans: Behrouz, Zhong, Mirrokni, arXiv 2501.00663
*Read: §3.1, §3.3, §4, §5.3–5.4. Targeted.*

- Trained from scratch (170M–760M parameters, 15–30B tokens). The memory is an MLP whose weights are updated at test time by a gradient "surprise" with momentum and a forgetting gate α_t. There are also learnable data-independent "persistent memory" tokens.
- **Read options.**
  - MAC: retrieve h_t = M*(q_t) and prepend it, with the persistent tokens, to the segment's attention. The paper's claim: "Attention by having both historical and current context, has the ability to decides whether given the current data, the long-term memory information is needed" (p.9).
  - MAG: SWA output ⊗ memory output via a normalised gate.
  - MAL: memory as a layer.
- **Numbers.** S-NIAH at 16K: MAC 98.4 / 97.4 / 95.2 (PK/N/W), against Mamba2 5.4 / 0.0 / 0.0 (Table 2). BABILong: beats GPT-4 in the fine-tuned setting (Fig. 6).
- **For our candidates.**
  - **(2)** MAC is the architectural form of our slice channel: retrieved memory appears as extra attention entries, and attention decides whether to use them. It works because attention was trained with those entries present. Our frozen model was not, which is why the licence is missing.
  - **(1)** Surprise-weighted writes are another learned salience signal.
  - Not a frozen-backbone result.

### 2.14 RF-Mem: Zhang et al., ICLR 2026, arXiv 2603.09250
*Read: abstract, §1, the gating section (§3), result headlines.*

- No LLM training. A training-free retriever over *text* memories (PersonaMem, PersonaBench, LongMemEval).
- **Gate.** Familiarity = the mean top-K similarity s̄ plus the entropy H(p) of the softmaxed scores. Above θ_high, use direct top-K; below θ_low, run iterative "recollection" (cluster plus query mixing); in between, decide by H(p) ≤ τ (eq. on p.5–6). Gains are small (for example about +0.02 Recall@5 on LongMemEval).
- **For our candidates.** **(3)** A ready-made, training-free confidence rule over retrieval scores (mean plus entropy). It applies directly to our subject-word key scores as the first, untrained version of "verify or abstain". RF-Mem uses low familiarity to search harder, never to abstain.

### 2.15 Tensor Memory: Swain et al., arXiv 2605.27686
*Read: §1–3, Table 1.*

- Trained from scratch on small models (8-layer, d = 384 LM; ViTs). A 3D voxel ConvLSTM memory, with a Gaussian "splat" write and a trilinear read.
- **Fusion.** `x̂ = x + σ(γ)·D(m)` with one *global* learned scalar γ. "tasks that do not require memory drive γ negative" (p.5).
- WikiText-2 test PPL 130.30 → 100.07 (25.9M → 28.3M parameters).
- **For our candidates.** Off-topic. A single global gate cannot express "this query, this memory".

### 2.16 MemoryLLM: Jaiswal et al. (Apple), "Plug-n-Play Interpretable Feed-Forward Memory for Transformers", arXiv 2602.00398
*Read: abstract, §1–2.2, §3.2 (Table 1).*

- Trained from scratch (250M, 750M, 1B). The FFNs see only the token embeddings X₀, never the residual stream, so they become context-free token-indexed lookups that can be precomputed and offloaded.
- Down-scaling the FFNs (α = 0.5) hurts recall-heavy tasks far more than reasoning ones: LAMBADA −69.9%, ARC-E −38.5%, against BoolQ −1.2% (Table 1).
- **For our candidates.** Off-topic. As noted above, this is not the memory-pool MemoryLLM the README lists.

---

## 3. What is still open / ours

**Not done in any paper in this cluster:**

1. **A trained read path over *native* cached K,V slices for a frozen model, across sessions.** The nearest relatives:
   - Larimar: one trained KV slot from a trained encoder; decoder fine-tuned.
   - Jeong: learned KV prefixes, which failed at 1× capacity.
   - Titans MAC: retrieved memory as attention entries, trained from scratch.
   - TransMem: re-reads states still in context, last 4 layers only.

   Ours is the only setup where the content already works untrained (0.86 clean use, against 0.27 for the same tokens as text) and only the *licence* is missing.
2. **Trained abstention conditioned on memory.** No paper trains "answer from the slice, or say you don't know, depending on whether the retrieved slice actually answers the question". TransMem drops the adversarial category; Pepper's adapter answers a zero vector; ReFT trains blanket refusal; Larimar's gate only routes. Our setup can supply the negatives cheaply: no slice, a wrong retrieved slice, lure slices, and dreamed unanswerable questions. Our LongMemEval refusal rate (0.70–0.79 with gold slices) gives a clear target: lower refusal on gold, no rise in confident wrong answers on wrong slices.
3. **Content vs licence as separate quantities in a memory read path.** Ferrando gives the mechanism (entity features gate extraction-head attention and refusal); Afendulev gives the hybrid split (state = mode). Nobody measures, *in a memory system*, value probability unchanged while answering changes, or shows that the subject position carries the licence. A trained licence adapter placed at the subject positions, compared with one at the question end, would test Ferrando's mechanism in our setting.
4. **Verification from the recurrent state.** On names, our state-only yes/no is item-specific (balanced 0.96 against lures). No paper uses a hybrid's state read-out to verify a retrieved KV slice before answering. That is a training-free candidate-3 component unique to the hybrid. Afendulev's lure result means it must be re-checked on non-name content.
5. **Amortised position selection across sessions.** SleepGate learns per-entry keep scores, but within one stream, on a 793K toy, for supersession. FwPKM's gate-weighted writes are within a stream and trained from scratch. Learning to predict S-loo (which 2 positions of a past turn will carry the answer in a future session) from position features is not done anywhere.
6. **Binding-safe consolidation.** None of these papers touch our binding result (storage order; the k-th memory as the k-th fact). SleepGate's supersession tagger is the nearest idea (entries with the same "slot" signature conflict). For candidate 4, open questions: does consolidating N separately stored slices into one adapter reproduce the swaps, and does replaying them in joint encoding (FwPKM-style re-reading, or Lee-style offline passes) remove them?

**Concrete design choices this cluster suggests (to argue about, not a plan):**
- **Candidate 2.** Start with a LoReFT (rank 4, untied, at the restored subject + value positions, a few mid layers) on Qwen2.5-1.5B. Use a TransMem-style KL teacher, with the model reading the fact as text in context. Data: dreamed answerable questions plus negatives (no slice, wrong slice, lure). On Qwen3.5-2B, add an S0-style learned state offset. Report generation and judge scores, not teacher-forced ones. Ablate to bias-only to see whether the adapter conditions on content or is just "comply". Pepper's bias was 85% of the gain; ReFT's bias-only lost about 10 points on arithmetic (App. E). Which of the two our case resembles is an empirical question.
- **Candidate 3.** Baseline = Larimar-style external-embedding 1-NN with a threshold, plus RF-Mem's mean and entropy over our subject-key scores. Learned upgrade = an abstain head fed by key score, our state recognition read-out (2B), and Ferrando's end-of-instruction uncertainty direction. Optionally apply the verification score as a SleepGate-style attention-logit bias on the slice, not a hard switch.
- **Candidate 4.** Try a rank-1/rank-4 ReFT or an S₀ state per user as well as LoRA. Expect LoRA instability with few dreamed questions (S0, Table 2). Include paraphrases (Larimar) and replay of earlier users' or facts' data (FwPKM's retention failure).
- **Candidate 1.** Model it on SleepGate's gate: features of a position (its k/v, received attention, token type, entity tag) → keep score. Train against S-loo labels, with a ranking loss on the ~2 positions per turn.
