# 1. The core idea

> **Seahorse**: "hippocampus" is Greek for seahorse. The project builds a small, fast,
> experience-driven memory that sits beside a large, frozen language model, the way the
> hippocampus sits beside the neocortex.

This document explains *why* the project exists, what it is trying to build, where the
ideas come from, and how it relates to existing work. The concrete mechanism is in
[02-method.md](02-method.md), the code in [03-code-map.md](03-code-map.md), results in
[04-experiments.md](04-experiments.md), and open directions in
[05-next-moves.md](05-next-moves.md).

---

## 1.1 The problem: LLMs have records, not memories

A language model has two kinds of "memory".

| | Weights | Context / KV cache |
|---|---|---|
| Analogy | Neocortex | Sensory buffer |
| Speed of learning | Slow (pre-training, fine-tuning) | Instant (whatever is in the prompt) |
| Content | General knowledge, consolidated | Verbatim history of the current session |
| Lifetime | Permanent | Gone when the session ends |

There is **nothing in between**: no fast, persistent, selective memory that forms from
experience. Animals have exactly this in the hippocampus.

### The KV cache is a record, not a memory
Three observations, made while the project was being framed, drove the design.

1. **The KV cache contains no information beyond the tokens.**
   *The Residual Stream Is All You Need* (Qasim et al., 2026, arXiv 2603.19664) shows that
   keys and values are deterministic projections of the residual stream (K = RoPE(RMSNorm(h)·W_k),
   V = RMSNorm(h)·W_v), reconstructable bit-exactly. Given the weights, the cache is a
   deterministic function of the text.
   **Consequence:** the lossless compression of a KV cache is *just the text*. Storing
   activations instead of text only helps if you either (a) skip recomputation (an
   engineering gain), or (b) store something *lossy* that keeps what text does not express
   compactly. Memory lives in (b), so **deciding what to lose is the whole problem**.
2. **The limit is the cost of reading.** Every new token attends over everything before it.
   A store whose lookup cost grows with its size stops being usable long before it is full.
   There's also an attention-quality ceiling: the more there is to attend over, the blurrier
   retrieval gets (Hopfield metastable averaging; "lost in the middle").
3. **The cache is made by one glance and never revisited.** Each token gets one forward pass
   with a fixed compute budget. Because of the causal mask, the representation of token 10
   never learns what token 1000 revealed. Nothing is re-read with hindsight, consolidated,
   merged or forgotten.

The KV cache is **cheap to write and expensive to read**. A real memory is the reverse:
**expensive to write** (you pay compute to decide what matters) and **cheap to read** (the
result is small and self-contained).

### "Session-bound" is really "context-bound"
A cache can be saved to disk, as prompt caching already does. The real limitation is
**composability**: each cached entry is only valid inside the exact prefix that produced it.
Its key has its position baked in by RoPE, and its content was computed while attending to
its neighbours. Moved into a new session, it's an orphan. An entry that is **self-contained**
("the user is vegetarian" rather than "token 4812 as seen by token 4811") is portable. So
**abstraction and persistence are the same problem**.

---

## 1.2 What we want: memory as animals have it

The target is a memory system with the properties biological memory has.

| Property | Meaning for an LLM |
|---|---|
| **Experience-driven** | Nobody calls `write()`. Memories form as a by-product of the model processing its own interactions. |
| **Reconstructive, not verbatim** | Recall regenerates content through the model's own priors (Bartlett), rather than replaying a transcript. |
| **Salience-gated** | What gets written depends on internal signals (surprise, confusion, emotional weight), not on an operator. |
| **Cue-driven recall** | Memories come back when the present *resembles* the past, without an explicit query. |
| **Compressed relative to priors** | Store only what the model could not already predict. The rest is regenerated from the weights. |
| **Revisable** | Contradicting evidence rewrites a memory (reconsolidation). Old memories can be suppressed without being erased (extinction). |
| **Persistent** | Survives across sessions, because entries are self-contained. |

Two directions were discussed and **parked**:
- **Memories created by reflection:** the model re-reading its past with hindsight and storing its *conclusions*. Parked until there's a reliable mechanism that shows memories can be created at all.
- **Cold start:** the first experiences write everything, because memory predicts nothing yet. Parked until something works.

---

## 1.3 Inspirations from neuroscience and psychology

### Reconstructive remembering (Bartlett, 1932)
In *Remembering*, Bartlett showed that recall is reconstruction, not replay: people rebuild
memories from fragments plus their own schemas, and details that don't fit get dropped or
reshaped. This is the central behavioural target.

A design question follows directly: **when a memory conflicts with the model's priors, which
should win?** Larimar always lets memory win (faithful, but gullible). Humans let priors
reshape the memory (reconstructive, but distorting). A biological system presumably weighs
the two by confidence, surprise and salience.

### Complementary learning systems (McClelland, McNaughton & O'Reilly, 1995; Kumaran, Hassabis & McClelland, 2016)
The brain pairs a **fast hippocampal system** (one-shot storage of episodes) with a **slow
neocortical system** (gradual extraction of structure). Replay during sleep lets the
hippocampus teach the cortex without catastrophic interference. In Seahorse, the frozen LLM
is the neocortex and the memory operator is the hippocampus. Consolidation into weights is a
parked future step.

### Hippocampal indexing theory (Teyler & DiScenna, 1986)
The hippocampus stores an **index** that re-evokes cortical activity patterns, not the
content itself. Recall is **reinstatement**: the cortex re-enters its earlier state, and
fills gaps from its own knowledge. That is exactly what makes remembering reconstructive.
This motivates (a) storing *compact cues* rather than full states, and (b) the finding that
**the same cortex that perceived should be the one that is indexed**, rather than a
separate encoder as in Larimar.

### Pattern separation, pattern completion, and the CA1 comparator
- **Dentate gyrus:** expands and sparsifies inputs, so similar episodes are stored distinctly (**pattern separation**).
- **CA3:** a recurrent auto-associative network that completes a whole memory from a partial cue (**pattern completion**, an attractor).
- **CA1 comparator** (Lisman; Hasselmo): compares what CA3 recalls (the prediction from memory) with what arrives from the cortex (reality). A mismatch signals novelty and drives encoding.

The **delta-rule write** in Seahorse is the CA1 comparator: the memory stores only the part
of an experience it could not already predict (§2.5).

### Surprise, uncertainty and neuromodulators
Encoding strength in animals is modulated by diffuse chemical signals.

| Signal | Neuromodulator | Seahorse analogue |
|---|---|---|
| Unexpected change, surprise | Noradrenaline (Yu & Dayan, 2005: *unexpected* uncertainty) | Prediction error measured against memory (built into the delta rule) |
| Known unreliability, "I don't know" | Acetylcholine (Yu & Dayan, 2005: *expected* uncertainty) | The **confusion signal**: the entropy of the next-token distribution (the user's proposal) |
| This matters, reward, novelty | Dopamine, amygdala | Valence and arousal directions (the emotion paper) |

Two behavioural findings shape how uncertainty should gate writing:
- **Curiosity** (Gruber et al., 2014): a state of curiosity *before* an answer boosts memory for the answer (via the dopamine–hippocampus loop). The confused state primes encoding of whatever resolves it.
- **Hypercorrection** (Butterfield & Metcalfe, 2001): errors held with *high* confidence are remembered best once corrected. Entropy is silent here (the distribution is sharp); the surprise-against-memory term catches it instead.

### Reconsolidation and extinction
- **Reconsolidation** (Nader et al., 2000): a recalled memory becomes labile and is re-stored, possibly modified. In Seahorse a contradiction produces a large delta-rule error, which rewrites the memory.
- **Extinction learning:** when a feared cue becomes safe, the fear memory is not erased. A new *inhibitory* memory suppresses it, and the old one can return under stress. So a memory system probably needs both excitatory steers ("do this") and inhibitory projections ("not this"), which is analogous to abliteration.

### Predictive coding and model-relative information
From the model's point of view, a conversation's information content is the sum of
per-token surprisal (−log p). Everything predictable can be regenerated from the weights;
this is literally how arithmetic coding with a language model works. So a memory should hold
the **residual between what happened and what the model expected**. Surprise-based writing
isn't a heuristic borrowed from neuroscience; it's the information-theoretically correct
thing to do.

---

## 1.4 Premises from the LLM literature

These are the empirical facts the design is built on.

### P1: The residual stream is the only causal channel
*The Residual Stream Is All You Need* (arXiv 2603.19664) proves the Markov property:
everything downstream of layer ℓ is a function of the residual stream at ℓ. Replacing the
recipient prompt's residual wholesale with a donor prompt's makes the output the donor's
(KL = 0).
- **So any memory that affects behaviour must act by writing into the residual stream.**
- **Adding a full stored state hijacks the computation** rather than adding to it. So store *deltas*, not states (§2.5).
- Attention itself is a dynamic steer: each head adds a context-dependent vector.

### P2: Transformers keep no persistent state in the residual stream
*Emotion Concepts and their Function in a Large Language Model* (Sofroniew et al., Anthropic,
2026, arXiv 2604.07729) finds that emotion representations are **locally scoped**: they
encode the operative concept at each token position, not a persistent state. Long-range
tracking happens by **attending to representations cached at earlier positions**. The authors
suggest that any chronic state "is likely represented… implicitly in the model's key and
value vectors… recalled when needed by the model's attention mechanism."

This premise turned out to predict the v0.1 result: the steer carries associations, but not
premises the model can reason with ([04-experiments.md](04-experiments.md)).

### P3: Abstract state lives at mid-to-late layers and is linearly steerable
Same paper:
- Early layers encode literal, token-level content ("sensory").
- Mid-to-late layers encode integrated, contextual meaning ("action"): negation is resolved there, as is "8000mg of this drug means danger".
- Integrated state collects at boundary tokens (the Assistant ":" probe predicts response emotion at r=0.87, versus 0.59 on the user's turn).
- Adding concept directions at about two-thirds depth causally changes behaviour.

Related: Contrastive Activation Addition (Rimsky et al., arXiv 2312.06681), ActAdd (Turner et
al., arXiv 2308.10248), abliteration / refusal direction (Arditi et al., arXiv 2406.11717).
*(IDs for papers not read in this project are cited from memory; verify before relying on them.)*

### P4: Attention is associative-memory retrieval
*Hopfield Networks is All You Need* (Ramsauer et al., 2020, arXiv 2008.02217): the
transformer attention update is the modern Hopfield update.
- Retrieval is sharp only when stored patterns are **well separated**. Near-duplicate keys collapse into a metastable average. So merging near-duplicates loses little, which gives a principled compression rule.
- Capacity is exponential in dimension for *random* patterns. But real LLM keys are low-rank and anisotropic (effective rank ~30% of head dimension, per the Residual Stream paper), so practical capacity is much lower.
- Early-layer heads mostly average globally, so memory belongs in the middle and upper layers.

### P5: Writes can be untrained; reads usually need training
- **Larimar** (Das et al., ICML 2024, arXiv 2403.11901) writes in closed form, but trains the encoder, prior memory and decoder jointly. Their own note: a memory model "did not show competitive recall… because it was not trained to perform memory-conditioned generation."
- **Trained Persistent Memory for Frozen Decoder-Only LLMs** (Jeong, 2026, arXiv 2603.22329) uses fixed *random* write projections, but trains the read side. Its claim: "without trained projections, accumulated cache states dilute attention."
- **TransMem** (Lei et al., 2026, arXiv 2607.29032): removing its learned transformer block drops performance exactly to the no-memory baseline (40.79).

### P6: Compression trades off against native readability
- **Native activations** (InfLLM's stored KV) are written in the model's own attention language, so any model reads them without training. But they're uncompressed: one entry per token.
- **Invented latent codes** (Larimar's z) compress a fact into one vector, but writer and reader must be trained together ("married").

Biology resolves this because the hippocampus and cortex **develop together**: the index code is native to the cortex that reinstates it.

### P7: Training-free retrieval into attention already works
- **InfLLM** (Xiao et al., 2024, arXiv 2402.04617) stores evicted KV in blocks, retrieves the relevant blocks with the model's own query·key scores, and splices them into attention. Training-free: Mistral-7B ∞-Bench average 25 → 58, and passkey retrieval at 1M tokens.
- **EM-LLM** (Fountas et al., ICLR 2025, arXiv 2407.09450) adds surprise-based event segmentation (surprise sets *segment boundaries*; everything is still stored) and graph-refined boundaries. It beats RAG and full context on LongBench, with passkey at 10M.

So "store activations and retrieve them through attention" isn't new on its own. What these systems lack is compression, cross-session persistence, and anything learned.

---

## 1.5 The design that follows

Putting the premises together gave the working design, the **modulatory channel** that v0
and v0.1 test.

1. **Memory is a steer** (P1), applied at **mid-to-late layers** (P3), where concepts live and lexical noise (typos) has been filtered out.
2. **Fast weights, not a filing cabinet:** an operator M applied as **h ← h + α·M·k(h)**, rewritten by experience. The current state *is* the retrieval cue, so recall happens by resemblance without explicit queries. It's a plastic form of abliteration (many directions, learned on the fly, no weight changes), and it belongs to the fast-weight lineage (Schmidhuber; linear attention; the linear case of Titans).
3. **Store deltas, not states** (P1). Measure the change an experience caused against **what memory itself predicted** (the delta rule; the CA1 comparator). Habituation and reconsolidation emerge.
4. **Salience from inside the model:** surprise against memory, uncertainty (entropy), and valence/arousal.
5. **Frozen cortex:** the base model is never modified. Training is allowed where needed (the read path).

The mechanism is in [02-method.md](02-method.md). The experiments (v0, v0.1) found that this
channel carries **which** fact and **which way** a preference points, but not **premises the
model reasons with**. That points to a second, **episodic** channel that reinstates content
where attention can use it ([05-next-moves.md](05-next-moves.md)).

---

## 1.6 Related work: how Seahorse differs

| Work | What it stores | Write | Read / fusion | Trained? | Relation to Seahorse |
|---|---|---|---|---|---|
| **KV cache** | Every token's K,V, per layer | Automatic, verbatim | Attention | No | A record, not a memory. Holds no information beyond the tokens (P1). |
| **The Residual Stream Is All You Need** (2603.19664) | One residual per token (KV-Direct) | Checkpoint | Recompute K,V | No | Source of P1. Its "one 5KB vector serves all layers" claim only holds with full recomputation. Per-layer residuals cost L·d·b, which can exceed the KV cache for GQA models (e.g. ~170KB vs 136KB per token on Gemma-3-4B). |
| **InfLLM** (2402.04617) | Evicted KV blocks + representative tokens | Automatic | Top-k blocks spliced into attention; fixed position | No | The training-free baseline for an *episodic* channel. Uncompressed, single session. |
| **EM-LLM** (2407.09450) | KV grouped into surprise-bounded events | Automatic | kNN + contiguity buffer, per layer | No | Surprise sets boundaries only; nothing is filtered or compressed. |
| **Larimar** (2403.11901) | Fixed K×C matrix (512×768) of *encoder* latents | Closed-form pseudo-inverse; recursive-least-squares update; exact forgetting with α=−1 | One KV slot per decoder layer + added to input embeddings | Encoder, prior memory, decoder jointly (decoder fine-tuned) | Episodic *mechanics*, semantic *content*: stores explicit facts handed to it. Separate encoder ("someone else's senses"); memory always overrides priors. Reference code in `../larimar/` (sequential write and forgetting are *not* in the released code). |
| **Titans** (2501.00663) | An MLP memory updated at test time | Gradient of ‖M(k)−v‖² with momentum and weight decay ("surprise") | Memory-as-context / as-gate / as-layer | Trained from scratch | Closest mechanical lineage: its linear case is the delta rule. Seahorse keeps the backbone frozen. |
| **Trained Persistent Memory, decoder-only** (2603.22329) | Persistent memory bank across sessions | Attention-coupled write with fixed random projections | KV prefix / parallel cross-attention / gated branch | Read path trained; GPT-2 124M | The closest *setup* (frozen decoder, cross-session). Evidence that the read side needs training (P5). Single author, one seed, one benchmark. |
| **TransMem** (2607.29032) | ~4 segment-end hidden states | None (reuses states) | Learned block + gate in the last 4 layers | Module trained by evidence-conditioned distillation | The distillation objective is a strong idea. Its gains may not come from memory content: changing the number of segments from 4 to 32 has no effect, and there is no random-memory control. |
| **AGCLR** (2606.07720) | One d-vector across ≤6 latent reasoning passes | Learned gates | Added to the hidden state | Whole GPT-2 fine-tuned | Latent reasoning, not persistent memory. Memory resets on every forward call, and the "forget" gate acts on the hidden state, not on memory. |
| **Recurrent Memory Transformer** (2207.06881) | Memory tokens passed between segments | Learned | Attention | Trained (backprop through time) | A trained baseline for segment recurrence. |
| **Tensor Memory** (2605.27686) | 3D voxel ConvLSTM state | Learned Gaussian splat | Gated residual | Trained from scratch; tiny models | Vision-oriented; weak long-context evidence. |
| **Hopfield Networks is All You Need** (2008.02217) | — | — | Attention = Hopfield retrieval | — | Theory for capacity, separation and metastable averaging (P4). |
| **Emotion concepts** (2604.07729) | — | — | — | — | Source of P2 and P3, the neuromodulator side idea, and the internal "surprised" signal. |
| **Rethinking Memory in LLM-based Agents** (Du et al., 2505.00675) | Survey | | | | A taxonomy; pointers to KV-selection work (Memorizing Transformers, RetrievalAttention, ArkVale, …). |

Also relevant but not read in this project (arXiv IDs from memory):
- kNN-LM (1911.00172)
- Memorizing Transformers (2203.08913)
- Unlimiformer (2305.01625)
- Function Vectors (2310.15213) and Task Vectors (2310.15916): single stored hidden states that reproduce behaviour
- Patchscopes (2401.06102)
- Feng & Steinhardt on binding IDs (2310.17191)
- Kadavath et al., "Language Models (Mostly) Know What They Know" (2207.05221)
- Farquhar et al., semantic entropy (Nature, 2024)
- Schlag et al., fast weights (2102.11174)
- DeltaNet (2406.06484)

The PDFs of the papers that were read are in `../papers/` (not in the repo).

---

## 1.7 Side idea: steering vectors as neuromodulators (parked)

> "Can we use certain steering vectors as 'neurotransmitters' in LLMs to steer the responses,
> just like a neurotransmitter would influence a human's behaviour?"

It came from mapping salience signals onto brain chemistry (§1.3) and then asking the
reverse: can steering vectors act like neuromodulator *chemicals*? Neuromodulators carry
little content; they change *how* circuits process (gain, plasticity, exploring versus
exploiting, mood). They're diffuse and global, which is how steering vectors behave.
"Neuromodulator" is the better term: transmitters (glutamate, GABA) are point-to-point,
closer to attention, whereas modulators broadcast, closer to steering.

| Neuromodulator | Human role | Possible LLM direction |
|---|---|---|
| Dopamine | Reward, motivation, novelty | "excited" / "curious" directions |
| Serotonin | Patience, impulse control | "calm" / "patient" directions |
| Noradrenaline | Arousal, surprise | the "surprised" vector |
| Acetylcholine | Attention, uncertainty | confusion / "don't know" directions |
| Oxytocin | Social bonding | the "loving" vector |
| Cortisol (hormone) | Stress | the "desperate" vector (linked to reward hacking) |

What real modulators have and steering lacks: endogenous release, dynamics (reuptake, decay,
tolerance), receptor specificity (layer-specific effects), nonlinear dose-response (inverted
U), and interactions. It could work in two modes: **exogenous** (inject, drug-like) and
**endogenous** (the model's own internal state releases it, then it decays).

**Update from the experiments:** v0.1 found that the memory steer behaves like a *modulatory*
channel (it biases what comes to mind) rather than an episodic one. So the side idea and the
memory project share machinery: both read internal state and then steer.
