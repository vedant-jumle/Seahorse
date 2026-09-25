# 2. The method: how Seahorse stores and recalls memory

This document specifies the mechanism exactly: where it attaches to the LLM, what the memory
*is*, how it is written, how it is read, and how it is evaluated. It matches the code at
`src/seahorse/` and `experiments/v0*/` (see [03-code-map.md](03-code-map.md)). For the
motivation, see [01-core-idea.md](01-core-idea.md).

---

## 2.0 Overview

```
                    SESSION 1  (experience; memory is written)
  "I'm vegetarian. I'm planning my meals for next week."   <- with experience
  "I'm planning my meals for next week."                   <- without experience
                  │                          │
          frozen LLM (Qwen2.5-1.5B)   frozen LLM
                  │ capture h at block ℓ     │ capture h at block ℓ
                  ▼                          ▼
             h_with[t]                  h_without[t]      (shared suffix tokens t)
                  └────────┬─────────────────┘
                 Δ_t = h_with − h_without     k_t = key(h_without)
                           │
                 delta-rule write:  M += η·g_t·(Δ_t − M k_t) k_tᵀ
                           │
                     M  (d × d matrix = the memory)

                    SESSION 2  (fresh context; memory is read)
  "Any ideas for what I should cook for dinner tonight?"
                  │
          frozen LLM ── at block ℓ, every position:  h ← h + α · M · key(h)
                  │
               output  (compared against: no memory, and experience-in-context)
```

- **One layer ℓ**, one matrix M, and no training anywhere: only forward passes and matrix arithmetic.
- The base model's weights are never touched. Memory acts purely by adding a vector to the residual stream.

---

## 2.1 Notation

| Symbol | Meaning |
|---|---|
| L, d | Number of decoder blocks (28), residual width (1536) for Qwen2.5-1.5B-Instruct |
| h_t^ℓ ∈ ℝ^d | Residual stream at position t after decoder block ℓ |
| μ_ℓ ∈ ℝ^d | Mean residual at block ℓ over generic prompts (centring vector) |
| k(h) | Key: `normalize(h − μ_ℓ)`, a unit vector |
| M ∈ ℝ^{d×d} | The memory (one per layer and per experiment condition) |
| Δ_t | Experience delta at a write token: `h_with − h_baseline` |
| e_t | Write error: `Δ_t − M k_t` (what memory could not already predict) |
| g_t ∈ [0,1] | Salience gate (per write token) |
| η | Write rate (1.0 in all experiments) |
| α | Read strength |

---

## 2.2 How Seahorse hooks into the LLM (`src/seahorse/residual.py`)

### The hook point
Seahorse attaches at the **output of decoder block ℓ**, `model.model.layers[ℓ]` (a
`Qwen2DecoderLayer`). That output is the residual stream after block ℓ's attention and MLP
have both been added in. The **same point** is used to capture states (writing) and to
inject recall (reading), so writing and reading always speak the same "coordinates".

### Why plain PyTorch forward hooks
- **No extra dependency**, and numerics identical to Hugging Face.
- **Rejected alternatives:**
  - **TransformerLens** reimplements the model, and by default processes the weights (folding LayerNorm, centring the writing weights). That changes the residual stream's values, which are exactly the numbers the method depends on.
  - **nnsight** wraps the real model and is a good option, but isn't needed for one hook point. Revisit it if the work scales to large models through NDIF remote execution.
- **Why not `output_hidden_states=True`:** one mechanism serves both capture and injection, and HF's last hidden state is taken *after* the final norm (not the raw residual), which is a subtle trap.

The hook handles both output forms: transformers 4.x decoder layers return a tuple whose
`[0]` is the hidden states, and newer versions return a tensor.
```python
def _hidden(out):  return out[0] if isinstance(out, tuple) else out
def _replace(out, h): return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
```

### Capture (used when writing)
```python
with capture(model, layers=[20, 23, 26]) as store:   # one hook per layer
    model(input_ids)                                  # one forward pass
store[23]  # -> [seq, d] residual after block 23 (detached copy)
```
One forward pass records every layer of interest, so memories for all swept layers are built
from the same two (or three) passes. Batch size 1 is asserted.

### Inject (used when reading)
```python
with inject(model, layer=23, memory=M, alpha=2.0):
    logits = model(input_ids).logits
```
The hook returns a modified output: for every position,
**h' = h + α · M · k(h)**. Everything downstream (later blocks, final norm, LM head) sees h'.
The context manager removes the hook on exit, so configurations in a sweep can't leak into
each other (tested).

**During generation** the hook fires on the prompt pass *and* on every decode step. The
keys and values that later layers cache already include the injected change, so the cache
stays consistent.

### Numerics
fp32 throughout, `torch.inference_mode()`, and no gradients anywhere. fp32 is required on
V100 (no bf16; Qwen overflows in fp16), and it keeps the delta and key arithmetic clean.

---

## 2.3 The centring mean μ_ℓ

Raw residual vectors are **anisotropic**: they share a large common component (and a few
massive-activation dimensions), so any two raw residuals have high cosine similarity. With
uncentred keys, every key would match every other key and the memory would fire everywhere.

**Estimate:** run ~100 generic prompts (`experiments/v0/generic_prompts.txt`) through the
chat template, capture every layer of interest, and average over all tokens **except the
shared template prefix** (system prompt, start-of-turn tokens and the attention-sink token,
whose states are huge and identical in every prompt). The prefix length is the common prefix
of all templated sequences.

---

## 2.4 The memory operator in detail (`src/seahorse/memory.py`)

```python
class FastWeightMemory:
    def __init__(self, d, mu, ...):
        self.mu = mu                      # centring vector for this layer
        self.M  = zeros(d, d)             # the memory: starts empty

    def key(self, h):                     # [..., d] -> unit key
        return normalize(h - self.mu)

    def predict(self, k):                 # recall: M k  (row-vector form k @ Mᵀ)
        return k @ self.M.T

    def read(self, h, alpha):             # the steer applied at the hook
        return h + alpha * self.predict(self.key(h))

    def write(self, delta, h_key, gate=None, eta=1.0):
        keys = self.key(h_key)
        for t in range(T):                # strictly sequential, in token order
            k = keys[t]
            e = delta[t] - self.M @ k     # error: what memory can't yet predict
            g = 1.0 if gate is None else gate[t]
            self.M += (eta * g) * outer(e, k)
        return err_norms, delta_norms     # surprise diagnostics
```

### What M is
M is a **linear associative memory**: a map from *situations* (keys) to *changes* (deltas).
Unrolling the writes shows it is a **sum of rank-1 imprints**:

 **M = Σ_t η g_t · e_t k_tᵀ**

Each write stamps one outer product: "when the state points in direction k_t, add e_t". There
are no slots and no list of items. Every memory is superimposed in the same d×d matrix,
which is the "synapses, not a filing cabinet" design (§1.5). It's 1536² fp32 ≈ **9.4MB** per
memory.

### Reading = similarity-weighted recall of imprints
 **M k(h) = Σ_t η g_t · e_t · (k_tᵀ k(h)) = Σ_t η g_t · e_t · cos(situation_t, now)**

Every stored imprint contributes in proportion to how similar the present state is to the
moment it was written. That is **cue-driven recall with no explicit query**: the current
residual *is* the cue. Because the key is normalised, only the *direction* of the state
matters, not its magnitude.

Compare softmax attention: it retrieves by similarity too, but the softmax makes items
**compete** (weights sum to 1, and near-duplicates average). Here there's no competition;
all similar imprints **add up**. That makes the memory simple and linear, but it's also the
root of interference and leakage (see capacity below).

### Writing = the delta rule (error-correcting, not Hebbian)
The update **M ← M + η g (Δ − M k) kᵀ** is the **delta rule** (Widrow–Hoff / LMS). It is one
gradient step on the associative loss ½‖Δ − M k‖²: the gradient with respect to M is
−(Δ − M k)kᵀ.

A plain **Hebbian** memory would do **M ← M + Δ kᵀ** ("store what happened"). The delta rule
stores **only the error**, "what happened *minus what memory already expected*". This is the
CA1 comparator from §1.3, and it's what makes the properties below emerge.

### Properties (each covered by a unit test in `tests/test_memory.py`)
With a unit key (‖k‖ = 1), after one write: M'k = M k + η g e (kᵀk) = M k + η g (Δ − M k).

1. **Exact one-shot storage** (`test_exact_recall_of_written_key`). With η g = 1: M'k = Δ. The memory reproduces the delta exactly for that situation after a single exposure, as a hippocampus does.
2. **Graded storage via the gate** (`test_zero_gate_writes_nothing`). In general M'k = (1 − ηg)·M k + ηg·Δ, an interpolation between what memory believed and what happened. **The salience gate is a per-moment learning rate**: g=0 ignores a moment, g=1 imprints it fully.
3. **Habituation** (`test_repeat_write_has_zero_error`). Repeat the same (Δ, k) and e = Δ − Δ = 0, so nothing changes. Familiar experiences don't re-write memory.
4. **Reconsolidation / overwrite** (`test_contradiction_overwrites`). A new Δ' in the same situation gives e = Δ' − Δ, and memory now returns Δ'. Contradictions rewrite.
5. **Independence of orthogonal situations** (`test_orthogonal_keys_do_not_interfere`). For another key k', M'k' = M k' + η g e (kᵀk'). If kᵀk' = 0, nothing changes.
6. **Read scaling** (`test_read_adds_recalled_delta_scaled_by_alpha`). read(h, α) = h + α·Δ at an exactly matching key, and α=0 is the identity.

### A worked toy example (d = 2)
Start with M = 0 and μ = 0.
1. **Write Δ₁ = (3, 0) at key k₁ = (1, 0):** e = Δ₁, so M = Δ₁k₁ᵀ = [[3,0],[0,0]].
2. **Read:**
   - At h = (5, 0) → k = (1,0) → M k = (3,0). Full recall, regardless of ‖h‖.
   - At h = (0, 2) → k = (0,1) → M k = 0. An unrelated situation gets nothing.
   - At h = (1, 1) → k = (0.707, 0.707) → M k = (2.12, 0). A partly similar situation gets partial recall.
3. **Write the same again:** e = Δ₁ − M k₁ = 0, so M is unchanged (habituation).
4. **Write Δ₂ = (0, 4) at k₂ = (0.6, 0.8)**, a *correlated* key (cos = 0.6 with k₁):
   - Error: e = Δ₂ − M k₂ = (0,4) − (1.8, 0) = (−1.8, 4). Memory "expected" (1.8, 0) here, because it has seen a similar situation.
   - M += e k₂ᵀ = [[−1.08, −1.44],[2.4, 3.2]], giving M = [[1.92, −1.44],[2.4, 3.2]].
   - Now M k₂ = (0, 4) ✓ (exact for the latest write), but **M k₁ = (1.92, 2.4) ≠ (3, 0)**. The second write **disturbed the first**, because the keys overlap.

Step 4 is interference in miniature. It's why capacity depends on how **distinct** the
situations are, not on d.

### Capacity and interference
- **Hard limit:** rank(M) ≤ d. At most d fully independent situations.
- **Practical limit is far lower**, because residual keys are correlated even after centring (anisotropic, low effective rank). Each write perturbs every earlier memory in proportion to key overlap.
- The sequential delta rule with η=1 and unit keys is the **Kaczmarz method** for solving M·K = Δ: each step projects onto the latest constraint and forgets earlier ones partially. Larimar instead uses **recursive least squares** (maintaining a key covariance C, with M ← M + C⁻¹Wᵀ(Z − WM)), which keeps the *global* least-squares solution over all writes. That costs more but interferes less. A possible upgrade.
- **Empirically** (v0/v0.1), write errors on later follow-ups were about 1 or higher when writing from every token (no shared structure; earlier writes made things worse). They were below 1 for pooled or top-k writes, and dispositions already collided at N=6 ([04-experiments.md](04-experiments.md)).

### Cost
- Write: O(d²) per written token (an outer product); the loop is sequential by design.
- Read: O(d²) per position per forward pass (one mat-vec).
- Memory footprint: d² floats (9.4MB at d=1536, fp32).

### Diagnostics returned by `write`
`err_norms[t] = ‖e_t‖` and `delta_norms[t] = ‖Δ_t‖`. The ratio ‖e‖/‖Δ‖ is the **relative
surprise** of each moment against memory: 1 means completely new, 0 means fully expected,
and above 1 means earlier writes point the wrong way. It's logged per follow-up as
`mean_rel_error` in `write_stats.jsonl`.

### Relation to known mechanisms
| Mechanism | Relation |
|---|---|
| Hebbian outer-product memory | Same form (sum of outer products) but without the error term, so no habituation or overwrite. |
| Delta rule / LMS (Widrow–Hoff) | Identical update. |
| Linear attention / fast weights (Schmidhuber; Schlag et al.) | Linear attention's state is Σ v kᵀ (Hebbian). DeltaNet uses this delta-rule update. Seahorse applies it to the residual stream of a *frozen* model rather than as a trained layer. |
| Titans (linear case) | One gradient step on ‖M k − v‖² is this update. Titans adds momentum and weight decay and learns the projections. |
| Larimar | A matrix memory with error-driven sequential updates, but recursive least squares over a *separate encoder's* latents, read through a trained KV slot. |
| Abliteration | A fixed operator (I − r rᵀ) applied everywhere. Seahorse's is a plastic operator (I + α M K(·)), written by experience. |

---

## 2.5 What gets written: the experience delta

### The session-1 protocol (`sessions.py::write_pair`, `run.py::collect_writes`)
For each scenario (e.g. "By the way, I'm vegetarian.") and each follow-up message (e.g.
"I'm planning my meals for next week."), build one **user turn** with the chat template and
the assistant header, and run it in several versions:

| Run | User message | Purpose |
|---|---|---|
| **with** | `"<experience> <follow-up>"` | The model as it actually lived the moment |
| **without** | `"<follow-up>"` | The same moment without the experience. Its keys are what future sessions will look like. |
| **counter** (v0.1) | `"<counter-experience> <follow-up>"` | Same concept, opposite relation or identity (e.g. "I really love peanuts") |

No assistant text is generated or teacher-forced in session 1. If the assistant's own words
were forced into both runs, the fact could leak into the "without" run through them.

### Write tokens: the longest common suffix
The memory is written on the tokens **after** the experience: the follow-up text, the
end-of-turn token and the assistant header. These are found as the **longest common suffix**
of the with/without (and counter) token sequences. That avoids tokenisation-boundary bugs;
for example, " I'm" after "vegetarian." and "I'm" after the newline are different tokens, so
the suffix starts one token later.

### The delta and the key
For each write token t and layer ℓ:
- **Δ_t = h_with[t] − h_without[t]** (the "without" baseline), or **Δ_t = h_with[t] − h_counter[t]** (the "contrastive" baseline, v0.1).
- **k_t = key(h_without[t])**: the key comes from the *without* run, because in session 2 the experience won't be in context, so its states will look like the without run's.

Δ is "how this moment was *changed* by having had the experience". Storing a change rather
than a state avoids the hijacking that full-state patching causes (P1). Measured sizes:
‖Δ‖/‖h‖ ≈ 0.19–0.24 across layers.

**Caveat (accepted):** in the with run the suffix tokens sit at later positions (offset by
the experience length), so Δ also contains a small RoPE position effect.

### Why the counterfactual pass is scaffolding
The with/without comparison assumes we know *what the experience was*. That's acceptable for
testing whether the channel can carry memory at all, but it isn't experience-driven. The
intended online version drops the counterfactual and relies only on **memory's own
prediction** as the baseline (the delta rule's e = h-change − M k). See
[05-next-moves.md](05-next-moves.md).

---

## 2.6 Write variants

### Baseline: `without` vs `contrastive` (v0.1)
- **without:** Δ = h(experience) − h(nothing). Keeps *everything* the experience added, which includes the **concept** itself. "Peanut-ness" is the largest part of "I have a peanut allergy", so the memory primed peanut butter.
- **contrastive:** Δ = h(experience) − h(counter-experience). The shared concept cancels, as in contrastive steering vectors (CAA), and the relation or identity remains. Counter-experiences used:

| Scenario | Experience | Counter |
|---|---|---|
| vegetarian | "By the way, I'm vegetarian." | "By the way, I love eating meat." |
| peanut_allergy | "I have a severe peanut allergy." | "I really love peanuts." |
| norway | "I live in Norway." | "I live in the United States." |
| dog_name | "My dog's name is Biscuit." | "My dog's name is Max." |
| sister_name | "My sister's name is Ines." | "My sister's name is Maria." |
| job | "I work as a deep-sea welder." | "I work as a school teacher." |

### Salience gate: `none` vs `entropy`
- **none:** g_t = 1.
- **entropy** (the confusion signal): H_t = the entropy of the next-token distribution at t in the **without** run ("how uncertain was I here without this knowledge?"). Then g_t = H_t / max_t H_t over that follow-up's write tokens, so the most uncertain moment writes fully and confident moments write less. Because g ≤ 1, it only ever scales writes down.

### Write position: which moments become memories (v0.1)
| Position | Tokens written | Key and delta |
|---|---|---|
| `all` | Every shared-suffix token (~14 per follow-up) | Per token, gated by g_t |
| `boundary` | Only the template tail after the user text (5 tokens: end-of-turn + assistant header). Length = common suffix of two templated messages. | Per token, gated |
| `topk` | The k=4 highest-entropy tokens | Per token, ungated (the selection *is* the gate) |
| `pooled` | One write per follow-up | Δ and h_key are g-weighted means over the suffix (uniform if gate=none); one key per moment |

### Mode: `isolated` vs `combined`
- **isolated:** one memory per scenario (N=1). Each scenario is probed with its own memory.
- **combined:** all 6 scenarios written into one memory (N=6), in order. This is the first interference and capacity test.

---

## 2.7 Reading: session 2

Session 2 contains **none** of session 1's text (the equal-input principle). The probe is a
single fresh user turn. With memory on, the inject hook applies h ← h + α·M·k(h) at block ℓ,
at **every position** (template prefix, probe tokens, assistant header, generated tokens).
Recall is therefore automatic, driven by resemblance, and nobody decides when to consult
memory.

Each probe is evaluated in three conditions:

| Condition | Input | Role |
|---|---|---|
| **baseline** | the probe, no memory | Floor |
| **memory** | the probe, memory injected | What's being tested |
| **ceiling** | `"<experience> <probe>"` in one user turn, no memory | What "remembering perfectly" looks like |

### Probe distances
Measured against session 1's follow-ups:
- **exact:** identical wording to a follow-up.
- **paraphrase:** same situation, different words.
- **related:** same topic, a situation not seen in session 1. For facts, this is the direct question ("What's my dog's name?").

Plus 8 shared **unrelated** probes (tax brackets, photosynthesis, …) for leakage.

---

## 2.8 Metrics (`src/seahorse/metrics.py`, `run.py`)

All metrics are **teacher-forced log-probabilities**: no generation-based judging, and they
are deterministic.

| Metric | Definition | Meaning |
|---|---|---|
| **KL along a continuation** | c = the ceiling's greedy response (20 tokens). For each condition, compute the next-token distributions predicting c, then the mean per-token KL(ceiling ‖ condition). | How far a condition is from "remembering perfectly" |
| **Gap closed** | 1 − ΣKL_memory / ΣKL_baseline over a group of probes (a ratio of sums, robust to probes with tiny KL_baseline) | 0 = no better than no memory, 1 = as good as having the experience in context, < 0 = pushed *away* from the ceiling |
| **Fact target** | log P(target \| probe + prefix), e.g. prefix "Your dog's name is", target " Biscuit". Reported as Δ = memory − baseline, next to the ceiling's Δ. | Is the specific fact recalled? |
| **Disposition contrast** | log P(a) − log P(b) after a prefix, e.g. "Sure! How about" + a=" a creamy mushroom risotto" vs b=" a juicy grilled steak" | Does behaviour lean the experience's way? |
| **Foils / specificity** (v0.1) | The same slot, other names (" Buddy", " Charlie"). specificity = Δtarget − mean Δfoils | This fact, or "state any name"? |
| **Relation probes** (v0.1) | A yes/no question answered right after the header, e.g. "Is a peanut butter cookie safe for me to eat? Answer yes or no." Score = log P(consistent answer) − log P(other). | Can the model *reason with* the memory? |
| **Leakage** | Mean KL(baseline ‖ memory) on unrelated probes, along the baseline's greedy continuation | Does memory fire where it shouldn't? |
| **Target fraction** | Δmemory / Δceiling (a ratio of means) | Share of the in-context effect recovered |

Greedy decoding uses `repetition_penalty=1.0`, overriding Qwen's default of 1.05, so it is
pure greedy.

---

## 2.9 Sanity checks and tests

- **Unit tests** (`tests/`, CPU, run at the start of every cluster job):
  - `test_memory.py`: the six properties in §2.4.
  - `test_hooks.py`, on a tiny random Qwen2 built from a config (no download):
    - capture equals `output_hidden_states[ℓ+1]`
    - α=0 or an empty memory is the identity
    - injection changes the logits and is removed on exit
    - injection leaves layers *before* ℓ untouched
- **Runtime assertions** in `run.py` (the job fails if violated):
  - α=0 reproduces the baseline KL and targets exactly (v0).
  - The most recent ungated write is recalled exactly (relative error < 1e-3). v0 checks this for gate=none; v0.1 checks it for topk and pooled.

---

## 2.10 Known limitations of the current method

1. **The counterfactual scaffold:** writes need a with/without (or counter) pass, so it isn't yet experience-driven.
2. **One layer, one additive vector.** v0.1 shows this carries associations but not premises.
3. **Linear, competition-free recall.** Similar imprints add rather than compete, which causes interference (dispositions collide at N=6) and leakage at high α.
4. **Kaczmarz-style sequential writes** disturb earlier memories with overlapping keys. Recursive least squares (as in Larimar) would be the principled fix.
5. **The RoPE position offset** leaks a little into Δ.
6. **Batch size 1**, a small scenario set, and one model (Qwen2.5-1.5B-Instruct).
