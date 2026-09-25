# 5. Next possible moves

Where things stand ([04-experiments.md](04-experiments.md)): the plastic steer carries
**which** fact and **which way** a preference points (specific, low leakage, salience helps),
but **not premises the model can reason with**, and dispositions **collide** at N = 6. That
points to a **two-channel memory**: a *modulatory* steer (what we have) and an *episodic*
channel that reinstates content where attention can use it.

The moves below are grouped by the question each one answers. Each lists why it matters,
what it would look like, and what the possible outcomes would tell us.

---

## A. The central fork: why can't the memory be reasoned with?

### A1. Is it *where* we steer? (cheap; decides the fork)
**Why:** relation probes were only measured at L20–26. In the ceiling, the implication
("vegetarian → burger is bad") is computed in the **middle** layers, as the question's tokens
attend back to the premise. A steer added *after* that computation can't take part in it.
v0 swept L6–26 but never measured relations.

**What:** rerun v0.1's relation probes and contrastive writes over **L8, 10, 12, 14, 17** (and
L20 as the reference), α {1, 2, 4}.

A variant: **multi-layer injection.** Write and read the same experience at several layers at
once (e.g. L12 + L17 + L23 with smaller α each), the way Larimar feeds a projection of its
readout into *every* decoder layer.

**Outcomes:**
- Relations flip at mid layers → the steer *can* be a premise; it was just placed too late. The single-channel design survives, with a different layer.
- They don't flip anywhere → a linear steer isn't the right carrier for premises. Go to A2.

### A2. An episodic channel: reinstatement into attention
**Why:** P2 in the core idea (the emotion paper) and hippocampal indexing both say content
comes back by **reinstatement**: the model *attends* to it. InfLLM and EM-LLM show that
retrieval into attention works without training. What they lack is compression, persistence
and salience.

**What (training-free first):**
- **Store:** only the **salient moments** (the top-k entropy tokens, which v0.1 showed carry nearly everything), and keep their residuals at **every** layer. That's compression by selection, not by averaging.
- **Recall:** in a new session, retrieve moments whose keys resemble the current state (the model's own query·key scores, as in InfLLM, or our centred keys). Splice their K,V (recomputed from the stored residuals with the frozen W_k, W_v) into attention as extra entries, with a fixed position (as InfLLM does).
- **Measure:** the same probes, *especially relation probes*, with and without the steer channel alongside.

**What (trained, if training-free is weak):**
- **Larimar-style read:** a pooled moment vector → a learned projection → one KV slot per layer.
- **Objective:** evidence-conditioned distillation (TransMem's idea). The teacher is the frozen model with the experience in context; the student is the frozen model with the memory. Minimise KL on the same probes.
- **Only the read projections are trained;** the backbone stays frozen. This is the "memory language" problem: teaching the model to read its own stored moments.

**Outcomes:**
- Relations transmit via reinstatement → the two-channel design is confirmed. The modulatory steer handles dispositions; reinstated moments handle facts and premises.
- They transmit only when trained → the episodic channel needs a learned read path, consistent with Larimar and Trained Persistent Memory.

---

## B. Capacity: why do dispositions collide at N = 6?

### B1. Diagnose key collisions (cheap, no GPU-heavy runs)
Measure the cosine similarity between the stored keys of different scenarios, per layer and
per write position. **Prediction:** vegetarian and peanut keys (both food contexts) overlap
strongly, and fact keys don't. That would confirm the collision explanation and point
directly at the fixes below.

### B2. Better writes: recursive least squares instead of Kaczmarz
The current sequential delta rule projects onto the *latest* write and partly forgets
earlier ones when keys overlap. Recursive least squares (Larimar's sequential update: keep a
key covariance C, and update M ← M + C⁻¹Wᵀ(Z − WM)) keeps the **global least-squares
solution** over all writes. The cost is a d×d covariance per memory, which is fine at
d = 1536.

### B3. Pattern separation (the dentate gyrus)
Make similar situations *less* similar before storing them. Expand keys into a larger, sparse
code (a random projection plus top-k sparsification, or sparse-autoencoder features of the
residual) and store in that space. Hopfield theory says retrieval is sharp only for
well-separated patterns (P4).

### B4. Competition at read time
Recall currently *adds* every similar imprint. A softmax over stored keys (attention-style
recall with a temperature) makes memories **compete**, so only the best match fires.
**Trade-off:** it moves from a linear operator towards a key-value store, which is closer to
the episodic channel.

### B5. Scale N properly
Templated facts ("My <relation>'s name is <name>", "I work as a <job>", …) to test N = 1, 5,
20, 50. Report where each variant breaks. This is the capacity curve the design needs.

---

## C. Remove the scaffold: truly experience-driven writing

**Why:** the with/without (and counter) passes assume we know what "the experience" was. A
real memory can't.

**What:** writing online, from a single pass:
- **Modulatory channel:** Δ measured against **memory's own prediction** (the delta-rule error already does this), or against a running average of recent states (habituation), or against the model's predicted next state (surprise in state space). See the baseline discussion in [01-core-idea.md](01-core-idea.md) and [02-method.md §2.5](02-method.md#25-what-gets-written-the-experience-delta).
- **Episodic channel:** no delta needed. Store the **states** of salient moments (reinstatement through attention doesn't hijack the way additive patching does). Salience decides *what* gets stored; the entropy gate and top-k already do this.
- **The open problem:** the contrastive write worked because it had a matched counter-experience. Online, nothing supplies one. Can memory's own prediction play that role, cancelling what's already known (the concept) and keeping what's new (the relation)?

---

## D. Salience: a richer "who calls write()"

### D1. A taxonomy of confusion (the user's idea, in the emotion paper's style)
**Why:** the entropy gate improved everything. But "flat distribution" mixes different states
that call for different memory operations:

| State | Memory operation |
|---|---|
| I don't know the fact | Store the answer when it arrives |
| The question is ambiguous / ill-posed | Store nothing, or store that it needs clarifying |
| I know it, but it's hard to compute | Store the result |
| My information conflicts | Reconsolidate (fix the old memory) |

**What:** build datasets for each state, and look for linear directions separating them in
the residual stream (as the emotion paper did for 171 emotion concepts). Also separate
epistemic from aleatoric uncertainty (semantic entropy): "many right answers" isn't
confusion.

**Related work:** Kadavath et al. (2022) on models knowing what they know; semantic entropy
probes (2024); known/unknown-entity features (Anthropic's "Biology of a Large Language
Model", 2025; Ferrando et al.).

### D2. Combine gates
Surprise against memory (built in), uncertainty (entropy or D1's directions), and
valence/arousal (the emotion vectors). Test whether each adds anything beyond the others.

### D3. The hypercorrection corner
Confident-but-wrong moments give a sharp distribution, so entropy stays silent. They should
be caught by surprise against memory. Build scenarios where the model confidently expects
one thing and is corrected, and check which gate writes them.

---

## E. Training (the user is open to it)

Where training should go, based on the evidence:
- **Not in the write.** Closed-form and delta-rule writes work, and salience selection works.
- **In the read path for episodic content (A2).** Learn projections from stored moments into KV slots or attention entries. The objective: distillation from an experience-in-context teacher (TransMem's evidence-conditioned self-distillation). This needs no labels and works on any text.
- **Possibly in the keys (B3).** A learned, separated key space.

The backbone stays frozen throughout.

---

## F. Evaluation hygiene (whatever direction is chosen)
- **More scenarios**, templated where possible, with several paraphrase sets per scenario.
- **Relation probes as a first-class metric:** they're the ones that separate "comes to mind" from "reasoned with".
- **A text-summary baseline:** the obvious rival once there are many memories.
- **InfLLM / EM-LLM** as the training-free episodic baselines.
- **Scale:** repeat the key results on Qwen2.5-7B (fp16/bf16 on DAIC's 96GB cards) to see whether a larger model carries more in a linear steer.
- **Real cross-session benchmarks** later: LoCoMo, LongMemEval.

---

## G. Parked (revisit when there's a working mechanism)
- **Memories created by reflection:** the model re-reading its past with hindsight and storing *conclusions* rather than observations. This would also address the "immutable past" problem (the causal mask means earlier states never learn what came later).
- **Cold start:** with an empty memory, every moment is maximally surprising.
- **Consolidation into weights:** replay stored moments into slow weights (complementary learning systems), or bake stable traits in, as abliteration bakes in a projection.
- **Reconstructive drift:** when memory conflicts with priors, should the prior reshape the memory (Bartlett) rather than memory always winning (Larimar)? Needs confidence-weighted reconciliation.
- **Steering as neuromodulators:** see [01-core-idea.md §1.7](01-core-idea.md#17-side-idea-steering-vectors-as-neuromodulators-parked). The v0.1 finding that the steer is modulatory makes this closer to the main line than it first looked.

---

## Suggested order
1. **A1** (layer sweep for relations; ~20 min on one MIG slice). It decides the fork.
2. **B1** (key-collision diagnostics; minutes, from stored states). It explains the capacity failure.
3. **A2, training-free** (salient-moment reinstatement into attention), with relation probes.
4. Depending on (1) and (3): **B2/B3** for the modulatory channel, or **E** (a trained read path) for the episodic channel.
5. **C** (remove the scaffold) once the channel(s) are settled, since the right online baseline depends on which channel it's for.
