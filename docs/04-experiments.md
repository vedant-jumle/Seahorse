# 4. Experiments and results

Two experiments so far, both on **Qwen2.5-1.5B-Instruct (frozen, fp32)** on one DelftBlue
A100 MIG slice (10GB). The method is in [02-method.md](02-method.md). Raw outputs are in
`results/v0_408770/` and `results/v0_1_415909/`.

**Common setup:** 6 scenarios (3 dispositions, 3 facts), 3 follow-ups each in session 1,
3 probes each in session 2 (exact / paraphrase / related), and 8 unrelated probes for
leakage. Session 2 contains none of session 1's text.

---

## 4.1 v0: can the channel carry memory at all?

**Job 408770** (commit `d6e5947`), 21m53s. All sanity assertions passed (α=0 is
exactly the baseline; the last write is recalled exactly).

**Sweep:** mode {isolated, combined} × gate {none, entropy} × layer {6, 10, 14, 17, 20, 23, 26}
× α {0, 0.5, 1, 2, 4}, with the "without" baseline and all-token writes.

### Gap closed by layer and α (isolated, gate = none)
Share of the ceiling's KL removed (0 = baseline, 1 = ceiling, <0 = pushed away).

**Dispositions**

| layer \ α | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|
| exact, L20 | 0.106 | 0.192 | **0.262** | 0.046 |
| exact, L23 | 0.106 | 0.183 | 0.247 | 0.136 |
| paraphrase, L23 | 0.105 | 0.185 | **0.261** | 0.177 |
| related, L23 | 0.113 | 0.188 | **0.239** | 0.063 |
| related, L14 | 0.002 | −0.008 | −0.068 | −0.560 |
| any distance, L6 | ~0 | ~0 | −0.04 to −0.05 | −0.1 to −0.16 |

**Facts**

| layer \ α | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|
| exact, L26 | 0.096 | 0.159 | **0.177** | −0.253 |
| related, L26 | 0.076 | 0.140 | **0.215** | 0.132 |
| paraphrase, L26 | 0.060 | 0.087 | 0.055 | −0.397 |

### Leakage (mean KL on unrelated probes; isolated, gate = none, L23)
| α | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|
| leak | 0.003 | 0.012 | 0.053 | 0.257 |

For scale, ceiling-vs-baseline KLs are 0.1–2.

### Target shifts (isolated, gate = none, α = 2)
Log-prob units. Δmemory versus Δceiling (mean over the three probes).

| Scenario | L23 Δmemory | L26 Δmemory | Δceiling |
|---|---|---|---|
| vegetarian (risotto − steak) | +2.40 | +1.40 | +1.78 |
| peanut (hummus wrap − peanut butter) | **−7.25** | −3.19 | −0.76 / −1.74 / +2.29 |
| norway (kroner − dollars) | +2.50 | +0.86 | ~+12.4 |
| dog (" Biscuit") | +0.30 | **+4.12** | ~+9.3 |
| sister (" Ines") | +0.79 | +3.20 | ~+13.5 |
| job (" deep-sea welder") | **+5.16** | +2.74 | ~+25.4 |

### Entropy gate vs no gate (isolated, L23, α = 2)
| | gap (dispositions) | Δtarget (facts) | leakage |
|---|---|---|---|
| gate = none | 0.251 | +2.08 | 0.053 |
| **gate = entropy** | **0.306** | **+3.39** | **0.025** |

Better recall *and* half the leakage. The gate only ever scales writes *down*, so if it were
acting as a weaker α, both would drop. It is **selecting better moments**.

### Combined (N = 6) vs isolated (L23, α = 2)
| | gap (dispositions) | fact gap / Δtarget | leakage |
|---|---|---|---|
| isolated, none | 0.251 | 0.115 / +2.08 | 0.053 |
| combined, none | **−0.174** | −0.008 / +0.90 | 0.153 |
| combined, entropy | −0.089 | **0.167 / +2.89** | 0.048 |

### Write statistics
- ‖Δ‖/‖h‖ ≈ 0.19–0.24 across layers: the experience changes later states by ~20% of their norm.
- Relative write error on follow-ups #0/#1/#2 ≈ 1.0 / 0.9 / 1.1 (1.16 / 0.98 / 1.42 at L23). **No habituation**: each follow-up's deltas are unpredictable from the previous ones, and later writes at L23/26 partly point the wrong way.

### Samples (L17, α = 1, isolated, none)
Memory outputs were almost word-for-word the baseline for every scenario. At this weak
setting, nothing reached the "related" probes behaviourally.

### v0 conclusions
1. The channel carries **something**, peaking at ~25–30% gap closed at L20–26, α 1–2. **Mid-to-late layers win**; L6 does nothing. **α = 4 breaks everything.**
2. **Leakage is low** at α ≤ 1 and moderate at α = 2.
3. **Dispositions** shift probabilities (vegetarian reaches the ceiling's contrast) without flipping greedy output.
4. **Facts partly work, which is better than predicted:** +3 to +5 nats on the target, but still far from being the answer.
5. **Peanut goes the wrong way:** the steer primes the *concept* (peanut), not the *relation* (allergic → avoid).
6. **The entropy (confusion) gate helps on every axis.**
7. **Capacity is tiny:** dispositions collapse at N = 6; facts partly survive with the gate.

---

## 4.2 v0.1: what kind of thing does the channel carry?

**Job 415909** (commit `098c317`), 25m13s. Exact-recall assertions passed for the ungated
(topk, pooled) writes.

**Four questions:**
1. **Specificity:** does a fact memory raise *this* name, or any name? (Foils.)
2. **Relations:** can the model *reason* with a memory? (Yes/no relation probes.) Does a contrastive write remove concept priming?
3. **Key granularity:** which moments should become memories? (Write positions.)
4. **Behaviour:** do samples change at the best settings?

**Sweep:** mode {isolated, combined} × gate {entropy} × position {all, boundary, topk, pooled}
× baseline {without, contrastive} × layer {20, 23, 26} × α {1, 2}.

### Q1: Specificity. Facts are specific.
Isolated, L26, α = 2. Δ target vs Δ ceiling, and mean Δ of the two foils.

| Fact | all / without | all / contrastive | Δceiling | Δ foils |
|---|---|---|---|---|
| " Biscuit" | +2.60 | **+3.93** | +9.27 | −0.45 to −0.23 |
| " Ines" | +2.26 | **+3.87** | +13.50 | −0.30 to +0.35 |
| " deep-sea welder" | +9.40 | **+11.33** | +25.35 | −0.67 to +0.19 |

Pooled over facts, all/contrastive: **Δtarget +6.37, Δfoils −0.19, specificity +6.56**, gap
0.261. The memory holds the **specific identity**, not a generic "state a name" mode, at
roughly **30–45% of the ceiling** in log space. The contrastive write adds 20–70% (job +21%, dog +51%, sister +71%). It's still
never enough to make greedy output say the name.

### Q2a: A contrastive write removes concept priming
Disposition contrasts, isolated, L26, α = 2.

| Scenario | all / without | all / contrastive | topk / contrastive | Δceiling |
|---|---|---|---|---|
| vegetarian | −1.14 | **+4.80** | +5.01 | +1.78 |
| peanut | **−4.23** | −0.78 | **+0.28** | −0.07 |
| norway | +8.17 | **+9.93** | **+10.26** | +12.37 |

- Peanut priming is gone (−4.2 → ≈0, matching the ceiling).
- Vegetarian flips from negative to above the ceiling.
- Norway reaches **~80% of the ceiling**.
- Contrastive writes also **leak 2–3× less** (L23 α = 2, all: 0.008 vs 0.025).

Subtracting the counter-experience cancels the concept and keeps the **direction**.

### Q2b: Relations do not transmit
Yes/no relation probes. Mean Δ(consistent − inconsistent answer), memory vs ceiling.

| | vegetarian | peanut | norway | mean |
|---|---|---|---|---|
| ceiling | +4.21 | +7.13 | +0.59 | **+3.98** |
| best memory (all / without, L20 or L26, α = 2) | up to +1.60 | up to +0.95 | up to +0.39 | **+0.98** |
| contrastive (any position) | −0.89 to −0.02 | −0.14 to +0.47 | −0.12 to +0.48 | **≈ 0** |

**Every sample answered "Yes."**, e.g. to "Would a beef burger be a good lunch for me?"
(vegetarian) and "Is a peanut butter cookie safe for me to eat?" (peanut allergy). The
ceiling answers "No." in each case.

### Q3: Write positions
Isolated, L23, α = 2, entropy gate, "without" baseline.

| Position | Tokens per write | gap (dispositions) | Δtarget (facts) | leakage | Relative write error (follow-ups #0/#1/#2) |
|---|---|---|---|---|---|
| all | ~14 | **0.306** | +3.39 | 0.025 | 1.26 / 1.27 / 1.60 |
| **topk (k = 4)** | 4 | 0.277 | **+3.40** | 0.029 | 0.96 / 0.85 / 0.87 |
| pooled | 1 | 0.228 | +1.44 | **0.004** | 1.00 / **0.79 / 0.82** |
| boundary | 5 | 0.118 | +0.06 | 0.003 | 0.98 / 0.93 / 1.01 |

- **The 4 highest-entropy tokens ≈ all tokens.** The confusion signal selects nearly the whole memory in a few moments.
- **Boundary tokens (the assistant header) carry almost nothing.** They fire only at the header positions, and the answer tokens that follow don't resemble them.
- **Pooled** (one write per moment) keeps most of the isolated disposition effect with the **lowest leakage**, and **habituation reappears** (write error < 1). Abstract keys let later writes partly predict earlier ones instead of fighting them.

### Q3b: Capacity (combined, N = 6, L23, α = 2)
| Position / baseline | gap (dispositions) iso → comb | fact gap / Δtarget, combined | leakage, combined |
|---|---|---|---|
| all / without | 0.306 → −0.089 | 0.167 / +2.89 | 0.048 |
| topk / without | 0.277 → **−0.163** | 0.151 / +2.96 | 0.060 |
| pooled / without | 0.228 → **+0.007** | 0.084 / +1.02 | **0.009** |
| all / contrastive | 0.190 → −0.029 | 0.151 / +3.45 | 0.016 |
| topk / contrastive | 0.161 → −0.075 | 0.135 / **+3.77** | 0.027 |

**Dispositions still collapse under combination for every variant.** Pooled comes closest to
harmless. **Facts survive.** A likely reason: the disposition follow-ups share topics
(vegetarian and peanut are both about food), so their keys collide; the fact contexts (dog,
sister, job) are distinct.

### Q4: Samples
Isolated, entropy gate, α = 2, L23 for dispositions and L26 for facts.

- **vegetarian**, "Any ideas for what I should cook for dinner tonight?"
  - all/without: "…I can suggest some **vegetarian and vegan recipes**… 1. Vegan Lentil Soup". **The first behaviour change seen.**
  - Other variants: chicken, pasta, meatballs (like the baseline).
- **peanut**, "What's a good sandwich to pack for lunch?"
  - all/without: "**A peanut butter and jelly sandwich is a classic and safe option**". Concept priming turned into harmful behaviour.
  - boundary/pooled: "turkey and avocado", the ceiling's first suggestion.
- **norway:** no visible change.
- **Relation probes:** all "Yes." (the ceiling says "No.").
- **Facts:** no variant states the name or the job. At most the refusal changes wording ("…based on the information you've provided").

### v0.1 conclusions
1. **Specificity holds:** facts are stored as *this* identity.
2. **A contrastive write fixes concept priming** and carries the *direction* of a preference strongly (up to ~80% of the ceiling), with less leakage.
3. **No variant carries a premise the model reasons with.** The memory changes *what comes to mind*, not *what the model concludes*.
4. **Salience selection works:** 4 surprising tokens ≈ the whole moment. Pooling into one memory per moment gives the lowest leakage and restores habituation.
5. **Capacity for dispositions is still tiny** (collapse at N = 6). Facts are robust.

---

## 4.3 Interpretation: a modulatory channel, not an episodic one

Across both experiments, the steer behaves like a **neuromodulator**: it biases associations
and preferences in open-ended generation, but it never becomes a premise that downstream
reasoning uses.

This is what the emotion paper's finding predicts (P2 in [01-core-idea.md](01-core-idea.md)):
transformers don't keep state in the residual stream. Anything they reason with, they
*attend to*, from representations cached at earlier positions. In the ceiling, the question's
tokens attend back to "I'm vegetarian", and the middle layers compute the implication. A
single vector added at L20–26 never enters that computation.

The same point follows from **hippocampal indexing**: the hippocampus doesn't inject a
correction into the cortex; it *reinstates* a pattern that the cortex then processes as
perception.

So the results suggest a **two-channel memory**:

| Channel | Carries | Mechanism | Status |
|---|---|---|---|
| **Modulatory** | Dispositions, preferences, "how I relate to this person" | The plastic steer (this work) | Works partially; capacity limited |
| **Episodic** | Facts and premises the model reasons with | Reinstating compressed, salient moments where attention can reach them | Next step ([05-next-moves.md](05-next-moves.md)) |

## 4.4 Caveats
- Small: 6 scenarios, 2–3 probes each, one run per configuration (decoding is deterministic, so reruns reproduce exactly; the v0.1 all/without/entropy numbers match v0's).
- The contrast metrics are brittle. At L23 the vegetarian *sample* switched to vegan recipes while its contrast score was slightly negative. Peanut's ceiling contrast is ≈0, so that metric is uninformative for it.
- The gap-closed metric only looks along the ceiling's 20-token continuation.
- The counterfactual write scaffold is still in place.
- One model size (1.5B). Larger models may carry more in a linear steer.
