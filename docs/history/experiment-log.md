# 4. Experiments and results

> **Scope.** This log covers v0, v0.1 and the diagnostics, all on the original v0/v0.1 design.
> The later stages (the attention test, benchmark v1, the gist-key / threshold / least-squares
> experiment and the `samples_v2` text run) are not logged here yet. They are summarised in
> [where-we-are.md §5](../where-we-are.md#5-what-we-found-the-story-in-order) and its appendix.

Two experiments and two diagnostic passes so far, all on **Qwen2.5-1.5B-Instruct (frozen,
fp32)** on one DelftBlue A100 MIG slice (10GB). The method is in [original-method.md](../reference/original-method.md).
Raw outputs are in `results/v0_408770/`, `results/v0_1_415909/`, `results/diag_dose_577895/`
and `results/diag_order_577896/`.

> **Read §4.5 alongside 4.1–4.2.** The diagnostics overturn three earlier readings: that facts
> survive combination, that the relation-probe gains were memory effects, and that key
> collisions by topic explain the capacity loss. Each is marked with a **Correction** note
> where it appears; the original text is left as it was.

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

> **Correction (2026-09-28, [§4.5.2](#452-diag_order-does-write-order-decide-who-survives)):**
> the facts that "partly survive" are **only the last-written fact**. The combined fact
> Δtarget of +2.89 (L23, entropy gate, in the table above) is the mean of dog −1.49, sister
> +0.29 and job +9.87. Job, written last, keeps all of its isolated effect; dog and sister
> lose most or all of theirs. With the write order reversed, job falls to −0.91. Capacity
> loss follows **write order** (recency) plus **crosstalk at read time**, not the type of
> memory.

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

> **Correction (2026-09-28, [§4.5.2](#452-diag_order-does-write-order-decide-who-survives)):**
> much of the "best memory" row is a **template artefact**. The relation score is read at the
> assistant header, a template-tail position where every memory fires, and **all six relation
> probes have "No" as the consistent answer**, so any general shift towards "No" counts as a
> relation gain. Rewriting the same memories without the template tail (L26, all/without,
> α = 2) shrinks the mean gain from +0.98 to **+0.33** (vegetarian +1.60 → +0.43, peanut
> +0.95 → +0.60, norway +0.39 → −0.03). That is about a twelfth of the ceiling's +3.98, not a
> quarter. The conclusion that relations do not transmit stands, and is stronger. Memory
> does fire on these probes ([§4.5.1](#451-diag_dose-how-big-is-the-steer-and-where-does-it-land)),
> so this is not a retrieval failure.

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

> **Correction (2026-09-28, [§4.5.2](#452-diag_order-does-write-order-decide-who-survives)):**
> - **"Facts survive" was job alone**, the last-written scenario. The all/without combined
>   fact Δtarget of +2.89 is the mean of dog −1.49, sister +0.29 and job +9.87. With the write
>   order reversed, job falls to −0.91, and vegetarian (now written last) keeps its gap closed
>   (retention +0.90, against −0.83 when written first).
> - **The key-collision explanation is only partly right.** A food write after vegetarian does
>   hurt it (gap retention −0.40 to −1.45), while a non-food write leaves +0.43 to +0.90. But
>   the keys do not separate by topic or by type: content keys of *any* two scenarios have a
>   mean cosine of 0.15–0.23 after centring, about the same as within one scenario, and
>   vegetarian–peanut is the highest pair by only ~0.01. The topic effect shows up at read
>   time (crosstalk), on top of a strong write-order effect.

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

> **Correction (2026-09-28, [§4.5.2](#452-diag_order-does-write-order-decide-who-survives)):**
> "Facts are robust" is wrong. Only **job, the last-written scenario**, survives combination.
> Across the four settings tested (L23/L26 × without/contrastive), dog and sister keep at
> most 45% of their isolated target gain, and at L23 it can reverse. Capacity loss follows
> write order (recency) plus crosstalk at read time.

---

## 4.3 Interpretation: a modulatory channel, not an episodic one

Across both experiments, the steer behaves like a **neuromodulator**: it biases associations
and preferences in open-ended generation, but it never becomes a premise that downstream
reasoning uses.

This is what the emotion paper's finding predicts (P2 in [core-idea.md](../reference/core-idea.md)):
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
| **Modulatory** | Dispositions, preferences, "how I relate to this person" | The plastic steer (this work) | Works partially; recall unselective, capacity set by write order (§4.5) |
| **Episodic** | Facts and premises the model reasons with | Reinstating compressed, salient moments where attention can reach them | Next step ([next-moves-v0.1.md](next-moves-v0.1.md)) |

**Update after the diagnostics (§4.5).** The two-channel reading survives, and the diagnosis
of the modulatory channel is sharper. Its problem is **selectivity, not strength**. At α = 2
it already brings back most of a stored imprint on related prompts, but unrelated prompts get
about half as much, largely through the chat template, and in a combined memory the last
write wins. The premise failure is real, because memory does fire on relation probes. The
small relation gains seen before were mostly the template tail, so the evidence that the
steer carries no premise is now stronger.

## 4.4 Caveats
- Small: 6 scenarios, 2–3 probes each, one run per configuration (decoding is deterministic, so reruns reproduce exactly; the v0.1 all/without/entropy numbers match v0's).
- The contrast metrics are brittle. At L23 the vegetarian *sample* switched to vegan recipes while its contrast score was slightly negative. Peanut's ceiling contrast is ≈0, so that metric is uninformative for it.
- The gap-closed metric only looks along the ceiling's 20-token continuation.
- The counterfactual write scaffold is still in place.
- One model size (1.5B). Larger models may carry more in a linear steer.
- **The relation-probe metric is confounded** (found in §4.5). It is read at the template tail, where every memory fires, and all six probes have "No" as the consistent answer, so a general yes/no bias scores as a relation effect. Relation probes need both answer polarities and a no-tail baseline.
- **Low leakage is not selectivity.** The KL leakage on unrelated probes is small, but the memory still acts on them: recall is about 0.4 of an imprint, mostly on the template (§4.5.1). Small KL means the output tolerates that at α ≤ 2, not that recall is selective.
- Retention ratios (§4.5.2) are noisy where the isolated effect is small (dog at L23), and are not computed where it is ≈0 or negative.
- The diagnostics used write position `all` only. `topk`, `pooled` and `boundary` were not re-examined.

---

## 4.5 Diagnostics: dose, selectivity and write order

Two measurement passes, run to decide between three readings that v0/v0.1 left open: the
steer is **too weak** (dose), memory **doesn't fire** on relation probes
([open-notes.md](open-notes.md) N1), or dispositions **collide by topic** (Q3b). Neither
pass changes the method. Both reuse the v0.1 pipeline (entropy gate, write position `all`),
both ran at commit `1d0628e`, and each took about 6 minutes on one MIG slice after the unit
tests.

A read-only reanalysis of the existing v0/v0.1 outputs came first ("step 0"; its scripts
were not kept). In those runs write order and scenario identity are confounded, because the
order was always vegetarian > peanut > norway > dog > sister > job. diag_order was designed
to separate them, and its results supersede step 0.

**Headline: the memory isn't too weak, it's unselective.** At α = 2 it already recalls most
of a stored imprint on related prompts, but also about half as much on unrelated ones, much
of it on the chat template. **Capacity loss is driven by write order** (the last write
survives: RNN-style recency) **plus crosstalk at read time**. **The relation-probe gains were
partly template artefacts.**

### 4.5.1 diag_dose: how big is the steer, and where does it land?
**Job 577895.** Baselines without (primary) and contrastive; modes isolated and combined;
layers 14, 17, 23, 26; α {1, 2, 4}. A logging copy of the inject hook records the steer
s = α·M·k(h) and its statistics at every position of every input:
- the scenario probes, teacher-forced with the ceiling's continuation
- the relation probes, prompt only
- the unrelated probes, with the baseline's continuation

The hook matched `memory.read` exactly on all 2,496 calls. It also gave logits identical to
the library's `inject` (max difference 0.0 over 48 end-to-end checks).

Positions are split into five kinds:
- the **template head**: the 24-token shared template prefix (system prompt and start-of-turn tokens), the same in every prompt
- **user text**
- the **template tail**: 5 tokens (end-of-turn + assistant header), also the same everywhere
- the continuation
- the **answer position**: the last prompt token, where yes/no is scored

Three quantities are used:
- **Steer ratio** ‖s‖ / ‖h − μ‖: the steer against the part of the state that differs from the average state.
- **Norm change** ‖h + s‖ / ‖h‖ − 1.
- **Recall fraction** ‖M·k(h)‖ / mean ‖Δ‖: how much of a stored imprint comes back (1 ≈ a full one). It does not depend on α.

#### Dose (isolated, without; median over all inputs and positions)
| Layer | steer ratio, α = 1 / 2 / 4 | norm change α = 2 (median / p90) | norm change α = 4 (median / p90) |
|---|---|---|---|
| L14 | 0.105 / 0.210 / 0.421 | +0.9% / +5.9% | +4.6% / +21.2% |
| L17 | 0.100 / 0.200 / 0.399 | +0.6% / +5.2% | +3.8% / +19.8% |
| L23 | 0.121 / 0.243 / 0.485 | +2.6% / +10.2% | +10.1% / +30.6% |
| L26 | 0.152 / 0.304 / 0.608 | +2.6% / +10.6% | +10.1% / +32.9% |

At α = 2 the steer is **about a quarter of ‖h − μ‖** and changes the state's norm by 1–3%.
At α = 4 it doubles. In the late layers it then reaches the size of ‖h − μ‖ where it
concentrates. At L23/L26 the median ratio is 1.0–1.2 on template-tail positions and 1.1–1.3
on related probes' user text, and tail positions grow in norm by a median of 23–33%. **That is
where α = 4 breaks.**

#### Recall is strong, and unselective
Recall fraction on user-text positions (isolated, without; median):

| Layer | exact | paraphrase | related (IQR) | relation | unrelated |
|---|---|---|---|---|---|
| L14 | 1.04 | 0.70 | 0.59 (0.50–0.86) | 0.45 | 0.38 |
| L17 | 0.99 | 0.71 | 0.60 (0.49–0.82) | 0.44 | 0.38 |
| L23 | 0.98 | 0.75 | 0.79 (0.53–1.13) | 0.50 | 0.38 |
| L26 | 0.99 | 0.85 | 0.87 (0.55–1.18) | 0.61 | 0.47 |

- A related prompt gets **0.6–0.9 of a full imprint**, but an unrelated one still gets **about 0.4**, only 1.5–2.1× less.
- On the template, recall is the same for every prompt: 0.29–0.54 on the head and 0.47–0.86 on the tail, rising with layer. Tail positions match a stored tail key at cosine 0.84–0.99 whatever the prompt.
- **Why:** the keys never get far apart. Unrelated user text still matches some stored content key at cosine 0.31–0.37 (related: 0.47–0.60). The read is linear in the match, so a 0.3 match returns a real fraction of the imprint. Content keys of *different* scenarios have a mean cosine of 0.15–0.23 even after centring (§4.5.2).
- On the write side, Δ is 38–42% of ‖h_without − μ‖ on content tokens but only 9–18% on tail tokens. The tail's deltas are small, but its keys match everything.

#### Where the steer lands on unrelated probes
Share of the steer energy Σ‖s‖² by position (isolated, without; independent of α):

| Layer | template head | user text | template tail | continuation | head + tail |
|---|---|---|---|---|---|
| L14 | 0.480 | 0.168 | 0.153 | 0.198 | 0.633 |
| L17 | 0.410 | 0.191 | 0.198 | 0.201 | 0.608 |
| L23 | 0.400 | 0.206 | 0.262 | 0.132 | 0.662 |
| L26 | 0.467 | 0.189 | 0.223 | 0.121 | 0.690 |

**About two-thirds of the steer on unrelated prompts lands on the chat template** (0.68
pooled over layers). The tail alone carries 15–26%, and the 24-token head carries the most
(40–48%).

#### Combining makes recall louder and less like itself
In combined mode, each scenario's recall is compared with what its own isolated memory
recalls at the same state. Own-scenario inputs, median; the cosine is taken on user text, the
size ratio over all positions:

| Layer | cos(combined, isolated): without / contrastive | combined ÷ isolated steer size: without / contrastive |
|---|---|---|
| L14 | 0.57 / 0.19 | 1.25 / 1.58 |
| L17 | 0.50 / 0.15 | 1.34 / 1.71 |
| L23 | 0.32 / 0.11 | 1.12 / 1.67 |
| L26 | 0.27 / 0.08 | 1.11 / 1.62 |

- A scenario's combined recall has only **0.27–0.57 cosine with its isolated recall** and is 1.1–1.3× louder (1.1–1.6× on unrelated inputs).
- With contrastive writes it is almost unrelated to its isolated recall (cosine 0.08–0.19) and 1.6–1.7× louder, reaching 2.7–4.2× for dog and sister.
- The exception is **job, the last write**. Its combined recall stays close to its isolated one: cosine 0.86–0.90 without, 0.93–0.98 contrastive. This is §4.5.2's recency effect, seen from the read side.

#### Memory does fire on relation probes
On user text, relation probes get **63–77% of the related probes' recall** (0.44–0.61 vs
0.59–0.87 in the table above). At the answer position the steer ratio is 0.21–0.33 on
relation probes, against 0.31–0.45 on related disposition probes. So the premise failure in
Q2b is **not a firing failure** (N1).

Two qualifications apply:
- Relation probes' user text matches the stored content keys barely better than unrelated text does (0.35–0.43 vs 0.31–0.37).
- At the answer position, the best match is always a template-tail key.

So what fires where the answer is read is mostly the shared template recall. §4.5.2 removes
exactly that.

#### Pre-registered predictions (isolated, without, α = 2)
| | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | steer ≈ 0.5–1× ‖h − μ‖ | 0.24 (0.20–0.30 by layer) | **Wrong:** the steer is smaller |
| P2 | norm change < 10% | median +1.4%, p90 +8.1% | **Right** |
| P3 | the template tail carries most of the steer on unrelated probes | tail 0.23; head + tail 0.68 | **Half right:** the whole template does |
| P4 | recall fraction on related probes ≈ 0.3 | 0.71 (IQR 0.51–0.98) | **Wrong:** recall is strong |

Taken together: the steer is small in absolute terms and still recalls most of what was
stored. **Dose is not the bottleneck; selectivity is.** Turning α up only brings the template
and question positions to the size of the state's own signal.

### 4.5.2 diag_order: does write order decide who survives?
**Job 577896.** α = 2, L23 and L26, baselines without and contrastive. Only the set and order
of the writes change, and each condition builds a fresh memory:

| Condition | Writes |
|---|---|
| `iso` | One memory per scenario: the reference for retention |
| `a_comb_orig` | All six, in the v0.1 order vegetarian > peanut > norway > dog > sister > job |
| `b_comb_rev` | All six, reversed |
| `c_veg_<x>` | Vegetarian first, then one interferer: peanut (food), dog (non-food fact) or norway (non-food disposition) |
| `d_comb_notail` | Original order, with writes that skip the 5 template-tail tokens |
| `d_comb_notail_renorm` | As `d`, with the entropy gate renormalised over the kept tokens |
| `iso_notail`, `c_veg_<x>_notail` | No-tail references |

**Retention** = combined effect ÷ isolated effect (1 = kept, 0 = lost, < 0 = reversed). It
isn't computed where the isolated effect is ≈0 or negative. The run also records
**forgetting curves** and **key overlap**:
- **Forgetting curves:** recall fidelity at each scenario's own write keys after every scenario's writes.
- **Key overlap:** the mean cosine between the write keys of each pair of scenarios.

**The reproduction check passed.** `a_comb_orig` and `iso` match v0.1's `results.jsonl` on
all 192 rows (max difference 0.0) and on all 8 leakage groups.

#### Reversing the order flips who survives
Gap-closed retention, L23, without. Write position in brackets.

| Scenario | original order | reversed order |
|---|---|---|
| vegetarian | **−0.83** (1st) | **+0.90** (6th) |
| peanut | −0.03 (2nd) | −0.16 (5th) |
| norway | +0.40 (3rd) | −0.28 (4th) |
| dog | −0.94 (4th) | −2.15 (3rd) |
| sister | +0.62 (5th) | −0.72 (2nd) |
| job | **+1.14** (6th) | **−0.01** (1st) |

The flip holds in all four settings (L23/L26 × without/contrastive):
- **Vegetarian:** between −0.43 and −1.13 when written first; between +0.90 and +1.03 when written last.
- **Job:** between +1.03 and +1.14 when written last; between −0.16 and +0.06 when written first.

**Whatever is written last survives almost whole; earlier memories are lost or reversed.**
This is RNN-style recency. It isn't strictly monotone in position: dog's isolated effect is
small (+0.08), so its ratio is noisy.

#### "Facts survive" was the last-written fact
Fact Δtarget (log-prob of the name or job), isolated → combined in the original order:

| Fact (position) | L23, without | L26, without | L26, contrastive |
|---|---|---|---|
| dog (4th) | +0.21 → −1.49 | +2.60 → +0.19 | +3.93 → +0.74 |
| sister (5th) | +1.12 → +0.29 | +2.26 → +0.43 | +3.87 → +1.74 |
| job (6th) | +8.84 → +9.87 | +9.40 → +9.70 | +11.33 → +11.58 |

The v0.1 pooled figure (combined fact Δtarget +2.89, all/without, L23) is the mean of
−1.49, +0.29 and +9.87: **it is job alone**. With the order reversed, job's own Δtarget falls
to −0.91 (L23) and −2.02 (L26). Specificity (target minus foils) holds up better for sister,
because its foils fall too, but its target barely rises.

#### Topic matters at read time, not in the keys
Vegetarian written first, then one interferer. Vegetarian's gap-closed retention:

| Interferer | L23, without | L26, without | L23, contrastive | L26, contrastive |
|---|---|---|---|---|
| peanut (food) | −0.40 | −0.98 | −1.20 | −1.45 |
| dog (non-food fact) | +0.43 | +0.67 | +0.53 | +0.52 |
| norway (non-food disposition) | +0.50 | +0.90 | +0.54 | +0.58 |

- **A food write after vegetarian ruins it.** A non-food write leaves +0.43 to +0.90.
- Yet vegetarian's **own contrast** (risotto − steak) mostly survives. With contrastive writes, where its isolated contrast is clearly positive, it keeps 0.72–0.94 of it after peanut, and 0.61–0.77 after the non-food writes. The stored direction is still there. What breaks the match to the ceiling is peanut's recall firing on vegetarian's food probes and adding its own change on top: **crosstalk at read time more than erasure at write time.**
- The forgetting curves agree (L23, without). At vegetarian's own write keys, the recall's cosine to its stored Δ drops from 0.62 to 0.39 at the peanut write. Over the next four writes it only drifts to 0.31. Meanwhile its relative error rises above 1 (0.98 → 1.65). Foreign content is being added on top; the trace is not simply fading to zero.
- **The keys don't separate by topic.**
  - Content keys of different scenarios have a mean cosine of 0.15–0.20 (L23) and 0.18–0.23 (L26) after centring.
  - That is about the same as within one scenario: 0.17–0.21 and 0.20–0.24.
  - Vegetarian–peanut is the highest pair, but only by ~0.01.
  - The same template-tail token has cosine 0.85–0.90 between any two scenarios, so the tail keys are practically one shared key.

#### The template tail: no rescue for capacity, but it drove the relation gains
Writing without the 5 template-tail tokens (`d_comb_notail`):
- **It doesn't rescue early memories.** Vegetarian's gap retention is −0.72 / −1.13 / −0.54 / −0.40 without the tail and −0.83 / −1.13 / −0.62 / −0.43 with it (L23 without, L26 without, L23 contrastive, L26 contrastive). Dog gets worse: −1.53 vs −0.94 at L23, without.
- **Self-recall looks better, but only because the tail is gone.** Mean relative error at a memory's own write keys falls from 1.35 to 0.61 (isolated, L23, without; 1.54 → 0.67 at L26). The content tokens' error is 0.57–0.66 either way. The tail tokens' own error is 1.7–3.4, worse than recalling nothing, because each scenario makes 15 tail writes under near-identical keys.
- **It removes the relation gains in combined mode, and most of them in isolated mode.**

Δ relation score (consistent − inconsistent answer), without baseline:

| Scenario, layer | isolated | isolated, no tail | combined | combined, no tail |
|---|---|---|---|---|
| vegetarian, L23 | +0.77 | +0.24 | +1.12 | −0.12 |
| peanut, L23 | +0.82 | +0.36 | +1.52 | +0.37 |
| norway, L23 | −0.10 | −0.10 | +0.72 | −0.36 |
| vegetarian, L26 | +1.60 | +0.43 | +1.98 | −0.09 |
| peanut, L26 | +0.95 | +0.60 | +2.53 | +0.08 |
| norway, L26 | +0.39 | −0.03 | +1.60 | −0.28 |
| **mean, L23 / L26** | **+0.49 / +0.98** | **+0.17 / +0.33** | **+1.12 / +2.03** | **−0.04 / −0.09** |

The ceiling's mean is +3.98.

- In combined mode the relation gains are **larger than in isolation**, and **vanish** without the tail.
- In isolation the mean gain shrinks by about two-thirds. Peanut at L26 keeps the most (63%).
- **Why this is an artefact:** the score is read at the answer position, a template-tail position where every memory fires, and all six relation probes have "No" as the consistent answer. A general shift towards "No" therefore scores as a relation gain.
- With contrastive writes, the sign of that shift follows the last write. At L23 the original order (job last) gives +0.58 to +1.00 on all three dispositions, and the reversed order (vegetarian last) gives −0.93 to −1.00 on all three.
- **The renormalised gate changes nothing, by construction.** No follow-up has its entropy maximum inside the tail (0 of 18), so renormalising over the kept tokens leaves every gate value unchanged. `d_comb_notail_renorm` is identical to `d_comb_notail`.
- Combined leakage (mean KL on unrelated probes) is about twice the isolated mean, with or without the tail (L23, without: 0.048 and 0.045 vs 0.025).

### 4.5.3 What the diagnostics change
1. **Not too weak, but unselective.** At α = 2 the steer is a quarter of ‖h − μ‖ (a 1–3% norm change). Yet it brings back 0.6–0.9 of an imprint on related prompts and about 0.4 on unrelated ones, and two-thirds of the steer on unrelated prompts sits on the chat template. More α doesn't help: α = 4 breaks because the steer reaches ‖h − μ‖ at template and question positions.
2. **Capacity loss comes from write order plus read-time crosstalk.** The last write survives whole (recency), and same-topic memories add onto each other's probes. "Facts survive" (v0 conclusion 7, v0.1 conclusion 5) was job, the last write. Keys don't separate by topic or type: all content keys share about 0.2 cosine after centring, and the tail keys are one shared key.
3. **Relation gains were partly template artefacts.** The tail carried a shared yes/no bias. Without it, the isolated relation gain is about a twelfth of the ceiling's, and the combined one is zero. Memory fires on relation probes (63–77% of related), so the premise failure is real. This answers N1 in [open-notes.md](open-notes.md).
4. **For the next steps** ([next-moves-v0.1.md](next-moves-v0.1.md)), selectivity comes before dose. Candidate fixes are no read or write at template positions, whitened keys, recursive least squares and a match threshold. Relation probes must be re-baselined without the template tail.
