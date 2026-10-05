# 6. Open notes

Observations made while re-reading docs 01–05 and the neuromodulator side idea
(2026-09-25). None of them has been acted on yet. They are here to come back to: some sharpen
moves already in [next-moves-v0.1.md](next-moves-v0.1.md), some add new ones. (Since then, the
diagnostics in [experiment-log §4.5](experiment-log.md#45-diagnostics-dose-selectivity-and-write-order)
have settled N1 and part of N8; see the notes under them.)

---

## 6.1 On the memory results

### N1. The relation-probe failure may mix "not used" with "not retrieved"
The v0.1 conclusion ([experiment-log.md](experiment-log.md), Q2b) is that the steer carries
no premise the model reasons with. But a relation probe is scored **right after the
assistant header**: log P("Yes"/"No") is predicted at the header positions. Q3 found that
header positions carry almost nothing, and relation probes ("Is a peanut butter cookie safe
for me to eat? Answer yes or no.") are worded unlike any session-1 follow-up. So the
recalled vector at the positions that matter may simply be small. The docs don't report how
strongly memory fired on relation probes.

These are two different failures:
- **Fired, not used:** ‖α·M·k(h)‖ on relation-probe tokens is comparable to the related probes, yet the answer doesn't move. Then the steer really isn't a premise.
- **Didn't fire:** recall is small on those tokens. Then it's a retrieval failure, and says nothing yet about whether a steer *could* carry a premise.

**What would settle it:** log ‖α·M·k(h)‖ / ‖h‖ and the max cosine to stored keys, per
position, for relation probes vs related probes. It costs about as much as B1.

> **Resolved by the diagnostics (2026-09-28, [experiment-log §4.5](experiment-log.md#45-diagnostics-dose-selectivity-and-write-order)):
> fired, not used.** On user text, memory fires on relation probes at **63–77% of its
> related-probe level** (recall fraction 0.44–0.61 vs 0.59–0.87, L14–26). At the answer
> position the steer is 0.21–0.33 of ‖h − μ‖, against 0.31–0.45 on related disposition
> probes. So the premise failure is real, not a retrieval failure. Two refinements:
> - What fires at the answer position is mostly the **template-tail** recall, which is shared by every prompt. Relation probes' own text matches the stored content keys barely better than unrelated text does (0.35–0.43 vs 0.31–0.37).
> - That tail recall carried a **spurious yes/no bias**. All six relation probes have "No" as the consistent answer. Without the tail, the isolated relation gain shrinks by about two-thirds (L26 mean +0.98 → +0.33) and the combined gain vanishes. Relation probes need re-baselining without the tail, and with both answer polarities, before they're used again.

### N2. Where the premise enters: treat multi-layer injection as a main arm of A1
In the ceiling, the premise is *tokens*. The question's tokens attend back to them, and the
implication is computed in the middle layers. The steer instead is an additive vector on the
probe's *own* positions, from layer ℓ upward.

A single earlier layer (A1) does put the change into K/V for every layer above it. But v0
already shows single early-layer steers do nothing (L6) or hurt (L14 at α ≥ 2). Injecting at
several layers with a small α each (A1's multi-layer variant, Larimar-style) may get the
change into the middle layers without the damage one large early steer causes. So it
deserves to be a main arm of A1, not a side variant.

Even then, an additive steer takes a different path from "attend to the premise". If neither
arm flips relations, that's the argument for A2 (reinstatement) rather than for more steer
tuning.

### N3. The online-baseline problem (C) is harder than it looks
Contrastive writes worked because the counter-experience cancelled the **shared concept**
("peanut-ness") and kept the relation. For online writing, C proposes memory's own
prediction as the baseline (e = Δ − M k). But M k only cancels what **memory** already holds.
The concept isn't in memory; it's in the **model's priors**, in what the words themselves do
to the residual stream. So on first exposure, memory's prediction can't cancel concept
priming, and the "without"-style failure (priming peanut butter) comes back.

Whatever replaces the counter-experience has to come from the model's priors, not from
memory. Rough candidates:
- a generic concept direction for the salient word (a CAA-style mean over neutral text containing it), projected out of Δ
- the model's own prediction of its next state, so only what the model didn't expect is stored

Open: which of these, if any, also keeps the relation.

### N4. Numbers checked
The derived figures in 04 match their tables:
- contrastive gains of +21% / +51% / +71%
- pooled fact Δtarget of +6.37
- contrastive leakage 0.008 vs 0.025

No action needed.

---

## 6.2 On the neuromodulator side idea

Relates to [core-idea.md §1.7](../reference/core-idea.md#17-side-idea-steering-vectors-as-neuromodulators-parked)
and the fuller side-idea note in the parent `docs/` folder (not in the repo).

### N5. Additive vs multiplicative is the line between content and modulation
A steering vector **adds a direction**, which is structurally *content*. That's consistent
with v0.1: the steer biases what comes to mind. Real neuromodulators mostly change **gain**:
how strongly circuits respond, not what they represent. The closer LLM analogue is
multiplicative:
- per-head attention temperature (sharpen vs broaden what is attended to)
- per-layer or per-head scaling of MLP or attention outputs
- per-layer residual gain

This fits the adaptive-gain account of noradrenaline (Aston-Jones & Cohen, 2005; cited from
memory): high gain = exploit / focus, low gain = explore. "Receptor specificity" then becomes
*which* heads or layers a gain factor applies to. It answers the side idea's first open
question ("processing style vs content") by the **form of the operation**, not by the choice
of direction.

### N6. Endogenous release adds tonic state, which transformers lack
P2 (the emotion paper): emotion representations are locally scoped, so there's no chronic
mood in the residual stream. The endogenous mode is therefore more than "steering, closed
loop". It adds a **state variable outside the forward pass** that carries across tokens and
has release, decay and tolerance. That's a new channel, not a reframing.

It maps onto the biological tonic/phasic distinction:
- **phasic:** a brief release at a moment of surprise or confusion
- **tonic:** a slowly drifting baseline level

### N7. A closed loop needs reuptake to stay stable
Probe reads "desperate" → release the desperate vector → the probe reads more desperate →
runaway. Biology prevents this with **reuptake** (decay) and **tolerance** (a weaker response
to repeated exposure). So those dynamics aren't cosmetic; they are what keeps the controller
stable.

The same risk applies to Seahorse if salience-gated writing ever goes online: write → state
shifts → more writes.

**Safety:** the endogenous mode is where the risk sits. An external "drug" is visible and
controlled; a self-releasing loop tied to the "desperate" vector is exactly the
reward-hacking route the emotion paper describes.

### N8. Memory steers already behave pharmacologically
Two of the "missing" properties the side idea lists already show up in v0:
- **Inverted-U dose-response:** α = 2 is best, α = 4 breaks everything, and leakage grows nonlinearly (0.003 → 0.257 from α = 0.5 to 4).
- **Receptor specificity:** the same memory steer does nothing at L6, is negative at L14, and helps at L23–26.

> **Update (2026-09-28, [experiment-log §4.5.1](experiment-log.md#451-diag_dose-how-big-is-the-steer-and-where-does-it-land)):**
> the top of the inverted U is now located. At α = 2 the steer is about a quarter of
> ‖h − μ‖ (a 1–3% norm change). At α = 4, in L23/L26, it reaches the size of ‖h − μ‖ on
> template-tail and question positions (median ratio 1.0–1.3), and tail positions inflate by
> a median of 23–33%. That is where it breaks.

### N9. A unifying picture: three tiers on the same machinery
| Tier | What it is | Gated by | Status |
|---|---|---|---|
| **Tonic modulators** | Global state (gain or broad steers) with release and decay | Internal signals (confusion, surprise, valence) | Side idea |
| **Modulatory memory** | The v0/v0.1 steer: a *learned, cue-dependent* modulator | Resemblance to stored situations | Works partially |
| **Episodic memory** | Stored moments reinstated into attention | Retrieval by key match | Next step (A2) |

On this view the Seahorse steer is closer to a **conditioned response** (a smell bringing
back a mood) than to a stored fact. That fits why it carries dispositions and identities but
not premises.
