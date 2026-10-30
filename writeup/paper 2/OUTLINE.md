# Paper 2: outline and claims (draft for approval)

*Written 2026-10-30 from `docs/dreaming/01-binding.md` (Results, addenda 1–9), `docs/dreaming/literature-exp1.md` and the report files in `results/dream_bind_*`. Nothing here is drafted prose yet: this is the claims → evidence map and the section plan, to agree before writing. Addendum 9 (storage order with 3 and 5 facts) is in (C9).*

---

## Working title

- **"What Binds a Memory? Cache Slices, Recurrent States, and Why Separately Stored Facts Swap"** (preferred)
- Alternatives: "Two Tokens and an Order Tag: What a Reinstated Memory Needs" / "Storing Facts as Cache Slices: What Carries the Value, What Recognises It, and What Binds It"

## The paper in three sentences

A fact told to a model in one session can be carried into the next by keeping a tiny slice of the model's own key/value cache (the value's position and the full stop after it), which works where pushing a vector does not and beats the same tokens as text, on three model families. On a hybrid model the two kinds of memory split the work: the recurrent state recognises a name it is shown, the attention entries select the right one, and a mostly generic "answer personal questions" gate explains why attention entries alone refuse direct questions. Slices stored in separate sessions swap partners when used together because each was encoded as "the first fact"; storing a new memory as a later fact removes most swaps for up to three memories, and joint encoding writes the binding at the subject's positions.

## Claims → evidence

Status: **P** = pre-registered rule, decisive; **E** = exploratory addendum with a stated rule (reported, rule verdict given); **R** = reported only. Models: Q = Qwen2.5-1.5B, H = Qwen3.5-2B (hybrid), L = Llama-3.2-3B, 9B = Qwen3.5-9B (teacher-forced only).

| # | Claim | Key numbers | Status | Models | Source |
|---|---|---|---|---|---|
| C1 | **The value is read from its own position, late; the subject holds no copy.** So a write at the subject cannot work, and neither can a push at the answer. | Patching the question's subject moves ≤ 0.02 of the preference, the attribute 1.00; knockout answer→attribute in late blocks costs 0.59 of the gain (Q). Subject write: clean use 0.00. Paper 1: pushed facts ≤ 0.15 clean use. | P (stage 0, subject arm) | Q, H, L (+ pool) | stage0, gen_0038, add7 D |
| C2 | **Two stored positions (value + full stop) carry the fact into a new session**, near the sentence in context, gated by a subject match; the user-turn segment keeps ownership. | Q: R-min 0.86 vs ctx 0.90, R-full 0.87, harvested after an unrelated chat 0.80; unrelated prompts token-identical 1.000 gated (0.064 ungated); self-claim 0. 52 pool facts' use prompts 0.97 (Q) / 0.98 (L). | P (arm R) + E (pool) | Q, L; H with state | gen_0038, add7 C |
| C3 | **The slice carries more than its tokens.** | Same 2 tokens as text: Q 0.27, H 0.24, L 0.24 vs slice 0.86 / 0.97 / 1.00. Verification (two-choice): L 0.95 vs 0.68. | E (rule "slice > text" holds on all) | Q, H, L | add5, add7 E, add8 6 |
| C4 | **On a hybrid, the state recognises and the attention entries select** (double dissociation). Recognition is item-specific (rejects look-alike and same-theme lures), across sessions, and does not bind two facts. | H: attention only yes/no 0.50 / two-choice 0.91; state only 0.93 / 0.48; both 0.97 use. Lures: false yes 0.00–0.04 vs 0.07 unrelated (7 items), −0.01 diff on 38 pool facts. Joint state of 2 facts: balanced 0.50. Recurrent part, not the conv buffer (0.86 vs 0.50). 9B: attention half replicates; state margins item-specific but below the "No" prior. | E (item recognition: holds twice) | H (+9B TF) | state2b, add6 C, add7 A/F, add8 2a/6 |
| C5 | **Attention entries alone hold the value but refuse direct questions; the block is mostly a generic response gate.** | H: TF +10.0 nats (ctx +10.65) but direct question 0.12–0.23. A ctx − nomem "told" direction lifts it to 0.76–0.83 (held out); a comply direction from personal vs general questions with no memory lifts it to 0.67. Told − comply: H +0.09 (generic comply), Q +0.09, L +0.15. On held-out facts told costs use prompts (H −0.14, Q −0.18). | E (verdicts mixed; see framing) | H, Q, L | add7 B, add8 2b/3 |
| C6 | **Separately stored slices swap partners when used together**; shifting positions does not help; joint encoding fixes it. | Partner confusion sep vs joint: Q 0.37 vs 0.05, H 0.33 vs 0.05, L 0.41 vs 0.01 (8 pairs); 34 pool pairs 0.16–0.22 vs 0.00. Different-subject pairs too (0.20–0.26). | E (rule holds on all, both pair sets) | Q, H, L | add5, add7 E, add8 2c |
| C7 | **The cause is storage order: every separately stored slice is "the first pair".** Storing the new memory after a dummy fact (dummy dropped) removes most swaps; an unrelated sentence helps less, so it is not just "not at the start". | Confusion sep → stored second: Q 0.37 → 0.09, H 0.33 → 0.15, L 0.41 → 0.09; pool 0.16–0.22 → 0.01. Placebo sentence instead: 0.26 / 0.19 / 0.31 (pool 0.07 / 0.05 / 0.13). Caveat: both stored second recreates swaps only on Q (pool 0.20 vs 0.22; L 0.03). | E (rule: holds Q, half H, fails L on the bothtagged clause) | Q, H, L | add8 1a |
| C8 | **Joint encoding writes the binding at the subject's positions.** | 2 positions + subject span from a joint harvest = joint (Q 0.03, L 0.01 vs 0.20, 0.17 for 2 positions; pool 0.00); the same positions from separate harvests don't help (0.34, 0.29). | E (holds on both pure transformers, both pair sets) | Q, L (H reported) | add8 1b |
| C9 | **Storage order is a consolidation without re-encoding, up to a limit.** Storing the k-th memory as the k-th fact reaches joint encoding for different subjects (N = 3 and 5) and removes most swaps among 3 same-kind memories; at 5 same-kind memories it halves them, and the rest are neighbours swapping. The stored tag counts, not the placement order. | Confusable N = 3: Q 0.47 → 0.09, H 0.38 → 0.04, L 0.41 → 0.09 (joint 0.00–0.01); N = 5: 0.54 / 0.55 / 0.51 → 0.22 / 0.23 / 0.30 (joint 0.00–0.03). Different subjects: ordered 0.00–0.03 everywhere. Reversed placement: −0.02 to +0.03. Neutral sentences instead: 0.27 / 0.20 / 0.34 (N = 3). | E (holds Q, H at N = 3; fails all at N = 5; holds all 6 different-subject cells) | Q, H, L | add9 |
| N1 | Negative: adding the binding-ID direction to a stored slice does not repair swaps. | Direction real (flips 0.75 vs 0 random), repair −0.02 [−0.14, 0.11]. | E | Q | add6 A |
| N2 | Retrieval by the subject's words scales to 59 facts; a wrong pick gives a confident wrong fact. | Right slice 0.85–0.92 at N = 10–59; wrong slice confab 0.86–0.90. | E | Q | add6 B |

## What we do not claim (framing guards)

- **Not a new caching mechanism.** Reinstating a turn equals prompt caching exactly (asserted to 1e-4). Prompt Cache, CacheBlend, EPIC, InfLLM do parts of it. The contribution is *which part carries what, and why combining fails*.
- **Not "recognition from the state" as such** (Afendulev et al., Sept 2026, show it within one session, as gist familiarity). Ours: across sessions, item-specific (lures rejected), and the double dissociation with attention.
- **C5 is not "a missing told signal".** Addendum 8 weakened it: say "a mostly generic response gate" (link to refusal-direction work); the memory-specific part is small and model-dependent. Keep it as a secondary result.
- **C9 has a limit:** five same-kind memories still swap (neighbouring order tags blur); say it.
- **C7: storage order is most of the effect, not all.** The placebo control rules out "only not-at-the-start" (the EPIC / attention-sink reading), but "both stored second" brings the swaps back only on Qwen2.5. Say both.
- **Gates are oracle in the collision runs** (forced open); retrieval is a separate result (N2).
- Small models (1.5–3B in generation; 9B teacher-forced only); synthetic personal facts (names and attributes), single-sentence memories.

## Section plan (workshop main text ~4 pages + appendix; an arXiv version can carry the appendix tables in full)

1. **Introduction** (0.5 p). Memory for a frozen model: what has to be stored for a fact to be *used* later? Paper 1: a pushed vector is not a store. Here: cache slices and hybrid states. Contributions = C2/C3, C4, C6–C8 (+ C9).
2. **Setup** (0.5 p). Harvest a session, keep a slice, reinstate it in a later session's user turn, gate by subject match. Conditions table (ctx, R-full, R-min, T-min, kv6 / state / both, R2-*). Items: 7 hand-written + 52 pool facts, 8 + 34 pairs. Judges: majority of three families; teacher-forced margins; balanced yes/no; bootstrap CIs; pre-registration.
3. **What carries a fact** (0.75 p). C1 → C2 → C3. Figure 2: clean use by condition, 3 models (slice vs text vs in context).
4. **What a hybrid adds** (0.75 p). C4 (Figure 3: the 2×2 yes/no vs two-choice dissociation), lures, no conjunction; C5 briefly (told vs comply vs random).
5. **Two memories** (1.25 p). C6 → C7 (with the placebo control and the bothtagged caveat) → C8 → C9. Figure 4: partner confusion by condition × model (sep, placebo, stored second, both second, joint, 2-pos, 2-pos + subject). N1 in a sentence.
6. **Related work** (0.25 p, longer in appendix). Binding IDs / lookbacks (Feng & Steinhardt; Prakash; Dai); read site (Geva vs in-context); cache reuse (Prompt Cache, CacheBlend, EPIC/HYPIC, InfLLM, LinearKV); hybrid states (Afendulev, DASC, KV-State Grafting); trained compression (Cartridges, Memory Inception); known-entity and refusal directions (Ferrando; refusal-direction work — **to add to literature-exp1**).
7. **Limitations and outlook** (0.25 p). As in the guards; the offline ("dreaming") phase as the place where joint encoding / ordering happens.

**Appendix:** A pre-registration map (which rule, which addendum, verdict); B items, lures, pairs, prompts; C full tables per model; D retrieval (N2) and the ID repair (N1); E conv vs recurrent; F the 9B margins; G judge panel agreement; H example answers per condition.

## Figures

1. Schematic: session 1 → harvest → keep 2 positions (+ state) → session 2 reinstated in the user turn; the collision case with two slices both tagged "first".
2. Clean use, slice vs text vs in context, 3 models (bars with CIs).
3. 2B dissociation: 2×2 (attention / state × yes/no / two-choice).
4. Partner confusion across the collision conditions, 3 models, both pair sets.
5. Confusion vs N (2, 3, 5) for stored alone / stored as k-th / neutral sentences / joint, confusable and different subjects, 3 models; inset: confusion by position k at N = 5.

## Open decisions for the user

1. **Venue and length:** ICLR 2027 workshop (list out 29 Nov, ~1 Feb), 4-page main + appendix, with an arXiv version alongside? Or arXiv first, workshop cut later (as paper 1)?
2. **Experiment 2 (self-written references):** leave out (recommended: it belongs with paper 1's steer channel) or one paragraph?
3. **C5 placement:** a short subsection in §4 (recommended) or appendix only?
4. **The 9B margins:** appendix (recommended) or §4?
5. **Template:** reuse paper 1's ACL style for the arXiv version, ICLR workshop style for the cut.
