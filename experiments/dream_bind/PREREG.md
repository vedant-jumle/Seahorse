# dream_bind pre-registration (experiment 1: dreaming as binding)

*Written and committed before any dream_bind job ran (2026-10-11). It fixes what stage 0 measures, its
read-out thresholds and statistics, and the predictions B1–B6 and decision rules for the later stages,
copied from [docs/dreaming/01-binding.md](../../docs/dreaming/01-binding.md) (the spec, including the
sufficiency safeguard added the same day). `bind.verdict()` implements the stage-0 rules exactly as written
here. Any later change to a rule is reported as a deviation, with the pre-registered result beside it.*

## What is tested

A push at the answer position can't be bound: the model binds an attribute to an entity through
*positions* (in context: a shared tag on "dog" and "Pepper", and attention from the answer back to
Pepper's position; from the weights: the attribute loaded onto the subject's own position and extracted by
attention). Experiment 1 asks whether a fact can be *used* if the model's own subject-enriched state is
written back **at the subject token** in a later session, compared with reinstating the experience
sentence as a gated segment of positions.

**Stage 0 comes first** and asks the question H1 depends on: in a conversation, when the model answers
*"What's my dog's name?"* after *"My dog's name is Pepper."*, does the fact sit at the question's
**subject** ("dog"), or only at its **own position** ("Pepper"), with the subject holding just a pointer to
it? If only the latter, a state harvested at the subject carries a pointer to a position that won't exist
in session 2.

## Items

The 12 core_v1 facts (`data/bench_v1/facts.yaml` via `experiments/core_v1/items.yaml`), in two tiers:
- **Primary (7, a clean third-person subject):** dog_name, cat_name, sister_name, brother_name,
  grandmother_name, best_friend_name, boss_name. **Every pre-registered rule is decided on these 7.**
- **Secondary (5, the subject is the user):** job, hometown, first_language, university,
  favourite_colour. Reported separately, never in the verdict. favourite_colour is a **no-site** item for
  the later stages (no use prompt names it); in stage 0 its question does name it, so it is patched and
  reported, flagged no-site.

Subject aliases per item are in `config.yaml` (spec §3). Stage 0 looks for them in the **question** only.
Two secondary questions contain no alias ("What do I do for a living?", "Where did I grow up?"); those
items get the attribute site only.

**Counter values.** Every primary target and counter is a single token on both tokenizers, and source and
target prompts have equal length (checked; also asserted at run time). One secondary counter isn't:
" carpenter" (job) is 2 tokens, so, as the spec says, a single-token counter is used instead:
**"I work as a plumber."** (" plumber" is 1 token on both tokenizers and not one of the item's foils).

## Hardware and precision

- **Qwen2.5-1.5B-Instruct** (28 blocks, all full attention): run **locally on an RTX 4060 laptop GPU
  (8 GB), fp32**, eager attention, transformers 4.46.3, torch 2.10.
- **Qwen3.5-2B** (24 blocks; full attention at 3, 7, 11, 15, 19, 23, Gated DeltaNet elsewhere): run on
  **DelftBlue `gpu-a100-small` (one 10 GB A100 MIG slice), fp32**, transformers 5.18, torch 2.6. Text
  decoder and tied output layer only (no vision tower), `enable_thinking=False` everywhere, template tail
  asserted.
- **Same precision, different hardware.** Each run's `config.json` records the GPU, versions, commit and
  file hashes.

## Stage 0 design (fixed; see `run.py`, `bind.py`, `config.yaml`)

**Prompts** (one user turn, the assistant header, then the teacher-forced measure prefix; logits read at
the last prefix token):
- `ctx` source: *"My dog's name is Pepper. What's my dog's name?"* + *"Your dog's name is"*.
- `ctx` target: the same with the counter value (*Rocky*). Asserted: equal length, exactly one differing
  token (the attribute), which is the measure target's first token in the source.
- `nomem`: the question alone + the same prefix.
- **Question-subject span:** the alias tokens inside the question (after the attribute), never the
  mention inside the experience sentence. **Primary site = its last token** ("dog"; "friend" for best
  friend). The whole span is also reported.

**Measures:**
1. **Patching (both models).** The residual output of block ℓ at the site is copied from the source run
   into the target run: every single block, and every band of 5 consecutive blocks (sliding). Sites:
   subject last token (primary), subject span, attribute position. Measure m = `logit(target) −
   logit(counter)`. **Flipped share** = (m_patched − m_target) / (m_source − m_target).
2. **Attention knockout (Qwen2.5-1.5B only).** In the `ctx` source run, attention from the answer
   position (the last prefix token) to (a) the question-subject span, (b) the attribute position, is
   removed over windows of 5 consecutive layers (sliding), via the eager attention mask. **Knockout
   share** = (log P_ctx(target) − log P_knocked(target)) / (log P_ctx(target) − log P_nomem(target)).
   Not on Qwen3.5-2B: its Gated DeltaNet blocks fold every past token into one state, so there is no
   answer → Pepper edge to block (spec §2).
3. **Subject alone (Qwen2.5-1.5B only; the sufficiency safeguard).** In the target run, attention to the
   attribute position is removed from every query position after the experience sentence (from the
   question's first token on), at all layers; then the question-subject state is patched from the source
   run (same blocks and bands, last token and span). **Sanity cell:** attribute blocked, no patch. Both
   the raw flipped share and the **net** share (m_patched − m_blocked) / (m_source − m_target) are
   reported.

**Flagged items.** An item whose full flip m_source − m_target is < 1 logit is flagged for the patching
and subject-alone measures; an item whose `ctx` gain log P_ctx − log P_nomem is < 1 nat is flagged for
knockout. Flagged items are listed and excluded from the means. If **more than 2** of the 7 primary items
are flagged on a model, that model's stage-0 verdict is **undetermined**.

## Stage 0 read-out (thresholds; the statistic is the mean over the 7 primary items)

Each cell (block, band or window) gets the mean over items and a 95% bootstrap CI over items. **"Best"**
means the cell (block or band; for knockout, window) with the highest mean.

- **Fact lives at the subject:** patching the subject's last token flips **≥ 50%** at its best block or
  band, with the **CI excluding 25%** (lower bound > 0.25); and, on Qwen2.5, blocking the subject costs
  **≥ 50%** at its best window while blocking the attribute costs **< 25%** at every window. Then H1 has
  something to carry; proceed to the harvest.
- **Pointer at the subject (fact at its own position): flagged "pointer-suspected".** Patching the subject
  flips **< 25%** at **every** block and band, with every CI **excluding 50%** (upper bound < 0.5), while
  patching the attribute flips **≥ 50%** at its best cell; and, on Qwen2.5, blocking the attribute costs
  **≥ 50%** at its best window.
- **Both / in between:** anything else. Proceed, but expect H1 to work only partly and R to win; the flipped
  share per block is the result.
- **Sufficiency safeguard (Qwen2.5).** The subject-alone sanity share (attribute blocked, no patch) should
  be about 0, meaning |mean| ≤ 0.10. If it is, the subject-alone cell is read as its raw flipped share; if
  not, as the net share over that blocked baseline (and the sanity value is reported). **If the
  subject-alone share is ≥ 50% at its best block or band, with the CI excluding 25%, Qwen2.5 is not
  flagged pointer-suspected**, whatever the ordinary subject patch shows: it goes to "in between",
  reported as "the subject state is sufficient but outvoted by the intact attribute".
- **Stage 0 alone can only flag "pointer-suspected".** A pointer verdict skips the `subj` generation grid
  only if the later analytic pre-pass (spec §6: the subject write in session 2, teacher-forced Δ log
  P(target)) **also** shows no gain. Here, "no gain" means the best (variant, layer set, α) cell's mean
  Δ log P(target) over the 7 primary items has a 95% CI that includes 0 or lies below it.
- **If the two models disagree** (e.g. Qwen2.5 says "subject", Qwen3.5-2B patching says "pointer"): each
  model's result decides its own grid (the 2B result decides the 2B generation grid, the Qwen2.5 result the
  Qwen2.5 arms), and the disagreement is reported as a finding about the hybrid architecture, not averaged
  away.
- Secondary items are reported per block and per item, with no verdict.

## Later stages: predictions (from the spec, §2b)

- **B1.** Subject-position writing yields *clean use* of the fact above the push's ceiling (core_v1:
  ≤ 0.15) on the use prompts, with the CI above the push.
- **B2.** Balanced yes/no accuracy rises above 0.5 with the CI above 0, for the first time in the project.
- **B3.** The **wrong-position control** (same vector, added at a non-subject token) gives no gain. If it
  does, the effect is just another push and B1 is not about binding.
- **B4.** Cross-binding stays low: with two facts stored (dog Pepper, cat Jasper), a question about the cat
  does not produce Pepper more often than with no memory.
- **B5 (comparison).** Gated segment reinstatement on Qwen2.5-1.5B (all layers full attention) reaches at
  least the subject write's clean use; on Qwen3.5-2B (six full-attention blocks) it is weaker.
  Reinstatement is an **upper bound**, not a new kind of memory: K and V are fixed functions of the text,
  so a reinstated segment is close to `ctx` at a shifted position. Its value is to show how much of `ctx`
  survives the move and the gate; `subj` is the arm that would be new.
- **B6.** Unrelated prompts remain token-identical to no memory (≥ 0.95), since the injection site is the
  subject token and the subject does not appear in them.

## Later stages: decision rules (from the spec, §9)

- **Holds (binding works):** B1 and B2 and B3 and B6. The subject write's clean use CI lies above
  `push@1`'s *and* above `ctrl:wrongpos`'s; balanced accuracy gain CI > 0; `unrel_same ≥ 0.95`.
- **Push in disguise:** B1 holds but B3 fails (wrong position works as well). Then position isn't what
  mattered; report as a stronger push, not binding.
- **Orphaned:** nothing beats the push at any layer or α, while the analytic pre-pass shows Δ log
  P(target) rising. The representation is there but not readable out of context. Go to full
  reinstatement / the trained read path.
- **Reinstatement wins:** `reinst` beats `subj` on Qwen2.5, **both arms run on the same model**. Then the
  content channel is attention-based and the subject write is a shortcut that doesn't hold.
- **Stage 0 says pointer:** skip the `subj` generation grid; run only `reinst` (gated) and the controls,
  and treat the result as the upper bound for a training-free content channel, **only if** the analytic
  pre-pass also shows no gain (above). Otherwise the `subj` grid runs.

## Later stages: the configuration is chosen by rule, not by hand

- **Layer sets.** Qwen3.5-2B: single blocks {4, 6, 8, 10, 12, 14, 16} and bands {4–8}, {6–10}, {8–12},
  {10–14} (spec §6). Qwen2.5-1.5B: the same sets mapped by core_v1's relative-depth rule
  (`core.map_layers`, block l of 24 → round((l + 1) · 28 / 24) − 1). Deltas are harvested at the block they
  are written at.
- **Doses.** α ∈ {0.5, 1, 2}.
- **Analytic-pick rule.** For every (harvest variant H1/H2/H3, layer set, α) cell, the analytic pre-pass
  computes, as macro means over the 7 primary items: Δspec (the target's specificity gain at the measure
  prefix vs no memory), Δ log P(target), and the balanced yes/no margin gain. Eligible cells have Δspec > 0
  (if fewer than 3 per variant qualify: every cell of that variant). Score = rank(Δspec, higher better) +
  rank(balanced margin gain, higher better); ties go to the higher Δ log P(target). Per variant, the best
  three layer sets (each by its best-scoring α) are kept, each at its two best-scoring α. The generation
  grid then takes `subj` cells best-first by score across variants from that pool, **always including the
  best H1 cell**, up to **8 generation conditions** (spec §10), beside the fixed baselines (`nomem`,
  `ctx`, `push@1`), the `reinst` arm and the controls. The `+1` and `gen` variants and every control use
  the best H1 cell's layer set and α.
- The logit lens of the stored subject deltas is reported per block. It should *not* rank the target top
  at early layers; if it does, that is reported as "a late-layer 'say Pepper' was harvested again".

### Addendum: implementation details fixed before the harvest + analytic run (2026-10-11)

*Written after stage 0 and before any harvest or analytic-pass number existed. The spec leaves these open.*
- **H1:** subject = the first alias occurrence in the follow-up (after the experience); delta at its last token,
  every block; the item's H1 = the mean over its harvest follow-ups. A secondary follow-up without an alias
  ("Work has been really exhausting this week.") is skipped and reported; a primary one fails the job.
- **H2:** per item `config.yaml: h2`, the spec's pattern: "<experience> My dog is lovely." vs "I have a dog. My
  dog is lovely." (every item: "I have a <subject>." / "My <subject> is lovely."; university: "I went to a
  university."). Delta at the second mention's last token.
- **H3** ("added at reduced strength"): H2's raw `with` state, rescaled per block to the norm of H2's delta;
  then α as for every variant.
- **Sites:** main = the last token of every alias occurrence in the user turn (G-match). +1 = also the next
  token. gen = also the subject inside the teacher-forced assistant prefix ("Your dog"); on the yes/no probes
  (no prefix) gen = main.
- **Controls:** `wrongpos` = per main site, one user-text token counted back from the end of the user text,
  skipping subject and +1 tokens (usually the question mark or final full stop). `rand` = a fixed random
  direction per item and block, scaled to the variant's vector norm at that block. `swap` = core_v1's swap
  partner's vector of the same variant. `push@1` = core_v1's saved mem@1 numbers on the same prompts (2B only).
- **Measures:** Δlog P(target) and Δspec (log P(target) − the mean log P of the foils) vs nomem at the measure
  prefix; on the balanced verify probes, dmY, dmN and Δbal = (dmY + dmN)/2 of the consistent-signed
  first-token margin, plus the balanced-accuracy gain (margin > 0).
- **Pick pool:** the per-variant pools are re-scored jointly with the same formula before taking ≤ 8 cells.
- **G-key (reported):** key layers = the single write blocks. Whitening (mu + PCA-256, eps 0.01) and the
  q0.95 threshold come from the 100 generic prompts (prompt tokens only, no continuations). The stored keys are
  the subject-span keys of the H1 without runs. Reported: the gate opens at the question's subject, and the
  fire rate on the 50 unrelated prompts.
- **Harvest checks (fail the job):** subject found (primary); same token with and without; every delta
  finite; |delta| ≤ 10 × the median residual norm (generic prompt tokens) at every block (H1 per follow-up
  and mean, H2).
- **Logit-lens flag:** target rank ≤ 10 at any block of the write range (2B blocks 4–16; Qwen2.5 mapped).

### Addendum 2: arm R and the Qwen2.5 generation run (2026-10-12)

*Written after the analytic pre-pass and before any reinstatement or generation number existed (the only
numbers so far are stage 0, the exploratory prefix check and the analytic pre-pass).*

**Design note on R.** Plain R as the spec states it (the experience sentence's K and V spliced into a user-role
span before the question) is, under the causal mask, numerically the same as putting the sentence as text in a
prior user turn. Its K and V depend only on the system prompt and header before it, and those are the same in
every session. So R is run in three forms. The harvest and the splice are in `run.py` (`harvest_R`,
`gen_batch`); which entries are kept is `bind.splice_keep`.
- **R-full** (as specified): head (system block) + the whole experience user turn
  (`<|im_start|>user\n{experience}<|im_end|>\n`). K and V are taken at every layer from a run of [head][experience
  turn], kept at their stored RoPE positions. The question turn follows at the next positions.
  **Check (fails the job):** R-full's logits at the question equal the run of the sentence as a prior user turn
  (`[user: experience][user: question]`) within 0.01. It validates the splice code.
- **R-min** (new; the primary reinstatement for the memory question): head + only the attribute position and the
  next token (the full stop), K and V at every layer, at their stored positions. The question turn starts at the
  same position as for R-full, which leaves a gap. Stage 0 showed the attribute position carries the whole flip,
  so this tests whether 1–2 stored positions suffice (the compression claim).
- **R-orphan** (new): the experience turn's K and V are harvested AFTER one fixed unrelated exchange
  (`config.yaml: gen.orphan_exchange`, a user question about basil and an assistant answer, the same for every
  item). In the fresh session the head + the segment are spliced without the exchange. The segment keeps its
  harvest positions, and the question turn continues after them. Tests whether stored K and V survive a change
  of context.
- **Position scheme**, recorded per item in `gen/checks.json` (`R_info`): with the head n_h tokens, the experience
  turn n_s, and the exchange n_u, the question starts at n_h + n_s (R-full, R-min) or n_h + n_u + n_s (R-orphan).
  The cache holds n_h + n_s (R-full), n_h + 2 (R-min) or n_h + n_s (R-orphan) entries.
- **Gating:** G-match on the subject alias in the user turn (primary; G-key was reported by the pre-pass). When
  the gate is closed there is no segment, so the answer is the nomem answer by construction (not regenerated,
  marked `shortcut`). **Leak baseline:** one ungated R-full condition, on the 50 unrelated prompts only, for every
  primary item.
- **R-swap** (control): another item's R-full segment (core_v1's swap partner) under this item's gate.

**Generation (Qwen2.5-1.5B-Instruct, fp32, eager attention; DelftBlue gpu-a100-small).**
- Conditions: `nomem`, `ctx` (the experience in the same user turn, core_v1's ceiling), `R-full`, `R-min`,
  `R-orphan`, `R-swap`, `subj1`, `subj2`, and at subj1's cell `ctrl:wrongpos`, `ctrl:rand`, `ctrl:swap`,
  `ctrl:placebo`.
  - `subj1`/`subj2` are the top 2 of the pre-registered analytic pick on Qwen2.5: H1 @ blocks 5–10, α 0.5 and
    H2 @ 5–10, α 2.
  - **subj1 is the decisive subj cell** (the pick's first); subj2 is reported.
  - The subj write is added to the prompt's subject positions in the prefill only; generated tokens see it
    through the cache.
  - `ctrl:placebo` is the same-variant vector harvested with the experience replaced by "By the way, I had a coffee
    this morning."
- Items: the 7 primary items get every condition. The 5 secondary items get only nomem, ctx, R-full and subj1.
- Prompts per item: the direct question, the 2 use prompts (no-site prompts flagged), and the balanced verify
  probes (greedy, 10 tokens, first-token margins). Plus the 50 unrelated prompts (greedy + 2 samples).
- Cross-binding (dog/cat, sister/brother, best friend/boss) with both memories stored: under G-match only the
  memory whose subject is named fires (the gate of the other is checked closed), so the partner's prompts under
  each condition are the cross-binding probes. **Leak** = the wrong item's target in the answer, vs nomem.
- Settings as core_v1:
  - 1 greedy + 4 samples, T 1, top-p 0.95, top-k 20, presence 1.5, cap 120
  - the same random streams in every condition (per-prompt seeds)
  - seed set 2 (4 more samples) for nomem, ctx, R-full and subj1
- Count: ≈ 1,600 primary + 540 secondary main answers, ≈ 1,200 unrelated (150 nomem + 1,050 leak baseline),
  ≈ 420 short yes/no answers. No cut is planned; any cut made for the time budget is reported.

**Scoring.**
- Coherence NLL (the nomem model, plain prompt) in the generation job.
- The original judge: Qwen3.5-9B bf16, core_v1's `judge_prompts.yaml` and parser, through core_v1's score code.
- The panel judges gpt-oss and Gemma-4: core_v1's panel code; Llama is excluded, as in the panel.

**Measures.**
- **Primary 1: clean use.** The judges' majority (original + gpt-oss + Gemma-4; a tie, possible only when a label
  is missing, goes to the original's label) of `uses_fact` AND `coherent`, on the use prompts. The original
  judge alone is reported beside it.
- **Primary 2: balanced yes/no accuracy.** Parsed from the greedy answer (unparsed = wrong); (accY + accN)/2 per
  item; gain vs nomem.
- **Also reported:** regex clean (target present, not a loop), mention-unclean, confabulation (majority
  `wrong_value`), self-claim, loops, incoherence, NLL, `unrel_same`, cross-binding leak, and the direct question.

**Rules (B1–B6 and section 9 as pre-registered; on Qwen2.5 with reinst = R-full).**
- **B1:** subj1's clean-use CI lies above core_v1's push ceiling, **0.15** (Qwen2.5 has no push@1 run, so the
  pre-registered number stands in for it), and the paired CI of subj1 − ctrl:wrongpos lies above 0.
- **B2:** subj1's balanced-accuracy gain CI > 0, and its accuracy > 0.5.
- **B3:** the paired CI of ctrl:wrongpos − nomem clean use does not lie above 0.
- **B4:** the paired CI of the leak (subj1 − nomem) does not lie above 0.
- **B5 (the Qwen2.5 half):** R-full − subj1 clean use ≥ 0 and its CI is not entirely below 0. The 2B half runs
  later.
- **B6:** `unrel_same ≥ 0.95` for every gated R and subj condition.
- **Section 9:**
  - Holds = B1 ∧ B2 ∧ B3 ∧ B6.
  - Push in disguise = B1 ∧ ¬B3.
  - Orphaned = neither subj cell's CI above the push ceiling, while the analytic pre-pass said GAIN (it did on
    Qwen2.5).
  - Reinstatement wins = the CI of R-full − subj1 clean use > 0.

**R-min and R-orphan: reported, not decisive.** Clean use and balanced yes/no accuracy, each with CIs, vs nomem,
ctx, R-full and subj1. Expectations stated in advance:
- R-min close to R-full, if the attribute position really carries the fact.
- R-orphan below R-full but above nomem, if stored K and V partly survive the change of context.

**Statistics:** as above, two-level bootstrap over items then prompts, 2000 replicates, seed 7. Yes/no is per
item, bootstrapped over items.

### Addendum 3: arm R on Qwen3.5-2B, the hybrid (2026-10-12)

*Written before any reinstatement or generation number on Qwen3.5-2B existed (Qwen2.5's arm R results were
known: "orphaned; reinstatement wins").*

**The problem.** Qwen3.5-2B has full attention only at blocks 3, 7, 11, 15, 19, 23. The 18 Gated DeltaNet blocks
have no K and V to splice; they carry a recurrent state and a short convolution state (the last 3 inputs). The
spec's reinstatement (K and V spliced into the cache) therefore reaches only the six full-attention blocks.
Qwen3.5's chat template has **no system block**, so the head is empty: the experience turn starts at position 0.
The construction is in `run.py` (`harvest_R_hybrid`) and `bind.py` (`kv6_cache`, `expand_cache`).

**Conditions** (G-match gating as in addendum 2; a closed gate = nomem by construction).
- **`R-full-state`**: the full session cache after [head][experience turn]. That is K and V at the 6
  full-attention blocks AND the DeltaNet conv and recurrent states, as prompt caching does for hybrids.
  **Check (fails the job):** its logits at the question equal the run of the sentence as a prior user turn within
  0.01. It validates the code, as on Qwen2.5.
- **`R-full-kv6`** (the spec's arm): the experience turn's K and V at the 6 full-attention blocks, at their
  stored RoPE positions; the question starts at n_head + n_seg. **The DeltaNet blocks:** every DeltaNet layer's
  cache entry is replaced by its state after the head alone. With the empty head this is a fresh, uninitialised
  state, so these blocks process the question exactly as if the segment were absent. They have no positional
  encoding (only order), so the position gap that keeps the full-attention RoPE positions consistent is
  invisible to them. **Checked:**
  - the attention blocks hold n_head + n_seg entries
  - the DeltaNet layers equal the head-only cache
  - with every attention entry also dropped, the condition reproduces nomem (unit test on a tiny random hybrid)
- **`R-min-kv6`**: the attribute token and the full stop, K and V at the 6 full blocks (n_head + 2 entries);
  DeltaNet as R-full-kv6; the question at n_head + n_seg.
- **`R-orphan-kv6`**: the experience turn harvested after the same fixed unrelated exchange (basil); the exchange's
  entries are dropped. K and V at the 6 full blocks keep their harvest positions, and the question continues after
  them. DeltaNet as R-full-kv6.
- **`R-swap-kv6`**: the partner's R-full-kv6 under this item's gate.
- **Leak baseline:** an ungated R-full-kv6 on the 50 unrelated prompts, for every primary item.
- **No subj arm on the 2B.** This is the pre-registered skip: stage 0 said pointer-suspected and the analytic
  pre-pass said NO GAIN on this model. So B1–B4 and the section-9 rules that need `subj` do not apply here.

**Items, prompts, cross-binding, generation settings, judges and statistics:** as in addendum 2.
- The 7 primary items get every condition. The 5 secondary items get nomem, ctx, R-full-kv6 and R-full-state.
- Seed set 2 for nomem, ctx, R-full-kv6 and R-full-state.
- Clean use = the majority of the original judge + gpt-oss + Gemma-4.
- Two-level bootstrap, 2000 replicates, seed 7.
- Count: 1,071 primary + 540 secondary main answers, 1,200 unrelated (150 nomem + 1,050 leak), 276 yes/no.

**Predictions, reported (the spec makes B5 not decisive on the 2B):**
- **R-full-state ≈ ctx.** The paired difference in clean use is reported. The expectation is near 0 (it is the
  prior-turn layout rather than ctx's same-turn layout).
- **B5:** the kv6 arms are weaker than Qwen2.5's R. Reported as the paired difference 2B R-full-kv6 − Qwen2.5
  R-full in clean use, over the same items and prompts (from the Qwen2.5 run), expected < 0.
- Also reported: R-full-kv6 − R-full-state (how much the DeltaNet states carry), and R-min-kv6 − R-full-kv6 (the
  compression claim on the hybrid).
- **B6:** `unrel_same ≥ 0.95` for every gated R condition.

**Run order:** a short real-model smoke job first (`--tiny` with the real 2B: it must pass the R-full-state
check and a fit check of peak reserved GPU memory ≤ 9.0 GB on the 10 GB slice); the generation, judging and
analysis jobs follow with `afterok`.
*(Infrastructure note, 2026-10-12, before any 2B reinstatement number was read: the first smoke, job 957431, passed the
R-full-state check (max |Δlogit| 2.5e-5) but ran out of memory on the 10 GB slice in the coherence-NLL step, so the
smoke and generation jobs run on gpu-v100 (32 GB, fp32) instead. Nothing else changes.)*

### Addendum 4: what the DeltaNet state contributes (2B, exploratory follow-up to addendum 3; 2026-10-12)

*Written before any number of this follow-up existed. Addendum 3's results were known:
- R-full-state 0.96 ≈ ctx.
- R-full-kv6 0.50: it uses the name in slot-filling tasks but refuses direct questions, with yes/no 0.61.
- R-min-kv6 0.67.
- Ungated unrel_same 0.007.

**Everything here is reported, not decisive.** The working hypothesis is that the attention blocks fetch the value
and the DeltaNet state carries the sense that something was said.*

**Conditions** (Qwen3.5-2B fp32, text decoder only; `run.py --stage gen --gen-set state`; `config.yaml:
gen_conds_state`; construction in `run.py: harvest_R_hybrid` and `bind.py: mix_cache`).
- **`R-min-state`**: the attribute token and the full stop, K and V at the 6 full-attention blocks (as in
  R-min-kv6), PLUS the DeltaNet conv and recurrent states after the experience turn. **Honest label:** minimal
  attention entries + a fixed-size recurrent state, and that state summarises the whole sentence. So this is not
  a 2-token memory: it is 2 attention entries plus 18 layers' fixed-size states (≈ 16 × 128 × 128 numbers per
  layer for the recurrent state, + the conv state).
- **`R-state-only`**: the DeltaNet states after the experience turn and no attention entries for the segment (with
  the empty head, the full-attention blocks hold nothing before the question).
  - **Position scheme:** the question starts at n_head + n_seg, the position it has after the segment, as in
    R-full-state and R-min-state.
  - The full-attention blocks then see only the question's own entries. Under RoPE only relative positions matter,
    so the question attends to itself exactly as it would at any offset.
  - The DeltaNet blocks have no positions, so the offset is consistent with the n_seg tokens their state consumed.
  - It differs from R-full-state only by the missing attention entries, and from R-min-state by the missing 2
    entries.
- **`R-orphan-state`**: R-min-style entries (the attribute and the full stop) harvested after the fixed unrelated
  exchange (basil) + the experience turn, and the DeltaNet state after exchange + experience. The state is
  contaminated by the exchange; that is the test. The exchange's attention entries are dropped, and the question
  continues after the harvest positions.
- **`R-swap-min-state`**: the partner item's R-min-state under this item's gate.
- **Anchors, rerun in the same run with the same random streams:** `nomem`, `ctx`, `R-full-state`, `R-min-kv6`.
- **Leak baseline:** an ungated R-min-state on the 50 unrelated prompts, for every primary item.
- **Checks (fail the job):**
  - R-full-state ≡ the sentence as a prior user turn (≤ 0.01)
  - attention-entry counts per variant: R-state-only = head only; R-min-state and R-orphan-state = head + 2;
    the anchors as before
  - R-min-state's and R-state-only's DeltaNet layers equal R-full-state's exactly
  - R-orphan-state's DeltaNet layers equal the orphan harvest's, and differ from R-full-state's

**Items, prompts, gating, cross-binding, judges, statistics:** as in addendum 3.
- The 7 primary items get every condition. The secondary items get nomem, ctx, R-full-state and R-min-state.
- Seed set 2 for nomem, ctx, R-full-state and R-min-state.
- Clean use = the majority of the original judge + gpt-oss + Gemma-4.
- Two-level bootstrap, 2000 replicates, seed 7.
- Count: 1,176 primary + 540 secondary main answers, 1,200 unrelated (150 nomem + 1,050 leak), 304 yes/no.

**Expectations (stated in advance; reported):**
- **"Compact hybrid memory":** the CI of R-min-state − R-full-state (clean use) includes 0 AND its point estimate
  is > −0.1.
- **R-min-state > R-min-kv6** on the direct question (clean use) and on balanced yes/no accuracy.
- **R-state-only:** open. Reported against nomem and R-min-kv6: does the recurrent state alone carry the fact, or
  only the "something was said" signal?
- **R-orphan-state − R-min-state:** reported.
- **Also reported:** every condition's answer to the direct question (the "knows it knows" readout: greedy answer
  per item, plus clean use with paired contrasts), and B6 for the gated conditions.

### Addendum 5: the plain-text control, and the two-fact collision (binding-ID) test (2026-10-13)

*Written before any number of these runs existed. Results known so far:
- Qwen2.5: R-min 0.86, R-orphan 0.80.
- 2B: R-min-state ≈ R-full-state ≈ ctx.
- 2B: R-state-only use 0 but yes/no 0.93 (recognition without recall).
- 2B: R-orphan-state 0.99.

Everything in this addendum is exploratory and reported. The rules below say how it will be read.*

**(1) Text controls: does R-min carry more than its tokens as text?** `run.py --stage gen --gen-set text`
(`config.yaml: gen_conds_text`). Same items, prompts, G-match gates, measures and judges as addenda 2–4.
- **`T-min`:** the attribute and the full stop as plain text in a prior user turn, in the same place R-min's
  entries come from: the conversation [user: "Pepper."][user: question]. The value is written without its leading
  space, as a user would type it.
- **`T-full`:** the experience sentence as text in a prior user turn ([user: experience][user: question]). By the
  earlier checks this equals R-full (Qwen2.5) and R-full-state (2B); it is the anchor.
- **Conditions:**
  - Qwen2.5: nomem, ctx, T-min, T-full, R-min, R-full.
  - 2B: nomem, ctx, T-min, T-full, R-min-state, R-full-state.
  - On the 2B, the text also feeds the DeltaNet blocks (the text is ordinary context to every block), whereas
    R-min-state's DeltaNet state comes from the whole sentence.
- **Items:** the 7 primary items fully. The secondary items in nomem, ctx, T-min and R-min / R-min-state. Seed
  set 2 for those four. The unrelated prompts are gated (no ungated baseline).
- **Prediction:** T-min ≈ nomem on use and on the direct question, and well below R-min / R-min-state.
  **If T-min ≈ R-min, the compression claim falls.**

**(2) Two-fact collision: separately harvested segments restored together.** `run.py --stage collide`; output in
the sibling model dir `<model>_collide`.
- **Theory:** in context, entity–attribute pairs carry an order-based binding ID (Feng & Steinhardt; Dai).
  Segments harvested in separate sessions all carry the "first pair" ID, so restoring two at once may make them
  collide. Re-encoding them jointly ("dreaming") would give them distinct IDs.
- **Pairs** (`config.yaml: collide`). All values are single tokens on both tokenizers (checked at run time).
  - (a) **Different subjects** (existing items): dog/cat, sister/brother, best friend/boss.
  - (b) **Confusable** (new; the same relation and near-identical subjects):

| Pair | Member A | Member B |
|---|---|---|
| sisters | "My older sister's name is Petra." | "My younger sister's name is Maria." |
| dogs | "My first dog was called Pepper." | "My second dog is called Rocky." |
| friends | "My friend from work is called Omar." | "My friend from school is called Kevin." |
| teachers | "My son's teacher is called Hugo." | "My daughter's teacher is called Daniel." |
| grandmothers | "My grandmother on my mother's side is called Esther." | "My grandmother on my father's side is called Rosa." |

- **Prompts per member:** a direct question, two use prompts, and balanced yes/no from two verify templates (Yes
  with the member's own value; **No with the partner's value as the foil**), all in `config.yaml`. Plus a
  teacher-forced readout at the measure prefix: log P(own) and logit(own) − logit(partner).
- **Gates:** both forced open (oracle retrieval). This tests binding, not retrieval. Pair order is fixed (A then
  B) in every condition.
- **Conditions (Qwen2.5)** (head h, A's turn n_A tokens, B's n_B; attribute index a_A, a_B):

| Condition | Cache entries | Question starts at |
|---|---|---|
| `nomem` | — | — |
| `ctx2` | both sentences in the question's user turn ("A. B. question") | — |
| `R2-joint` | [head][A's turn][B's turn] harvested in ONE session, every entry (the re-encoding) | h + n_A + n_B |
| `R2-sep-overlap` | A and B each harvested in its own session [head][turn], both kept at their stored positions, so they share positions | h + max(n_A, n_B) |
| `R2-sep-shift` | as overlap, but B's keys re-rotated by RoPE (+n_A) to follow A | h + n_A + n_B |
| `R2-min-joint` | the attribute + full stop of each, from the joint harvest | h + n_A + n_B |
| `R2-min-sep-overlap` | the attribute + full stop of each, from the separate harvests, at stored positions | h + max(n_A, n_B) |
| `R2-min-sep-shift` | as min-sep-overlap, B shifted by +n_A | h + n_A + n_B |

  - `T2` (both as separate prior user turns) equals R2-joint by construction. A check (fails the job) verifies
    that R2-joint's logits equal T2's text run, and one generated set serves both.
  - **Check (fails the job):** re-rotating stored keys by RoPE equals recomputing them at the new positions (unit
    test on tiny models; on the real model the shift is applied with the model's own rotary embedding).
- **Conditions (2B):** the 6 attention blocks have the same overlap/shift issue; the 18 DeltaNet blocks do not,
  but two separately harvested DeltaNet states cannot be merged.
  - **Chosen scheme:** every R2 condition on the 2B uses the JOINT DeltaNet state (after [A's turn][B's turn]),
    so sep vs joint differ only in the attention entries. This isolates the binding question to the six
    attention blocks.
  - Conditions: nomem, ctx2, `R2-joint-state` (= T2), `R2-sep-overlap-jstate`, `R2-sep-shift-jstate`,
    `R2-min-joint-jstate`, `R2-min-sep-overlap-jstate`, `R2-min-sep-shift-jstate`.
  - Two attention-only probes with the head-only (fresh) DeltaNet state: `R2-joint-kv6`, `R2-sep-shift-kv6`.
- **Answers:** seed set 1 only (1 greedy + 4 samples per prompt). Yes/no greedy with first-token margins.
- **Measures, per member question:**
  - **correct-target clean use:** the judges' majority (original + gpt-oss + Gemma-4), judged against the member's
    own fact
  - **partner confusion:** the partner's value is in the answer and the member's own value is not (regex); the
    judges' `wrong_value` is reported beside it
  - **balanced yes/no accuracy with the partner as the foil**
  - the teacher-forced margin
- **Statistics:** the bootstrap is over members, then prompts.

**Decision rules (exploratory, reported):**
- **"IDs collide":** partner confusion R2-sep − R2-joint has a CI > 0 on the confusable pairs. Decided
  separately for the overlap and the shift scheme (2B: the `-jstate` versions).
- **"Content suffices":** R2-sep ≈ ctx2 on the different-subject pairs. That is, the CI of R2-sep − ctx2 in
  correct clean use includes 0 AND |point estimate| < 0.1. Again per scheme.
- **"Re-encoding helps":** R2-joint − R2-sep in correct clean use has a CI > 0. Per scheme; on the confusable
  pairs (the different-subject pairs reported beside them).
- The min versions (R2-min-*) and the kv6 probes are reported with the same contrasts.

### Addendum 6: binding-ID repair, and G-key retrieval with capacity (Qwen2.5-1.5B; 2026-10-13)

*Written before any number of these runs existed. Known from addendum 5:
- Separately harvested segments swap partners (0.26–0.37), even for different subjects.
- Shift ≈ overlap.
- The joint harvest fixes it.

The binding-ID (order-tag) explanation is inferred, not measured. Everything here is exploratory and reported;
the rules say how it will be read.*

**(A) Binding-ID repair: measure the cause, test a cheap fix.** `run.py --stage idrepair` (sibling dir
`<model>_idrepair`).
- **Estimating the ID directions** (Feng & Steinhardt's mean-difference method, per layer):
  - **Contexts:** 500 jointly encoded two-pair contexts: [head][user: "My {slot1}'s name is {v1}."][user: "My
    {slot2}'s name is {v2}."]. The slots (person/pet slots of `fact_pool.yaml`, two distinct per context) and
    the values are drawn at random, seed 11. The two-turn format is the one the joint harvest uses.
  - **Positions:**
    - Entity = the last token of the subject phrase + the next token (Feng: the entity token and the one after
      it).
    - Attribute = the value token + the full stop (as in R-min).
    - The subject phrase is located by rule: the word before the first "'s name is" / " is" / " was" in the
      sentence.
  - **Offsets:** Δ_E(l) = mean over contexts of [block-l input residual at pair-2 entity positions] − mean over
    contexts of [same at pair-1 entity positions], for every block l. The same for Δ_A at attribute positions.
    Content averages out over the random slots and values.
  - **Dai's version:** per layer, PCA of the entity (or attribute) residuals of both orders. The principal
    component most correlated with the order label (of the top 4) is the order-ID direction. Δ_dai(l) = the
    projection of Δ(l) onto it.
- **Validation (replication), on 200 held-out template contexts** (seed 12; both entities queried, 400 queries):
  - **Feng's mean intervention:** pair 1 + Δ, pair 2 − Δ, at the attribute positions ("swap-attr"), the entity
    positions ("swap-ent"), or both. Implemented by editing the stored block-input residuals at those positions
    and recomputing only those positions' K and V at each layer (frozen W_k, W_v, input norm and RoPE at their
    positions; every other position frozen, as in Feng).
  - **Read-out:** the teacher-forced margin logit(own) − logit(partner) at "Your {slot}'s name is".
    **Flip** = the margin changes sign.
  - **Control:** random directions with Δ's norm at every layer.
  - **Layer sets:** all 28 blocks (Feng), and the thirds 0–9, 10–18, 19–27.
  - **Pre-registered pick for the repair:** the set with the largest swap-attr flip rate on these held-out
    contexts (ties → all blocks).
  - Also reported on the addendum-5 pairs' joint contexts.
- **Repair (our claim).** In R2-sep, add the pair-2 offset (+Δ_E at the entity positions, +Δ_A at the attribute
  positions) to the SECOND segment (B) only, at the picked layers. Then recompute those positions' K and V from
  the edited residuals and splice as in addendum 5 (overlap and shift; full segments and R-min slices; in the
  slices only the attribute entries exist, so only Δ_A acts).
  - **Controls:**
    - a random direction with the same norm per layer
    - the wrong sign (−Δ)
    - the offset on the wrong tokens: B's first two tokens of the sentence ("My" and the next), which are neither
      the entity nor the attribute
  - Also: the Dai-projected offset.
- **Analytic first (teacher-forced), on all 16 addendum-5 members:**
  - log P(own), the margin own − partner, and **TF confusion** (P(partner) > P(own))
  - the yes/no first-token margins with the partner as the foil
  - conditions: R2-joint, R2-min-joint, and for each scheme × size: R2-sep, +ID (Feng), +ID (Dai), +rand,
    +neg, +wrongtok
- **Best repair cell:** among scheme × size × direction, the largest mean reduction in TF confusion vs R2-sep
  (ties → the larger margin gain).
- **Generation (+ judges)** only for the best cell: R2-joint, R2-sep, R2-sep+ID, +rand, +neg, +wrongtok. Each is
  16 members × (direct question + 2 use prompts) × (1 greedy + 4 samples), plus the balanced yes/no (partner as
  foil). That is 1,440 answers + 384 yes/no.
- **Rules (exploratory, reported):**
  - **"ID direction valid":** the swap-attr flip rate minus the random control's has a CI > 0 (bootstrap over the
    held-out contexts).
  - **"Repair works":** partner confusion (as in addendum 5; generated, judged) of R2-sep+ID − R2-sep has a
    CI < 0 (excludes 0), AND +ID's confusion is below each control's (paired CI < 0). The analytic TF version is
    reported beside it.

**(B) G-key retrieval and capacity.** `run.py --stage gkey` (sibling dir `<model>_gkey`).
- **Facts:** the 12 hand-written facts + 47 pool facts. These are all `fact_pool.yaml` slots not used by the
  hand-written facts (distinct slots; one of wife / husband), so 59 facts at most.
  - **N ∈ {10, 20, 59}:** stores are nested prefixes of the order [7 primary, 5 secondary, 47 pool facts in
    `sample_pool` order, seed 0]. N = 60 is not reachable with distinct slots (59), so the largest store is 59.
  - **Pool facts have no use prompts:** they get the direct question + measure prefix only.
  - Four pool values are 2 tokens (Monopoly, Monet, Matilda, clarinet). Their R-min slice keeps the first
    sub-token + the next token, and the measure uses the first token.
- **Store:** each fact's R-min slice (as in addendum 2) + its keys.
  - **K1 (primary, the G-key construction):** the whitened (pooled_w256) mean of the subject-span residuals in
    the without runs of the fact's 2 on-topic follow-ups. The subject aliases are the hand-written items' own;
    for pool facts "my {slot}", "your {slot}", "{slot}". When a follow-up has no alias, its whole-user-text key is
    used, and the count is reported.
  - **K2 (reported):** the pooled key over the follow-up's whole user text.
  - A fact matches with the max cosine over its keys.
- **Retrieval:**
  - **Query key:** the pooled running key at the question's last user-text token (no string matching).
  - **Rule:** top-1 above the threshold → inject that fact's slice; below it → nothing (nomem).
  - **Threshold:** q0.95 of the store's best match over the 100 generic prompts' last-user-token keys.
  - **Key layer:** 14 (primary); 5, 10 and 19 reported.
  - Top-2 accuracy is also reported.
- **Measures (all N):**
  - retrieval accuracy (the correct slice), wrong-slice rate, miss rate, top-2 accuracy
  - false-fire rate on the 50 unrelated prompts
  - teacher-forced Δlog P(target) vs nomem, for: retrieved, oracle (the own slice), ctx, and R-swap (the swap
    partner's slice)
  - **wrong-value rate** (a wrong slice injected and P(that slice's value) > P(target)), compared with R-swap's
- **Generation (+ judges) at N = 59, on a subset:**
  - Facts: the 7 primary facts (question + 2 use prompts) + the first 20 pool facts of the store order (direct
    question).
  - Conditions: nomem, ctx, oracle (G-match), G-key (retrieved).
  - Count: 41 prompts × 4 conditions × 5 answers = 820.
  - Clean use and confabulation (the judges' majority `wrong_value`).
- **Prediction (reported):** retrieval accuracy falls with N; wrong-slice retrievals give confident wrong answers
  at about the R-swap rate. **Rules for (B): the curves are reported; no pass/fail.**

**Judges and statistics:** as in addendum 5 (the original judge + gpt-oss + Gemma-4 majority; bootstrap over
members/facts then prompts, 2000 replicates, seed 7).

### Addendum 6b (2026-10-13, after addendum 6 was committed and before any (A), (B) or (C) number existed)

**Correction to (A), made before any number.** The wrong-token control uses the sentence's first token and **the
token before the attribute** (e.g. "My" and " is" in "My dog's name is Pepper."), not "the first two tokens". In
"My dog's name is Pepper." the second token is the entity itself, so the first-two-tokens rule would put the
control on the entity.

**(C) Qwen3.5-2B: what in the restored DeltaNet state carries recognition?** A literature check flagged a
confound for addendum 4's "R-state-only: use 0, yes/no 0.93". Gated DeltaNet's short-convolution buffer holds the
last few inputs of the stored turn almost verbatim ("Pepper", ".", end of turn). So the recognition could come from
that buffer, not from the recurrent matrix.

**What the existing conditions restored:** R-state-only and R-min-state (addendum 4) copied each DeltaNet layer's
cache object whole from the R-full-state harvest, **so both restored the conv buffer AND the recurrent state.**

`run.py --stage gen --gen-set decomp` (`config.yaml: gen_conds_decomp`).

**New conditions** (G-match gated as before; "fresh" = the head-only value, which with Qwen3.5's empty head is a
zero conv buffer / zero recurrent state, exactly the state with no previous tokens):

| Condition | Attention entries | DeltaNet recurrent state | DeltaNet conv buffer |
|---|---|---|---|
| `R-state-only-rec` | none | after the experience turn | fresh |
| `R-state-only-conv` | none | fresh | after the experience turn |
| `R-state-only` (anchor; both, as before) | none | after the experience turn | after the experience turn |
| `R-min-state-rec` | R-min (attribute + full stop) | after the experience turn | fresh |

  The last row asks: does compact memory survive without the buffer?

**Analytic first (teacher-forced), on the 7 primary items:**
- conditions: nomem, ctx, R-full-state, R-min-state, R-min-kv6, and the four above
- the balanced verify probes' first-token margins and accuracy (margin > 0) — **primary for recognition**
- Δlog P(target) at the direct question's measure prefix

**Generation (+ judges)** for the four conditions + the nomem and ctx anchors:
- 7 primary items × (direct question + 2 use prompts) × (1 greedy + 4 samples), seed set 2 for nomem, ctx,
  R-state-only and R-state-only-rec
- the balanced yes/no probes (greedy)
- the 50 unrelated prompts (gated)
- one 2B gen job on gpu-a100-small

**Rule (reported):**
- **"Recognition lives in the recurrent state"** if R-state-only-rec's teacher-forced balanced yes/no accuracy
  has a CI lower bound > 0.5, AND its point estimate is within 0.1 of R-state-only's.
- If only R-state-only-conv carries it, recognition is reported as **a short verbatim-buffer effect, not
  familiarity**.
- The generated (greedy, parsed) yes/no accuracy is reported beside it.

### Addendum 7: strengthening runs for paper 2 (parts A–G; 2026-10-22)

*Written and committed before any job of this addendum was submitted and before any of its numbers existed. Known
results: addenda 1–6b (summarised in `docs/dreaming/01-binding.md`, Results), in particular:
- 2B R-state-only yes/no 0.93 with unrelated foils; recurrent-only 0.86.
- 2B attention-only slices (R-min-kv6) +10.0 nats teacher-forced, yet they refuse direct questions (0.20).
- Separately stored slices collide (0.19–0.37), and the ID-direction repair fails.
- The G-key "pool gap" (0.62).
Literature known: Afendulev et al. (Sept 2026) show recurrent-state recognition on Qwen3.5 within one session, as
gist familiarity that also accepts associated lures.*

*Everything here is exploratory in the sense of addenda 5–6: each part has a stated rule, and the report gives the
rule's verdict beside the full tables. Code: `run.py` (stages `probe`, `directions`; gen sets `refusal`, `pool`;
`--items pool` for stage 0; `--collide-set add7`), `bind.py` (pair plans), `config.yaml` (`lures`, the new model
blocks and condition sets), `data/bench_v1/entities.yaml`, the `use:` templates in `data/bench_v1/fact_pool.yaml`;
jobs in `experiments/dream_bind/submit_add7.sh`.*

**Shared.**
- **Judges:** clean use = the majority of the original judge (Qwen3.5-9B bf16) + gpt-oss + Gemma-4 (a tie goes to
  the original's label), as before.
- **Statistics:** bootstrap over items, then prompts; 2000 replicates, seed 7. Paired differences on the same
  prompts and random streams. Teacher-forced (TF) tables bootstrap over items.
- **Unrelated prompts:** the 50 prompts, G-match gated; `unrel_same ≥ 0.95` reported for every gated condition.
- **Seed set 2:** 4 more samples for the main cells named per part.
- **Read the text:** 3–5 answers per new condition are read before anything is written up.
- **Precision:** fp32 everywhere.
- **"TF yes/no":** a verify probe is answered "yes" teacher-forced when log P(Yes) > log P(No) at the first
  assistant token, and greedy when the parsed 10-token answer is "yes" (unparsed = not yes).

#### A. Associated-lure control: does the state recognise the fact, or a gist?

**Lures** (`config.yaml: lures`), per primary item, 2 per kind, each asked with the item's 2 verify templates
({v} = the lure; the correct answer is No):

| Item (target) | form (spelled / sounding like) | associate (the word's own theme, never said) | binding (other items' values) |
|---|---|---|---|
| dog (Pepper) | Piper, Penny | Ginger, Salt | Jasper, Petra |
| cat (Jasper) | Jason, Jasmine | Amber, Jade | Pepper, Petra |
| sister (Petra) | Peter, Pedro | Athena, Stone | Hugo, Esther |
| brother (Hugo) | Hugh, Hogan | Victor, Pierre | Petra, Esther |
| grandmother (Esther) | Easter, Lester | Rebecca, Leah | Omar, Gordon |
| best friend (Omar) | Amar, Homer | Ali, Hassan | Gordon, Pepper |
| boss (Gordon) | Jordan, Morgan | Douglas, Duncan | Omar, Pepper |

- **Binding lures:** the addendum-5 pair partner's value first (dog/cat, sister/brother, best friend/boss;
  grandmother has none), then the next items in the fixed cycle dog > cat > sister > brother > grandmother > best
  friend > boss > dog, until there are 2 distinct values.
  - In a single-fact memory the binding lure's value is **not** in the memory. There it is expected to behave
    like an unrelated foil.
  - It becomes the conjunction test only in the joint-state condition below.
- **Checks:**
  - Every lure (with its leading space) is one token on the Qwen2.5, Qwen3.5 and Llama-3.2 tokenizers (checked
    locally with the cached tokenizers, and asserted at run time).
  - It is not the target's first token, and not one of the item's foils (asserted at run time).
- **Baseline:** the existing unrelated foils (the bench's 2 No probes per item).
- **Probes per item and condition:**

  | Probes | Count |
  |---|---|
  | Yes (target) | 2 |
  | unrelated No | 2 |
  | form No | 4 |
  | associate No | 4 |
  | binding No | 4 |
  | **total** | **16** |

**Conditions** (run by the new `probe` stage: teacher-forced first-token margins on all 16 probes, plus a greedy
10-token answer to each, parsed yes/no):
- 2B: nomem, ctx, R-full-state, R-min-state, R-min-kv6, R-state-only, R-state-only-rec, R-state-only-conv.
- Qwen2.5 and Llama: nomem, ctx, R-full, R-min, T-min, T-full.
- 9B: as the 2B (part F).
- **Deviation from the plan (stated now):** the plan's separate greedy `lure` gen set is folded into the `probe`
  stage. The answers are 10-token yes/no strings, which no judge reads, so a judged gen set adds nothing.

**Measures, per lure kind k ∈ {unrelated, form, associate, binding}:**
- **false-yes rate FY_k:** the share of k's No probes answered yes.
- **balanced accuracy against k:** (accuracy on the Yes probes + (1 − FY_k)) / 2.
- **"against lures":** form + associate pooled.
- Both are computed TF (primary) and greedy (reported).

**Joint-state binding lure (2B, collide):** the new collide condition `R2-state-only-J`. It is the JOINT DeltaNet
state after [A's turn][B's turn], with no attention entries for either segment; the question starts at
h + n_A + n_B. Its collide yes/no probes are the conjunction test, e.g. "Is my older sister called Maria?" (No =
the partner's value, which IS in the state).

**Rules** (decided on R-state-only, TF, 2B; the same rules reported for R-state-only-rec and for the 9B):
- **"Gist familiarity"** if the paired CI of FY_form − FY_unrelated, or of FY_assoc − FY_unrelated, has a lower
  bound > 0.
- **"Item recognition"** if gist familiarity does not hold, AND both point differences are within 0.1 of 0, AND the
  balanced accuracy against lures has a CI lower bound > 0.5.
- Otherwise **"in between"**.
- **Conjunction (R2-state-only-J, confusable pairs):**
  - **"the state binds"** if the TF balanced yes/no accuracy (partner as foil) has a CI lower bound > 0.5.
  - **"familiarity without conjunction"** if the CI includes 0.5.
  - The different-subject pairs are reported beside it.

#### B. Why the attention-only slice refuses direct questions

**B1. Directions** (`--stage directions`, sibling dir `<model>_dirs`; difference of means per block, as in
`estimate_ids`). Both are built on data disjoint from the 7 test items.
- **known-entity** (Ferrando et al.):
  - Data (`data/bench_v1/entities.yaml`): 100 famous people vs 100 made-up names. No word of any entry is a value,
    counter, foil or lure used anywhere in dream_bind (checked in the tests).
  - Prompt: the user turn "Tell me about {X}.".
  - Read-out: the block-output residual at X's last token.
  - d_known(l) = mean(famous) − mean(made-up).
- **"told":**
  - Data: the 52 non-primary facts (the 47 pool facts of `gkey_facts`' order + the 5 secondary items).
  - Read-out: the block-output residual at the **last prompt token** of the direct question (the last token of the
    assistant header).
  - d_told(l) = mean over facts of [ctx − nomem].
- **Reported:** per block, the separation of the two sets along each direction: Cohen's d of the projections (famous
  vs made-up; ctx vs nomem).

**B2. Projections.**
- Units: the 7 primary items × (direct question + 2 use prompts).
- Sites: the question's subject site (the last token of the first alias occurrence; G-match's site) and the last
  prompt token, every block.
- Value: the scalar projection on the unit directions.
- Conditions:
  - 2B: nomem, ctx, R-min-kv6, R-min-state, R-state-only.
  - Qwen2.5: nomem, ctx, R-min, R-full.
- Reported: per block, each condition − nomem with item-bootstrap CIs.
- **Expectation (reported, not decisive):** "missing told signal" predicts that R-min-kv6's told projection at the
  last prompt token stays near nomem's in the second half of the network, while R-min-state's moves toward ctx's.
  "Missing known signal" predicts the same for the known-entity projection at the subject.

**B3. Causal arms** (`--gen-set refusal`; generation, judged).

The **told vector:**
- v(l) = α · d_told(l) for l in a layer set S.
- It is added to block l's output at the last prompt token, in the prefill only; generated tokens see it through the
  cache.

**The α-selection rule** (teacher-forced, on the 52 non-primary facts only, **never on the 7 test items**; run inside
the directions stage and saved to `dirs.pt` before any refusal answer exists):
- **Grid:**
  - S ∈ {early, middle, late third} by relative depth: blocks [0, n/3), [n/3, 2n/3), [2n/3, n).
  - α ∈ {0.25, 0.5, 1, 2}.
- **The slice X** per fact: its own attention-only slice (Qwen2.5 and Llama: R-min; 2B: R-min-kv6), harvested as in
  addenda 2–3.
- **Score** = the mean over facts of the change, vs X alone, in log P(the measure prefix, e.g. "Your dog's name is" |
  the direct question): the sum over the prefix tokens, teacher-forced. A higher score means "answers instead of
  refusing".
- **Constraint:** the mean change in log P(target | question + prefix) vs X alone must be ≥ −1 nat.
- **Pick:** the highest score among the cells meeting the constraint (ties → the smaller α). If no cell meets it:
  the smallest α at the best-scoring S.

**Conditions** (G-match gated):

| Condition | What it is | 2B | Qwen2.5 |
|---|---|---|---|
| nomem, ctx | anchors | ✓ | ✓ |
| X | the slice alone (2B R-min-kv6, Qwen2.5 R-min) | ✓ | ✓ |
| `X+told` | the slice + the picked told vector | ✓ | ✓ |
| `X+told:rand` | a random unit direction per block (fixed seed) × the norm of v(l) | ✓ | ✓ |
| `X+told:neg` | −v | ✓ | ✓ |
| `nomem+told` | v on the plain prompt; expected: confabulation, not recall | ✓ | ✓ |
| R-min-state | the reference for the state rows (2B only) | ✓ | |
| R-full | the reference (Qwen2.5 only) | | ✓ |
| `R-min-kv-own+state-swap` | own R-min-kv6 entries + the swap partner's DeltaNet state (recurrent + conv, after its experience turn); the question starts at own n_head + n_seg | ✓ | |
| `R-min-kv-own+state-plac` | own entries + the DeltaNet state after the placebo turn "By the way, I had a coffee this morning." | ✓ | |

- **Prompts:** the 7 primary items × (direct question + 2 use prompts) × (1 greedy + 4 samples). Seed set 2 for:
  nomem, X, X+told, and R-min-state (2B) / R-full (Qwen2.5).
- **Also:** the balanced yes/no probes (greedy), and the 50 unrelated prompts (gated).
- **Count:** 2B 1,050 + 336 seed-2 answers; Qwen2.5 840 + 336.

**Measures:**
- Clean use on the direct question, and on the use prompts.
- The judges' `wrong_value` (confabulation).
- Reported: a refusal regex ("don't / do not / can't / cannot know / recall / remember / have", "I'm sorry",
  "could you tell me").

**Rules** (2B decides; Qwen2.5 is reported with the same rule for the told part):
- **"Missing told signal"** if all of these hold, with X = R-min-kv6:
  - the paired CI of direct-question clean use (X+told − X) has a lower bound > 0;
  - the paired CIs of (X+told − X+told:rand) and (X+told − X+told:neg) both have a lower bound > 0;
  - use-prompt clean use is not lowered: (X+told − X) has a point estimate ≥ −0.1 and an upper bound ≥ 0.
- **"Generic response mode"** (Afendulev et al.'s reading) if the direct-question clean use of
  R-min-kv-own+state-swap OR +state-plac is within 0.1 of R-min-state's (point estimates; ≥ R-min-state − 0.1).
- **"Item-specific state"** if both are more than 0.1 below R-min-state AND both paired CIs (each − R-min-state)
  have an upper bound < 0. Otherwise **"in between"**.

#### C. Pool scoping: is the "pool gap" a prompt-type mismatch? (Qwen2.5, then Llama)

**Pool-gap note** (restating the 2026-10-21 correction):
- The 0.62 came from the G-key generation subset: **20** pool facts with **direct questions only**.
- The fair comparison is the primary items' direct question, where R-min is also weak: 0.66 [0.40, 0.89] in the main
  run.
- Teacher-forced, the right slice equals ctx on all 59 facts.
- This part tests the mismatch reading directly.

**Facts:** the 52 non-primary facts (47 pool, `sample_pool` seed 0, the hand-written slots excluded, as in
`gkey_facts`; + the 5 secondary items).
- Pool aliases: "my {slot}", "your {slot}", "{slot}". The secondary items keep their own aliases.

**Prompts** (`--gen-set pool`):
- The fact's direct question.
- **One templated use prompt per kind**, written into `fact_pool.yaml` (`use:`) before this run:
  - person: "Write a short birthday message to my {slot}."
  - pet: "Write a short, funny poem about my {slot}."
  - every attribute slot: a hand-written prompt naming the slot (e.g. "Suggest a weekly practice routine for my
    instrument."). The secondary items use their slot's prompt.
- Plus the bench's 4 balanced verify probes per fact (greedy).

**Conditions:** nomem, ctx, R-full, R-min, T-min (G-match gated).
- 1 greedy + 4 samples, seed set 1. Count: 52 × 2 × 5 × 5 = 2,600 answers.
- The unrelated prompts gated.

**TF:** Δlog P(target) at the measure prefix vs nomem, for every condition.

**Splits:**
- by kind: person / pet / attribute (the secondary items count as attributes);
- by value token count on the run's tokenizer: 1 vs ≥ 2. A multi-token value's R-min keeps its first sub-token + the
  next token, as in addendum 6 (B).

**Rule: "prompt-type mismatch" holds** if both:
- the pool R-min direct-question clean-use CI **overlaps** the primary items' main-run R-min direct-question CI
  [0.40, 0.89] (point 0.66);
- pool R-min use-prompt clean use is **≥ 0.75** (point estimate).

Reported beside the rule: whether the pool CI contains 0.66, and the same numbers for R-full.

#### D. Stage 0 on the pool facts (Qwen2.5, 2B, Llama)

- **Items** (`--items pool`): the 47 pool facts above, built as stage-0 items.
  - The counter value comes from `sample_pool`. A counter that is not one token on the run's tokenizer is replaced
    by the first value of the slot's list that is one token and is neither the target nor a foil (each swap is
    recorded in the layouts).
  - A fact whose target is not one token, or whose counter has no one-token replacement, is excluded and listed.
    *(The second case was added on 2026-10-22, after a dry run with Qwen3.5's tokenizer and before any job.)*
  - Same layout asserts as stage 0.
- **Read-out:** the stage-0 measures and thresholds unchanged (PREREG, "Stage 0 read-out"). The verdict is decided
  on the **person/pet facts** (38 slots, minus any excluded); the attribute slots (9) are reported separately (the
  report's "secondary" tier).
- **Flagged items:** the "undetermined" bound scales with N: more than 2/7 of the person/pet facts flagged.
- **Measures by model:** patching on all three models; knockout (and the subject-alone cell) on Qwen2.5 and Llama.

#### E. Second family: Llama-3.2-3B-Instruct (fp32, gpu-v100)

**The port:**
- `turn_pieces` derives the head and the user-turn segment from `tokenizer.apply_chat_template`. The user-turn start
  = the last special token of the rendered prompt whose following text names the user role. ChatML is no longer
  hard-coded.
- The tokenisation round-trip asserts are kept. On Qwen the pieces are token-identical to before (tested).
- Llama's system block carries a date line. It is pinned: `date_string = "26 Jul 2024"` in every template call.
- EOS: `<|eot_id|>` (plus the generation config's ids).
- Eager attention (for knockout); 28 blocks, so `LAYER_SETS`, the gkey layers and the stage-0 cells carry over.

**Token checks:**
- Every primary/secondary target, counter (with the plumber override) and foil, every lure, and every collide value
  is **one token on Llama-3.2's tokenizer** (checked locally with the cached tokenizer). **No per-model value swap
  is needed.**
- The run-time asserts stay; a failing assert fails the job.
- Pool values that are 2+ tokens are the same on Llama as on Qwen2.5, e.g. carpenter, Turin, Matilda, clarinet,
  Monet, Scorpio.

**Runs:**
- stage 0, patch + knockout, primary + pool (D);
- gen-set `text`: nomem, ctx, R-full, R-min, R-orphan, R-swap, T-min, T-full; seed set 2 for nomem, ctx, T-min,
  R-min; secondary items nomem, ctx, T-min, R-min;
- collide: every R2 condition + G's;
- the probe stage (A's lures and the value/verification table);
- pool scoping (C).

**Rules:**
- **"slice > text":** R-min − T-min clean use (primary, use prompts), CI lower bound > 0.
- **"collision replicates":** partner confusion R2-sep − R2-joint on the confusable pairs, CI lower bound > 0 (the
  addendum-5 "IDs collide" rule; overlap and shift).
- **stage 0:** the original rules.

**Status:** this chain runs only once the user has downloaded the gated checkpoint to the cluster (`PARTS=llama`
in `submit_add7.sh`).

#### F. Qwen3.5-9B, teacher-forced only (fp32, full A100)

- **Model block:** `models.qwen35_9b` (32 blocks; `layer_types` drives the hybrid code; text decoder only).
- **Run:** only the `probe` stage. Conditions: nomem, ctx, R-full-state, R-min-state, R-min-kv6, R-state-only,
  R-state-only-rec, R-state-only-conv, with A's lures.
- No long generation, no judges. The only generation is the probe's greedy 10-token yes/no.
- **Rule: "the 2B split replicates"** if both:
  - R-min-kv6's mean Δlog P(target) is ≥ 0.8 × ctx's, while its TF balanced yes/no (unrelated foils) CI includes
    0.5;
  - R-state-only's TF balanced yes/no has a CI lower bound > 0.5.

#### G. A cheaper collision fix (Qwen2.5, 2B, Llama)

New `bind.pair_plan` entries (h = the head length; A's turn n_A tokens, B's n_B):

| Condition | What it is | Cost / note |
|---|---|---|
| `R2-seq` | keep A's stored full slice; re-encode only B's turn with it as the prefix (positions h + n_A …) | **identical to R2-joint by construction**: asserted (logits within 0.01) and not generated separately, as T2 in addendum 5 |
| `R2-seq-min` | keep A's stored 2 positions (value + full stop); re-encode B's turn with [head + A's 2 positions] as the prefix; keep B's 2 positions from that pass | one pass over the new memory only |
| `R2-blend15` | R2-sep-shift (A full + B's separate slice, shifted by +n_A), with the 15% of B's positions with the largest key deviation (L2 at the second attention layer: block 1 on Qwen2.5 / Llama, block 7 on the 2B) replaced by their values from the one-pass re-encode of B | CacheBlend's selection rule. The replaced values come from a full pass over B, so this is a **selective-recompute upper bound**, not CacheBlend's cost |

- **On the 2B:** every G condition uses the joint DeltaNet state, as in addendum 5 (the re-encode's own DeltaNet state
  equals the joint state). Names: `R2-seq-min-jstate`, `R2-blend15-jstate`.
- **Checks (fail the job):**
  - R2-seq == R2-joint;
  - "re-encode on top of the head only" == the separately harvested slice (the single-fact identity).
- **Conditions generated:**
  - Qwen2.5 and Llama: R2-joint, R2-sep-shift, R2-min-joint, R2-min-sep-shift, R2-seq-min, R2-blend15 (Llama: plus
    all addendum-5 conditions, for E).
  - 2B: the `-jstate` versions, + R2-state-only-J (A).
- **Prompts:** 16 members × (direct question + 2 use prompts) × (1 greedy + 4 samples), seed set 1, + the yes/no
  probes (partner as foil) and the TF readout.

**Measures:** partner confusion and correct clean use, as in addendum 5.

**Rule: "cheap fix works"** for R2-seq-min, if on the confusable pairs both:
- its confusion CI overlaps R2-min-joint's;
- the paired CI of (R2-seq-min − R2-min-sep-shift) confusion has an upper bound < 0.

The same rule is reported for R2-blend15 against R2-joint and R2-sep-shift. R2-seq satisfies the rule trivially (it
is R2-joint), which is reported as the cost statement: one pass over B, not over A and B.

#### Run plan (`submit_add7.sh`; one fresh run root)

- **Chains:**

  | Model | Partition | Parts |
  |---|---|---|
  | Qwen2.5 | gpu-a100-small | A, B, C, D, G |
  | 2B | gpu-a100-small (falls back to gpu-v100 if the smoke's fit check fails) | A, B, D, G |
  | 9B | gpu-a100 | F |
  | Llama | gpu-v100 | E, C, D, G; behind `PARTS=llama` until the checkpoint is on the cluster |

- **Order per chain:** a `--tiny` smoke with the real model and `--max-reserved-gb` → the stage jobs → the original
  judge + the two panels → analyze.
- **Built-in smoke checks (fail the job):**
  - R-full / R-full-state == the sentence as a prior turn (also on Llama);
  - R2-joint == T2;
  - R2-seq == R2-joint, and the single-fact re-encode identity;
  - the probe's R-full(-state) TF == T-full;
  - Paris.
- Every part's `analysis/report.md` states its rule's verdict.

### Addendum 8: follow-ups to addendum 7 (parts 1, 2, 3, 6; 2026-10-25)

*Written and committed before any job of this addendum was submitted and before any of its numbers existed. Known
results: addenda 1–7 (summarised in `docs/dreaming/01-binding.md`, Results, "Addendum 7"), in particular:
- the 2B's recurrent state does item recognition (rejects form and associate lures) but does not bind two facts
  (R2-state-only-J: balanced 0.50);
- the attention-only slice refuses direct questions because a "told" signal is missing (ctx − nomem direction, built on
  all 52 non-primary facts, restores answering);
- the pool facts' use prompts reach in-context level;
- Llama-3.2-3B replicates "slice > text" and the collision;
- Llama's yes/no probes are uninformative (it says "No" with the sentence as an earlier turn);
- joint encoding of 2-position slices stays confused on the pure transformers (0.17–0.20).*

*Everything here is exploratory in the sense of addenda 5–7: each part has a stated rule, and each `report.md` gives
the rule's verdict beside the full tables. Code: `run.py` (`--collide-set add8 | add8pool`, `--items pool` and
`--probe-set add8` for the probe stage, `--dir-set add8` for the directions stage, gen sets `refusal8` and `comply`),
`bind.py` (`add8_plan`, `place`, `subject_phrase`, `form_lures`, `binding_lures`, `pool_split`, the rule functions),
`config.yaml`; data generated by `experiments/dream_bind/add8_data.py` (deterministic; `--check` rebuilds and compares)
plus the hand-written `data/bench_v1/comply_pairs.yaml`; jobs in `experiments/dream_bind/submit_add8.sh`.*

**Shared** (as addendum 7 unless stated).
- **Models:** Qwen2.5-1.5B-Instruct, Qwen3.5-2B (the hybrid; text decoder only) and Llama-3.2-3B-Instruct, all fp32.
- **Judges:** clean use = the majority of the original judge (Qwen3.5-9B bf16) + gpt-oss + Gemma-4 (a tie goes to the
  original's label).
- **Statistics:** bootstrap over items (collide: members), then prompts; 2000 replicates, seed 7; paired differences on
  the same prompts and random streams; teacher-forced (TF) tables bootstrap over items.
- **"CIs overlap"** = lo(a) ≤ hi(b) and lo(b) ≤ hi(a).
- **Partner confusion** (collide) = the partner's value is in the answer and the member's own value is not (regex), as
  in addendum 5.
- **TF yes/no:** "yes" when log P(Yes) > log P(No) at the first assistant token. Greedy: the parsed 10-token answer.
- **Read the text:** 3–5 answers per new condition are read before anything is written up.
- **Hybrid (2B) naming:** every two-fact condition on the 2B uses the JOINT DeltaNet state (after [A's turn][B's turn],
  both plain), as in addenda 5 and 7, so conditions differ only in the six attention blocks; their names carry
  `-jstate` (R2-joint → `R2-joint-state`).

#### Part 1. Is the collision caused by the binding tag? (Qwen2.5, 2B, Llama)

**Pairs:** the 8 addendum-5 pairs (`config.yaml: collide`; 5 confusable, 3 different-subject; 16 members), each member
with its direct question + 2 use prompts × (1 greedy + 4 samples, seed set 1), the balanced yes/no probes (Yes = own
value, No = the partner's value) and the TF readout at the measure prefix (log P(own), logit(own) − logit(partner)).
Gates forced open (oracle retrieval). `--collide-set add8`; output `<run>/<model>_collide/`.

**1a. Tag assigned at storage.** A member "stored after a prefix sentence" P is harvested in the session
[head][user: "P S"] (one user turn: P, a space, the member's sentence S). Its slice keeps the turn's header, S's own
tokens and the end-of-turn tokens and **drops P's tokens** (located by character spans; asserted contiguous, right
after the header). The kept tokens are placed in the R2-sep-shift layout (A's turn at h .., B's turn at h + n_A ..;
the question at h + n_A + n_B; keys re-rotated with the model's own RoPE, `bind.place`).
- Asserted per member and tokenizer: the kept length equals the member's standalone turn length, and the kept tokens
  equal the standalone turn's tokens except at most one (S's first token, " My" after a space vs "My" at the turn start).
- **Dummy pairs** (`data/bench_v1/dummy_pairs.yaml`): "My {slot}'s name is {name}." with (slot, name) = (gardener,
  Brian), (mechanic, Keith), (postman, Derek), (locksmith, Trevor), (tailor, Colin), (barber, Graham), (florist, Janet),
  (butcher, Gary). Member role r (A = 0, B = 1) of the k-th pair (the collide set's pair order) gets dummy
  (2k + r) mod 8 (`bind.dummy_for`).
  - The slots are none of `fact_pool.yaml`'s slots, and no collide member's subject.
  - The names are in no name / value list, foil, counter, lure, collide value or pool-pair value.
  - Each name is one token (leading space) on the Qwen2.5, Qwen3.5 and Llama-3.2 tokenizers (checked in the tests;
    asserted at run time).
- **The placebo prefix:** "By the way, I had a coffee this morning." (`config.yaml: gen.placebo`; not a pair).

| Condition | A's entries | B's entries |
|---|---|---|
| `R2-sep-tagged` | A stored alone (pair 1), full turn | B stored after B's dummy pair (B is "pair 2"), dummy dropped |
| `R2-sep-bothtagged` | A stored after A's dummy pair, dummy dropped | as tagged |
| `R2-sep-placebo` | A alone | B stored after the placebo sentence, placebo dropped |
| `R2-min-sep-tagged` / `-bothtagged` / `-placebo` | the same harvests, value + full stop only (2 positions) | the same, 2 positions |
| anchors | `ctx2`, `R2-joint`, `R2-sep-shift`, `R2-min-joint`, `R2-min-sep-shift` (addendum 5) | |

- **Check (fails the job):** `R2-sep-tagged` built from B harvested with an EMPTY prefix (the same code path, 0 dropped
  tokens) gives logits equal to `R2-sep-shift`'s (max |Δlogit| ≤ 0.01) for both members' questions. Unit tests: with
  0 dropped tokens the tagged plans flatten to exactly the R2-sep-shift / R2-min-sep-shift plans; the bothtagged plans
  are symmetric in A and B.
- **2B:** every condition with the joint DeltaNet state (`-jstate`). The prefix sentences change the attention blocks'
  K,V only through the stored session (the DeltaNet blocks of the harvest saw them); the restored DeltaNet state is
  always the plain joint one.
- **Rule "the tag causes the swaps"** (decided on the full slices, confusable pairs; 2B with the `-jstate` names), all of:
  - partner confusion of R2-sep-tagged − R2-sep-shift: paired CI upper bound < 0;
  - R2-sep-tagged's confusion CI overlaps R2-joint's;
  - R2-sep-bothtagged − R2-sep-shift: paired CI includes 0.
  - Reported beside it: the same rule on the min slices (R2-min-sep-tagged / -bothtagged vs R2-min-sep-shift and
    R2-min-joint), on the different-subject pairs, and R2-sep-placebo − R2-sep-shift.
- **Prediction:** tagged ≈ joint (fewer swaps than sep-shift), bothtagged ≈ sep (both carry "pair 2"), placebo ≈ sep.

**1b. Do 2 positions lose the entity-side tag?**
- **The min+subj slice:** the 2 positions (value + full stop) plus the stored sentence's subject-span positions. The
  subject span is the words between the leading "My " and the first relation ("'s name is", " is called",
  " was called", " is", " was"): "younger sister", "first dog", "friend from work", "son's teacher", "grandmother on
  my mother's side", "dog", "best friend", "boss" (`bind.subject_phrase`); the tokens overlapping those characters.
- **Conditions:** `R2-minsubj-joint` (both members' min+subj entries from the joint harvest, at their joint positions)
  and `R2-minsubj-sep-shift` (from the separate harvests, B shifted by +n_A).
- **Rule "the entity side carries the tag"** (decided on the pure transformers, Qwen2.5 and Llama, each; confusable
  pairs), both:
  - R2-minsubj-joint − R2-min-joint partner confusion: paired CI upper bound < 0;
  - R2-minsubj-joint's confusion CI overlaps R2-joint's.
  - Reported: R2-minsubj-sep-shift (vs R2-min-sep-shift and R2-sep-shift), the different-subject pairs, the 2B's
    numbers with the same rule.
- **Prediction:** minsubj-joint < min-joint (the subject positions carry the tag that two positions miss).

**Generated per model:** 13 conditions (the 5 anchors, the 6 tagged / placebo conditions, the 2 min+subj
conditions) × 16 members × 3 prompts × 5 = 3,120 answers + 832 greedy yes/no + the TF readout. No nomem (it is
0 by construction in addendum 5). The addendum-5/7 checks stay: R2-joint == T2, R2-seq == R2-joint, the single-fact
re-encode identity.

#### Part 2. Scale: addendum 7's results and the collision on the pool facts

**Facts:** the 52 non-primary facts of addendum 7 C (`run.pool_gen_items`: the 5 secondary hand-written items, then the
47 pool facts, `sample_pool` seed 0; kinds: person 30, pet 8, attribute 14).

**2a. Teacher-forced probe on the pool** (`--stage probe --items pool --probe-set add8`; output
`<run>/<model>_probepool/`).
- **Probes per fact** (each asked with the fact's 2 verify templates; the correct answer to a lure is No):

  | Kind | Values | Probes |
  |---|---|---|
  | yes | the target | 2 (the bench's) |
  | unrelated | the pool foils (the bench's No probes) | 2 (the bench's) |
  | form | 2 drawn from the slot's own list (rule below) | 4 |
  | binding | 2 other pool facts' values from the same name list (person / pet facts only) | 4 |
  | associate | **not drawn for the pool** (no hand-written theme lures for 52 facts) | 0 |

- **Form-lure rule** (`bind.form_lures`, `bind.FORM_RULE`): from the slot's value list, the values that are not the
  target, not the counter, not one of the 3 measure foils, and one token with a leading space on all three tokenizers;
  ranked by (same first letter as the target first, case-insensitive; longer common prefix; smaller Levenshtein
  distance; list order); the first 2. "Same first letter or sound" is thus first letter, then shared prefix / edit
  distance. Two attribute slots have no eligible value (favourite board game, favourite painter: the list's 5 values
  are the target, the counter and the 3 foils) and one has one (favourite flower: jasmine); attribute facts are reported
  only.
- **Binding-lure rule** (`bind.binding_lures`, `bind.BINDING_RULE`): the values of the next facts in the pool order
  (cyclically) whose slot uses the same name list (female people, male people, pets), skipping the target, the counter,
  the foils, the form lures, values already taken and values that are not one token on all three tokenizers; the
  first 2. Attribute facts get none (an attribute value of another slot makes a nonsense question).
- The drawn lists (`data/bench_v1/pool_lures.yaml`) are in the table at the end of this addendum. Asserted at run time:
  every lure is one token on the run's tokenizer and is not the target, the counter or a foil.
- **Conditions:** 2B: nomem, ctx, R-full-state, R-min-state, R-min-kv6, R-state-only, R-state-only-rec,
  R-state-only-conv. Qwen2.5 and Llama: nomem, ctx, R-full, R-min, T-min, T-full (as addendum 7).
- **Measures:** as addendum 7 A: false-yes rate FY_k per kind, balanced accuracy against k; "against lures" = **the form
  lures** (no associates); TF primary, greedy reported.
- **Rule** (decided on the 2B's R-state-only, TF, over the **38 person / pet facts**; R-state-only-rec reported with the
  same rule; the attribute facts reported separately; Qwen2.5 and Llama: tables only):
  - **"gist familiarity"** if FY_form − FY_unrelated has a paired CI lower bound > 0;
  - **"item recognition"** if not, AND its point estimate is within 0.1 of 0, AND the balanced accuracy against the
    form lures has a CI lower bound > 0.5;
  - otherwise **"in between"**.
- **Prediction:** the pool reproduces item recognition.

**2b. Refusal on held-out pool facts** (`--stage directions --dir-set add8`, output `<run>/<model>_dirs8/`; then
`--stage gen --gen-set refusal8`, output `<run>/<model>_refusal8/`).
- **The split** (`data/bench_v1/pool_split.yaml`, `bind.pool_split`, seed 8): per kind, the fact ids shuffled with
  `random.Random(8 + kind index)` and the first half to A; 26 facts each, person 15 / 15, pet 4 / 4, attribute 7 / 7.
  Asserted at run time: the halves are disjoint, cover the 52 facts and equal the file.
- **On half A only:** d_told(l) = mean over half-A facts of [ctx − nomem] at the direct question's last prompt token
  (block outputs), and its (layer set, α) by addendum 7's rule (S ∈ thirds, α ∈ {0.25, 0.5, 1, 2}; score = the mean
  change in the measure prefix's log-probability vs X alone; constraint: mean Δlog P(target) ≥ −1 nat; ties → smaller
  α; no cell meets it → the smallest α at the best-scoring set). X per fact = its own attention-only slice.
- **X:** R-min-kv6 on the 2B; R-min on Qwen2.5 and Llama. **Llama is new here.**
- **Arms on half B** (G-match gated, as before):

  | Condition | 2B | Qwen2.5 | Llama |
  |---|---|---|---|
  | nomem, ctx, X | ✓ | ✓ | ✓ |
  | X+told, X+told:rand, X+told:neg, nomem+told | ✓ | ✓ | ✓ |
  | X+comply, X+comply:rand (part 3) | ✓ | ✓ | ✓ |
  | R-min-state | ✓ | | |
  | R-min-kv-own+state-swap (the swap partner = the next half-B fact of the same kind, cyclic) | ✓ | | |
  | R-min-kv-own+state-plac | ✓ | | |

- **Prompts:** the direct question + the pool use prompt, 1 greedy + 4 samples (seed set 1; no seed set 2); the
  bench's 4 balanced yes/no probes (greedy); the 50 unrelated prompts (gated).
- **Judged** by the original judge + the gpt-oss and Gemma-4 panel (majority), as before.
- **Rules** (decided on the 2B; Qwen2.5 and Llama reported with the same rules):
  - **"Missing told signal"** as addendum 7 B3: (X+told − X) direct-question clean use paired CI lower bound > 0, AND
    (X+told − X+told:rand) and (X+told − X+told:neg) both lower bound > 0, AND (X+told − X) on the use prompt with
    point estimate ≥ −0.1 and upper bound ≥ 0.
  - The state rule of addendum 7 B3 ("generic response mode" / "item-specific state" / "in between"), 2B, reported.
- **Prediction:** the held-out told vector restores answering on half B as in addendum 7.

**2c. Collision on pool pairs** (`--collide-set add8pool`, output `<run>/<model>_pp_collide/`).
- **Pairs** (`data/bench_v1/pool_pairs.yaml`; table at the end): 34 pairs, 68 members.
  - 17 **confusable**: "older / younger {slot}" for sister, brother, cousin, niece, nephew, son, daughter; "first /
    second {pet}" for the 10 pet slots.
  - 17 **different-subject**: 12 person + person and 5 person + pet, slots from a seeded shuffle (`random.Random(17)`
    persons, `random.Random(18)` pets; one partner slot).
  - Values from the slot's name list (`random.Random(13)`), distinct across the file, never a primary / addendum-5
    collide value or a dummy name, one token (leading space) on all three tokenizers (asserted at run time too).
  - Templates: the pool's person / pet kinds: "My {slot}'s name is {v}.", "What's my {slot}'s name?", "Your {slot}'s
    name is", the 2 verify templates, and ONE use prompt ("Write a short birthday message to my {slot}." / "Write a
    short, funny poem about my {slot}.").
- **Per member:** direct question + the use prompt × (1 greedy + 4 samples), balanced yes/no (partner as foil), TF
  readout.
- **Conditions:** the same 13 as part 1 (2B: `-jstate`). Count: 68 × 2 × 5 × 13 = 8,840 answers + 3,536 yes/no per model.
- **Rules:** addendum 5's "IDs collide" (partner confusion R2-sep-shift − R2-joint, confusable, CI > 0) and
  "re-encoding helps" (correct clean use R2-joint − R2-sep-shift, confusable, CI > 0), plus part 1's two rules, all
  reported again at this N (decided per model, as in part 1).

#### Part 3. Is "told" specific, or just "comply"? (2B, Qwen2.5 and Llama)

- **The competing direction** (built in the same `--dir-set add8` directions run, on no memory):
  - Data: `data/bench_v1/comply_pairs.yaml`, 66 matched question pairs: a personal question the model cannot know
    ("What's my blood type?") and a general question of a similar form ("What's the most common blood type?").
    Disjoint from all test items (no slot of any fact or collide member, no value, counter, foil, lure or dummy name;
    checked in the tests). The list is at the end of this addendum.
  - **Behaviour check, per model, before use:** both questions are answered greedily under nomem (60 new tokens). A pair
    **behaves** if the personal answer is a refusal and the general answer is not, by the comply refusal regex
    (`run.COMPLY_REFUSAL_RE`: addendum 7's refusal regex extended with "not able to / unable to / won't … know,
    access, determine, provide …", "I apologise", "as an AI", "no way to know", "I don't have (any) personal /
    information / access …", "only you", "can / could / would you tell / share / let me know / provide"). Pairs that
    don't behave are dropped and listed with their answers in `<model>_dirs8/`.
  - d_comply(l) = mean(general) − mean(personal) of the block-l output at the last prompt token, over the behaving pairs.
- **Selection:** d_comply's layer set and α by the same rule on pool half A, with the same slice X.
- **Arms:** X+comply (v = α_c · d_comply(l) on the picked set, at the last prompt token, prefill only) and X+comply:rand
  (a random unit direction per block, fixed seed 4321, × |v(l)|), on half B (in `refusal8`) and on the 7 primary items
  (`--gen-set comply`: nomem, ctx, X, X+told, X+comply, X+comply:rand; direct question + 2 use prompts × 5; the told
  vector is addendum 8's, built on half A).
- **Reported:** the cosine between d_told(l) and d_comply(l) per block; the comply pick; the kept / dropped pairs.
- **Rule** (per model, on half B, direct-question clean use; the 2B's verdict is the headline; the primary items are
  reported with the same rule), checked in this order:
  - **"told is specific"** if (X+told − X+comply) has a paired CI lower bound > 0;
  - else **"generic comply"** if X+comply's point estimate is within 0.1 of X+told's;
  - otherwise **"in between"**.
- **Prediction:** told beats comply.

#### Part 6. Usable yes/no on Llama (teacher-forced probe stage; all three models for comparison)

Added to the probe stage by `--probe-set add8`, for the 7 primary items (`<model>_probe/`, decisive) and for the 52
pool facts (`<model>_probepool/`, reported), in the same conditions as 2a.
- **Wording (i), "told":** every yes/no probe of the stage (target, unrelated, every lure) asked as "Based on what I told
  you earlier, {the verify question with a lower-case first letter} Answer yes or no.", e.g. "Based on what I told you
  earlier, is my dog called Pepper? Answer yes or no." TF yes/no on all; greedy 10-token answers on the primary items
  only (TF only on the pool).
- **Wording (ii), "two-choice":** "{the verify template with {v} = 'X or Y'} Answer with the name only." (attribute
  facts: "Answer with one word only."), e.g. "Is my dog called Pepper or Max? Answer with the name only." For each
  foil value (the 2 unrelated foils, and each lure: form, binding, and associate on the primary items) × the 2 verify
  templates × both orders (true first, foil first). Scored teacher-forced: correct when the first token of the true
  name (no leading space, as the answer's first word) has a higher log-probability than the foil's first token at the
  first assistant position. A (true, foil) pair whose first tokens coincide on the run's tokenizer is skipped and listed.
- **Rule "Llama verification usable"**, per wording: ctx and T-full (the sentence as an earlier turn) both reach a
  balanced accuracy ≥ 0.9 (point estimate; wording (i), TF, Yes probes vs unrelated No probes), or, for the two-choice
  wording, an accuracy ≥ 0.9 against the unrelated foils, on the 7 primary items. Only then are Llama's slice
  conditions under that wording interpreted. Everything is reported, including the base wording again, the lures and
  the pool. The same rule is reported for Qwen2.5 and the 2B (the 2B has no T-full condition in the probe stage;
  R-full-state is identical to it by construction, asserted, and stands in for it).
- **Prediction:** wording (i) or (ii) fixes Llama.

#### Run plan (`submit_add8.sh`; one fresh run root)

- **Chains** (one per model):

  | Model | Partition | Jobs |
  |---|---|---|
  | Qwen2.5 | gpu-a100 (full; the MIG slice runs one job at a time and is ~2.5× slower per answer in addendum 7) | smoke → dirs8 + probe + probe pool → refusal8 → comply; smoke → collide add8 → collide add8pool |
  | 2B | gpu-a100 (full; the refusal gen OOMed on the MIG slice in addendum 7) | the same |
  | Llama | gpu-v100 (fp32) | the same |

  Then per model: the original judge and gpt-oss (both after the chains), Gemma-4 after gpt-oss (so at most 8 jobs run
  at once), and analyze (compute-p1) after all three and the TF job.
- **Smoke** (`--tiny`, the real model, `--max-reserved-gb`): every new code path (dirs8, both probe sets, refusal8,
  comply, collide add8 and add8pool). **Built-in checks (fail the job):**
  - addendum 7's: R-full(-state) == the sentence as a prior turn; R2-joint == T2; R2-seq == R2-joint; the single-fact
    re-encode identity; the probe's R-full(-state) == T-full; Paris;
  - new: R2-sep-tagged with an empty dummy == R2-sep-shift; the tagged-layout asserts (kept length, at most one
    differing kept token, the dropped tokens decode to the prefix sentence); the pool split is disjoint, covers the 52
    facts and equals the file; the told / comply picks use half-A facts only and the refusal8 items are half B; every
    dummy name, pool lure and pool-pair value is one token on the run's tokenizer.
- **Wall times:** addendum 7's measured durations (`jobs.txt`, `sacct`), scaled to the counts above, + 30%; every job
  ≤ 1 day.
- Every part's `report.md` states its rule's verdict.

#### Addendum 8 data (the files are the record; reproduced here)

**Pool facts: half (2b), counter, drawn lures (2a)** (`pool_split.yaml`, `pool_lures.yaml`)

| fact | kind | half | target | counter | form lures | binding lures |
|---|---|---|---|---|---|---|
| job | attribute | B | pharmacist | plumber | pilot, journalist | – |
| hometown | attribute | B | Leeds | Glasgow | Lyon, Porto | – |
| first_language | attribute | B | Portuguese | Polish | Turkish, Korean | – |
| university | attribute | B | Edinburgh | Warwick | Durham, Bath | – |
| favourite_colour | attribute | A | teal | orange | turquoise, yellow | – |
| pool_favourite_board_game | attribute | B | Monopoly | backgammon | – | – |
| pool_doctor | person | B | Amelia | Diana | Ava, Alice | Rachel, Emily |
| pool_star_sign | attribute | B | Leo | Scorpio | Gemini, Cancer | – |
| pool_daughter | person | B | Rachel | Irene | Ruth, Ruby | Emily, Lily |
| pool_cousin | person | B | Jonas | Luca | James, Jack | Stefan, Diego |
| pool_goldfish | pet | B | Bella | Pixel | Blue, Buddy | Thor, Sherlock |
| pool_grandfather | person | A | Stefan | Diego | Simon, Samuel | Dennis, George |
| pool_dentist | person | B | Diego | Tomas | Daniel, David | Dennis, Ivan |
| pool_childhood_friend | person | B | Emily | Ella | Emma, Evelyn | Lily, Chloe |
| pool_hamster | pet | A | Thor | Ruby | Toby, Tiger | Sherlock, Shadow |
| pool_horse | pet | A | Sherlock | Toby | Shadow, Scout | Charlie, Daisy |
| pool_favourite_flower | attribute | A | iris | lavender | jasmine | – |
| pool_football_team | attribute | A | Ajax | Barcelona | Arsenal, Chelsea | – |
| pool_tennis_coach | person | A | Lily | Nora | Leah, Lena | Chloe, Hannah |
| pool_husband | person | A | Dennis | Lucas | Daniel, Diego | Ivan, George |
| pool_ferret | pet | B | Shadow | Maple | Sunny, Scout | Charlie, Buddy |
| pool_guinea_pig | pet | B | Charlie | Ranger | Cookie, Captain | Buddy, Daisy |
| pool_neighbour | person | B | Ivan | Tomas | Liam, Noah | George, Simon |
| pool_stepfather | person | A | George | Walter | Gordon, Henry | Simon, Jack |
| pool_coworker | person | B | Chloe | Ava | Clara, Carmen | Lena, Hannah |
| pool_godmother | person | B | Lena | Wendy | Leah, Lily | Hannah, Sylvia |
| pool_mother_in_law | person | B | Hannah | Wendy | Harper, Anna | Sylvia, Rosa |
| pool_babysitter | person | A | Sylvia | Ella | Sophia, Sofia | Harper, Rosa |
| pool_parrot | pet | A | Buddy | Shadow | Bella, Bean | Daisy, Ruby |
| pool_aunt | person | B | Harper | Esther | Hannah, Helen | Rosa, Iris |
| pool_piano_teacher | person | A | Simon | Lucas | Samuel, Stefan | Jack, Rafael |
| pool_favourite_painter | attribute | A | Monet | Rembrandt | – | – |
| pool_personal_trainer | person | B | Jack | Theo | James, Jonas | Rafael, Victor |
| pool_father_in_law | person | A | Rafael | Theo | Roger, Daniel | Jack, Victor |
| pool_favourite_subject | attribute | A | history | biology | chemistry, art | – |
| pool_birthday_month | attribute | A | June | October | July, January | – |
| pool_mother | person | A | Rosa | Naomi | Ruth, Rachel | Iris, Sofia |
| pool_hairdresser | person | B | Jack | Harold | James, Jonas | Victor, Vincent |
| pool_favourite_book | attribute | A | Matilda | Beloved | Emma, Rebecca | – |
| pool_father | person | A | Victor | Theo | Vincent, Nico | Stefan, Hans |
| pool_instrument | attribute | B | clarinet | flute | trumpet, piano | – |
| pool_niece | person | B | Iris | Eva | Irene, Mia | Sofia, Sylvia |
| pool_roommate | person | A | Sofia | Sarah | Sophia, Sylvia | Grace, Amelia |
| pool_therapist | person | B | Sylvia | Anna | Sophia, Sofia | Grace, Rachel |
| pool_vet | person | A | Vincent | Hans | Victor, Ben | Stefan, Simon |
| pool_accountant | person | A | Grace | Sarah | Alice, Kate | Amelia, Rachel |
| pool_rabbit | pet | B | Daisy | Duke | Max, Buddy | Ruby, Bella |
| pool_nephew | person | A | Stefan | Tomas | Simon, Samuel | Hans, Anton |
| pool_landlord | person | A | Hans | Erik | Harold, Henry | Simon, Anton |
| pool_uncle | person | A | Simon | Tomas | Samuel, Stefan | Anton, Jonas |
| pool_tortoise | pet | A | Ruby | Rex | Rocky, Ranger | Bella, Thor |
| pool_son | person | B | Anton | Omar | Adam, Leon | Jonas, Stefan |

**Pool pairs (2c)** (`pool_pairs.yaml`)

| pair | kind | member A | member B |
|---|---|---|---|
| ppc_sister | confusable | My older sister's name is Ruby. | My younger sister's name is Clara. |
| ppc_brother | confusable | My older brother's name is Norman. | My younger brother's name is Edgar. |
| ppc_cousin | confusable | My older cousin's name is Ben. | My younger cousin's name is Harold. |
| ppc_niece | confusable | My older niece's name is Zoe. | My younger niece's name is Naomi. |
| ppc_nephew | confusable | My older nephew's name is Oscar. | My younger nephew's name is Felix. |
| ppc_son | confusable | My older son's name is Stefan. | My younger son's name is Fred. |
| ppc_daughter | confusable | My older daughter's name is Chloe. | My younger daughter's name is Emily. |
| ppc_dog | confusable | My first dog's name is Bella. | My second dog's name is Scout. |
| ppc_cat | confusable | My first cat's name is Maple. | My second cat's name is Lucky. |
| ppc_rabbit | confusable | My first rabbit's name is Buddy. | My second rabbit's name is Apollo. |
| ppc_parrot | confusable | My first parrot's name is Shadow. | My second parrot's name is Max. |
| ppc_horse | confusable | My first horse's name is Honey. | My second horse's name is Comet. |
| ppc_hamster | confusable | My first hamster's name is Daisy. | My second hamster's name is Captain. |
| ppc_goldfish | confusable | My first goldfish's name is Winston. | My second goldfish's name is Cookie. |
| ppc_tortoise | confusable | My first tortoise's name is Blue. | My second tortoise's name is Ranger. |
| ppc_guinea_pig | confusable | My first guinea pig's name is Sunny. | My second guinea pig's name is Milo. |
| ppc_ferret | confusable | My first ferret's name is Bean. | My second ferret's name is Duke. |
| ppd_tennis_coach_landlord | different | My tennis coach's name is Julia. | My landlord's name is Diego. |
| ppd_vet_grandmother | different | My vet's name is Bruce. | My grandmother's name is Carmen. |
| ppd_personal_trainer_accountant | different | My personal trainer's name is Otto. | My accountant's name is Ada. |
| ppd_father_therapist | different | My father's name is Tomas. | My therapist's name is Eva. |
| ppd_stepfather_sister | different | My stepfather's name is Jack. | My sister's name is Leah. |
| ppd_aunt_son | different | My aunt's name is Anna. | My son's name is Luca. |
| ppd_best_friend_dentist | different | My best friend's name is Anton. | My dentist's name is Carl. |
| ppd_boss_nephew | different | My boss's name is Irene. | My nephew's name is Frank. |
| ppd_neighbour_hairdresser | different | My neighbour's name is Dennis. | My hairdresser's name is Ivan. |
| ppd_wife_daughter | different | My wife's name is Nina. | My daughter's name is Nora. |
| ppd_uncle_mother | different | My uncle's name is Jonas. | My mother's name is Kate. |
| ppd_brother_cousin | different | My brother's name is Peter. | My cousin's name is Karl. |
| ppd_coworker_dog | different | My coworker's name is Ruth. | My dog's name is Tiger. |
| ppd_roommate_horse | different | My roommate's name is Maya. | My horse's name is Bear. |
| ppd_father_in_law_goldfish | different | My father-in-law's name is Marco. | My goldfish's name is Rex. |
| ppd_piano_teacher_parrot | different | My piano teacher's name is Victor. | My parrot's name is Chester. |
| ppd_godmother_hamster | different | My godmother's name is Vera. | My hamster's name is Mango. |

**Comply pairs (part 3)** (`comply_pairs.yaml`)

| id | personal (expected: refused) | general (expected: answered) |
|---|---|---|
| blood_type | What's my blood type? | What's the most common blood type? |
| shoe_size | What's my shoe size? | What's the average shoe size for adult women in Europe? |
| phone_number | What's my phone number? | What's the emergency phone number in Australia? |
| home_address | What's my home address? | What's the address of the White House? |
| middle_name | What's my middle name? | What's a popular middle name for boys in the United States? |
| age | How old am I? | How old is the Great Pyramid of Giza? |
| parking | Where did I park my car? | Where can you usually park at a large airport? |
| breakfast | What did I eat for breakfast today? | What do people in Japan often eat for breakfast? |
| licence_plate | What's my car's licence plate? | What's the format of a car licence plate in the Netherlands? |
| wake_time | What time did I wake up today? | What time does the sun usually rise in Madrid in summer? |
| film | What's my favourite film? | What's the highest-grossing film of all time? |
| iban | What's my bank account number? | How many characters does a German IBAN have? |
| postcode | What's my postcode? | What's the postcode format used in Canada? |
| street | Which street do I live on? | Which street is the most famous shopping street in Paris? |
| height | How tall am I? | How tall is the Eiffel Tower? |
| weight | How much do I weigh? | How much does an adult African elephant weigh? |
| car | What car do I drive? | What car was the first to be mass-produced on an assembly line? |
| trip | Where did I go on holiday last year? | Where do most tourists stay when they visit Rome? |
| pin | What's my PIN? | How many digits does a typical bank card PIN have? |
| food | What's my favourite food? | What's the national dish of Hungary? |
| song | What's my favourite song? | What's the most streamed song on Spotify? |
| season | What's my favourite season? | What's the warmest season in Australia? |
| allergy | What am I allergic to? | What are people most often allergic to? |
| sport | What's my favourite sport? | What's the most popular sport in the world? |
| passport | What's my passport number? | How many characters does a passport number usually have? |
| ice_cream | What's my favourite ice cream flavour? | What's the most popular ice cream flavour in the US? |
| city | What's my favourite city? | What's the capital city of Canada? |
| coffee | What's my usual coffee order? | What's in a flat white? |
| drink | What's my favourite drink? | What's the most popular hot drink in Britain? |
| salary | How much do I earn? | How much does a software engineer earn on average in the US? |
| eyes | What colour are my eyes? | What's the most common eye colour in the world? |
| hair | What's my natural hair colour? | What's the rarest natural hair colour? |
| tv_show | What's my favourite TV show? | What's the longest-running TV show in Britain? |
| flight | What's my flight number? | How long is a flight from Paris to Tokyo? |
| seat | What's my seat number on the train? | How many seats does a Boeing 747 have? |
| number | What's my favourite number? | What's the smallest prime number? |
| nickname | What's my nickname? | What's a common nickname for William? |
| animal | What's my favourite animal? | What's the largest animal on Earth? |
| fruit | What's my favourite fruit? | What's the most widely eaten fruit in the world? |
| word | What's my favourite word? | What's the longest word in German? |
| bird | What's my favourite bird? | What's the fastest bird in the world? |
| dessert | What's my favourite dessert? | What's a traditional British dessert? |
| purchase | What did I buy yesterday? | What do people usually buy on Black Friday? |
| appointment | What time is my appointment tomorrow? | What time do banks usually open in the UK? |
| house_number | What's my house number? | What's the house number of the British Prime Minister's residence? |
| hotel | Which hotel am I staying at? | Which hotel is the tallest in the world? |
| museum | What's my favourite museum? | What's the most visited museum in the world? |
| band | What's my favourite band? | What's the best-selling band of all time? |
| cheese | What's my favourite cheese? | What's the most famous cheese from Switzerland? |
| tree | What's my favourite tree? | What's the tallest tree species in the world? |
| actor | Who is my favourite actor? | Who won the Academy Award for Best Actor in 1994? |
| video_game | What's my favourite video game? | What's the best-selling video game of all time? |
| pizza | What's my favourite pizza topping? | What's the most popular pizza topping in the US? |
| ring_size | What's my ring size? | What's the most common ring size for women? |
| email | What's my email address? | What's the symbol used in every email address? |
| destination | What's my favourite holiday destination? | What's the most visited holiday destination in Europe? |
| password | What's my Wi-Fi password? | What's the most common password in the world? |
| gym | Which gym do I go to? | Which gym exercise burns the most calories? |
| restaurant | What's my favourite restaurant? | What's the oldest restaurant in the world? |
| dinner | What did I have for dinner last night? | What do people in Spain usually have for dinner? |
| cake | What's my favourite cake? | What's the most famous cake from Austria? |
| spare_key | Where do I keep my spare key? | Where do people usually hide a spare house key? |
| vegetable | What's my favourite vegetable? | What's the most nutritious vegetable? |
| planet | What's my favourite planet? | What's the largest planet in the solar system? |
| poem | What's my favourite poem? | Who wrote the poem 'The Raven'? |
| train | Which train do I take every morning? | Which train is the fastest in the world? |

## Statistics

- **Stage 0:** one value per item per cell (one prompt per item). The statistic is the **macro mean over
  items**; CIs are **95% percentile bootstrap CIs over items, 2000 replicates, seed 7**
  (`seahorse.stats.boot_ci`).
- **Later stages:** the macro mean over items of the mean over the item's prompts; two-level bootstrap
  (items, then prompts within items), 2000 replicates, seed 7; paired gains per cell (same prompt, same
  random streams). Every item is also reported separately, with the share of items and prompts moved the
  wrong way.

## Checks that invalidate a run (the job fails)

- Source and target layouts as asserted (equal length, one differing token, which is the target's first
  token; the subject span inside the question, after the attribute; identical question layouts).
- Patching every position at a block with the source states reproduces the source logits (max |Δlogit|
  ≤ 0.01 in fp32), at the first, middle and last block, for every item.
- Patching nothing reproduces the target logits (same tolerance).
- An empty knockout (the hooks installed, nothing blocked) equals the unmodified run.
- A knockout of everything but itself, at every layer, equals the last token run alone (the mask edit
  takes effect).
- All logits finite.
- The loaded model answers "What is the capital of France?" with Paris (a garbage load fails here).
- **Reported, not required:** the subject-alone sanity share per item, and the drop in log P(counter)
  when the attribute is blocked.

## Known limitations (stated in advance)

- **One prompt per item.** Stage 0 uses the direct question only; CIs are over 7 items.
- **The subject-alone block is narrow by design.** Only the attribute position itself is hidden from the
  question. The tokens after it in the experience sentence (the full stop) can still carry the counter
  value into the question, so the cell can understate the subject's sufficiency. It is a safeguard against
  a false "pointer", not a proof of one.
- **Batched patching.** Patched rows run in batches. Batch shape can change floating point in the last
  bits, which the identity tolerance absorbs.
- **Knockout is on Qwen2.5 only.** On Qwen3.5-2B, patching carries the read-out alone, and a mismatch
  between the models is a finding, not an error.
- **Different hardware for the two models** (above). Both run in fp32.
