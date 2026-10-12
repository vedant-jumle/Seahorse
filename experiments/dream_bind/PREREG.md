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
