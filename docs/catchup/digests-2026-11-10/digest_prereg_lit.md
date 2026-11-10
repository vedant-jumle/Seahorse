# Digest: dream_bind PREREG (add. 1–9), literature.md, literature-exp1.md, dream_refs, core_v1, dream_subj

Sources read in full (2026-11-10): experiments/dream_bind/PREREG.md, docs/reference/literature.md,
docs/dreaming/literature-exp1.md, experiments/dream_refs/{PREREG,README}.md, experiments/core_v1/{PREREG,README}.md,
experiments/dream_subj/PREREG.md. The PREREGs record rules, not results. Verdicts below marked [R] come from
docs/dreaming/01-binding.md (Results) or the run's report.md/C7.md, which I read to fill them in.
Seahorse = an independent side project (never call it a thesis).

## 0. Shared conventions (all dream_bind addenda unless stated)
- Stats: macro mean over items of the per-item prompt mean. 95% percentile CIs from a two-level bootstrap (items, then
  prompts), 2000 reps, seed 7 (`seahorse.stats.boot_ci`). Paired differences use the same prompt and random streams.
  Teacher-forced (TF) tables bootstrap over items. Collide: members, then prompts.
- Generation (core_v1 settings): 1 greedy + 4 samples, T 1, top-p 0.95, top-k 20, presence 1.5, cap 120. Per-prompt
  seeds, identical streams across conditions. Seed set 2 = 4 more samples for the named main cells.
- Judges: clean use = majority of the original judge (Qwen3.5-9B bf16, core_v1 `judge_prompts.yaml` + parser) +
  gpt-oss + Gemma-4 (Ollama panel; a tie goes to the original's label). `uses_fact` AND `coherent` on use prompts.
  Llama-3.1 is excluded as a judge (it failed core_v1's sanity check).
- Yes/no: balanced (accY + accN)/2. Greedy 10 tokens, parsed (unparsed = wrong). "TF yes/no" = log P(Yes) > log P(No)
  at the first assistant token.
- 50 unrelated prompts, G-match gated. `unrel_same` ≥ 0.95 (B6). A closed gate means the answer is nomem by construction
  (`shortcut`, not regenerated).
- Fail-the-job checks: R-full(-state) == the sentence as a prior turn (≤ 0.01); R2-joint == T2; R2-seq == R2-joint;
  single-fact re-encode identity; probe R-full(-state) TF == T-full; "Paris"; finite logits.
- "CIs overlap" = lo(a) ≤ hi(b) and lo(b) ≤ hi(a). add9: |CI bound| < 1e-9 counts as 0 (add8 printed a 4e-18 as > 0).
  Read 3–5 answers per new condition before writing up (add7+).

## 1. dream_bind PREREG: base (2026-10-11), stage 0 and the B1–B6 plan
**Question.** A push at the answer position can't bind. Can the model's own subject-enriched state, written back at
the subject token in session 2 (`subj`), be used? Compare with reinstating the experience as a gated segment (`reinst`).
Stage 0 asks first: does the fact sit at the question's subject ("dog") or only at its own position ("Pepper")?
- **Items.** The 12 core_v1 facts. **Primary 7** (all rules): dog_name, cat_name, sister_name, brother_name,
  grandmother_name, best_friend_name, boss_name. **Secondary 5** (reported only): job, hometown, first_language,
  university, favourite_colour (no-site in later stages). The job counter is overridden to "I work as a plumber."
  (carpenter is 2 tokens). Every primary target and counter is 1 token on both tokenizers, with equal-length prompts.
- **Models.** Qwen2.5-1.5B-Instruct: 28 blocks, all full attention, fp32, eager, local RTX 4060, transformers 4.46.3.
  Qwen3.5-2B: 24 blocks, full attention at 3, 7, 11, 15, 19, 23 (Gated DeltaNet elsewhere), fp32, DelftBlue
  gpu-a100-small (10 GB MIG), transformers 5.18. Text decoder only, `enable_thinking=False`.
- **Stage 0 prompts.** Source: "My dog's name is Pepper. What's my dog's name?" + prefix "Your dog's name is". Target:
  the same with the counter (Rocky), exactly 1 differing token. nomem = the question alone. Primary site = the last
  token of the question-subject alias span (not the mention in the experience).
- **Measures.**
  1. Patching (both models): residual output of block ℓ copied source → target, per block and per sliding 5-block band.
     Sites: subject last token, subject span, attribute. m = logit(target) − logit(counter).
     Flipped share = (m_patched − m_target)/(m_source − m_target).
  2. Knockout (Qwen2.5 only): remove answer→subject-span and answer→attribute attention over 5-layer windows via the
     eager mask. Share = (lpctx − lpknocked)/(lpctx − lpnomem). Not run on the 2B (no answer→Pepper edge in DeltaNet).
  3. Subject-alone safeguard (Qwen2.5): block attention to the attribute from every query after the experience, at
     all layers, then patch the subject. The sanity cell (blocked, no patch) should be |mean| ≤ 0.10, else read the
     net share.
- **Flags.** Full flip < 1 logit or ctx gain < 1 nat: the item is excluded from the means. More than 2/7 flagged
  makes that model undetermined.
- **Rules.** *Fact at subject*: subject patch ≥ 50% at its best cell, CI lo > 0.25; on Qwen2.5 also subject knockout
  ≥ 50% and attribute knockout < 25% everywhere. *Pointer-suspected*: subject < 25% at every cell (every CI hi < 0.5),
  attribute patch ≥ 50%, and on Qwen2.5 attribute knockout ≥ 50%. Otherwise *in between*. If subject-alone is ≥ 50%
  (CI excludes 25%), Qwen2.5 is not pointer. A pointer verdict skips the `subj` grid ONLY if the analytic pre-pass
  also shows no gain (the best cell's mean ΔlogP(target) CI includes or lies below 0). If the models disagree, each
  decides its own grid.
- **Predictions.** B1: subj clean use is above the push ceiling (core_v1 ≤ 0.15). B2: balanced yes/no > 0.5, CI > 0.
  B3: the wrongpos control gives no gain. B4: no cross-binding leak (dog/cat). B5: reinst ≥ subj on Qwen2.5 and
  weaker on the 2B (reinstatement = upper bound ≈ ctx at a shifted position). B6: unrel_same ≥ 0.95.
- **§9 rules.** Holds = B1∧B2∧B3∧B6. Push in disguise = B1∧¬B3. Orphaned = nothing beats the push while the
  pre-pass ΔlogP rises. Reinstatement wins = reinst > subj on the same model. Stage-0 pointer → reinst + controls only.
- **Config by rule.** 2B sets {4,6,…,16} + bands {4–8, 6–10, 8–12, 10–14}; Qwen2.5 round((l+1)·28/24) − 1; α {0.5, 1,
  2}. Pick: Δspec > 0; score = rank(Δspec) + rank(balanced margin); ≤ 8 subj cells incl. the best H1.
- **[R] Stage 0.** *Pointer-suspected* on both models. Subject patch 0.00 everywhere; attribute patch 1.00. Qwen2.5:
  attribute knockout 0.59 [0.41, 0.81] in the last window, subject knockout 0.00. Subject-alone: the blocked baseline
  already costs 0.45 [0.40, 0.49]; net gain 0.00. The answer reads the value late (2B by block 19; Qwen2.5 window
  23–27). Pre-pass: NO GAIN on the 2B (subj skipped); a nominal GAIN on Qwen2.5, matched by the rand/swap controls,
  so subj ran there.

### Addendum 1 (2026-10-11): implementation before harvest
- **H1**: the delta at the last token of the first subject alias in each follow-up, every block, averaged over the
  item's harvest follow-ups (an alias-less secondary follow-up is skipped; an alias-less primary one fails the job).
- **H2**: "<experience> My dog is lovely." vs "I have a dog. My dog is lovely.", taken at the 2nd mention's last
  token. **H3**: H2's raw `with` state rescaled per block to the norm of H2's delta.
- **Sites**: main = the last token of every alias in the user turn (G-match); +1 = also the next token; gen = also
  the subject in the TF assistant prefix.
- **Controls**: wrongpos (a user token near the end), rand (norm-matched), swap (core_v1 partner), push@1 (2B).
  G-key: mu + PCA-256 whitening, q0.95 over 100 generic prompts. Check: |delta| ≤ 10× median residual norm.

### Addendum 2 (2026-10-12): arm R + Qwen2.5 generation
- Design note: plain R under the causal mask == the sentence as a prior user turn. So R runs in 3 forms
  (`run.py: harvest_R, gen_batch`; `bind.splice_keep`):
  - **R-full**: head (system block) + the whole experience turn `<|im_start|>user\n{exp}<|im_end|>\n`, K,V at every
    layer at stored RoPE positions. The question follows at n_h + n_s. Check: == prior-turn text within 0.01.
  - **R-min**: head + the attribute position + the next token (the full stop), K,V at every layer, stored positions.
    The question still starts at n_h + n_s (a gap). Cache = n_h + 2 entries.
  - **R-orphan**: the turn harvested AFTER a fixed unrelated exchange (basil Q+A, `gen.orphan_exchange`). The
    exchange is dropped and the segment keeps its harvest positions; the question starts at n_h + n_u + n_s.
  - **R-swap**: the partner's R-full under this item's gate. **Leak baseline**: ungated R-full on 50 unrelated × 7.
  - Gating: G-match on the subject alias in the user turn.
- Conditions (Qwen2.5, DelftBlue gpu-a100-small): nomem, ctx, R-full, R-min, R-orphan, R-swap, subj1
  (H1 @ blocks 5–10, α 0.5; decisive), subj2 (H2 @ 5–10, α 2), ctrl:wrongpos/rand/swap/placebo (placebo = "By the way,
  I had a coffee this morning."). Subj is added in prefill only. Secondary items: nomem, ctx, R-full, subj1.
  Cross-binding pairs: dog/cat, sister/brother, best friend/boss.
- Rules: B1 = subj1 clean CI > 0.15 and subj1 − wrongpos > 0. B2 = gain CI > 0, acc > 0.5. B3 = wrongpos − nomem not
  > 0. B4 = leak CI not > 0. B5 = R-full − subj1 ≥ 0. B6. Reinstatement wins = R-full − subj1 CI > 0.
  R-min and R-orphan are reported (expected: R-min ≈ R-full; nomem < R-orphan < R-full).
- **[R]** B1 ✗, B2 ✗, B3 ✓, B4 ✓, B5 ✓, B6 ✓ → **"orphaned; reinstatement wins"** (R-full − subj1 +0.87 [0.71, 1.00]).
  Use prompts: subj1/subj2/controls 0.00. ctx 0.90. R-full 0.87 [0.71, 1.00]. R-min 0.86 [0.63, 1.00] (yes/no 0.86).
  R-orphan 0.80 [0.56, 1.00]. R-swap 0.00 (writes the other value in 0.41 of answers). Direct question: R-full 0.91,
  R-min 0.66 [0.40, 0.89], R-orphan 0.97. Ungated R-full unrel_same 0.064 vs gated 1.000. Self-claim ≈ 0.01 (= ctx).

### Addendum 3 (2026-10-12): arm R on the 2B hybrid
- Qwen3.5 has no system block, so the head is empty and the experience starts at position 0. DeltaNet blocks have
  recurrent + conv (last 3 inputs) states and no K,V (`harvest_R_hybrid`, `bind.kv6_cache`, `expand_cache`).
- **R-full-state**: the full session cache (K,V at the 6 attention blocks + DeltaNet conv + recurrent), as in prompt
  caching. Check: == prior turn (smoke max |Δlogit| 2.5e-5).
- **R-full-kv6**: K,V at the 6 attention blocks only; DeltaNet = head-only (fresh, zero) state; q at n_head + n_seg.
  **R-min-kv6** (attr + full stop), **R-orphan-kv6**, **R-swap-kv6**; leak = ungated R-full-kv6. No subj arm (skip).
  Predictions: R-full-state ≈ ctx; 2B kv6 < Qwen2.5 R-full. Ran on gpu-v100 (10 GB slice OOM).
- **[R]** Use / direct / yes-no: ctx 1.00 / 0.94 / 1.00. R-full-state 0.96 [0.89, 1.00] / 0.94 / 1.00.
  R-full-kv6 0.50 [0.23, 0.77] / 0.23 / 0.61. R-min-kv6 0.67 [0.41, 0.91] / 0.20 / 0.50. kv6 refuses direct
  questions ("I don't know your best friend's name! Could you tell me…"). 2B R-full-kv6 − Qwen2.5 R-full −0.37
  [−0.66, −0.09]. Ungated kv6 unrel_same 0.007.

### Addendum 4 (2026-10-12): what the DeltaNet state contributes (2B; exploratory)
- **R-min-state** = R-min-kv6 entries + the DeltaNet conv + recurrent state after the experience turn. Honest label:
  2 attention entries + 18 layers of fixed-size state (≈16×128×128 per layer; ≈4.7M numbers, ≈19 MB fp32), not a
  2-token memory.
- **R-state-only** = the DeltaNet state and no segment attention entries; the question still at n_head + n_seg.
- **R-orphan-state** = min entries + the state after the basil exchange + experience (contaminated state).
  **R-swap-min-state**. Anchors rerun: nomem, ctx, R-full-state, R-min-kv6. Leak = ungated R-min-state.
- Checks: entry counts (state-only = head only; min = head + 2); the DeltaNet layers of min/state-only == R-full-state's.
- Expectations: "compact hybrid memory" if R-min-state − R-full-state CI ∋ 0 and point > −0.1. R-min-state > R-min-kv6
  on direct and yes/no.
- **[R]** R-min-state 0.97 [0.91, 1.00] / direct 0.89 / yes-no 0.93. R-state-only **0.00 / 0.00 / 0.93**
  [0.86, 1.00]: recognition without recall. R-orphan-state 0.99 / 0.91 / 1.00. R-swap-min-state 0.00 (wrong value
  0.47). R-min-state − R-min-kv6: +0.30 [0.06, 0.57] use, +0.69 [0.51, 0.86] direct. R-min-state − R-full-state
  0.01 [−0.06, 0.09] (compact claim holds).

### Addendum 5 (2026-10-13): text control + two-fact collision
**(1) Text controls** (`--gen-set text`). **T-min**: [user: "Pepper."][user: question], the value without its leading
space. **T-full**: [user: exp][user: q] (== R-full / R-full-state). Qwen2.5: nomem, ctx, T-min, T-full, R-min, R-full.
2B: the same with R-min-state / R-full-state. Prediction: T-min ≈ nomem ≪ R-min; "if T-min ≈ R-min the compression
claim falls".
- **[R]** T-min use 0.27 [0.06, 0.51] (Qwen2.5), 0.24 (2B); direct 0.00 / 0.11. R-min 0.87, R-min-state 0.97.
  T-min − R-min −0.60 [−0.86, −0.34], −0.73 [−0.93, −0.50]. **Compression survives (~3×).** T-full == R-full exactly
  → full reinstatement = prompt caching across sessions.

**(2) Collision** (`--stage collide`, `<model>_collide`). Theory: an order-based binding ID (Feng & Steinhardt; Dai).
Separately harvested segments all carry the "first pair" ID.
- Pairs: (a) different subjects: dog/cat, sister/brother, best friend/boss. (b) confusable: older sister Petra /
  younger sister Maria; first dog Pepper / second dog Rocky; friend from work Omar / from school Kevin; son's teacher
  Hugo / daughter's teacher Daniel; grandmother mother's side Esther / father's side Rosa. Total 8 pairs, 16 members.
- Per member: direct question, 2 use prompts, balanced yes/no (No = the partner's value as foil), TF readout
  logP(own) and logit(own) − logit(partner). Gates forced open (oracle); order A then B.
- Qwen2.5 conditions: nomem; ctx2 ("A. B. question"); **R2-joint** ([h][A][B] in one session; == T2, asserted);
  **R2-sep-overlap** (each alone, stored positions, overlapping; q at h + max(n_A, n_B)); **R2-sep-shift** (B's keys
  re-rotated by RoPE +n_A; q at h + n_A + n_B); **R2-min-joint**, **R2-min-sep-overlap**, **R2-min-sep-shift**
  (value + full stop each). The RoPE re-rotation equals recompute (unit test).
- 2B: every R2 uses the JOINT DeltaNet state (merging is impossible), so sep vs joint differ only in the 6 attention
  blocks: `R2-joint-state`, `R2-sep-overlap-jstate`, `R2-sep-shift-jstate`, `R2-min-*-jstate`. Probes with a fresh
  state: `R2-joint-kv6`, `R2-sep-shift-kv6`.
- Measures: correct clean use; **partner confusion** = the partner's value in the answer AND own value absent (regex);
  judges' wrong_value beside it.
- Rules: "IDs collide" = confusion sep − joint CI > 0 (confusable; per scheme). "Content suffices" = |sep − ctx2| < 0.1
  with the CI ∋ 0 (different-subject pairs). "Re-encoding helps" = correct joint − sep CI > 0.
- **[R]** Qwen2.5 confusable correct / confusion: ctx2 0.93/0.01, joint 0.89/0.05, sep-overlap 0.51/0.35,
  sep-shift 0.54/0.37, min-joint 0.61/0.20, min-sep 0.46/0.28–0.32. Different-subject sep confusion 0.26–0.28.
  2B: joint 0.89/0.05, sep 0.57–0.59/0.28–0.33, **min-joint 0.73/0.06**, min-sep 0.56/0.23–0.24; kv6 probes weak
  (0.19, 0.16). IDs collide +0.30 [0.13, 0.48] (Qwen2.5), +0.23 [0.07, 0.42] (2B). Re-encoding helps +0.38
  [0.21, 0.57], +0.29 [0.12, 0.49]. **Shift ≈ overlap (not positional).** Qwen2.5 min IDs-collide is n.s. (0.08
  [−0.08, 0.27]) because min-joint is itself confused.

### Addendum 6 (2026-10-13; Qwen2.5): (A) binding-ID repair, (B) G-key capacity; 6b correction + (C)
**(A) Repair** (`--stage idrepair`).
- ID directions, Feng's mean-difference method: 500 joint 2-pair contexts [h][user: "My {slot1}'s name is
  {v1}."][user: "My {slot2}'s name is {v2}."], seed 11. Entity = the subject phrase's last token + next; attribute =
  value + full stop. Δ_E(l), Δ_A(l) = mean(pair-2) − mean(pair-1) of the block-l input residual. Dai: the projection
  on the top-4 PC most correlated with order.
- Validation: 200 held-out contexts (seed 12, 400 queries). Feng mean intervention (pair 1 +Δ, pair 2 −Δ) at attr /
  ent / both; K,V recomputed at those positions only. Flip = the margin changes sign. Layer sets: all 28, thirds
  0–9, 10–18, 19–27. Pick = the largest swap-attr flip.
- Repair: +Δ_E/+Δ_A on segment B only, recompute its K,V, splice (overlap/shift; full/min). Controls: random (norm),
  −Δ, wrong tokens (6b: the sentence's first token + the token before the attribute, e.g. "My" and " is").
- Rules: "ID direction valid" = swap-attr flip − random CI > 0. "Repair works" = generated confusion (sep+ID − sep)
  CI < 0 AND below every control.
- **[R]** Flip rates: late (19–27) Feng 0.75 [0.69, 0.81], random 0.00, Dai 0.25; all layers 0.64; early/mid 0.00
  (mid 0.43 when swapped at both). Pick = late. TF: joint confusion 0.00 (margin 13.30); sep 0.38 (1.25); +ID 0.44
  (1.77); +rand 0.31; −Δ 0.38 (0.66). The best cell was min + Dai. Generated: 0.28 vs 0.30 (Δ −0.02 [−0.14, 0.11]);
  min-joint 0.19. **"Repair works" does NOT hold.**

**(B) G-key retrieval/capacity** (`--stage gkey`). 59 facts (12 hand + 47 pool), N ∈ {10, 20, 59} nested; 2-token
values keep the first sub-token + next. **K1** = whitened (pooled_w256) mean of subject-span residuals (`without`
runs of 2 follow-ups); **K2** = whole user text. Query = the running key at the last user token; top-1 above q0.95;
key layer 14 (5, 10, 19 reported). Generation N = 59 (7 primary + 20 pool, direct only). No pass/fail.
- **[R]** K1 correct at N = 59: 0.92/0.88/0.88/0.85 (layers 5/10/14/19); K2 0.42–0.61; top-2 ≈ 0.9; miss 0.00;
  false fire 0.02–0.04. TF gain: ctx 9.48, oracle 9.47, G-key 8.32, swap −2.62 (wrong value 0.86–0.90). Generation:
  primary use oracle 0.87 vs G-key 0.49 (wrong value 0.16); pool oracle 0.62 = the later-corrected "pool gap".

**(C) 6b, 2B: conv buffer vs recurrent state** (`--gen-set decomp`). "Fresh" = zero conv / zero recurrent.
`R-state-only-rec` (recurrent only), `R-state-only-conv` (conv only), R-state-only (both, anchor), `R-min-state-rec`
(2 entries + recurrent). Rule: "recognition lives in the recurrent state" if R-state-only-rec TF balanced yes/no CI
lo > 0.5 AND within 0.1 of R-state-only.
- **[R] Holds.** TF yes/no / gain: R-state-only 0.93 / 4.24 nats; -rec 0.86 [0.71, 0.96] / 3.74; -conv 0.50 / 0.66;
  R-min-state 0.93 / 11.11; R-min-state-rec 1.00 / 10.79 (generation 0.97 use, 0.97 direct). R-min-kv6 TF +10.01
  nats with yes/no 0.50 (value present, verification at chance).

### Addendum 7 (2026-10-22): parts A–G for paper 2 (Qwen2.5, 2B, 9B TF, Llama-3.2-3B)
Code: `run.py` stages `probe`, `directions`; gen sets `refusal`, `pool`; `--items pool`; `--collide-set add7`;
`submit_add7.sh`. Known before: Afendulev et al. (Sept 2026) show recurrent-state recognition = gist familiarity.
- **A. Lure control.** Per primary item, 2 lures per kind × 2 verify templates (answer No); `config.yaml: lures`,
  e.g. dog Pepper: form Piper, Penny | associate Ginger, Salt | binding Jasper, Petra. 16 probes per item: 2 Yes,
  2 unrelated, 4 form, 4 assoc, 4 binding. FY_k = false-yes rate; balanced vs k = (accYes + 1 − FY_k)/2;
  "against lures" = form + associate. Conditions: 2B nomem, ctx, R-full-state, R-min-state, R-min-kv6, R-state-only,
  -rec, -conv; Qwen2.5/Llama nomem, ctx, R-full, R-min, T-min, T-full. Joint-state conjunction: **R2-state-only-J** =
  the joint DeltaNet state of [A][B] with no attention entries, q at h + n_A + n_B.
  Rules (2B R-state-only TF): "gist familiarity" = FY_form − FY_unrel or FY_assoc − FY_unrel CI lo > 0. "Item
  recognition" = not gist, both diffs within 0.1, balanced vs lures CI lo > 0.5. Conjunction: "state binds" =
  R2-state-only-J balanced (partner foil) CI lo > 0.5; "familiarity without conjunction" if the CI ∋ 0.5.
  - **[R] Item recognition**: FY form 0.00, assoc 0.04, unrelated 0.07; balanced vs lures 0.96 [0.88, 1.00].
    **Familiarity without conjunction**: R2-state-only-J 0.50 [0.40, 0.60] (27/32 yes on both names).
- **B. Refusal of the attention-only slice.** Directions: **d_known** (Ferrando) = famous (100) − made-up (100) names
  at X's last token in "Tell me about {X}."; **d_told** = mean over 52 non-primary facts of [ctx − nomem] at the
  direct question's last prompt token (block outputs). Told vector v(l) = α·d_told(l) on layer set S at the last
  prompt token, prefill only. α-selection on the 52 facts only: S ∈ thirds, α ∈ {0.25, 0.5, 1, 2}; score = Δ log P(the
  measure prefix | question) vs X; constraint ΔlogP(target) ≥ −1 nat; ties go to the smaller α. X = R-min-kv6 (2B) /
  R-min (Qwen2.5, Llama). Arms: X, X+told, X+told:rand, X+told:neg, nomem+told, R-min-state (2B), R-full (Qwen2.5),
  **R-min-kv-own+state-swap**, **R-min-kv-own+state-plac** (2B). Refusal regex reported.
  Rules: "missing told signal" = (X+told − X) direct CI lo > 0, beats rand and neg (lo > 0), use-prompt Δ ≥ −0.1 with
  hi ≥ 0. State: "generic response mode" if swap or plac ≥ R-min-state − 0.1; "item-specific state" if both are
  > 0.1 below with CI hi < 0.
  - **[R] Missing told holds (2B, Qwen2.5)**: 2B direct 0.23 → 0.83 (rand 0.26, neg 0.11); Qwen2.5 0.66 → 0.94.
    nomem+told = confabulation ("Alex"). Told projection at block 23: R-min-state 5.47 vs ctx 5.21 vs R-min-kv6 0.96.
    d_known shows no such pattern. **"Item-specific state"** by the rule, but own entries + swap state 0.71, + plac
    0.57, own 0.89, none 0.23: most of the lift is generic. Told makes yes/no say yes to everything (it licenses
    answering, not verification).
- **C. Pool scoping.** 52 non-primary facts; one templated use prompt per kind (person "Write a short birthday message
  to my {slot}."; pet "Write a short, funny poem about my {slot}."; attributes hand-written). nomem, ctx, R-full,
  R-min, T-min. Rule "prompt-type mismatch" = the pool R-min direct CI overlaps [0.40, 0.89] AND pool R-min use ≥ 0.75.
  **[R] Holds (Qwen2.5, Llama)**: use R-min 0.97 / 0.98 vs ctx 0.96 / 0.98, T-min 0.34 / 0.32; direct 0.65 / 0.51.
- **D. Stage 0 on 47 pool facts** (decided on the 38 person/pet). **[R]** Subject patch ≤ 0.02, attribute 1.00 on
  every model. Verdicts "in between" on the Qwen2.5 pool and Llama only because attribute knockout < 0.5 (0.48;
  0.05–0.06). 2B pool: pointer-suspected.
- **E. Llama-3.2-3B port.** `turn_pieces` from apply_chat_template; date pinned "26 Jul 2024"; EOS `<|eot_id|>`; 28
  blocks (layer sets carry over). Every value/lure/collide value is 1 token. Rules: "slice > text" (R-min − T-min CI
  lo > 0), "collision replicates". **[R] Both hold**: R-min 1.00, R-orphan 1.00, T-min 0.24 (+0.76 [0.46, 0.97]),
  R-swap 0.00 (wrong 0.33); sep confusion 0.41 [0.23, 0.61] vs joint 0.01. Llama yes/no is uninformative (T-full
  Yes accuracy 0.21).
- **F. Qwen3.5-9B, TF probe only** (32 blocks, full A100). Rule "2B split replicates" = R-min-kv6 ΔlogP ≥ 0.8×ctx with
  yes/no CI ∋ 0.5, AND R-state-only yes/no CI lo > 0.5. **[R] Does not hold**: kv6 +9.83 vs ctx +10.80 (yes/no at
  chance), but a strong "No" prior (margin ≈ −3.9); the state gives the true name −1.27 vs −3.15 unrelated (+2.6 nats
  item-specific, too weak to flip).
- **G. Cheaper collision fix.** **R2-seq** = keep A's stored full slice and re-encode only B with it as prefix (≡
  R2-joint, asserted). **R2-seq-min** = A's 2 positions as prefix for re-encoding B, keep B's 2. **R2-blend15** =
  sep-shift with the 15% of B's positions with the largest key deviation (L2 at the 2nd attention layer: block 1
  Qwen2.5/Llama, block 7 2B) replaced from a re-encode (CacheBlend rule; a selective-recompute upper bound). 2B
  `-jstate`. Rule "cheap fix works" = R2-seq-min confusion CI overlaps R2-min-joint AND (seq-min − min-sep-shift)
  CI hi < 0. **[R] Fails on every model**: seq-min reaches min-joint (2B 0.09 vs 0.06; Qwen2.5 0.20 vs 0.20; Llama
  0.23 vs 0.17), but the paired diff is n.s. (2B −0.15 [−0.33, 0.02]). Joint 2-position slices stay confused on
  pure transformers (0.17–0.20). blend15 picks template tokens and barely helps. Full-slice R2-seq = joint costs one
  pass over the new memory only.

### Addendum 8 (2026-10-25): parts 1, 2, 3, 6 (Qwen2.5, 2B, Llama; fp32)
Code: `--collide-set add8|add8pool`, `--probe-set add8`, `--dir-set add8`, gen sets `refusal8`, `comply`;
`bind.add8_plan, place, subject_phrase, form_lures, binding_lures, pool_split`; `add8_data.py --check`;
`submit_add8.sh`. Partitions: Qwen2.5 and 2B on full gpu-a100, Llama on gpu-v100.
- **Part 1a, tag at storage.** Harvest in [head][user: "P S"]; drop prefix P's tokens; keep header + S + end-of-turn;
  place in sep-shift (`bind.place`). Asserted: kept length = standalone, tokens equal except at most S's first.
  **Dummies** (`dummy_pairs.yaml`): "My gardener's name is Brian." + mechanic Keith, postman Derek, locksmith Trevor,
  tailor Colin, barber Graham, florist Janet, butcher Gary; dummy (2k + r) mod 8. Placebo = the coffee sentence. Conditions: **R2-sep-tagged** (A alone, B after its dummy), **R2-sep-bothtagged** (both after
  dummies), **R2-sep-placebo**, `R2-min-sep-tagged/-bothtagged/-placebo`; anchors ctx2, R2-joint, R2-sep-shift,
  R2-min-joint, R2-min-sep-shift. Check: tagged with an empty prefix == sep-shift (≤ 0.01). 2B `-jstate` (the
  restored DeltaNet state is always the plain joint one). Rule "the tag causes the swaps" (full, confusable):
  tagged − sep-shift CI hi < 0; tagged overlaps joint; bothtagged − sep-shift CI ∋ 0. Prediction: tagged ≈ joint,
  bothtagged ≈ sep, placebo ≈ sep.
- **Part 1b, entity side.** **min+subj** = 2 positions + the stored sentence's subject span (the words between "My "
  and the first relation: "younger sister", "first dog", "friend from work", "son's teacher", "grandmother on my
  mother's side", "dog", "best friend", "boss"). **R2-minsubj-joint**, **R2-minsubj-sep-shift**. Rule "the entity side
  carries the tag" (Qwen2.5 and Llama each): minsubj-joint − min-joint CI hi < 0 AND minsubj-joint overlaps R2-joint.
  13 conditions × 16 members × 3 prompts × 5 = 3,120 answers.
- **Part 2.** 52 non-primary facts (kinds: person 30, pet 8, attribute 14). **2a** TF probe on the pool: yes 2,
  unrelated 2, form 4, binding 4, no associates. Form lures by `FORM_RULE` (same first letter > common prefix >
  Levenshtein > list order); binding lures = the next same-list pool facts' values. Rule on 2B R-state-only over 38
  person/pet: gist / item recognition (vs form lures) / in between. **2b** held-out refusal: `pool_split.yaml` seed
  8, 26/26 facts (person 15/15, pet 4/4, attribute 7/7); d_told and its (S, α) built on half A; arms on half B: X,
  X+told, :rand, :neg, nomem+told, X+comply, X+comply:rand, R-min-state, own+state-swap (the next half-B fact of the
  same kind), own+state-plac. Direct question + pool use prompt × 5. **2c** 34 pool pairs (17 confusable "older/younger
  {sister…daughter}" + "first/second {10 pets}"; 17 different), 68 members, the same 13 conditions; 8,840 answers +
  3,536 yes/no per model.
- **Part 3, told vs comply.** `comply_pairs.yaml`: 66 personal/general pairs (e.g. "What's my blood type?" / "What's
  the most common blood type?"). Behaviour check: personal refused and general answered under nomem (60 tokens,
  `COMPLY_REFUSAL_RE`); keep only pairs that behave. d_comply = mean(general) − mean(personal) at the last prompt
  token. (S, α) chosen by the same rule on half A. X+comply:rand seed 4321. Also on the 7 primary items (`--gen-set
  comply`). Rule (direct, half B): "told is specific" if X+told − X+comply CI lo > 0; else "generic comply" if
  within 0.1; else in between.
- **Part 6, Llama yes/no wording.** (i) "Based on what I told you earlier, {q} Answer yes or no." (ii) two-choice
  "Is my dog called Pepper or Max? Answer with the name only.", both orders, scored TF on the first token of the true
  name vs the foil. Rule "Llama verification usable": ctx and T-full ≥ 0.9.
- **[R] Results.**
  - 1a storage order. Confusable confusion, sep → B stored second (joint): Qwen2.5 0.37 → 0.09 (0.05), pool
    0.22 → 0.01 (0.00); 2B 0.33 → 0.15 (0.05), pool 0.16 → 0.01; Llama 0.41 → 0.09 (0.01), pool 0.20 → 0.01.
    Bothtagged 0.21/0.19/0.16 (pool 0.20/0.09/0.03); placebo 0.26/0.19/0.31 (pool 0.07/0.05/0.13). The rule holds on
    Qwen2.5 (both sets) and on the 2B add5 pairs; it fails on the 2B pool and on Llama (always the bothtagged clause).
    Reading: being stored second, after a different fact, is most of the fix, and part of it is generic.
  - 1b holds on both pure transformers: minsubj-joint 0.03 vs min-joint 0.20 (Qwen2.5), 0.01 vs 0.17 (Llama); correct
    0.93 / 0.90 vs 0.60 / 0.52. From separate harvests no help (0.34, 0.29). → the entity side of the tag sits at the
    subject positions and is written only by joint encoding.
  - 2a item recognition at scale (FY form 0.01 vs unrelated 0.03; balanced 0.94 [0.90, 0.97]). 2c IDs collide and
    re-encoding helps on all 3 models (sep − joint 0.22, 0.16, 0.20). 2b told lifts half B (2B 0.12 → 0.76, Qwen2.5
    0.72 → 0.89, Llama 0.62 → 0.88), but "missing told" FAILS on the 2B and Qwen2.5 (use prompts drop −0.14
    [−0.25, −0.02], −0.18); Llama holds. State: item-specific again (swap 0.65, plac 0.60, own 0.88).
  - 3: cos(told, comply) 0.4–0.65 (mid/late). Half B, comply vs told: 0.67 vs 0.76 (2B), 0.80 vs 0.89, 0.73 vs 0.88.
    **2B = "generic comply"** (the report printed "specific" from a CI lo of 4e-18; corrected), Qwen2.5 +0.09
    [0.03, 0.15] and Llama +0.15 = told specific. Primary 7: generic comply on the 2B and Qwen2.5; in between on Llama.
  - 6: two-choice makes Llama usable (ctx 0.98, T-full 0.93; R-min 0.95 vs T-min 0.68). Wording (i) fails. 2B double
    dissociation: R-min-kv6 yes/no 0.50 but two-choice 0.91; state-only yes/no 0.93 but two-choice 0.48 → familiarity
    in the state, selection in the attention entries.
  - A panel-coverage bug was fixed (a82244b, `--relink-only`). No verdict changed (≤ 0.02).

### Addendum 9 (2026-10-30): storage order with N = 3 and N = 5
- `--collide-set add9` (`stage_collide9`, `harvest_group`; `bind.group_plans, group_layout, group_confusion,
  order_rule`); `groups9.yaml` from `add9_data.py` (FEMALE9 48, MALE9 48, PET9 71 new names, none used before;
  `random.Random(9)`; different-subject slots shuffled with `Random(10)`). 30 groups / 114 members: N3 confusable
  12 (eldest/middle/youngest × 7 persons; first/second/third × 5 pets), N3 different 6, N5 confusable 8 pet groups,
  N5 different 4. Templates: "My {subject}'s name is {v}.", direct, prefix, one use prompt. No yes/no.
- Layout: all N members in shift layout (q at h + Σn_k), RoPE re-rotated; head from member 1. Conditions: **ctxN**,
  **RN-joint** (== T-N), **RN-sep-shift** (each alone = "first"), **RN-sep-ordered** (member k harvested after
  dummies 1..k−1 in the same user turn, dummies dropped), **RN-sep-ordered-rev** (the same slices in reverse layout),
  **RN-sep-placebo** (k−1 placebos: coffee; "The weather was quite windy today."; "I went for a long walk after
  lunch."; "I still need to do the laundry tonight."). Dummies = the first 4 of `dummy_pairs.yaml`. 2B `-jstate`.
  6,840 answers + 684 TF rows per model. TF margin = logit(own) − max over the others.
- Rule "storage order scales" (confusable, per N, per model): ordered − shift CI hi < 0 AND ordered overlaps joint.
- **[R]** Confusable shift → ordered (joint): N3 Qwen2.5 0.47 → 0.09 (0.01) holds; 2B 0.38 → 0.04 (0.01) holds;
  Llama 0.41 → 0.09 (0.00) fails (2nd clause). N5: 0.54 → 0.22, 0.55 → 0.23, 0.51 → 0.30, all fail clause 2.
  Different-subject: ordered 0.00–0.03 everywhere (holds in 6/6). Rev changes −0.02 to +0.03 (the stored order
  counts, not placement). Placebo helps less (N3 0.27/0.20/0.34). The N5 residual is between neighbours (positions
  3–5; close tags blur, as Feng predicts).

## 2. literature.md (17 papers, groups A–F) — claim / what remains ours
- **A. Reliability of steering** (Tan et al. NeurIPS'24; Braun 2025; Subbiah et al. 2026; Lindsey 2026). Our
  failures are normal steering: ~half of prompts move the wrong way; Yes/No tilt (→ balanced probes); steerability is
  per concept; whole-answer traits steer, one-slot content doesn't; expression vs coherence (core_v1 α 2: 41–61%
  incoherent); combining loses ≥ 15%. Lindsey: best layer ≈ ⅔ depth, later = "say the word", overdose, countries
  hardest, injected states read as the model's own (≈ self-claim).
- **B. Conditional/context/user steering** (CAST; ICV; BiPO; CoS; NPM; Suri). CAST = our gate; ours = a condition from
  ONE experience, whitened keys, q0.95, a many-pair store. ICV: context → vector carries style (C2 expected); "the
  attention mix depends on the query" = our theory's core. BiPO: raw differences miss the behaviour (trained fix).
  CoS states concept-vs-direction + a weak negative λ (must cite; needs the context present). **NPM (closest)**:
  retrieved steering vectors as *procedural* memory, declarative assumed to need text; ours = the declarative side.
- **C. Where facts live** (Geva; ROME; MEMIT). Subject enrichment → relation → question-driven upper-attention
  extraction (attribute rank ≈ 1,000 at the subject). ROME/MEMIT write optimised values into mid-layer MLPs at the
  subject; late edits regurgitate. Our store = ROME's maths in activations with observed values → no usable fact.
- **D. Negation** (Jang 2022): negated prompts ≈ the original (inverse scaling) → "I can't stand jazz" ≈ jazz (C5).
- **E. Binding** (Feng & Steinhardt ICLR'24; Dai et al. 2024; Prakash et al. ICLR'24 entity tracking; + Geva re-read).
  Feng: a binding-ID vector at the entity (+ next token) AND at the attribute; mean intervention flips binding (0.99 →
  0.00); position-independent; close IDs are confused; the query mechanism is open. Dai: the ID is an *ordering* ID →
  context-relative ("2nd pair here"), so a tag written in session 1 means nothing in session 2. Prakash: 72-head
  circuit (structure readers, position detectors, position transmitters, value fetchers); binding = transporting the
  *position*. "Whose" (speaker) is positional too. Geva route: the attribute loaded onto the subject's own position →
  the "subject-position write" test (update 2026-10-13: tested, failed).
  - **"Why a push can't bind" table.** To the *question*: value fetchers need a position to fetch from; the push sits
    at the query → reinstate positions. To the *subject*: a shared order tag (in context) or subject-position loading
    (weights); the push is pooled and tags can't be stored → reinstate a contiguous segment, or write at the subject
    early-middle. To the *owner*: turn structure; a vector has no "I" → reinstate inside a user turn. Consequences:
    composition by addition doesn't bind (concept vectors = keys/tags, not summands); "the partner position is the
    thing to supply".
- **F. FwPKM** (Zhao & Jones, Sakana 2026): trained-from-scratch PKM (512² slots, top-8) with delta-rule fast weights
  and a next-token target; re-reading 2–4× raises NIAH < 10% → > 70%; the gate ≈ 0 under full attention; retention
  flushes. Supports P5 "writes training-free, reads need training" (→ a trained read adapter). Must cite.
- **Theory.** In context the effect is recomputed per query (ICV, Geva); a stored shift is one fixed average, so only
  question-independent content survives: leanings (narrow α), not facts ("say Pepper" only late, C6; floods, C3) or
  premises (C4). **Concept vs direction** (what / which way / whose): opposite keeps direction; centroid and `without`
  keep concept → dislikes flip (opposite +0.23 / +0.25; centroid −0.38 / −0.37 every item; without −0.25 / −0.31;
  2B / 9B; panel centroid −0.42 to −0.53).
- **Prior vs ours.** NPM procedural → ours declarative/user. ROME maths in activations, observed values. C1 = CAST +
  one-experience whitened keys + many-memory store. C2 nothing new. C3/C4/C6, C5: ours = the systematic pre-registered
  2-scale demonstration. Binding: untested before Exp 1.

## 3. literature-exp1.md (36 papers, 2026-10-13; updates 10-21) — five findings
| # | Finding | Verdict | Closest |
|---|---|---|---|
| 1 | Recall reads the value late from the attribute position; the subject holds no copy | Known (we add the chat setting + a GDN hybrid) | Lookbacks, Entity Tracking, Retrieval Head |
| 2 | A 2-position gated slice carries the fact (0.86 vs 0.87); the same tokens as text 0.27 | Partly known: sparse KV memory exists only with training; slice-vs-text is new | Memory3, KBLaM, Gather-and-Aggregate, Michalak & Abreu |
| 3 | Slices harvested after unrelated content work at other positions | Mostly known as a property; the cross-session framing is ours | Prompt Cache, Feng, Memory3, PIC |
| 4 | State = recognition w/o recall; KV = slot-filling but refuses direct questions; both = ceiling | Recall half known (DASC on Qwen3-Next: zeroing KV → NIAH 0.000, zeroing the state → 0.990). Recognition no longer new (Afendulev 2026) | DASC, Michalak, Bick, LatentPort, Hymba, **Afendulev** |
| 5 | Separately cached facts swap (0.26–0.37), even across subjects; shift doesn't help; joint fixes | Failure class known; the swaps + binding-ID explanation are new | CacheBlend, Prompt Cache, EPIC; Feng, Dai, Lookbacks |
- **A. Hybrids/recall.** Michalak & Abreu: no attention → 0% NIAH; "Only-Keep" 85.7% vs 50.3% (in-session analogue of
  R-min). Bick G&A: Gather heads summarise a segment into its last token (→ why the full stop matters; our inference).
  Systematic Analysis: recall improves below 3:1 (Qwen3.5 is 3:1). **LatentPort**: same-model restore exact; KV-only
  NLL 3.115 ≈ empty 3.246 (native 2.161); a wrong donor state is worse than none. LinearKV: any chunk's state works.
  Hymba: "snapshot vs fading gist". State Soup: separate states < joint. S0 Tuning: state = "low-magnitude directional
  bias". WriteSAE: out-of-context targets never reach top-1. Log-Linear: pure GDN can recall → our 0.00 = division
  of labour, not capacity. JRT does NOT say recognition is cheaper.
- **B. Cache reuse.** Prompt Cache: discontinuous positions fine; **"scaffolding" = our joint encoding**. CacheBlend:
  missing cross-attention, explicitly not positions; recomputes ≈ 15% high-deviation tokens; no swaps. EPIC: attention
  sinks, recompute k ≤ 32 initial tokens. Memory3 (trained, 8/128 tokens per head); Remark 3 = why a slice beats its
  text. KBLaM: weak on exact names. No cache paper mentions binding/swaps; EPIC's "each chunk thinks it is at the
  start" ≈ our "each fact is the first pair" (narrows CacheBlend).
- **C. Binding.** Lookbacks (Prakash 2025, ICLR'26): the attribute token holds payload + address (character/object
  OIs) side by side → the 2-position slice carries value + tag, which explains both cross-session binding and
  collisions; the answer is fetched late (layers 52–56/80). Feng: IDs numbered by order; position independence; close
  IDs ≈ 50%. Dai: an ordering ID "no matter bound or not", found at the query entity token. Entity Tracking App. J:
  "enumerating from the start token" → in separate sessions every fact is #1. Cross-subject swaps are predicted by
  lookbacks (the address holds OIs, not identities). **Wording caveat:** say "the subject carries no copy of the
  value", NOT "nothing the answer needs" (it may carry an order-ID pointer; the collision shows a pointer matters).
- **D. Familiarity.** "Bogacz" PDF = Cortes et al. 2007 (one Hopfield net, two read-outs; capacity N² vs N): our analogy.
- **Added 10-21.** **Afendulev et al. (arXiv 2609.04434)**, split-prefill/state-swap on Qwen3.5-4B/9B/27B, Falcon-H1:
  KV-only retrieval 0.64–0.98 of full, recurrent-only 0.00; language/persona via the state (KV-only "names the
  language but answers in English"); recurrent-only TPR 0.93–0.98 but DRM lures 0.87–0.89 (gist). **Ferrando et al.**
  known-entity latents steer refusal (→ add7 B; not the cause). Also: Gottesman & Geva, Brzezinka, Ruscio & Thompson,
  Kadavath, Lindsey "Biology", Arditi; Cartridges (trained dreaming-into-cache), Gist/ICAE/xRAG etc. (trained),
  LLMLingua, SnapKV/H2O; **Memory Inception** (training-free KV banks > CAA; must cite); KV-State Grafting.
- **Positioning line, paper 2.** Trained methods compress by optimisation; eviction picks positions per query
  in-session; Memory Inception and grafting reuse KV without training for steering or whole blocks. Ours keeps 2 of
  the model's own positions, chosen without a query, across sessions, and explains which positions carry the value and
  why separate slices collide.
- **Still ours.** (1) Cross-session recognition from a stored state + 2 positions, with the conv control and the TF
  split (kv: 10.0 nats / yes-no chance; state: yes-no 0.93 / 4.2 nats); add7: item-specific (rejects lures, 0.96),
  conjunction failure (0.50), the told/licence refusal mechanism. (2) The full dissociation; the KV-only refusal
  qualifies DASC. (3) The training-free 2-position slice ≈ 3× text. (4) Swaps across subjects + the binding-ID link
  (write "consistent with Feng", not "caused by"). (5) The read site in chat on a GDN hybrid (setting only).
  **Cannot claim:** cross-session KV reuse is new; joint encoding is a new fix; recall needs attention in hybrids;
  the subject "carries nothing".
- **Must cite** (beyond the table's closest papers): F4 + Systematic, GDN, S0, Cortes; F5 + HYPIC/LinearKV, Entity
  Tracking App. J; state-as-memory: State Soup, LatentPort, Marconi, Tail-Replay.

## 4. core_v1 (PREREG 2026-10-07; Qwen3.5-2B V100 + 9B A100, both fp32, thinking off)
- **Design.** Read layers: 2B {20, 21, 23} (linear, linear, full); 9B {27, 28, 31} via round((l+1)·32/24) − 1.
  Write: one entropy-weighted pooled moment per follow-up (template tail excluded), delta rule, isolated memory per
  item. Facts store plain (with − without); preferences store the **opposite** (with − counter); `without`/`centroid`
  (with − mean of 8 same-frame alternatives) are mechanism references. Keys pooled_w256, hard threshold q0.95
  per item and layer, two-pass gating, h ← h + α·gate·M·q. α ∈ {0.5, 1, 2, 3}. Controls at α = 2: rand, swap,
  placebo (coffee), gate_on. Items: 15 leanings, 8 dislikes, 6 one-of-many, 12 facts (xlayer_v1's), 50 unrelated.
- **Measures.** Preferences: `j_lean` (+1 toward, −1 away, 0 neutral/incoherent; primary). Facts: `use_judge_clean`
  on 2 use prompts (primary), `mention_unclean`, TF ΔlogP and Δspec. Yes/no balanced, dmY/dmN. Selectivity:
  unrel_same, open rate, KL. Logit lens per block. Exclusion: ctx lean − nomem lean < 0.3 → flagged.
- **Rules.** C1: every gated condition unrel_same ≥ 0.95 and gate_on lower (non-overlapping). C2: leaning gain CI > 0
  at α = 1 AND 2, each control ≤ ⅓ of mem@2 with a paired CI excluding 0. C3: at every α, mem ≤ ½ ctx (CI of
  ctx − mem excludes 0), and some α ≥ 2 has mention_unclean > nomem. C4: balanced gain CI ∋ 0 and the margin is a tilt
  ((dmY+dmN)/2 CI ∋ 0 or |dmY + dmN| < 0.25(|dmY| + |dmN|)). C5: on dislikes at the loop-matched α (the highest with
  loop ≤ 10%), opposite > 0 and centroid, without < 0. C6: median lens rank ≤ 10 only at depth ≥ 0.75, and > 100 at
  every block ≤ 0.5. C7: each decided per model.
- **[R] C7.md.** C1 ✓✓ (unrel gated 0.986/0.984; gate_on 0.057/0.101; false fire 0.035/0.040). **C2 ✗✗** (judge-
  sensitive per the panel). C3 ✓✓ (ctx 0.96/0.98 vs mem@0.5 0.04/0.08). **C4 ✗ 2B / ✓ 9B** (2B fails on a
  technicality). C5 ✓✓ (2B opposite +0.23 [0.05, 0.44] α 1; centroid −0.38). C6 ✓✓ (rank ≤ 10 at blocks 19–23 on the
  2B, 27–31 on the 9B; min rank at depth ≤ 0.5 ≈ 70k). Panel (2026-10-10): C3 and C5 robust; Llama-3.1 fails the
  sanity check.

## 5. dream_refs (Exp 2; PREREG 2026-10-11 + Amendments 1–2 + PrefEval addendum)
- **Question.** Can a 9B dreamer (sees only the experience) write the references for the 2B? Does concept × sign help?
- **Fixed design.** core_v1's 2B (layers 20/21/23; state.pt reused). 29 preference items (15/8/6). Dreamer: frame JSON
  {value, frame, key_words, category} → opposite (no new negation) → 8 alternatives (12 × 12 samples, by count) →
  paraphrases (lens only) → null_frame values. Prompts v3 FINAL. "Dreaming failed" (opposite fails; < 4 valid
  alternatives) excludes the item from the dependent variants.
- **Variants.** hand:opposite; gen:opposite; gen:centroid; gen:para (lens only); **cs:concept_x_sign** = s_ℓ ·
  gen:centroid_ℓ with s_ℓ = sign(⟨plain_ℓ, v_ℓ⟩); cs:oracle = true sign × gen:centroid. v_ℓ = the mean (like − hate)
  over 40 dreamer topics × 3 frames, follow-up "What should I do this weekend?". Grid: nomem, ctx + 5 variants ×
  α {0.5, 1, 2} = 17 conditions; 1 related + 5 ambiguous prompts; seed set 2 for nomem, ctx, the α = 1 gen cells.
- **Rules.** S1: per item and layer, pass iff cos(gen, hand) ≥ 0.8 and exceeds both nulls (null_swap, null_frame) by
  ≥ 0.2; indeterminate if a null ≥ 0.8; a layer holds at ≥ 80% of determinate items (≥ 15 determinate). S2: dislikes
  at loop-matched α, gen:opposite > 0 and gen:centroid < 0 (CIs exclude 0). S3: the valence reader accuracy ≥ 0.9 on
  23 signed items at some layer, with the signed-margin CI > 0. S4: cs on dislikes > 0 AND on leanings ≥ the lower
  CI of gen:opposite. S5: `without` needs no generation. Exclusion: ctx − nomem j_lean < 0.3. The original judge
  decides; a verdict that differs under the majority = judge-sensitive.
- **[R] (run dream_refs_20261015_1545).** S1 **fails** (opposite 10/29 pass per layer; centroid 12–13/25). S2
  **holds** under every judge (dislikes gen:opposite +0.190 [0.010, 0.395]; gen:centroid −0.386 [−0.552, −0.214], every
  item wrong). S3 **fails** (accuracy 0.43–0.52; it misreads mostly leanings). S4 **fails**: dislikes cs +0.386 [0.186,
  0.590] at α 2 (= oracle), but leanings −0.093 [−0.290, 0.093] → "the sign reader is the weak part" (oracle works).
  S5 holds. Flagged: hates_horror; dreaming failed: has_young_kids (centroid). Formal decision: scaffold_removed false
  (S1 fails), sign_readable false, concept_x_sign_works false. Practical reading: generated refs ≈ hand behaviourally.
- **PrefEval addendum** (explicit split, N = 200 stratified topic × polarity, seed 20261015: 132 avoid, 43 mixed,
  19 like, 6 other). Dreamed opposite/alternatives + neutral follow-ups (≥ 2 required; the test question is never used
  for writing or keys). 11 conditions, cap 200. Accuracy = helpful AND not violating (PrefEval's rubrics). P1:
  gen:opposite − nomem CI > 0 (all). P2: on avoid, cs − nomem > 0 AND gen:centroid − nomem < 0.
  **[R]** P1 holds (small): nomem 0.096, ctx 0.530 (+0.434), gen:opposite@1 +0.031 [0.011, 0.051]. P2 fails under
  the original judge (centroid not < 0; −0.026 at α 2 n.s.) but holds under gpt-oss/Gemma/majority (judge-sensitive);
  cs@1 on avoid +0.049 [0.018, 0.082]. Gate open on the questions 0.96; false fire 0.039.

## 6. dream_subj (PREREG 2026-10-22; Qwen2.5-1.5B fp32 eager, local, teacher-forced only, no judges)
- **Question.** When a slice is restored, does a direct question need the subject's own position (an entity-tracking
  two-step lookup), or does any extra position help ("more evidence")?
- **Conditions.** Harvest [head][experience turn], K,V every layer, stored RoPE positions, question at n_head + n_seg.
  nomem; ctx; R-full; **R-min** (value + next; == `bind.splice_keep("R-min")`); **R+subj** (subject last token + value
  + next); **R+fill** ("is" + value + next); **R+rel** ("name" + value + next); **R-subj2** (subject + value);
  **T-min** ("Pepper." as text); **T-subj3** ("dog Pepper." as text). Subject = the last token of the first alias in
  the experience sentence (e.g. "er" for hairdresser, "-law"). It must equal `bind.ea_positions`' entity token.
- **Items.** 7 primary + 38 pool person/pet = 45 in the rule (attribute facts reported separately).
- **Measures.** Primary **answer score** = log P(measure prefix "Your dog's name is" | direct question), summed.
  Also: target gain (saturates), TF yes/no, attention from the last token and the question's subject to the restored
  subject vs value entries.
- **Rule.** "Subject position matters" = lo > 0 for R+subj − R-min, − R+fill, − R+rel. "More evidence" = R+subj −
  R-min lo > 0 but fill or rel ≈ (CI ∋ 0, |point| < 0.25×). "Neither" = no 3-position beats R-min. Else in between.
- **[R] (results/dream_subj_20261022_1755/report.md) "Subject position matters (two-step lookup)".** R+subj − R-min
  +1.31 [1.01, 1.63]; − R+fill +1.24 [0.97, 1.52]; − R+rel +1.52 [1.15, 1.88]; R+fill − R-min +0.07; R+rel − R-min
  −0.21. It closes ≈ 81% of the R-min → R-full gap (R-full − R+subj 0.31 [0.18, 0.48]). Refusal first-token mass
  0.15 → 0.02. Yes/no 0.82 → 1.00. R-subj2 (2 positions) beats R-min +0.51 [0.19, 0.84]. The value's own probability
  is unchanged (≤ 0.1 nat): the subject decides *whether* to answer. Attention: layers 3–13 question "dog" → restored
  "dog" (0.07–0.17); layers 23–27 answer → name. R+subj − T-subj3 +3.53. Pool carries it (+1.47); primary 7 n.s.
  (+0.46 [−0.20, 1.13]) but yes/no +0.14. Can't separate pointer from familiarity; no generation.

## 7. Builder cheat-sheet
- Working slice: own K,V at every layer (Qwen2.5/Llama) or the 6 attention blocks + the recurrent state (2B; conv not
  needed), stored RoPE positions, inside a user turn, question at n_head + n_seg, G-match or G-key (K1, layer 14,
  q0.95). Keep subject + value + full stop. Value = attribute position, read late; subject = licence/lookup + the
  entity side of the tag.
- Several facts: store the k-th as the k-th fact (dummies dropped); else re-encode jointly (R2-seq = one pass over the
  new memory). ID-direction addition and blend15 don't fix it. KV-only refusal ≈ a generic comply gate (2B).
