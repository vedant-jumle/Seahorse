# Digest: Seahorse core/history docs (read 2026-11-10)

Scope: docs/where-we-are.md, where-it-fails.md, how-it-works.md, reference/original-method.md,
history/experiment-log.md, history/open-notes.md, history/next-moves-v0.1.md, reference/code-map.md
(+ experiments/core_v1/README.md, mentioned by code-map). All other .md links in these files point to
README/HANDOVER/dreaming/*/core-idea/metrics/glossary/literature/PREREG (covered elsewhere).
Seahorse = independent side project (never "thesis").

## 0. Freshness map (most current first)
- **where-it-fails.md**: last update 2026-10-29/30 (dream_bind add. 8 + 9). The only doc here that knows the
  KV-slice channel. Most current.
- **where-we-are.md**: 2026-09-30, updated 10-08 and 10-11. Stops at core_v1 + judge panel. STALE on: "facts
  and premises not at all" (slices fix both), "capacity beyond 6 unknown" (slices: 59 facts), "dreaming
  specified but not yet built" (exp1/exp2 done 10-12..10-30), ownership (fixed by user-turn slice).
- **experiment-log.md**: v0 → core_v1 + panel (10-11). No dreaming entries. Stages 4–6 (diag_attn,
  bench_v1_calib, diag_keys, samples_v2) only summarised in where-we-are.
- **code-map.md**: core_v1 era. STALE: no dream_* experiment folders; says dreaming "specified, not built".
- **how-it-works.md**: 2026-09-30, Qwen2.5-1.5B steer design (L26, α=2). STALE vs Qwen3.5 (layers {20,21,23}
  on 2B / {27,28,31} on 9B, α default 1 per layer after core_v1). Does not describe slices.
- **original-method.md**: v0/v0.1 spec, archival (delta rule still = single-memory write).
- **open-notes.md**: 2026-09-25 (+09-28 resolution of N1). Ideas N3, N5–N9 still unacted.
- **next-moves-v0.1.md**: archived (after v0.1, priority update 09-28). Its "still open A2" is now done by
  dream_bind.

---

## 1. where-we-are.md — plain-language project report
**Purpose:** step-back narrative: question, prior work, theory, test protocol, stages 1–10, meaning, limits,
next steps, appendix of real outputs (samples_v2).

**Question (§1):** can a frozen LM get a small fast memory formed from its own experiences, selective, fading,
changing later behaviour? Memory not transcript.

**Prior work (§2):** InfLLM/EM-LLM = uncompressed single-session filing cabinet; trained modules (Larimar,
Titans, TransMem): "writing simple, reading needs learning"; steering changes dispositions not knowledge (ReFT:
one trained vector reproduces 2,000 tokens); DeltaNet = same maths as M; neuro index/pattern separation/
comparator/neuromodulators. **Theory (§3):** store the *shift*, only surprise (delta rule), recall by
resemblance; Store/Read/Modulator (modulator never built); expected dispositional, not premises.
**Protocol (§4):** session 1 → fresh session 2; no-memory / memory / in-context ceiling; facts, preferences,
yes/no, leakage, capacity. Qwen2.5-1.5B (S1–6), Qwen3.5-2B (7+), 9B (10); bench 48 items + 60-fact pool.

**Stages (§5):**
- S1–S3 (v0, v0.1, diagnostics): ~¼ gap closed, mid-late layers, entropy gate helps, peanut concept priming;
  facts specific; contrastive fixes priming; reasoning never moves; unselective not weak; last write wins;
  yes/no "gains" = all-"No" artefact (details §5 below).
- S4 diag_attn: attention-/confusion-/random-picked tokens work equally → **unit = moment, not word**;
  attention = relevance not importance.
- S5 bench_v1: facts clean; "No" bias real (placebo pushes "No"); preferences hard even for ceiling (21/48
  flagged); emotional phrasing no absorption change.
- S6 key exp (diag_keys): gist key (mean of user-token states, hum removed, whitened) + threshold →
  leakage ~0, fact recall ~doubled (~6 nats vs ~3); least squares → order-free; 6 memories keep ~½ solo
  strength; soft threshold = hard; coin-flip recall no help.
- S7 think_v1 (Qwen3.5-2B): facts at L19–23; 3 layers {23,20,21} >> 1; short continuations 91% mention, 22%
  clean, rest loops; full answers ~4/44 clean vs ≥28 ctx; "84%" headline counted loops; thinking-only boost
  floods thinking ("Pepper Pepper…"), answer ignores; yes/no at chance; self-claims ("Teal is my favourite
  colour!"); vegetarian 12%→94% consistent, meat mentions 79%→3–15% (ctx ~half still mention meat); thinking
  budget 384 exhausted ~95% even nomem.
- S8 ref_v1: opposite = *which way* but loses *what* when both sides mention it (jazz → "podcasts");
  centroid = *what*, strongly, loses sign, floods; dislikes flip to likes under all refs except opposite
  ("Lo-fi Jazz", no_alcohol → "craft beer"); hum = topic/address only, fragments at high α; countries fail.
- S9 xlayer_v1: logit lens: fact shift's top word = the name at L20–23 (rank 1 of ~248k), rank ~30k–100k at
  L12–16; inject at L8 ~10% survives, L16 ~⅔; strong → garbles ("My name is Omar.", "I am an AI named
  Pepper", "It's a very Leeds."); yes/no unchanged. **Fact memory = intention to say a word.**
- S10 core_v1 (prereg, 2B + 9B; 15 leanings, 8 dislikes, 6 one-of-many, 12 facts; controls rand, swap,
  placebo, gate removed; judge): C1 selectivity ✅✅ (~98%; no gate 6–10%); C2 leanings ❌❌; C3 facts not
  clean (≤15% vs 96–98% ctx) ✅✅; C4 no premises ❌(2B technicality: worse at α=2) ✅; C5 dislike flip ✅✅;
  C6 late-word ✅✅. α=1 lean +0.2 (2B), +0.1 (9B) on −1..+1; α=2 61%/41% incoherent → coherent lean ≈0.
  Strong items: early riser, prefers quiet, vegan, vegetarian; none: celiac, lactose intolerant, has young kids.
  9B changed nothing but coherence.
- Panel update (2026-10-11): 34,200 answers re-judged by gpt-oss-20B, Gemma-4-12B, Llama-3.1-8B,
  Qwen3.6-35B-A3B (ref only). C3, C5 hold under all competent judges (gpt-oss/Gemma C5 centroid −0.42 to
  −0.53 vs −0.38). C2 fails each single judge, holds under majority votes; own judge strictest (62% coherent
  vs 72–76%); gpt-oss/Gemma 2B α=2 gain +0.36/+0.42 > α=1 +0.29/+0.24 → "narrow window" judge-sensitive.
  Llama excluded (92% "uses fact", target absent in 48%). Fleiss κ 0.90 coherence (facts); direction pairwise
  κ 0.45–0.69. No human labels.

**Lessons (§6):** keys > contents; key = gist (≈ index); unit = moment; selectivity not strength; measurement
fools; preference = *what / which way / whose* → concept × sign; steering suits whole-answer leanings; fact =
word → episodic channel; judge coherence.

**Limits (§7):** small scale; leanings modest; dislikes need opposite; countries fail; self-claims; capacity >6
unknown (STALE for slices); not self-driven (hand comparison); no premises; modulator paper-only.

**Next (§8, as of 10-11):** dreaming exps (1 subject-position state write → used facts; 2 model-generated
opposite/alternatives → concept × sign; 3 consolidation, needs 1); concept × sign with general like/dislike sign;
thermostat (target concept level, not fixed add); fair doses ~α=1, per-item doses, burst at thinking start;
hand facts to episodic channel; 60-fact capacity curve; rewrite weak items. Then neuromodulated memory (gates
whether/how strongly/fade/recall loudness at moment level); test "does emotion protect from overwrite?" (Biscuit
died vs named). Later: self-driven writing, mood SSM state with decay + bounded loop, combine with attention
channel, larger model. Paper-open items: second model family, human labels on subset.

**Appendix (samples_v2, job 873325, Qwen2.5 L26; 6 memories: veg, peanut, Norway, dog Biscuit, sister Ines,
deep-sea welder):** unrelated identical 24/24 (old 8/24; haiku match 0.04 vs thr 0.19; real ctx leaks
"Vegetarian waves"). Gist not fact ("welder at a welding equipment company", "What's her name?"); forced-prefix
/300: Biscuit nomem 0, old 2, current 7, six 1, ctx 294; Ines 0 (ctx 290); welder 0 (ctx 293). Veg no-meat /80
23/25/21(six)/35(plain)/76; peanut safe 68/72/41(plain)/80; Norway 0 (ctx 50), silent on grocery/coffee.
Biscuit ~5% next-word ≈ 6/100. Lesson: judge preferences by rates, not single-pair contrast.

**Cross-links:** how-it-works, where-it-fails, literature, metrics, dreaming/README.
---

## 2. where-it-fails.md — every failure, cause, fix (MOST CURRENT, to 10-30)
Header dates: written 10-01; +10-07 (13–16), +10-08 (xlayer/core), +10-11 (panel), +10-13 (dream_bind: 1, 2, 15
routed via slice; new 17), +10-19 (add. 6 + exp2: 7, 8, 9, 13, 17; new 18), +10-23/10-29 (add. 7/8 on 17),
add. 9 (10-30).

**All 18 (status — root cause):**
1. No premise — open for steer; WORKS with stored attention slice — a vector nudges, reasoning needs something to attend to.
2. Exact fact rare, gist common — open for steer; WORKS with slice — recall is a blend + target starts ultra-unlikely.
3. Concept not relation (peanut → PB) — fixed only with hand opposite — plain shift carries topic + point.
4. Preferences faint or crude — open — contrastive cancels too much; plain carries too much.
5. Fired everywhere — FIXED — keys alike, no bar (whitened gist + threshold).
6. Last write wins — FIXED — sequential delta overwrites (least squares).
7. Fires only where filed — open (partly fixed for facts: subject-span key) — filed under follow-ups not experience.
8. More memories weaker — partly open (slices hold to 59) — one shared table compromises.
9. Needs hand comparison — mostly fixed for effects (model writes refs); open for sign — shift = diff vs supplied counterfactual.
10. Too much strength breaks text — managed — added vector overpowers state.
11. Fooled by measurements — fixed — unbalanced Qs, template artefacts, narrow scores.
12. Generalisation unknown — partly open — few small models, few scenarios.
13. Dislikes → likes — fixed with true sign; open for reading sign — negation faint; most refs keep concept, lose direction.
14. Single-item prefs flood/do nothing — open — constant push at every token vs one-slot content.
15. Model claims memory as own — open for steer; fixed by user-turn slice — injected states read as own thoughts.
16. Memory during thinking floods thinking — open — constant push = overdose, not a thought.
17. Separately stored facts collide — open (joint re-encode fixes; tag repair fails; store-new-second mostly fixes) — every separately stored slice is "first pair" (storage order); binding written at subject positions by joint encoding.
18. Retrieval errors → confident wrong facts — open — wrong slice used as if right; no "about someone else" signal.

**Key numbers per failure (slice-relevant first):**
- F1 steer: fires on related at 60–80% strength yet yes/no at chance (2B layers 4–23, 1 or 3 layers, thinking
  or answer); mid-layer injection + 9B don't help; 2B α=2 yes/no worse. **dream_bind (10-13):** K,V of sentence
  restored + subject-match gate: balanced yes/no 0.46 → 1.00 whole turn (Qwen2.5-1.5B), 0.86 name + full stop,
  1.00 slice stored after unrelated chat; 2B with recurrent state 0.93–1.00; attention blocks alone 0.50–0.61;
  **state alone 0.93 yes/no but no recall** (recognition without recall). 10-21: Afendulev et al. 2026 same on
  Qwen3.5 within session, plus "yes" to associated never-seen words. Add. 7 lure control: 2B state recognition
  item-specific (balanced 0.96 vs lures) but doesn't bind (joint state of 2 facts → yes to both names for either
  person). Folders results/dream_bind_gen_20261012_0038/, dream_bind_state2b_20261012_1759/.
- F2 steer: Biscuit 7/300; base P(Biscuit) ~0.3%, Ines ~1/250,000; memory ×10–×80,000; think_v1 91%/22%;
  ~4/44 clean; core_v1 clean ≤15% at α 0.5–3 vs 96–98% ctx; α≥2 85–94% mention without clean use.
  **dream_bind:** stage 0 patching: question-subject state flips 0.00, attribute position flips 1.00 (value
  read late from attribute's own position). Clean use Qwen2.5: nomem 0.00, ctx 0.90, whole-turn K,V 0.87,
  **2 positions (name + full stop) 0.86**, after unrelated chat 0.80; 2B 2 pos + recurrent state 0.97.
  Same 2 tokens as plain text 0.27 (2B 0.24) → slice carries more than tokens. Wrong slice → confident wrong
  value in 41% of use answers. Unrelated unchanged 1.000 only with gate (6% change without). 2B state ≈4.7M
  numbers (~19 MB) regardless of text length → not compact for one fact.
- F3: plain peanut safe 68→41/80; contrastive 72/80. Proposed: topic baseline (model's typical state for topic).
- F4: veg 23/25/35/76 (nomem/contrastive/plain/ctx); Norway 0/80 vs 50; L26 of 28 too late. think_v1 veg
  12→94%; core_v1 α=1 +0.2/+0.1, α=2 41–61% incoherent; item-uneven.
- F5: ~40% unrelated recall, ~⅔ on template, key sim ~0.2 regardless → fixed 24/24. F6: veg −0.83 first / +0.90
  last; LS keeps ~½ each; cost: contradictions average, reconsolidation lost (modulator's job).
- F7: Norway silent on grocery/coffee; think_v1 gate opened 12/12 related prompts (close to follow-ups).
  **Add. 6 (10-19):** subject-own-words key ("my dog") finds right slice 0.85–0.92 (Qwen2.5, 59 facts) vs
  whole-experience-text key 0.42–0.61; 16/59 facts lacked subject alias → fell back to whole-text key.
- F8: Biscuit 7/300 alone → 1/300 with 5 others. **Add. 6:** slices: right slice 0.85–0.92 at N=10, 20, 59,
  no collapse; right slice = ctx (+9.47 vs +9.48 nats); cost = wrong picks 8–15% at N=59. Pool facts weaker in
  generation (0.62 vs 0.83–0.87) but explained by direct-question-only + 2-pos weakness (0.66–0.83); teacher-
  forced right slice = ctx on all 59 (correction 10-21). >59 untested.
- F9: fixes proposed: automate "without" (leave a sentence out, attention picks which), topic baseline, train
  predictor. **Exp 2 (10-19):** 9B "dreamer" writes opposite + 8 alternatives: 7 dislikes, generated opposite
  +0.19 [+0.01,+0.40] (hand +0.23), generated centroid −0.39 [−0.55,−0.21] (hand −0.38), all judges (S2).
  PrefEval 200 prefs: generated opposite +0.03 to +0.10 at α=1 vs ctx +0.43 (P1 holds). S1 cosine test fails
  (10/29 opposites reproduce hand shift). Dreamer errors (kept consequence clauses; has_young_kids no
  alternatives). Sign: general like/dislike direction fails, acc 0.44–0.52 on 23 signed items.
- F10: incoherent 2B/9B: nomem 14%/3%, α1 22%/9%, α2 61%/41%, α3 85%/80%; placebo at α2 equally damaging →
  dose not content; default now α=1 per layer (was 2). Norm-preserving rotation reads available.
- F11: traps yes/no all-"No", template, single-pair, averaging, random baseline, loop, coherence → judge +
  prereg + CIs. F12: carried to 2B/9B unchanged except layers (Llama-3.2-3B in add. 7 not reflected).
- F13: core_v1 loop-matched α1 judged gain: opposite +0.23 (2B)/+0.25 (9B); centroid −0.38/−0.37 (all items
  wrong); without −0.25/−0.31. **Exp 2:** concept × sign with oracle sign: dislikes +0.39 [+0.19,+0.59] at α=2
  (7/7 right); leanings +0.27 [+0.10,+0.45] (best leaning memory). Automatic sign (valence = mean "I love X" −
  "I can't stand X", 120 pairs) reads 8/8 dislikes negative but restriction leanings (vegetarian, vegan,
  lactose, celiac, diabetes, tight budget, young kids) negative too → acc 0.44–0.52; leanings −0.09
  [−0.29,+0.09]. Cause: avoid ≠ dislike; need sign relative to stored concept. PrefEval: auto sign calls nearly
  all negative (132/132 avoid right, 18/19 like wrong); avoid subset +0.05 to +0.13; like items −0.075 (ns);
  vs unsigned centroid judge-sensitive.
- F14: Norway loops or nothing; Singapore barely moves; fix ideas thermostat / mid-layer / episodic. F15: steer self-claims. **dream_bind:** user-turn slice self-claim 0.01 (= ctx 0.01), 2B 0.00–0.01; "whose"
  comes from placement. Swapped slice leaks wrong value.
- F16: thinking repetition 95–98%; fixes: phasic burst then fade, mid-layer, bigger budget.
- F17 (add. 5, 10-13; Qwen2.5 confusable pairs, 5 pairs): both in prompt 0.93 right/0.01 partner; stored
  together one pass 0.89/0.05; **separate, overlapping positions 0.51/0.35; separate, shifted 0.54/0.37**.
  Different subjects (dog/cat, sister/brother): confusion 0.26–0.28 (Qwen2.5), 0.19–0.22 (2B); right 0.62 vs
  0.91–0.97 in prompt. 2B: separate 0.57–0.59 right, 0.28–0.33 confusion; together 0.89/0.05. Two-pos slices
  even joint: Qwen2.5 0.61 right/0.20 confusion; 2B joint 2-pos + joint state 0.73/0.06. Folders
  results/dream_bind_add5_20261013_1255/{qwen25_15b_collide,qwen35_2b_collide}/.
  - Add. 6: binding-ID direction (Feng & Steinhardt mean 2nd−1st pair) real in L19–27: flips partner
    preference 75% [69,81] (random 0%, early/mid 0%). Repair fails: TF confusion as stored 0.38 (margin
    1.25), +ID 0.44 (1.77 [−0.23,4.02]), +random 0.31 (1.27), reversed 0.38 (0.66), joint 0.00 (13.30).
    Generation best cell −0.02 [−0.14,0.11]. → tag editing out; joint re-encode stays; selective recompute
    of tag positions UNTESTED.
  - Add. 7 (10-23): Llama-3.2-3B separate 0.41 [0.23,0.61] vs joint 0.01. Re-encode only new memory with old
    one's 2 positions as prefix = joint 2-pos level, but joint 2-pos still confused on pure transformers
    (0.17–0.20) → fails prereg. Full slices: re-encoding only the new memory == joint encoding (cost = one
    pass over new memory). 2B joint state w/o attention → yes to both partners.
  - Add. 8 (10-29): cause = storage order. Store new memory after a dummy fact (dummy tokens dropped):
    confusion 0.37/0.33/0.41 → 0.09/0.15/0.09 (Qwen2.5/2B/Llama); 34 pool pairs ~0.2 → 0.01. Not pure tag:
    storing both second brings swaps back only on Qwen2.5 (0.20 vs 0.22; Llama 0.03); unrelated sentence
    before also helps somewhat (generic part). Binding at subject: 2 pos + subject span from joint encoding =
    full joint (confusion 0.01–0.03 vs 0.17–0.20 for 2 pos). → storage-time fix: store k-th memory as k-th fact.
  - Add. 9 (10-30): different subjects reach joint at N=3 and 5; same-kind facts: N=3 0.38–0.47 → 0.04–0.09;
    N=5 → 0.22–0.30 (joint 0.00–0.03): neighbouring order tags blur.
- F18 (add. 6, Qwen2.5, 59 facts): right slice (subject-span key) 0.85–0.92; wrong 0.08–0.15; oracle +9.47 nats
  (ctx +9.48); swap −2.62 nats (worse than nomem); wrong value stated when wrong slice injected 0.86–0.90;
  use-prompt clean use right slice 0.87 vs key's pick **0.49**, confabulation 0.16 [0.00,0.43] (7 hand facts,
  wide CI); direct-Q style: 0.83 right, 0.74 key pick, no wrong values. Gate false fire 0–4% but doesn't
  stop wrong-memory-on-related. Fixes: verification (right slice in top-2 ~0.9; score candidates or yes/no
  probe — recurrent state supports this on 2B); better/learned key (untried); abstain when top candidates close.
  Folder results/dream_bind_add6_20261014_1608/qwen25_15b_gkey/.

**Root-cause clusters:** A adding a vector = nudge (1,2,4,10,14,16); B what gets written: what/which way/whose
(3,4,9,13,15); C filing/keys (5,6,7,8,18); D measurement (11,12); E binding (1,2,15,17,18): push has no
position → can't bind; slice has one → binds, but separately stored slices share a tag and collide; wrong slice
read as answer.
**Priority list:** 9 (now mostly sign + what-to-store), 13 & 15, 2 & 14, 7 & 18. F1 = other channel's scope.
**Cross-links:** dreaming/01-binding.md (add. 5–9), dreaming/02-scaffold.md, literature.md, core_v1 PREREG.

---

## 3. how-it-works.md — current STEER design end to end (2026-09-30, Qwen2.5)
- Pieces: state 1,536-d hooked between L26/L27 (of 28); hum mean; top-256 PCA dirs (~96% var) + spreads from ~100
  generic prompts (~1,300 states); key = 256-d whitened (÷√(spread + 1%-of-mean floor)); shift 1,536-d;
  M 1,536×256; key list; per-memory threshold. Whitening made keys discriminating.
- Key: mean of whitened user-token states (skip system prompt/template), unit norm; cosine: unrelated
  0.0–0.15, same-topic 0.4–0.8, verbatim 1.0.
- Shift: per follow-up token, contrastive = with − opposite (preferences), plain = with − without (facts);
  pooled to one per moment, weighted by entropy of "without" next-token dist (max-normalised). Key from
  "without" run. 3 (key, shift) pairs per memory.
- Write: delta rule M ← M + (s − Mk)kᵀ (single memory); least squares M = A B⁻¹, A = Σ s kᵀ, B = 0.1·I + Σ k kᵀ
  (several; order-free; ~½ strength at 6).
- Threshold: 95th pct of best match over calibration prompts + answers; 0.14–0.25 single, 0.38 for six.
- Read at every token: q = gist of user words so far; match = max cos to key list; gate hard or soft (width
  0.5 × spread of ordinary matches); h ← h + 2·gate·M q; blocked on system prompt. Examples: veg dinner 0.50 vs
  0.19 fires; haiku 0.04; Norway coffee below.
- Artificial bits (§10): hand comparison; key never contains experience (filed under follow-ups); a "small
  filing cabinet" crept in (key list decides *whether*, M decides *what* ≈ hippocampal index); gentle strength.
- Relevance to KV memory: the whitened-gist key + q0.95 threshold + soft gate is the reusable gating
  machinery (dream_bind's subject-match gate / G-key build on this lineage).
**Cross-links:** where-we-are.
---

## 4. reference/original-method.md — v0/v0.1 spec (archival)
- Hook at output of `model.model.layers[ℓ]`; plain PyTorch hooks (not TransformerLens: weight processing alters
  residuals; nnsight ok, revisit for NDIF; not output_hidden_states: last is post-norm). Injection during
  generation also alters later layers' cached K,V (consistent). fp32 (Qwen overflows fp16 on V100).
- μ_ℓ over ~100 generic prompts excluding template prefix (system, start tokens, attention sink: huge identical
  states). Raw residuals anisotropic → centring needed.
- FastWeightMemory: key = normalize(h − μ); M d×d (1536² fp32 ≈ 9.4 MB); read h + αMk(h) at every position;
  sequential delta write with gate g_t, η=1. M = Σ η g e kᵀ (sum of rank-1 imprints; no slots). Read =
  similarity-weighted sum with no competition (vs softmax attention) → interference + leakage root.
- Properties (unit-tested): exact one-shot recall; gate = per-moment learning rate; habituation; contradiction
  overwrite (reconsolidation); orthogonal independence; read scaling. Toy d=2 interference example.
- Capacity: rank ≤ d, practical far lower (correlated keys); sequential delta = Kaczmarz; Larimar uses
  recursive LS (C⁻¹ covariance) — upgrade (later adopted as LS). Write errors ~1+ for all-token writes, <1
  pooled/top-k; dispositions collided at N=6.
- Write protocol: with / without / counter; no assistant text generated or forced (would leak fact into
  without). Write tokens = longest common suffix (tokenisation-boundary safe). Δ = h_with − h_without or −
  h_counter; key from without. ‖Δ‖/‖h‖ ≈ 0.19–0.24. RoPE offset caveat (Δ includes small position effect).
- Variants: gate none/entropy (g = H/maxH, only scales down); position all (~14 tok) / boundary (5 template-tail
  tok) / topk (k=4 highest entropy) / pooled (1 per moment); isolated vs combined (N=6, in order).
- Read at every position incl. template; probes exact/paraphrase/related + 8 unrelated. Metrics teacher-forced:
  KL along ceiling's 20-tok continuation, gap closed = 1 − ΣKL_mem/ΣKL_base, fact Δlog P, contrast, foils
  specificity, yes/no margin, leakage KL; repetition_penalty 1.0.
- Limits: counterfactual scaffold; one layer additive; linear competition-free recall; Kaczmarz; RoPE; batch 1.
- Online-writing intent (§2.5): drop counterfactual, use memory's own prediction as baseline (see open-notes N3
  for why that fails for concept cancellation).
**Cross-links:** how-it-works, where-it-fails (5, 6), experiment-log, code-map, core-idea, next-moves-v0.1.

---

## 5. history/experiment-log.md — numbers for v0 → core_v1 (to 10-11)
Setup v0–diag: Qwen2.5-1.5B fp32, DelftBlue A100 MIG 10GB; 6 scenarios × 3 follow-ups × 3 probes + 8
unrelated. Results results/v0_408770/, v0_1_415909/, diag_dose_577895/, diag_order_577896/.

**§4.1 v0 (job 408770, commit d6e5947, 21m53s):** sweep mode × gate × L{6,10,14,17,20,23,26} × α{0,.5,1,2,4}.
- Dispositions gap closed: best ~0.26 (L20 exact α2 0.262; L23 para 0.261; related 0.239); L14 related α4 −0.56;
  L6 ~0. Facts L26 α2: exact 0.177, related 0.215; α4 negative.
- Leakage L23: α .5/1/2/4 = 0.003/0.012/0.053/0.257 (ceiling-vs-base KL 0.1–2).
- Target shifts α2 (L23 / L26 / ceiling): veg +2.40/+1.40/+1.78; peanut −7.25/−3.19/mixed; norway +2.50/+0.86/
  ~+12.4; dog +0.30/+4.12/~+9.3; sister +0.79/+3.20/~+13.5; job +5.16/+2.74/~+25.4.
- Entropy gate L23 α2: gap 0.251→0.306, Δtarget +2.08→+3.39, leak 0.053→0.025 (selects better moments).
- Combined N=6: gap 0.251 → −0.174 (none), −0.089 (entropy); fact +2.89 (entropy) — CORRECTED: = mean of dog
  −1.49, sister +0.29, job +9.87 (job written last; reversed order job −0.91).
- ‖Δ‖/‖h‖ 0.19–0.24; rel. write error ~1.0/0.9/1.1 (no habituation).
**§4.2 v0.1 (job 415909, commit 098c317, 25m13s):** positions × baselines × L{20,23,26} × α{1,2}.
- Specificity L26 α2 (without / contrastive / ceiling): Biscuit +2.60/+3.93/+9.27; Ines +2.26/+3.87/+13.50;
  welder +9.40/+11.33/+25.35; pooled contrastive Δtarget +6.37, foils −0.19, specificity +6.56 (~30–45% of
  ceiling in log space). Contrastive +21/+51/+71% (job/dog/sister).
- Contrast L26 α2 (without/contrastive/topk-contr/ceiling): veg −1.14/+4.80/+5.01/+1.78; peanut −4.23/−0.78/
  +0.28/−0.07; norway +8.17/+9.93/+10.26/+12.37 (~80%). Contrastive leaks 2–3× less (0.008 vs 0.025).
- Relations: ceiling mean +3.98; best memory +0.98 → CORRECTED to +0.33 without template tail (~1/12 ceiling).
- Positions L23 α2: all gap 0.306/Δt +3.39/leak 0.025; topk 0.277/+3.40/0.029; pooled 0.228/+1.44/0.004 (rel err
  <1, habituation returns); boundary 0.118/+0.06/0.003.
- Combined: dispositions collapse every variant (pooled ~0.007); "facts survive" = job only (CORRECTED).
- Samples: veg all/without "vegetarian and vegan recipes" (first behaviour change); peanut all/without "PB&J
  safe option"; facts never stated.
**§4.3–4.4:** steer = modulatory channel → two-channel memory (steer vs episodic reinstatement into attention;
hippocampal indexing = reinstatement). Caveats: small, brittle contrasts, relation metric confounded, low KL ≠
selectivity.
**§4.5 diagnostics (commit 1d0628e; ~6 min each):**
- diag_dose (job 577895): steer ratio ‖s‖/‖h−μ‖ at α1/2/4: L14 .105/.210/.421 … L26 .152/.304/.608; norm change
  α2 median +0.6–2.6%, p90 5–11%; α4 p90 20–33%; at L23/26 α4 ratio 1.0–1.3 on tail + related user text, tail
  norm +23–33% → breaks. Recall fraction: exact ~1.0, para 0.70–0.85, related 0.59–0.87, relation 0.44–0.61,
  **unrelated 0.38–0.47**; template head 0.29–0.54, tail 0.47–0.86; tail keys cos 0.84–0.99 to any prompt.
  Unrelated user text cos 0.31–0.37 to content keys (related 0.47–0.60). Steer energy on unrelated: head
  0.40–0.48, tail 0.15–0.26, head+tail 0.61–0.69 (0.68 pooled). Combined recall cos to isolated 0.27–0.57
  (without), 0.08–0.19 (contrastive), louder 1.1–1.7× (dog/sister 2.7–4.2×); job (last) 0.86–0.98.
  Relation probes get 63–77% of related recall → fired, not used. Prereg P1 wrong (0.24 not 0.5–1), P2 right,
  P3 half, P4 wrong (0.71 not 0.3). Dose not bottleneck; selectivity is.
- diag_order (job 577896): reverse order flips survival (veg −0.83 first → +0.90 last; job +1.14 last →
  −0.01 first); fact Δtarget iso→comb: dog +2.60→+0.19, sister +2.26→+0.43, job +9.40→+9.70 (L26 without).
  Food interferer after veg: retention −0.40 to −1.45; non-food +0.43 to +0.90; veg's own contrast kept
  0.72–0.94 → read-time crosstalk > write erasure. Forgetting curve: cos 0.62 → 0.39 at peanut write → 0.31.
  Content keys cross-scenario cos 0.15–0.23 ≈ within 0.17–0.24; tail cos 0.85–0.90 (one shared key). No-tail
  writes don't rescue capacity; tail tokens' own error 1.7–3.4 (15 writes/scenario under ~same key). Relation
  means L23/L26: iso +0.49/+0.98, iso no-tail +0.17/+0.33, comb +1.12/+2.03, comb no-tail −0.04/−0.09. Renorm
  gate identical (0/18 entropy max in tail).
- §4.5.3 fixes proposed: no read/write at template, whitened keys, RLS, match threshold; re-baseline relations.
**§4.6 later (Qwen3.5):**
- think_v1 (jobs 906983–906987; V100 fp32; pooled_w256 key, q0.95, α2; 4 facts Pepper/Petra/pharmacist/teal +
  4 prefs): stage 1 fact specificity L19 +4.9, L20 +5.7, L21 +5.4, L22 +5.3, L23 +6.2 (L4–18 ~0–1.3); pref
  contrast 2.1–3.0 late; selectivity all layers; Ledoit–Wolf key bad (gate opens 30–60% unrelated). Stage 2
  (612/cond) mention/clean/loop: L23 0.36/0.04/0.33; 3 layers split 0.21/0.16/0.05; **3 layers full (chosen)
  0.91/0.22/0.70**; combined RLS 0.43/0.15/0.28; ctx 0.95/0.95/0. Stage 3 facts (44 answers): answer-only 4/44
  clean (27 loops), both 1/44, think-only 4/0 or 6/0 3/44 with think repetition 0.95–0.98, 6/1 6/44, ctx 28/44;
  balanced yes/no 0.38–0.50 (ctx 0.82). Prefs consistent: veg 0.12→0.94; hiking 0.24→0.39–0.76; norway 0.82 but
  0.61 loops; jazz 0.21→0.18–0.36. Clean recount results/think_v1_20261003_0004/.
- ref_v1 (jobs 915094–915095; L23/20/21; shifts norm-matched to without; 11 prefs; refs opposite/without/hum/
  disclosure(24)/centroid(8)): α1 lean two-ended/one-of-many/hates_jazz/no_alcohol: opposite +0.59/+0.56/+0.48/
  +0.24; without +0.39/+0.49/−0.86/−0.79; hum +0.24/+0.30/−0.71/−0.56; disclosure +0.45/+0.56/−0.81/−0.85;
  centroid +0.48/+0.70(loops 0.70)/−1.00/−0.42. Logit lens L23 hates_jazz centroid −9.1. Hum shift cos 0.63–0.85
  with own key (re-injects topic); raw norms hum 1.8–3.2×, centroid 0.3–0.9× (norm-matching boosted centroid
  ~2.5×). Unrelated identical in all 110 cells.
- xlayer_v1 (jobs 921693–921695; 12 facts; two-pass gating; norm-ratio rescale ×0.15–×5): lens median target
  rank ~100k L12, ~30k L16, 1 at L20/L23; read L16 ≤ +1.6; read L23 α2 direct vs full: W8 +8.3/+0.7, W16
  +7.5/+4.8, W20 +7.2/+7.5 (later blocks damp, never amplify). Gen: {20,21,23}→{14,15,16} related 0.35 / use
  0.27 clean but garbled, 56% answers <40 words; current {20,21,23} 0.19/0.09 with loops 0.71/0.83.
- core_v1 (jobs 922999–923009): 2B L{20,21,23} V100; 9B L{27,28,31} A100 80GB, both fp32; judge Qwen3.5-9B bf16
  (j_lean, use_judge_clean); 1 greedy + 4 samples; two-level bootstrap. Leanings gain [CI] / incoherent:
  2B ctx +0.72/0.22, α0.5 +0.09, **α1 +0.20 [0.04,0.38]/0.22**, α2 +0.15/0.61, α3 +0.05/0.85; 9B α1 +0.11
  [0.004,0.24]/0.09, α2 +0.15/0.41; rand +0.01; placebo α2 −0.03/0.86 (2B). Lexicon lean keeps rising (2B +0.35
  α1, +0.51 α2). Per item 2B nomem→α1: early_riser +0.17→+0.77, prefers_quiet +0.27→+0.90, vegan −0.20→+0.47,
  bookworm +0.13→+0.50. Facts clean use 2B α0.5–3 0.04/0.15/0.02/0.03, 9B 0.08/0.12/0.02/0.01; TF Δlog P(target)
  +9 to +11, top-1 0.62–1.00 → "there as next word, not used". C4 2B fail: −0.15 [−0.27,−0.04]. C6 from L19
  (depth 0.83) 2B, L27 (0.875) 9B.
**Cross-links:** original-method, where-we-are §5, metrics, open-notes, next-moves, HANDOVER §6/§8, PREREG.
---

## 6. experiments/core_v1/README.md (via code-map)
- Stages smoke → prep → gen → score → analyze (+ combine → C7.md, ~200-answer audit sample). 9B layers by
  relative depth round((l+1)·32/24)−1 → {27,28,31} (full, linear, full; 2B linear, linear, full); thinking forced
  off. prep = μ + PCA-256 whitening, per-item thresholds, plain/opposite/centroid/placebo shifts, delta rule,
  logit lens. Results rsync from delftblue:/scratch/vvjumle/seahorse_runs/core_v1_<ts>.

---

## 7. history/open-notes.md — observations (2026-09-25; N1 resolved 09-28)
- N1 fired-vs-not-retrieved on relation probes → RESOLVED: fired, not used (63–77% of related recall; answer-pos
  steer 0.21–0.33 vs 0.31–0.45). Answer-position firing mostly template-tail recall carrying "No" bias.
- N2 multi-layer small-α injection as main arm of A1; if no arm flips relations → reinstatement (A2). (Later:
  xlayer negative; A2 = dream_bind positive.)
- **N3 online-baseline problem (still relevant):** memory's own prediction Mk can't cancel concept priming on
  first exposure (concept lives in model priors, not memory). Replacement must come from priors: CAA-style
  generic concept direction projected out of Δ, or model's predicted next state (store only unexpected).
  Open: which keeps the relation. (Exp 2 partly answers with model-generated refs; sign still open.)
- **N5 additive = content, multiplicative = modulation:** real neuromodulators change gain → per-head attention
  temperature, per-layer/head output scaling, residual gain; Aston-Jones & Cohen adaptive gain (high = exploit,
  low = explore); receptor specificity = which heads/layers.
- **N6** endogenous release needs a state variable outside the forward pass (tonic drift vs phasic burst).
- **N7** closed loop needs reuptake (decay) + tolerance to stay stable; same runaway risk for online
  salience-gated writing (write → state shift → more writes); safety: self-releasing "desperate" loop =
  reward-hacking route.
- N8 steers already pharmacological: inverted-U (α2 best, α4 breaks; leakage 0.003→0.257); layer specificity
  (L6 nothing, L14 negative, L23–26 helps); top of U located (α4 ratio 1.0–1.3, tail +23–33%).
- **N9 three tiers on same machinery:** tonic modulators (internal-signal gated) / modulatory memory (steer,
  resemblance-gated, ≈ conditioned response) / episodic memory (moments reinstated into attention, key-match
  gated) — the KV-slice channel is tier 3.
**Cross-links:** experiment-log §4.5, next-moves, core-idea §1.7, parent docs/ side-idea note (not in repo).
---

## 8. history/next-moves-v0.1.md — archived option list (post-v0.1; priority update 09-28)
Status header: done B1, B2 goal (LS), whitened keys + threshold (≈B4); partly F (bench_v1); open A1, A2, B3, B5,
C, D, E, G. (STALE: A2 now realised by dream_bind; A1 answered negative by xlayer_v1/core_v1; B5 for slices
done to 59 facts.)
- A1 layer sweep for relations (L8–17), multi-layer Larimar-style → answered: no.
- **A2 episodic reinstatement (the origin of the KV-slice channel):** store only salient moments (top-k entropy
  tokens), keep residuals at every layer (compression by selection, not averaging); recall by key resemblance
  (InfLLM query·key or centred keys); splice K,V recomputed with frozen W_k, W_v as extra attention entries at
  fixed positions (InfLLM-style); measure relation probes. Trained fallback: Larimar-style pooled vector →
  learned projection → KV slot per layer; TransMem evidence-conditioned distillation (teacher = ctx, student =
  memory, KL), only read projections trained.
- B1 key collisions (done: ~0.2 cos all pairs); B2 RLS (done as LS); **B3 pattern separation** (random
  projection + top-k sparsify, or SAE features; Hopfield: sharp retrieval needs separation); **B4 read-time
  competition** (softmax over stored keys, temperature → only best fires; moves toward KV store); **B5 capacity
  curve** N=1,5,20,50 templated facts.
- **C scaffold removal:** modulatory: Δ vs memory prediction / running average (habituation) / predicted next
  state; **episodic: no delta needed — store states of salient moments; salience decides what** (entropy gate,
  top-k).
- **D salience:** D1 confusion taxonomy (don't know fact → store answer; ambiguous → store nothing/clarify;
  hard to compute → store result; conflicting → reconsolidate), linear directions per state, epistemic vs
  aleatoric (semantic entropy); refs Kadavath 2022, semantic entropy probes 2024, known/unknown-entity
  features (Biology of an LLM 2025; Ferrando et al.). D2 combine gates (surprise, uncertainty, valence/arousal).
  **D3 hypercorrection:** confident-but-wrong moments invisible to entropy; need surprise-vs-memory gate.
- E training: not in write; in episodic read path (distillation); maybe keys (B3). Backbone frozen.
- **F eval hygiene:** more templated scenarios + paraphrase sets; relation probes first-class; **text-summary
  baseline**; **InfLLM / EM-LLM baselines**; scale to Qwen2.5-7B; **LoCoMo, LongMemEval** later.
- **G parked:** reflection memories (store conclusions with hindsight; fixes "immutable past" of causal mask);
  cold start (empty memory = all surprising); consolidation into weights (CLS replay); reconstructive drift
  (prior reshapes memory vs memory always wins; confidence-weighted reconciliation); steering as neuromodulators.
- Suggested order (as written): A1 → B1 → A2 training-free → B2/B3 or E → C.
**Cross-links:** where-we-are §8, where-it-fails, experiment-log §4.5, core-idea §1.7, original-method §2.5.
---

## 9. reference/code-map.md — repo layout (core_v1 era; STALE re dreaming)
- env: `seahorse` conda (py3.11, torch 2.6 cu126, transformers ≥5.18,<6 for Qwen3.5). (Local `torch` env differs.)
- src/seahorse/: memory.py (FastWeightMemory), residual.py (load_model, capture, inject; handles transformers-5
  tensor outputs), sessions.py (chat_ids, common_prefix/suffix_len, write_pair, ceiling_ids), metrics.py
  (cont_logprobs, kl, seq_logprob, entropy, greedy), stats.py (two-level bootstrap), bench/ (data, score
  word-list lean, validate). data/bench_v1/ YAML.
- experiments/: v0, v0_1, diag_dose, diag_order, diag_attn (no-go), **diag_keys (definitive current-design
  impl: KeySpace, whitening, unit_writes, delta/RLS build, gate_fn)**, bench_v1_calib, samples_v2, think_v1,
  ref_v1, xlayer_v1 (xl.py two-pass gating), core_v1 (+ panel_judge.py, analyze_panel.py). No dream_* listed.
- tests/ (CPU; tiny random Qwen2 for hooks); tools/manuscript (public page), writeup/ (main.tex, arxiv).
- slurm/: run.slurm (gpu-a100-small MIG 10GB, 4h), run_v100.slurm (gpu-v100 32GB fp32; Qwen3.5-2B exps),
  core_v1.slurm (A100 80GB for 9B + judge), panel.slurm. Gotchas: --ntasks=1; compute-p1 3996MB/CPU; no internet
  on compute nodes; maintenance cancels queued jobs.

---

## 10. Cross-cutting: still-open ideas relevant to a training-free KV / recurrent-state memory
Established facts to build on:
- Slice of model's own K,V (user-turn segment) + subject-match gate → exact fact use (0.86–0.87), premises
  (yes/no 1.00), correct ownership (self-claim 0.01); 2 positions (name + full stop) nearly = whole turn on
  Qwen2.5; text of same tokens only 0.27. Hybrid 2B needs recurrent state too (attention blocks alone 0.50–0.61).
- Recurrent (DeltaNet) state alone = item-specific recognition/familiarity without recall and without binding;
  ~19 MB on 2B regardless of length.
- Gate essential (6% unrelated change without). Subject-span key 0.85–0.92 retrieval at 59 facts.
- Binding lives at subject positions and in order tags; storage order is the main collision cause.
Open (training-free):
- F18 verification before use (top-2 ≈0.9; candidate scoring, yes/no probe via recurrent state) and abstention
  on close candidates; key for facts without subject alias (16/59 fell back).
- F17: k-th-memory-as-k-th-fact storage blurs at N=5 same-kind (0.22–0.30 vs joint 0.00–0.03); selective
  recompute of tag-carrying positions untested; joint re-encode = one pass over new memory (full slices) —
  dreaming consolidation job; >59 facts untested.
- Contradiction / update policy (LS averaged; slices: which wins?) — reconsolidation via modulator (F6).
- Salience/what to store: unit = moment (steer) but slices need specific positions (attribute + subject span);
  D1 confusion taxonomy, D3 hypercorrection, N3 prior-based baseline, emotion-protects-memory test.
- Steer-side leftovers that could pair with slices: concept × sign (sign relative to concept unsolved), thermostat,
  phasic burst for thinking, N5 multiplicative gain as true modulation, N9 three-tier picture.
- Baselines owed: text-summary, InfLLM/EM-LLM; real benchmarks LoCoMo/LongMemEval (cf. dream_select plan on
  LongMemEval in memory notes); second model family (Llama-3.2-3B already used in add. 7/8); human labels.
- Parked G: reflection memories (store conclusions; fixes causal-mask "immutable past"), cold start,
  reconstructive drift.
