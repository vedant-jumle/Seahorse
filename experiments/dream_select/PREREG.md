# dream_select pre-registration: can the slice's positions be chosen automatically, on natural text?

*Written and committed 2026-11-04, before the full run. The only numbers that exist are from a 3-item `--tiny` smoke
(k = 4 only; items e47becba, 118b2229, 51a45a95), used to check the code. Exploratory: two stated rules; everything
else is reported beside them. Code: `run.py` (model and forward passes; reuses dream_bind's loading, chat-turn pieces
and splice code by import; nothing under `experiments/dream_bind/`, `experiments/dream_subj/`, `src/` or `slurm/` is
changed), `select.py` (pure helpers, tested in `test_dream_select.py`), `analyze.py` (tables, rules), `prep_data.py`
(the data extract). Local only: Qwen2.5-1.5B-Instruct, fp32, eager attention, RTX 4060.*

## Known before this experiment

- **dream_bind** (`docs/dreaming/01-binding.md`): a stored K,V slice of a fact's user turn, put back before the next
  session's question at its stored positions, restores answering. R-full (the whole turn) equals the turn as a prior
  user turn (T-full) exactly; two hand-picked positions (value + full stop, R-min) already restore most of it.
- **dream_subj** (`results/dream_subj_20261022_1755/report.md`): on 45 templated facts ("My X's name is V."), adding
  the subject's position to R-min closes about 80% of the gap to R-full; "is" and "name" add nothing. Attention fits a
  two-step lookup: the question's subject token attends to the stored subject entry in layers ~3–13, the answer
  position attends to the stored value in layers ~23–27.
- In both, the positions were **hand-picked** on one sentence template. Here they are chosen by an automatic rule, on
  natural benchmark text.

## The question

Given the right memory (the user turn that holds the answer; chosen by the label, so only **positions** are tested),
can a selector that never sees the question or the answer pick k positions of that turn whose K,V slice restores
answering? In particular: does attention from **dreamed questions** (the model's own guesses at what the user might
later ask) pick better positions than random, and better than query-free attention?

## Data

LongMemEval, cleaned oracle split (Wu et al., ICLR 2025; MIT): `data/longmemeval/ssu70.json`, built by `prep_data.py`
(README beside it). The 70 `single-session-user` items. **Two facts about the file differ from the plan** (asserted in
`prep_data.py`, recorded in the extract):
- **6 are abstention items** (`_abs`; the answer is "You did not mention this information …") and flag no turn. They
  have no gold answer to score and no memory to store; they are not run. **N = 64 usable items; 56 are verbatim** (the
  answer string occurs in the turn, case-insensitive).
- **2 items also flag the assistant reply** that follows the user's evidence turn (it repeats the fact). Only the user
  turn is stored. No item has 2 user evidence turns.
- **Turn lengths:** 20 to 98 tokens (median 64). The longest is far below the 200-token S-loo cap, so S-loo runs on
  every item.

## Setup

- **Harvest:** one pass over [chat head][the evidence turn as a user turn] (`dream_bind.turn_pieces`). K,V at every
  layer. The head is always kept (as R-min). A slice keeps the head plus the selected positions, at their stored RoPE
  positions. The question turn (the benchmark question as a new user turn, then the assistant header) starts at
  n_head + n_seg, where the whole turn would end (as dream_subj).
- **Content positions** (`select.content_positions`): the tokens that spell the turn's text inside its user turn;
  not the head, not `<|im_start|>user\n`, not `<|im_end|>\n`. Only these can be selected.
- **Anchors:**
  - `nomem`: the question alone.
  - `ctx`: the turn and the question in the same user turn ("{turn} {question}"), as dream_bind's ctx.
  - `T-full`: the turn as an earlier user turn, as text.
  - `R-full`: the whole stored user turn (its role and end tokens included) as a slice. **Asserted R-full == T-full**
    (logits at every scored position within 0.01). This is the denominator of the gap closed.
  - `R-content` (added): the head + all content positions, without the turn's role/end tokens. Every selection is a
    subset of this, so it is the ceiling a selector can reach.
- **Budgets:** k ∈ {2, 4, 8} and k10 = max(2, round(0.10 × n_content)) (capped at n_content).

## Selectors (none sees the benchmark question or answer)

Every selector function takes `(C, store)` or `(C, store, dreams)`; the store is built from the turn alone
(`run.harvest(C, turn)`). `test_dream_select.py` checks the signatures (no question / answer / item / gold argument)
and that no selector body reads the benchmark item. Ties go to the earlier position.

1. **S-attn (attention received, no query).** In the harvest pass (eager), score content position j = the mean
   attention it receives from the later content tokens (mean over those queries, not the sum, to avoid H2O's
   early-token bias), averaged over heads and layers. The last content token has no later query and ranks lowest.
2. **S-dream (dreamed questions; the primary, pre-registered score).**
   - **Dreamer:** the same Qwen2.5-1.5B-Instruct, greedy, no memory, ≤ 96 new tokens. The prompt (`run.DREAM_PROMPT`)
     shows the turn and asks for 3 short memory-quiz questions that the user could later ask about facts from their
     own life in it, in the user's voice ("I", "my"), one per line. Two invented worked examples (not from
     LongMemEval) come first as earlier chat exchanges (`run.DREAM_EXAMPLES`).
   - **How the prompt was fixed (disclosed):** a first zero-shot prompt mostly echoed the turn (3 of 3 smoke items fell
     back). Three variants were then tried on the dreamed TEXT of the first 10 usable turns only (format: 3 lines with
     '?', first person); the one used here was the best on format. No score was used to choose it. The 3-item smoke
     scores of the first two variants were seen (they are noise at N = 3).
   - **Parse and validate** (`select.parse_dreams`): list markers are stripped; questions = the non-empty lines ending
     in "?". Strictly valid = exactly 3 lines, all questions. Fewer than 3 questions: retry once with a stricter prompt
     (the same plus "Output exactly 3 lines …"); still fewer: use what was parsed (recorded as a fallback; if no line
     ends in "?", the first non-empty lines).
   - **Score:** per dreamed question, [head][whole stored turn as a slice][the dreamed question as a user
     turn][assistant header], eager. Per question, the attention to each stored content position, averaged over
     heads, layers and the query set {the question's own text tokens} ∪ {the last prompt token}. Score = the sum over
     the questions. Top-k.
   - **S-dream-split (exploratory):** per question, 0.5 × (mean over the question tokens, layers 0–13) + 0.5 × (the
     last prompt token, layers 20–27), heads averaged; summed over the questions. Exploratory because the layer bands
     come from dream_subj's attention analysis.
3. **S-recon (reconstruction, KVzip-like).** After the whole stored turn: the user turn "Please repeat exactly what I
   just told you.", the assistant header, then the turn's own text teacher-forced as the reply. Score = the mean
   attention each stored content position receives from the reconstruction tokens (the reply's tokens as queries),
   averaged over heads and layers. Top-k.
4. **S-loo (leave-one-out reference, expensive).** Per dreamed question, the model's own greedy answer (≤ 24 tokens)
   with the whole slice. For each content position, drop it from the whole slice (R-full minus that entry; the
   question still starts at n_head + n_seg) and measure the drop in log P(that own answer | dreamed question); sum over
   the questions; top-k by drop. Turns over 200 content tokens would skip S-loo (none here).
5. **Baselines.**
   - `random`: 3 fixed seeds per item and budget (seed = (s, item index, k), s ∈ {0, 1, 2}); every measure is the
     mean over the 3 seeds.
   - `last-k`: the last k content tokens.
   - `gold` (uses the label; a reference only): the answer's tokens (first case-insensitive occurrence in the turn)
     plus the next token; verbatim items only; k = its own length + 1.
   - `gold@k`: the answer's first token, the next token and then the nearest preceding tokens, to fill k (later tokens
     if the turn's start is reached); verbatim items only.

## Measures (teacher-forced; no generation is scored, no judges)

- **Answer score (primary):** log P(gold answer tokens | question), the answer starting the assistant reply, summed
  over the answer's tokens. Also the per-token mean.
- **Refusal at the first word:** P("I") + P("Sorry") + P("As") at the first assistant token.
- **Gap closed:** per item, (cond − nomem) / (R-full − nomem) on the answer score. Reported as the mean and the median.
  Items with R-full − nomem < 0.5 nat are dropped from this measure (the count is reported).
- **Hit rate** (verbatim items): the share of items whose selection holds at least one token of the gold answer span
  (random: the mean over seeds).
- **Kept-token examples:** the first 8 usable items in file order: the turn with each selector's k = 4 picks marked,
  the dreamed questions and S-loo's own answers.

## Statistics

- Percentile bootstrap over items, 2000 replicates, seed 7 (95% intervals). Medians are bootstrapped the same way.
- Contrasts are paired: per-item differences on the same items.

## Rules (decided at k = 4, on the answer score, over all 64 usable items)

- **"Dreamed questions select well"** holds if the paired CIs of (S-dream − random) and (S-dream − S-attn) both have a
  lower bound > 0, AND S-dream's mean gap closed ≥ 0.5.
- **"Any attention selector works"** holds if (S-attn − random) has a CI lower bound > 0.
- Implementation: `analyze.rules`, tested on synthetic numbers.

**Reported beside them:** every selector at every k (answer score, per-token mean, refusal, gap closed mean and
median, hit rate); S-loo as the reference; gold and gold@k; S-dream-split (exploratory); R-content; the paired
contrasts at each k; splits verbatim (56) vs non-verbatim (8) and short (≤ 60 content tokens) vs long turns; the
dreamer's validity counts.

## Checks that fail the run

- R-full == T-full (logits within 0.01 at every scored position).
- Every slice keeps the head intact plus exactly k entries, and every selected index is a content position.
- Paris (the loaded model answers "What is the capital of France?").

## Known limitations (stated in advance)

- **The answer score counts one answer form.** It scores the bare gold string as the reply's first words
  ("Business Administration"). A model that answers "You graduated with a degree in Business Administration." scores
  low even with the whole turn. The gap closed normalises by R-full's score, which shares the problem; the refusal
  mass is reported beside it.
- **Non-verbatim answers** ("Target" when the turn never says it; "The painting is worth triple …") may not be
  recoverable from the turn at all; R-full − nomem can then be small or negative (dropped from the gap closed).
- **Positions only.** The memory turn is given by the label. Which turn to store is not tested.
- **The dreamer is weak** (a 1.5B model, greedy); its questions are sometimes about the help asked for, not the facts.
- **Scope:** one model, teacher-forced only, 64 items.
