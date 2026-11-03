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

---

## Addendum 1: DelftBlue replication + generation (2026-11-04)

*Written and committed 2026-11-04, after the local run (`results/dream_select_20261104_0032`) and before any cluster
job or cluster number exists.*

**Why this run exists.** The local primary rule failed ("Dreamed questions select well": S-dream − S-attn 0.04
[-1.13, 1.21]; S-dream's gap closed 0.25 < 0.5). "Any attention selector works" held. Before the local run, the user
had set a threshold for going to the cluster: if S-dream clearly beats random (the paired CI of S-dream − random
above 0), run a bigger model, a second family, and generation with judges. It did (1.18 [0.35, 2.08] nats), so this
run was triggered by that pre-stated threshold, although the local primary rule failed.

### Models
- **Qwen2.5-1.5B-Instruct:** generation only. The teacher-forced numbers are the local run's (reused, not re-run).
  Its generation uses the local run's selections and dreamed questions, copied to the cluster, so the generated
  answers belong to exactly the selections that were scored. On the cluster the content positions and the head
  are re-derived and asserted equal to the local run's. The 1.5B smoke re-runs the teacher-forced stage on its 3
  items on the cluster and reports how many of the k = 4 picks match the local ones (information only).
- **Qwen2.5-7B-Instruct** (the HF snapshot a09a354 in the cluster cache): teacher-forced + generation.
- **Llama-3.2-3B-Instruct** (a second family): teacher-forced + generation.
- **Precision:** fp32 everywhere, eager attention.
  - The 7B runs on a full A100 (`gpu-a100`) and Llama on a V100 (`gpu-v100`).
  - The 1.5B runs on the 10 GB A100 slice (`gpu-a100-small`), with a fit check of 9 GB peak reserved in the smoke.
    If it does not fit, it moves to a V100; that move is infrastructure only.
- **Each model dreams its own questions** with the frozen dreamer prompt (`run.DREAM_PROMPT`, the strict retry and
  the two worked examples, unchanged). Each model's selectors use its own attention.
- **Unchanged from the local run:** the 64 items, the selectors, the budgets (k = 2, 4, 8 and 10%), the anchors and
  the teacher-forced measures.
  - S-dream-split's layer bands (0–13, 20–27) are kept as they are. All three models have 28 blocks (asserted).
- **Chat templates:**
  - The head, the user turn and the content positions come from each tokenizer's own chat template
    (`dream_bind.turn_pieces` / `user_turn_start`, as in dream_bind's Llama port).
  - Llama's template writes today's date into the system block. It is pinned to dream_bind's string,
    "26 Jul 2024", in every template call (`dream_bind.date_pinned`).
  - The R-full == T-full identity (logits within 0.01 at every scored position) is asserted per item and per
    model: in the teacher-forced stage, and again at the first generation step in the generation stage.

### Generation
- **Prompt:** the benchmark question as the user turn, as in the teacher-forced measures. Slices: the question turn
  starts at n_head + n_seg.
- **Conditions (per item):**
  - `nomem`, `ctx` (the turn and the question in one user turn), `R-full` and `R-content`.
  - `gold` (the answer's tokens + the next token), on verbatim items only.
  - At k = 4 and at k = 8: S-dream, S-attn, S-recon, S-loo, random (seed 0 only, of the 3 teacher-forced seeds) and
    gold@k (verbatim items only).
  - That is 17 conditions on a verbatim item and 14 on a non-verbatim one.
- **Answers per condition:** 1 greedy + 4 samples, at most 64 new tokens. 64 items give 5,320 answers per model.
- **Sampler:** dream_bind's own (T = 1, top-p 0.95, top-k 20, plus its presence penalty of 1.5 on tokens already
  generated). The random streams are the same in every condition of an item: one generator per item, seeded from
  the question text (`core_v1.seed_of("dream_select", question, 0)`), with the j-th sample row using the j-th draw.

### Judging
- **The criterion is LongMemEval's own correctness check for single-session-user questions.**
  - The template is reproduced verbatim from the LongMemEval repo (`src/evaluation/evaluate_qa.py`,
    `get_anscheck_prompt`, the single-session-user branch; fetched 2026-11-04 from commit d6dc8b5, file sha256
    ecce9c4c…). It is not a reconstruction.
  - It is filled with the benchmark question, the gold answer and the model's response. An empty response is
    given as "(empty reply)", as in dream_bind.
- **Label:** LongMemEval's own parse rule: correct if "yes" occurs in the lower-cased judge reply.
  - A reply with neither "yes" nor "no" as a word is unparsed. It is retried once with another seed (panel judges);
    if it is still unparsed, it is a missing vote. The counts are reported.
- **Judges:**
  - The original judge: Qwen3.5-9B, bf16, thinking off, greedy, at most 10 new tokens (LongMemEval's max_tokens),
    run through core_v1's model loading and batched generation.
  - The panel, through Ollama, as dream_bind's panel does: gpt-oss (reasoning "low", up to 768 tokens with the
    reasoning) and Gemma-4 12B (thinking off, at most 10 tokens). Temperature 0, and core_v1's panel client
    (`panel_judge.chat`, `wait_ready`).
  - Each model's answers go to their own panel files (`<model>/panel/<judge>.jsonl`). That avoids addendum 9's
    shared-file problem; dream_bind's relink (a82244b) still runs on resume.
- **Primary label:** the majority of the three judges. A tie, or a missing vote that leaves a tie, goes to the
  original judge (`dream_bind.bind.majority`).
- **Also reported:**
  - Each judge alone, and the pairwise agreement.
  - The refusal regex of dream_bind addendum 7 (B3) (`run.REFUSAL_RE`: "don't / do not / can't / cannot / couldn't
    know / recall / remember / have / access / tell", "I'm sorry", "could you tell / share / remind").
  - Loops (the repetition rate ≥ 0.3, as dream_bind).
  - The share of answers that contain the gold answer string (verbatim items; a judge-free check).

### Rules (per model; decided at k = 4)
- **Teacher-forced (7B and Llama):** the same two rules as the local PREREG (`analyze.rules`), on each model's own
  teacher-forced run. For the 1.5B, the local verdicts stand.
- **Generation, primary: "S-dream beats random in judged correctness"** holds if the paired CI of (S-dream −
  random) on judged correctness has a lower bound > 0.
  - Per item, correctness is the mean over its 5 answers. There is one prompt per item, so the two-level bootstrap
    (`seahorse.stats.boot_ci`, prompts within items) reduces to an item bootstrap.
  - 2000 replicates, seed 7. A bound with |bound| < 1e-9 counts as 0 (as dream_bind addendum 9).
- **Also reported:**
  - S-dream − S-attn, every selector against random, at k = 4 and k = 8.
  - S-loo and gold@k as references.
  - The gap closed in judged correctness: (mean cond − mean nomem) / (mean R-full − mean nomem), with its CI from
    the same item bootstrap (the ratio per replicate). It is n/a when R-full − nomem < 0.05.
  - The refusal rates.
- **Replication across models:** "replicates" if the generation rule holds on at least 2 of the 3 models. A final
  cross-model step writes `summary.md` with each model's teacher-forced and generation verdicts and this verdict.

### Infrastructure (no change in behaviour)
- **The rename:** `select.py` is renamed `selectors_ds.py`, because the old name shadows Python's stdlib `select`
  module. Its contents are unchanged. The tests prove it, and re-running `analyze.py` on the local run reproduces
  its report.
- **What the code adds:**
  - `run.py` gains `--model` and `--stage` (tf, gen, judge, analyze-gen, summary).
  - `panel.py` is this experiment's yes/no panel client.
  - `jobs.py` runs pytest and several stages in one Python process (as `dream_bind/multi.py`).
  - New slurm scripts `slurm/dream_select*.slurm` and `experiments/dream_select/submit.sh`.
  - Nothing under `experiments/dream_bind/`, `experiments/dream_subj/`, `src/` or the existing slurm scripts is
    changed.
- **Smoke per model** (`--tiny`: 3 items, k = 4): the teacher-forced stage, generation and the generation analysis,
  with the identity checks and a memory reading.
  - One judge smoke (gpu-a100) runs the original judge, both panel judges (24 prompts each) and the analysis on the
    7B smoke's answers before any full judge job starts.
