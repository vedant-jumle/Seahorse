# dream_subj pre-registration: does a restored slice need the subject's position?

*Written and committed 2026-10-22, before the full run. The only numbers that exist are from a 2-fact `--tiny` smoke
(dog_name, pool_hairdresser), used to check the code. Exploratory: there is one stated rule; everything else is
reported beside it. Code: `run.py` (model and forward passes, reusing dream_bind's loading, turn pieces and splice code
by import; nothing under `experiments/dream_bind/`, `src/`, `data/` or `slurm/` is changed), `subj.py` (pure helpers,
tested in `test_dream_subj.py`), `analyze.py` (tables, rule).*

## Known before this experiment

From `docs/dreaming/01-binding.md` and `experiments/dream_bind/PREREG.md` (addenda 2, 5, 6, 7), on Qwen2.5-1.5B:
- **R-min** (the K,V of 2 positions of the experience turn, the value and the next token, at every layer, kept at
  their stored positions and put back before the user's question turn) gives clean use 0.86 on use prompts but only
  0.66 [0.40, 0.89] on direct questions ("What's my dog's name?").
- **R-full** (the whole experience turn) gives 0.91 on direct questions.
- Teacher-forced, the right slice gives about +9.5 nats on the target at the measure prefix, the same as ctx.
- Stage 0: the answer reads the value from its own position, late. The question's subject holds no copy of the value
  (it may hold an order-ID pointer).

## The question

When a stored slice is put back, does answering a direct question need the **subject's own position** ("dog" in "My
dog's name is Pepper.")? That would be the entity-tracking two-step lookup: find the entity's position, then fetch the
attribute that shares its order tag. Or does any extra stored position help equally ("more evidence")?

## Conditions

Qwen2.5-1.5B-Instruct, fp32, eager attention, local RTX 4060.
- **Harvest:** K,V at every layer, from one pass over [head][experience user turn], as R-min in dream_bind.
- **Placement:** entries are kept at their stored RoPE positions.
- **Question start:** the question turn starts at n_head + n_seg, where R-full's would.

| name | kept entries of the experience turn (besides the head) | count |
|---|---|---|
| nomem | none (the plain question) | 0 |
| ctx | the experience sentence in the same user turn as the question (dream_bind's ctx) | |
| R-full | the whole experience turn | n_seg |
| R-min | value + next token. Must equal `bind.splice_keep("R-min")` (asserted) | 2 |
| R+subj | subject's last token + value + next | 3 |
| R+fill | the "is" token right before the value + value + next | 3 |
| R+rel | the relation token "name" + value + next | 3 |
| R-subj2 | subject's last token + value | 2 |
| T-min | text control (dream_bind addendum 5): "Pepper." as plain text in an earlier user turn | |
| T-subj3 | text control: the same 3 tokens as R+subj (subject, value, next) as plain text in an earlier user turn, e.g. "dog Pepper." | |

**Positions** (`subj.slice_positions`, by token-offset matching like `bind.ea_positions`):
- **Subject:** the last token of the first alias occurrence inside the experience sentence. Examples: "dog"; "friend"
  for "best friend"; "er" for "hairdresser"; "-law" for "mother-in-law".
  - Aliases: dream_bind config for the primary facts; "my {slot}" / "your {slot}" / "{slot}" for the pool.
  - Asserted: it lies before the value and is not the value. For person and pet facts it also equals ea_positions'
    entity token.
- **Value:** the target's first token (found exactly once in the turn, as `harvest_R`).
- **Next:** value + 1. Multi-token values keep the first sub-token + the next token (addenda 6 and 7).
- **Fill:** value − 1 if that token is "is".
- **Rel:** the last "name" token between the subject and the value.
- **Missing positions:** a fact with no "is" or no "name" skips that condition, and the skip is recorded. Every person
  and pet fact here has the form "My {slot}'s name is {v}.", so none is expected to skip.

## Items

- **Primary:** the 7 primary facts (dog, cat, sister, brother, grandmother, best_friend, boss).
- **Pool:** the pool facts built exactly as addendum 7 C builds them (dream_bind `pool_gen_items`: `sample_pool` seed 0,
  the hand-written slots excluded, aliases as above).
  - Only the person and pet facts are in the primary analysis: 30 person + 8 pet = 38 on this sampling.
  - **Attribute facts** (the 5 secondary items and 9 pool attributes) are run and reported separately, never in the
    rule. Those whose experience sentence contains no subject alias (job, hometown, birthday month, instrument) are
    excluded and listed. For attribute facts R+rel is skipped (there is no relation token; the slot's last token is the
    subject itself).
- **Analysed set:** "all" = the 45 person and pet facts. Splits: primary 7 vs pool 38.

**Prompts per fact:**
- the direct question (bench `related`);
- the use prompts (primary: the 2 bench use prompts; pool: the templated `use:` prompt from `fact_pool.yaml`);
- the bench's 4 balanced verify probes (2 Yes, 2 No).

## Measures (all teacher-forced; no judges)

1. **Answer score** (primary): log P(the measure prefix, e.g. "Your dog's name is" | the direct question), summed over
   the prefix tokens, as addendum 7 B3 defines it. Higher means "answers instead of refusing". Also recorded,
   descriptive: the first-token probability of the prefix's first token ("Your"), of the bare value's first token
   ("Pepper": an answer that starts with the name), and of the refusal-like tokens "I" + "Sorry".
2. **Target gain:** Δ log P(target | question + measure prefix) vs nomem (the existing TF measure), on the direct
   question.
   - **Use prompts:** dream_bind has no use-prompt measure prefixes. The use prompts' target gain is therefore measured
     with the fact's measure prefix forced after the use prompt. It is descriptive, and it is the measure for the
     direct-vs-use contrast below.
3. **TF yes/no:** balanced accuracy from the first-token log P(Yes) vs log P(No) on the 4 verify probes; per fact,
   (accYes + accNo) / 2.
4. **Attention to the restored subject entry** (R+subj and R-subj2 only, descriptive):
   - eager attention, direct question;
   - queries: the last prompt token, and the question's subject token (the last token of its first alias occurrence);
   - per layer, averaged over heads: the attention mass on the restored subject entry vs the restored value entry
     (and on next, the head, and the question's own tokens).

**Read only, never scored:** a greedy 40-token answer to the direct question, for 5 example facts named now (dog_name,
sister_name, best_friend_name, pool_hairdresser, pool_mother_in_law), so the report can quote text.

## Statistics

- **Bootstrap:** over facts, 2000 replicates, seed 7 (`seahorse.stats.boot_ci`). For use prompts (1 or 2 per fact) the
  bootstrap is two-level: facts, then prompts.
- **Paired differences:** computed per fact on the same prompts.
- **95% intervals:** percentile.

## Rule (decided on the direct question's answer score, over all included person and pet facts)

Contrasts are paired CIs of the answer-score difference.
- **"Subject position matters (two-step lookup)":** the lower bounds of (R+subj − R-min), (R+subj − R+fill) and
  (R+subj − R+rel) are all > 0.
- **"More evidence":** (R+subj − R-min) has a lower bound > 0, but at least one of (R+subj − R+fill) and
  (R+subj − R+rel) has a CI containing 0 with |point| < 0.25 × the (R+subj − R-min) point.
- **"Neither":** no 3-position condition beats R-min: the CIs of (R+subj − R-min), (R+fill − R-min) and
  (R+rel − R-min) all include 0.
- Otherwise **"in between"**.
- Implementation: `subj.verdict`, tested on synthetic numbers. The rules are checked in the order listed.

**Reported beside it:**
- the same contrasts for target gain and TF yes/no;
- R-subj2 vs R-min (same size: does the subject replace the next token?);
- the direct-vs-use contrast: per fact, (R+subj − R-min) target gain on the direct question minus the same on the use
  prompts. Two-step predicts it is positive;
- R-full − R-min, R-full − R+subj, ctx − R-full;
- T-subj3 − T-min and R+subj − T-subj3 (does the subject entry carry more than its token?);
- everything split primary 7 vs pool 38, and the attribute facts separately.

## Checks that fail the run

- R-full TF == the sentence as a prior user turn (last-position logits within 0.01; the existing identity).
- R-min's kept indices == `bind.splice_keep("R-min")`, and R-full's == `bind.splice_keep("R-full")`.
- The subject index < the value index, for every included fact.
- Every condition keeps the head plus exactly the specified number of entries, inside the experience turn, with the
  question at n_head + n_seg.
- The subject and value tokens == `bind.ea_positions`' entity and attribute (person and pet facts).
- Paris (the loaded model answers "What is the capital of France?" correctly).

## Known limitations (stated in advance)

- **The target gain is expected to saturate.** Every slice holds the value, and teacher-forced with the prefix the
  value is read from its own position (stage 0). Its contrasts are expected near 0. The answer score is the measure
  that can move.
- **The answer score counts one answer form.** It measures "Your dog's name is …" only. An answer that starts with the
  name ("Pepper.") scores low although it answers. This is why P(value) first and P("I" + "Sorry") first are reported
  beside it.
- **Word-piece subjects.** For several pool slots the subject's last token is a word piece ("er", "ster", "oise"). The
  T-subj3 text then reads oddly ("er Jack.").
- **Scope:** one model, short single-sentence facts of one template, and no generation-based clean use.
