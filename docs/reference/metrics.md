# Seahorse: what the numbers in the tables mean

*A plain guide to the measures in the experiment reports, mainly `ref_v1` (which reference to subtract when writing a preference) and `think_v1` (memory during Qwen3.5's thinking). For each measure: what it counts, a worked example, and what it can't tell you. The examples are real outputs from those runs where possible. Written 2026-10-06; §7b (judged measures and statistics of the pre-registered `core_v1` run) added 2026-10-08.*

---

## 0. How to read a table

**Rows** are preferences (vegetarian, loves_jazz, …) or groups of them (two-ended, one-of-many). **Columns** are conditions.

### Condition names
A condition is written `<reference>@<strength>`, e.g. `centroid@1`.

The **reference** is what we subtract when writing the memory. The memory stores "the state with the experience, minus the reference":

| Reference | Subtracts | Example for "I'm a huge jazz fan." |
|---|---|---|
| `opposite` | an opposite statement | minus "I can't stand jazz music." |
| `without` | nothing: the same conversation with no experience | minus the bare follow-up |
| `hum` | the average state over ordinary text | minus the model's "background hum" |
| `disclosure` | the average over 24 *other* things a user might say | minus the average of "I live in Japan", "My dog is called Pepper", "I don't own a car", … |
| `centroid` | the average over 8 *alternatives in the same category* | minus the average of "I love rock", "I love classical", "I love hip-hop", … |

The **strength** (`@1`, `@2`) is how hard the memory is pushed in, at each of the three layers it's added to.

Two reference conditions bracket everything:
- **`nomem`:** no memory at all. This is the model's default behaviour.
- **`ctx`:** the experience written straight into the prompt ("I'm a huge jazz fan. What should I listen to…"). This is the ceiling: what the model does when it plainly knows.

### Where each number comes from
Each preference has **6 prompts**:
- **1 related:** "What should I listen to while I cook dinner?"
- **5 ambiguous:** "Recommend an album to relax to.", "Suggest a fun night out in the city.", …

Each prompt is answered **11 times**: 1 *greedy* (the model's single most likely answer) and 10 *sampled* (with some randomness, like asking it again). So **every cell in a preference table is 66 answers.**

Every condition uses the same random numbers, so differences come from the memory, not from luck in sampling.

---

## 1. Score (`lean`): which way do the answers lean?

### How it is counted
Each preference has two hand-written word lists. For jazz:

| List | Words |
|---|---|
| **consistent** (fits the preference) | jazz, saxophone, Miles Davis, Coltrane, bebop, swing, bossa nova, … |
| **inconsistent** (the other way) | pop, rock, hip-hop, EDM, Taylor Swift, Ed Sheeran, … |

Each answer gets **one label**, by counting words from each list:
- more consistent words → **consistent**
- more inconsistent words → **inconsistent**
- equal, including no list words at all → **neutral**

Matching uses whole words, ignores case and allows plurals. Overlaps go to the longer phrase: in the no-alcohol lists, "non-alcoholic beer" counts once as *non-alcoholic beer* (consistent), not also as *beer* (inconsistent).

**Score = share of consistent answers − share of inconsistent answers.** The `cons` and `inc` columns show those two shares separately.

### Worked example
Six made-up answers to "What should I listen to while I cook dinner?" with a jazz memory:

| # | Answer | Jazz words | Pop words | Label |
|---|---|---|---|---|
| 1 | "Try some Miles Davis or Coltrane while you chop." | 2 | 0 | consistent |
| 2 | "A mix of pop hits and a bit of jazz." | 1 | 1 | neutral |
| 3 | "Lo-fi beats or a good podcast." | 0 | 0 | neutral |
| 4 | "Taylor Swift or Ed Sheeran, something upbeat." | 0 | 2 | inconsistent |
| 5 | "Smooth jazz or some bossa nova." | 2 | 0 | consistent |
| 6 | "jazz jazz jazz jazz jazz jazz …" | many | 0 | consistent (but a loop, see §2) |

3 consistent and 1 inconsistent out of 6 → score = 3/6 − 1/6 = 0.50 − 0.17 = **+0.33**.

### Reading the score
- **+1:** every answer leans toward the preference.
- **0:** no lean overall (balanced, or nothing either way).
- **−1:** every answer leans the other way.

**The baseline is not 0.** With no memory, vegetarian scores **−0.29**, because the model suggests chicken and salmon by default. So always compare with the `nomem` column (the `d_lean` column is the score minus `nomem`), and with `ctx` for how far there is to go.

### Score with loops removed (`lean_clean`)
This is the same score, counted **only over answers that aren't loops**. In the example, drop answer 6: 2 consistent and 1 inconsistent out of 5 → **+0.20** instead of +0.33.

Answer 6 doesn't show a preference. It's a broken answer that happens to be made of jazz words, and the plain score rewards it. **`lean_clean` is the honest number.** The plain `lean` is still in the reports, but it's inflated whenever loops are common.

### Negated items: the sign flips
For **hates_jazz** ("I can't stand jazz music.") the lists are swapped:
- consistent = pop, rock, … (away from jazz)
- inconsistent = jazz words

So **+ means the memory steers away from jazz (correct), and − means it steers *toward* jazz (backfired).**

no_alcohol works the same way without any swap: consistent = mocktail, juice, tea, coffee, …; inconsistent = wine, beer, cocktail, …. So + is correct.

### What it can't tell you
It counts words; it doesn't understand them.
- **Mentioning is not recommending.** With "I can't stand jazz" in the prompt (`ctx`), the model answers *"That is completely understandable! Jazz can be incredibly complex…"*, then recommends classical. The jazz mention counts against it, so `ctx` scores **−0.50** on hates_jazz even though the model is doing exactly the right thing.
- **Puns count.** *"Cooking dinner is a fantastic way to unwind, jazz up your evening…"* (`without@1`) counts as jazz.
- **Anything not in either list is invisible.** *"Explore your favourite podcast series, audiobooks…"* is neutral, even though it clearly avoids music.

So when a score is surprising, read some of the actual answers before believing it.

---

## 2. Loop rate (`loop`): how often does the answer break into repetition?

### How it is counted
Slide a window of **4 tokens** (roughly 4 words) along the answer, one step at a time. **Repetition = 1 − (distinct windows ÷ all windows)**: the share of windows that are repeats of earlier ones. An answer counts as a **loop if its repetition is 0.3 or more**. The loop rate is the share of the 66 answers that are loops.

### Examples (counting words for simplicity)

| Answer | Windows | Distinct | Repetition | Loop? |
|---|---|---|---|---|
| "Try some smooth jazz while you cook tonight." | 5 | 5 | 0.00 | no |
| "I love jazz. I love jazz. I love jazz." | 6 | 3 | 0.50 | yes |
| "Norway Norway Norway Norway Norway Norway Norway Norway" | 5 | 1 | 0.80 | yes |

Real loops from `ref_v1`:
- *"Cooking dinner jazz jazz jazz jazz jazz…"* (jazz, `centroid@1`: 91% of answers loop)
- *"The price of a "decent" winter coat Norway Norway Norway…"* (Norway, `centroid@1`)

### Why it matters
A loop is the memory **overdosing**: it pushes one word so hard the model can't say anything else. Loops are full of the target word, so they inflate every word-counting measure.

In `think_v1`, the report said the answer named the fact 84% of the time. Most of those were loops like *"Petra Petra Petra…"*; only 4 of 44 answers were clean.

### What it can't tell you
- **Short broken answers slip through.** At `hum@2` the answers are fragments like *"vegetarian vegan keto"* or *"trying to pick only * one food out"*. They're too short to repeat, so they don't count as loops, but they're clearly broken. The length column (§3) catches these.
- **Mild repetition** below the 0.3 cut-off passes as normal.

---

## 3. Length (`len`)

The mean answer length in tokens. A token is roughly a word or part of one. Answers are capped at 200 tokens, and normal answers usually hit the cap (about 195).

A big drop means the answers are collapsing. For example, `hum@2` averaged **41** tokens on the two-ended preferences: a sign the memory is breaking the model, not guiding it.

---

## 4. Leaks: `unrel_same`, `contam` and `fire`

### Unrelated identical (`unrel_same`)
Each preference is also tested on **4 unrelated prompts**:
- "Explain how tax brackets work."
- "Explain recursion in programming with a simple example."
- "What is the capital of Australia?"
- "Write a haiku about the ocean."

The greedy answer with the memory is compared **word for word** with the answer without it. **1.00 means every one is identical**: the memory stayed silent where it had no business firing.

It's a strict test: a single changed word counts as different. In `ref_v1` it was 1.00 for every preference in every condition.

### Contamination (`contam`)
The share of unrelated answers that contain a word from the preference's consistent list, minus the same share without memory. **0 means nothing leaked**, e.g. no jazz in the haiku.

### Fired (`fire`)
The share of prompts where the memory's gate was open, i.e. the memory decided "this is relevant" and added itself. On the related and ambiguous prompts it should be close to 1.

---

## 5. Logit-lens score (`lex_gain`): what does the stored memory vector contain?

### The idea
This one involves **no generated text at all**.

The model's very last step turns its internal state into a score for every word in its vocabulary (about 248,000 of them). The score means "how much do I want to say this word next".

Here we take the **memory vector on its own** and push it through that last step as if it were a state. That shows which words the memory itself pushes toward. It's like tasting one ingredient on its own instead of the whole dish.

**Score = the average score of the consistent words − the average score of the inconsistent words.** Only the first piece of each word is used; "Coltrane" may be split into pieces like "Col" + "trane".

### Reading the units
These scores are *logits*. Every +1 multiplies the odds by about 2.7:

| Logit difference | Odds of a jazz word vs a pop word |
|---|---|
| +1 | about 3× |
| +3 | about 20× |
| +9 | about 8,000× |

From `ref_v1` (layer 23):

| Memory | `opposite` | `centroid` |
|---|---|---|
| loves_jazz | **+2.5**: mildly prefers jazz words | **+9.3**: overwhelmingly prefers jazz words |
| hates_jazz | +2.5: mildly prefers *away* words (the negated item's swapped lists) | **−9.1**: the "I can't stand jazz" memory overwhelmingly prefers *jazz* words. The sign is lost. |

The top 20 words each memory vector pushes are listed in `vocab.txt`.

### What it can't tell you
- **What the model actually does with the vector.** Centroid jazz scores +9.3 and also produces 91% loops: lots of content, and it floods.
- **It's a rough reading.** It treats a layer 20–23 vector as if it were the final layer. It shows what the vector *contains*, not how the answers will come out.

---

## 6. The shape of the memory: length ratio and cosines

### Length ratio
How long each reference's memory vector is **before rescaling**, relative to `without` (which is 1.00 by definition):
- **`hum`, about 2–3 (long).** Subtracting the hum leaves *everything* about the moment, including its whole topic.
- **`centroid`, about 0.3–0.7 (short).** Subtracting close alternatives cancels almost everything except the specific thing, e.g. "jazz rather than rock or classical".

All vectors are then **rescaled to the length of `without`**, so only their directions differ. One side effect: centroid gets boosted about 2.5×. Since it's concentrated content, that makes it a much stronger dose of the concept, which is part of why it loops so much. A fairer comparison would match strengths by loop rate, not by length.

### Cosine
Cosine measures how far two vectors point the same way. A compass gives the idea:

| Directions | Cosine |
|---|---|
| north vs north | 1 |
| north vs north-east | about 0.7 |
| north vs east | 0 (unrelated) |
| north vs south | −1 (opposite) |

The model's states have 2,048 directions, not 2, and two unrelated vectors land near 0.

Three cosines appear in the reports:
- **Between two references (`cos(a, b)`):** for the same memory. `without` vs `disclosure` is about 0.7, so they're similar. `opposite` vs `hum` is about 0.1, so they're unrelated.
- **With the key (`cos_key`):** the memory vector (transformed the same way keys are) against the memory's own stored key. **`hum` gives 0.6–0.85; every other reference about 0.**
  - High means the memory vector is the memory's *address*, its topic ("planning meals"), rather than the preference.
  - Firing it just re-injects the topic into a conversation that's already about that topic.
- **With the topic (`cos_topic`):** the same check without the transformation, against the follow-up's topic direction. `hum` gives about 0.9.

---

## 7. Measures for facts and yes/no questions (`think_v1`, benchmark reports)

| Measure | What it counts | Example | Watch out |
|---|---|---|---|
| **Fact named** (`ans_tgt`) | Share of answers that contain the exact fact | "Pepper" in answer to "What's my dog's name?" | Counts loops: *"Petra Petra Petra…"* scores. Recount clean answers by hand. |
| **Fact in thinking** (`think_tgt`) | The same, in the thinking text | | *"Thinking Pepper Pepper Pepper… (×383)"* counts as a hit |
| **Balanced yes/no** (`bal`) | Questions where Yes is right ("Is my dog called Pepper?") and questions where No is right ("Is my dog called Max?"); accuracy on each half, then averaged | 0.5 = chance, 1.0 = perfect | Why balanced: a model that always says "No" gets 100% of the No-questions right. Averaged with the Yes half, it correctly scores 0.5. |
| **Confabulation** (`confab`) | The answer states a *wrong* value | "Your dog's name is Max" | Only catches stated values, not vague guesses |
| **Thinking cut off** (`forced`) | Thinking hit the 384-token limit and was cut off | | In `think_v1` this happened about 95% of the time *even without memory*, so many "answers" are leftover thinking |

---

## 7b. Judged measures and statistics (`core_v1`)

Word lists and loop checks missed two things: mentions vs recommendations (§1), and text that is garbled but not repetitive (*"It's a very Leeds."*). So core_v1 adds a **judge**: a second language model (Qwen3.5-9B) reads every answer with a fixed rubric (`experiments/core_v1/judge_prompts.yaml`).

| Measure | What it counts | Example (made-up unless it's a quote from a run) |
|---|---|---|
| **Judged lean** (`j_lean`, the primary preference measure) | +1 if the answer *coherently recommends* toward the user's trait, −1 if it coherently recommends away, 0 if neutral **or incoherent** | *"Try a lentil curry or a mushroom risotto."* with a vegetarian memory → +1. *"vegetarian vegan keto"* → 0, because it's incoherent. |
| `j_dir` | the same direction, ignoring coherence | the fragment above → +1 here |
| **Incoherent** | the share of answers the judge calls incoherent: loops, fragments, word salad, off-task text | *"Since you do not your native language. Let you it in it more."* |
| **Clean use** (`use_judge_clean`, the primary fact measure) | on "use" prompts, the answer uses the *correct* fact, correctly, in a coherent answer | *"Hi Petra! Can't wait for your visit next weekend…"* → yes. *"Petra Petra Petra…"* or *"My name is Omar"* → no. |
| `mention_unclean` | the target is mentioned, but not in a clean use: loops, garble, misuse | the fact memory at α 2: 86–89% of answers |
| `self_claim` | the assistant claims the user's trait or fact as its own | *"Teal is my favourite colour!"* |
| **NLL** (coherence, no judge) | how surprising the answer is to the model *without* memory: the mean negative log-probability per token. Higher means stranger text. | no memory ≈ 0.9; placebo memory at α 2 ≈ 3.9 (2B) |

**Why "incoherent = 0" matters.** A dose that makes 60% of answers incoherent can still raise word counts, because the loops and fragments are full of target words. The judged lean doesn't rise. In core_v1, at α 2 on the 2B, the word-list lean went up (+0.51) while the judged lean didn't clear zero. Most of the extra "lean" was broken text.

**Statistics used in core_v1:**
- **Gain** = memory minus no memory, *on the same prompt with the same random numbers*, so prompt difficulty cancels.
- **Group number** = the average over items of each item's average over its prompts. Every item counts equally.
- **95% confidence interval** from a two-level bootstrap. Resample the items, then the prompts within each item, 2,000 times; keep the middle 95% of the resulting averages. **If the interval includes 0, we can't claim an effect.**
- **ctx rule:** a preference item whose in-context answers don't lean at least 0.3 more than no memory is flagged and set aside. If the model can't show the preference even when told, the item can't test memory.
- **Pre-registration:** the pass/fail rule for each claim was written before the run (PREREG.md). A claim that fails its rule is reported as failed, even if a different rule would have passed it.

**What the judge can't tell you:** whether it's right. It's a model too, and the larger of the two under test. 200 answers are set aside for a hand check. Until that's done, judged numbers are provisional.

---

## 8. A checklist before believing a number

1. **Compare with `nomem` and `ctx`**, not with 0.
2. **Use `lean_clean`, not `lean`**, and look at the loop rate beside it. A high score with a high loop rate is a flood, not a preference.
3. **Check length, judged incoherence and NLL.** A collapse in any of them means broken answers that the loop rate missed. Prefer the judged lean over word lists.
4. **Check `unrel_same` = 1.00**: the memory shouldn't touch unrelated questions.
5. **Read 3–5 actual answers.** Mentions, puns and negation all fool the word lists.
6. **Mind the sample size.** One preference's score rests on 66 answers and can move by about ±0.2 by chance. Trust differences bigger than that, or ones that repeat across several preferences.
