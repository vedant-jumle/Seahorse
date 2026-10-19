# Seahorse: where it fails, and why

*A companion to [where-we-are.md](where-we-are.md) and [how-it-works.md](how-it-works.md). It collects every way the memory has failed so far, with a real example of each, the best explanation we have, how we know, and what might fix it. Same plain style. Written 2026-10-01. Updated 2026-10-07 with the Qwen3.5 results (think_v1, ref_v1): failures 13–16, plus updates to 1, 2, 4 and 7. Updated 2026-10-08 with cross-layer injection (xlayer_v1) and the pre-registered run on Qwen3.5-2B and 9B (core_v1): updates to 1, 2, 4, 10, 11, 12 and 13. Updated 2026-10-11: failure 11's judge check was done by a cross-family panel instead of by hand (note at the end of failure 11). Updated 2026-10-13 with dreaming experiment 1 (dream_bind): failures 1, 2 and 15 have a working route through a stored attention slice, and there is a new failure 17 (separately stored facts collide). Updated 2026-10-19 with addendum 6 of experiment 1 and the finished experiment 2: updates to 7, 8, 9, 13 and 17, and a new failure 18 (retrieval errors become confident wrong facts). Related papers: [literature.md](reference/literature.md).*

---

## The short version

| # | Failure | Status | Root cause, in one line |
|---|---|---|---|
| 1 | It can't give the model a premise to reason with | **Open for the steer channel; works with a stored attention slice** | Adding a vector nudges; reasoning needs something the model can *attend to* |
| 2 | The exact fact rarely comes out; the gist does | **Open for the steer channel; works with a stored attention slice** | Recall is a blend, and the right answer starts out extremely unlikely |
| 3 | It stores the concept instead of the relation ("peanut" → peanut butter) | **Fixed only with a hand-made opposite** | The plain shift carries the topic as well as the point |
| 4 | Preferences are either faint or crude | **Open** | Clean writing cancels too much; plain writing carries too much |
| 5 | It fired everywhere | **Fixed** | Keys of unrelated situations looked alike, and nothing set a bar |
| 6 | The last memory written won | **Fixed** | Step-by-step writing lets new memories overwrite old ones |
| 7 | It fires only where it was filed | **Open (partly fixed for facts: a subject-span key finds the slice)** | A memory is filed under its follow-up sentences, not under the experience |
| 8 | More memories means weaker memories | **Partly open (slices: retrieval holds to 59 facts)** | All memories share one table and must compromise |
| 9 | It needs a hand-made comparison to write anything | **Mostly fixed for effects (a model writes the references); open for the sign** | A "shift" is defined as a difference from a counterfactual we supply |
| 10 | Too much strength breaks the text | **Managed** | An added vector can overpower the model's own state |
| 11 | We fooled ourselves with our measurements | **Fixed** | Unbalanced questions, template artefacts, narrow scores |
| 12 | We don't know how far any of this generalises | **Partly open** | Two small models, a handful of scenarios |
| 13 | Dislikes turn into likes | **Fixed with the true sign; open for reading the sign** | Negation is faint inside the model; most references keep the concept and lose the direction |
| 14 | Single-item preferences flood or do nothing | **Open** | A constant push at every word suits whole-answer leanings, not one word in one place |
| 15 | The model claims the memory as its own | **Open for the steer channel; fixed by the user-turn slice** | Injected states read as the model's own thoughts; nothing stores *whose* a memory is |
| 16 | Turning memory up during thinking floods the thinking | **Open** | A constant push on every thinking word is an overdose, not a thought |
| 17 | Separately stored facts collide when used together | **Open (joint re-encoding fixes it; the cheap tag repair fails)** | Slices stored in separate sessions carry the same order tag (the tag exists, but adding it back does not fix the swaps) |
| 18 | Retrieval errors become confident wrong facts | **Open (new)** | A wrong slice is used as if it were the right one: it pushes the right answer down and the model states the wrong value |

The failures cluster into five root causes (E, binding, was added with dreaming experiment 1; 18 joins E and C), discussed at the end:
- the limits of **adding a vector**
- **what gets written**
- **where it gets filed**
- **how we measured**

---

## 1. It can't give the model a premise to reason with

**What you see.** With the vegetarian memory on, *"Would a beef burger be a good lunch for me? Answer yes or no."* gets **"Yes."** in every design we've tried, old and new, one memory or six. With the experience in the prompt, the answer is "No." The same goes for "Is a peanut butter cookie safe for me to eat?".

**Why.** The memory works by adding a vector to the model's state at one layer. That can tilt *what comes to mind*. Reasoning works differently. To answer the burger question, the model has to *look back at* the fact "I'm vegetarian" while working out the answer, which is what attention does with words in the prompt. An added vector is not something the model can look back at. It's more like a mood than a sentence. This fits what research on emotions in these models found: they keep no lasting internal state, and longer-range information is carried by attending back to earlier words.

**How we know it's not a measurement problem.** The memory *does* fire on these questions, at 60–80% of its strength on related prompts. It's present; it just isn't used as a premise.

**What might fix it.** Probably nothing within this channel. The natural answer is a second, *episodic* channel that puts stored moments back where attention can reach them (the other session's "ladder"). This failure is the main reason to think of Seahorse as two channels: this one for leanings, another for premises.

**Update (Qwen3.5-2B, think_v1).** Still at chance on the newer model:
- at every layer from 4 to 23
- with one layer or three
- with memory during thinking, during the answer, or both

Even when the thinking was full of the right name, the yes/no answer didn't change.

Two untested escape routes remain, both from Lindsey 2026 ([literature.md](reference/literature.md)):
1. Injected concepts act as *thoughts* about two-thirds of the way through a model. We have always injected near the end, the "mouth". xlayer_v1 tests reading late and injecting in the middle.
2. Using an injected state to reason appeared mainly in large models, so part of this failure may be the 2B model's size.

**Update (xlayer_v1, core_v1).** Both routes are now tested, and neither helped:
- **Middle-layer injection** left yes/no unchanged.
- **The 9B model** behaves like the 2B.

In the pre-registered run, the balanced yes/no accuracy never rose on either model, at any dose. What the memory does do is tilt answers toward "Yes" or toward "No" across the board, which helps one kind of question exactly as much as it hurts the other. On the 2B, the fact memory at α=2 even made yes/no *worse*.

**Update (2026-10-13, dream_bind): the predicted route works.** The natural answer above was a second channel that puts a stored moment back where attention can reach it. Experiment 1 built it. With the model's own keys and values (K,V) for the sentence put back in a later session, gated on a subject match, balanced yes/no accuracy goes from 0.46 (no memory) to 1.00 for the whole turn (Qwen2.5-1.5B), 0.86 for just the name and the full stop, and 1.00 for a slice stored after an unrelated chat. On the hybrid Qwen3.5-2B the same holds when the recurrent state is included (0.93 to 1.00); the attention blocks alone give 0.50 to 0.61. The state alone gives 0.93 yes/no but no recall: it can recognise a fact without being able to say it. So premises were never out of reach for the model; they needed something to attend to. Folders: `results/dream_bind_gen_20261012_0038/`, `dream_bind_state2b_20261012_1759/`. Details: [01-binding.md](dreaming/01-binding.md).

---

## 2. The exact fact rarely comes out; the gist does

**What you see.**
- **Dog:** "Biscuit" appears in **7 of 300** samples (no memory 0; experience in context 294).
- **Sister and job:** "Ines" and "deep-sea welder" appear **0 times** in 300.

But something related often does:
- *"You work as a **welder** at a welding equipment company"* (without memory: "You work as a Qwen, created by Alibaba Cloud")
- *"That's great! **What's her name?**"* when the user mentions buying their sister a present

**Why. Two reasons on top of each other.**
1. **Recall is a blend.** All stored moments are mixed into the same table, and reading returns a similarity-weighted mix. A mix naturally lands *near* the stored content rather than exactly on it. That's why the gist ("welder") survives while the detail ("deep-sea") doesn't.
2. **The right answer starts from almost nothing.** Without memory, the model gives "Biscuit" about a 0.3% chance after "Your dog's name is", "Ines" about 1 in 250,000, and "deep-sea welder" effectively zero. The memory multiplies these by 10× to 80,000×, which is huge, but from such a low start the answer is still far from the top. Meanwhile the model's habit of answering "I'm sorry, as an AI I don't have access to your personal information…" is very strong. Instruction tuning put it there, and a nudge can't beat it.

Multi-word answers ("deep-sea welder") are hardest, because every word has to come out right.

**What might fix it.**
- More strength, but only where memory fires, since leakage is now controlled.
- Storing the fact where it's *needed* (at the point of answering "your dog's name is…") rather than as a general shift.
- A small trained read path that turns the gist into the exact answer.

**Update (Qwen3.5-2B, think_v1).** Using three late layers at once got the exact name out far more often, but mostly as loops:
- *"Pet Petra Petra Petra Petra…"*
- *"Thinking Pepper Pepper Pepper… (×383)"*

The counts:
- In short continuations, 91% mentioned the fact; 22% did so cleanly.
- In full answers, about 4 of 44 named it cleanly (with the fact in the prompt: 28 or more).

So more strength traded "never" for "flood", with a narrow window in between. Single words are the hardest kind of content for steering (failure 14). They may belong to the episodic channel. The next attempts are injecting in the middle layers (xlayer_v1) and a "thermostat" that sets the concept to a target level instead of adding to it.

**Update (xlayer_v1, core_v1): why it fails, and on two model sizes.** A fact memory stores *an intention to say a word*, not a statement:
- **Where the name exists:** read through the logit lens, the stored memory's top word is the name itself, but only in the last ~15% of the model's layers. At half depth it ranks around 70,000th. This holds for the 2B and the 9B alike.
- **What injection does:**
  - Injected late, the word comes out as loops.
  - Injected in the middle, it fades or turns into confused text: *"My name is Omar."*, *"I am an AI named Pepper"*, *"It's a very Leeds."*
- **The numbers:** the judge counts a "clean use" only when the fact is used correctly in a coherent answer. Clean use stays ≤ 15% at every dose (0.5–3) on both models, against 96–98% in context. At α ≥ 2, 85–94% of answers mention the fact without a clean use.

The fact most likely lives where it was said, reachable by attention. That makes it the episodic channel's job, not something to fix in this one.

**Update (2026-10-13, dream_bind): confirmed, and the channel works.** Stage 0 showed that in context the answer reads the value from the attribute's own position ("Pepper"), late: patching the state of the question's subject flips 0.00 of the preference on both models, patching the attribute flips 1.00. Putting back a stored slice of that position restores the exact fact. Clean use (Qwen2.5-1.5B, a judged correct, coherent use of the fact): no memory 0.00, experience in context 0.90, whole turn's K,V 0.87, **just the name and the full stop 0.86**, stored after an unrelated chat 0.80. On the 2B, two positions plus the recurrent state give 0.97. The same two tokens as plain text give only 0.27 (0.24 on the 2B), so a stored slice carries more than its tokens. A wrong slice gives a confident wrong value (41% of use answers). Unrelated prompts stay word-for-word unchanged (1.000) only with the gate; without it, 6% do. The state of the 2B is about 4.7M numbers (~19 MB) whatever the text length, so on a hybrid this is not compact for a single fact. Details: [01-binding.md](dreaming/01-binding.md), Results.

---

## 3. It stores the concept instead of the relation

**What you see.** With a plainly written peanut-allergy memory:
- *"What's a good sandwich to pack for lunch?"* → *"A **peanut butter and jelly sandwich** is a classic and delicious option…"*
- Across four open questions, safe answers drop from **68/80** (no memory) to **41/80**.

**Why.** The shift we store is "how the model's state changed after hearing *I have a severe peanut allergy*". The biggest part of that change is simply *peanuts are now on the table*: the topic. The smaller part is *and they're dangerous*: the relation. Adding the whole shift back primes the topic, and the model does what it normally does with peanuts.

**What fixes it (partly).** Contrastive writing: subtract the state after an *opposite* experience ("I really love peanuts"). The shared topic cancels and the relation survives. Safe answers go back to 72/80. But:
- (a) it needs a hand-made opposite (failure 9)
- (b) it overshoots into failure 4: the result is clean but faint

**What might fix it properly.** Replace the hand-made opposite with *the model's typical state when talking about this topic*, learned from ordinary text. That would cancel the topic using only data the system can collect itself.

---

## 4. Preferences are either faint or crude

**What you see.** Vegetarian memory, answers with no meat or fish across 80 sampled answers:

| no memory | contrastive (clean) | plain (crude) | experience in context |
|---|---|---|---|
| 23 | 25 | 35 | 76 |

- The clean version barely moves anything.
- The crude version moves more but sometimes breaks: *"What would you like to eat? vegetarian, vegetarian and vegetarian, vegetarian and vegetarian and vegetarian…"*
- The Norway memory never once produced kroner or "Norway" (0/80; in context 50/80).

**Why.**
- **Contrastive writing cancels too much.** "Vegetarian" and "loves meat" share a lot, including part of what makes the preference *act* like a preference. What's left is a thin direction.
- **Plain writing carries too much.** It includes the whole concept, so at full strength the concept floods the output.
- **We also overestimated this effect earlier.** Our old score compared the model's preference between *one* pair of phrases ("mushroom risotto" vs "grilled steak") and showed large shifts. Real sampled answers have endless ways to include meat, and the lean was much weaker than that one pair suggested.
- **Layer 26 of 28 is late.** There are only two layers left for the nudge to be integrated into what the model says.

**What might fix it.**
- Something between clean and plain (partial cancellation, or the topic-based baseline from failure 3).
- An earlier or multi-layer read.
- Judging preferences by **rates over many answers**, which the benchmark now supports, so we can see what actually works.

**Update (Qwen3.5-2B, think_v1 and ref_v1).** On the newer model with three layers, **vegetarian is no longer faint**:
- consistent answers rose from 12% to 94%
- answers mentioning meat fell from 79% to 3–15%, fewer than with the fact in the prompt

What remains is item-specific:
- Preferences that shape the whole answer (diet, hiking, budget) work.
- Single-item ones (a country) don't (failure 14).
- Dislikes need the opposite (failure 13).

ref_v1 also explained "faint vs crude": the clean (contrastive) version keeps the *direction* but can lose the *concept*; the crude (plain) version keeps the concept and loses the direction.

**Update (core_v1, judged): faint again, but honestly measured.** With a judge model that counts incoherent answers as zero, on 15 preferences:
- **α=1:** leanings move significantly but modestly, about +0.2 (2B) and +0.1 (9B) on a −1..+1 scale.
- **α=2:** word counts keep rising, but 41–61% of answers become incoherent, and the coherent lean no longer clears zero.

The pre-registered claim (a gain at both doses) failed on both models. The effect is uneven: some preferences move well ("early riser", "prefers quiet", "vegetarian"), others not at all ("celiac", "lactose intolerant"). Part of the stage-7 "94%" was text the judge would call incoherent. The open questions are a finer dose search around α=1, per-item doses, and a gentler way to apply the memory (the thermostat).

---

## 5. It fired everywhere *(fixed)*

**What you saw.** The old memory recalled about **40%** of a stored memory on completely unrelated questions, and changed answers it had no business touching: *"Ocean's vastness, Whispers of the deep, Peace in **every** wave."* About two-thirds of that stray recall landed on the chat's formatting tokens (system prompt, turn markers).

**Why.** The model's states all share a large common component and vary mostly along a few "loud" directions. Keys built from them looked alike for any two situations (similarity about 0.2 whether related or not). The memory had no threshold, so any resemblance produced recall.

**How it was fixed.**
- Keys became the **gist of the conversation**, with the loud shared directions scaled down (whitening).
- Recall happens only above a **threshold** set by how well ordinary text matches.

Result: unrelated answers are now **word-for-word identical** to no memory in **24 of 24** cases, even with six memories together. Recall on related prompts got *stronger* at the same time.

**What's left.** This was shown on six scenarios. It needs confirming on the larger benchmark.

---

## 6. The last memory written won *(fixed)*

**What you saw.** With six memories written one after another into the same table:
- the **last** kept all of its effect
- the **first** was not just faded but pushed the *wrong* way (vegetarian: −0.83 of its effect when written first, +0.90 when written last)

**Why.** The original writing rule stores each new memory by correcting the table so the new key returns the new shift *exactly*. Any older memory whose key overlaps gets partly overwritten in the process. Recurrent neural networks forget for exactly this reason (recent inputs overwrite older ones), and for us it was made worse because keys overlapped so much (failure 5).

**How it was fixed.** Write with **least squares**: keep running totals over all memories and solve for the table that serves *all* of them as well as possible. Totals don't care about order, so order no longer matters.

**What it cost.**
- Each of six memories now keeps about **half** its solo strength.
- **Contradictions now average instead of overwrite.** Writing "my dog is Max" after "my dog is Biscuit", under the same gist, would give a compromise rather than an update.

That second point gives up one of the original idea's hippocampal properties (reconsolidation). Deciding *when* a new memory should override an old one is a job for the planned modulator.

---

## 7. It fires only where it was filed

**What you see.** The Norway memory stayed completely silent on *"What's a reasonable monthly budget for groceries?"* and *"How much does a cup of coffee usually cost at a cafe?"*. Its match never crossed the threshold. Yet these are exactly the questions where living in Norway matters.

**Why.** A memory's key is the gist of the **follow-up sentences** it was written with, not of the experience. The Norway memory was written alongside "I want to buy a new winter coat", "I'm trying to plan a weekend trip" and "Can you explain what a mutual fund is?". So it's filed under *coats, trips and mutual funds*, and a coffee question doesn't resemble any of them. The fix for failure 5 (a strict threshold) makes this worse: selective sometimes means *too* selective.

**What might fix it.**
- Keys that describe the experience itself (possible once writing no longer needs the hand-made comparison).
- More, and more varied, moments per memory.
- A softer threshold.
- Recall that also considers the *kind* of question ("prices", "shopping").

**Update (Qwen3.5-2B, think_v1).** On the newer model, the gate opened on every related and ambiguous preference prompt tested (12 of 12). For example, the hiking memory fired on *"Suggest a birthday gift I'd enjoy."*. But those prompts were written to sit fairly close to the follow-ups. The grocery and coffee questions that Norway missed before haven't been retested.

**Update (2026-10-19, dream_bind addendum 6): for facts stored as slices, the filing problem is mostly solved by choosing the key.** A key built from the subject's own words in the experience ("my dog") finds the right slice 0.85 to 0.92 of the time (Qwen2.5-1.5B, 59 facts), against 0.42 to 0.61 for a key built from the whole experience text, which is the "filed under the follow-ups" key of this failure. So filing under *what the memory is about* works better than filing under *the situation it was written in*. It does not help the steer channel's leanings (not retested), and 16 of the 59 facts had no subject alias and fell back to the whole-text key. Details: [01-binding.md](dreaming/01-binding.md), addendum 6 B.

---

## 8. More memories means weaker memories

**What you see.** Alone, the dog memory yields "Biscuit" 7 times in 300. Stored together with the five others, **once**. Measured more broadly, each memory keeps about half its solo effect when six share the table.

**Why.**
- All memories live in one table and the least-squares solution has to compromise between them, especially where their keys overlap.
- The blend also means a recalled shift carries some of the *other* memories' content.

**What we don't know.** How this curve continues. Six is the most we've tested; the benchmark's 60-fact pool is ready for the real capacity curve.

**What might fix it.** Better-separated keys (so memories overlap less), separate subspaces per memory, or sparse, selective storage (store fewer, more salient moments). The neuromodulation idea was partly motivated by this.

**Update (2026-10-19, dream_bind addendum 6): the capacity curve for slices.** For the episodic channel (a stored slice per fact, Qwen2.5-1.5B), the right slice is found 0.85 to 0.92 of the time with 10, 20 and 59 facts stored, with no sign of collapse, and each slice is its own object rather than a share of one table. With the right slice, the answer is as good as having the sentence in context (+9.47 nats vs +9.48). The cost of more facts is wrong picks (8% to 15% at N = 59, failure 18), not weaker memories. Not measured: more than 59 facts, and the 52 pool facts are weaker even with the right slice (0.62 vs 0.83 to 0.87 for the 7 hand-picked ones; unexplained). The steer channel's six-memory curve above is unchanged.

---

## 9. It needs a hand-made comparison to write anything

**What you see.** Nothing in a live conversation. This failure is in the method itself. Every memory was written by comparing *three* versions of a message: with the experience, without it, and with its opposite. A real conversation only ever has the first.

**Why.** We defined "what to remember" as *the difference an experience makes*, and a difference needs something to compare against. We supplied that something by hand. That was fine for asking "can this channel carry memory at all?". It is not how a real memory could work.

**What might fix it.**
- **"Without"** can be automated: re-run the conversation with one earlier sentence left out, and use attention to choose which sentences are worth testing. It costs compute, perhaps in a consolidation pass after the conversation, but needs no human.
- **"Opposite"** is human knowledge and has no automatic equivalent yet. The most promising replacement is the topic baseline from failure 3.
- Alternatively, train a small network to predict the shift from the conversation alone, using our hand-made comparisons as training data.

This is the largest gap between the current system and the project's goal of "memory formed from experience".

**Update (2026-10-19, experiment 2): model-written references reproduce the effects.** A 9B model, shown only the experience sentence, wrote the opposite and the eight alternatives (the "without" needs no writing). The memories built from them behave like the hand-made ones. On 7 dislikes the generated opposite moves answers the right way (+0.19 [+0.01, +0.40]; the hand-written opposite +0.23) and the generated centroid the wrong way (-0.39 [-0.55, -0.21]; core_v1's hand-written one -0.38), under every judge (S2). On PrefEval (200 preferences, no hand-written references at all) the generated opposite improves preference-following by +0.03 to +0.10 at α = 1 depending on the judge, against +0.43 for the experience in context (P1 holds). It is small, but it needs no human. Two caveats:
- The generated references are different sentences from ours: only 10 of 29 opposites reproduce the hand-written shift closely (cosine test S1 fails), so "the same memory" is not the claim, "the same effect" is.
- The dreamer is a 9B model writing for the 2B, and it still makes mistakes (it kept consequence clauses: "I have celiac disease, so I can eat gluten", "I enjoy virtual reality experiences due to motion sickness issues"; one item, has_young_kids, produced no usable alternatives).

What is still not done without a human or an outside rule is the **sign**: whether a preference is a leaning toward or away from its concept (see failure 13). The earlier idea of a general like/dislike direction (ref_v1 note above) was built and **fails**: it reads the sign of only about half of the items (accuracy 0.44 to 0.52 on 23 signed items). Details: [02-scaffold.md](dreaming/02-scaffold.md), Results.

**Update (ref_v1).** Two pieces of the scaffold now have candidate replacements:
- **The *what*** can come from subtracting the average of *other* things a user might say, which needs no hand-written opposite. In ref_v1 this "disclosure" reference carried the concept about as well as "without".
- **The *which way*** could come from a general like/dislike direction, built once from many "I love X" vs "I hate X" pairs, instead of a hand-written opposite per memory. This is untested.

---

## 10. Too much strength breaks the text *(managed)*

**What you saw.** Doubling the read strength from 2 to 4 broke everything in v0. Even at the normal strength, the crude preference memory produced *"vegetarian, vegetarian and vegetarian…"*.

**Why.** At most positions the added shift is only 20–30% the size of the meaningful part of the model's state. But at some positions, or with a strong memory, it becomes as large as the state itself, and then the memory isn't nudging the model any more; it's overriding it. Mood-like and drug-like dose curves look the same: helpful in a window, harmful beyond it.

**How it's managed.** Strength stays at 2, and the threshold keeps the shift away from positions where it doesn't belong. Norm-preserving reads (rotating the state instead of adding to it) are available if we need more headroom.

**Update (core_v1): α=2 was already too much.** On Qwen3.5, with the memory added at three layers, a judge rated the answers:

| Strength | Incoherent answers (2B) | Incoherent answers (9B) |
|---|---|---|
| no memory | 14% | 3% |
| α=1 | 22% | 9% |
| α=2 | 61% | 41% |
| α=3 | 85% | 80% |

A placebo memory ("I had a coffee this morning") at α=2 is just as damaging, so it's the dose, not the content. The working window is around α=1 per layer; the default is now 1, not 2.

---

## 11. We fooled ourselves with our measurements *(fixed)*

This is the least glamorous failure and the most important lesson.

- **The yes/no trap.** All of our early reasoning questions had "No" as the right answer, and the model drifts toward "No" whenever anything in its state changes. Even a neutral placebo sentence does it. So any memory looked like "reasoning progress". The benchmark now balances Yes and No answers and scores them separately.
- **The template trap.** Writing memory on the chat's formatting tokens produced fake effects, including part of that "No" drift. Those tokens are now excluded.
- **The single-pair trap.** Preference strength measured by one pair of phrases suggested big effects that sampled answers don't show (failure 4). Preferences are now judged by rates over many answers.
- **The averaging trap.** "Facts survive when memories are combined" was really just *the last fact written* surviving. The average hid it.
- **Random baselines matter.** When choosing which words to store, picking words at *random* worked as well as our clever choices. Without that control, we'd have credited the clever choice.

**The fix** was a proper benchmark (48 items, 60 facts, balanced questions, a placebo condition for the noise floor) and the habit of checking text, not only scores.

**Two more traps, found later:**
- **The loop trap** (think_v1). "Names the fact" counted *"Petra Petra Petra…"*. Loops are now counted separately.
- **The coherence trap** (xlayer_v1, core_v1). "Not a loop" still isn't "makes sense". Garbled answers like *"It's a very Leeds."* passed the loop check.

The pre-registered run (core_v1) therefore judges every answer for direction *and* coherence with a judge model, fixes its pass/fail rules in advance (`experiments/core_v1/PREREG.md`), and puts confidence intervals on every number. 

**Update (2026-10-11):** the planned hand check of 200 answers was replaced by a cross-family judge panel (gpt-oss-20B, Gemma-4-12B, Llama-3.1-8B, Qwen3.6-35B-A3B re-judged all 34,200 answers; `results/core_v1_20261007_1820/panel/panel_report.md`). The fact and dislike-direction verdicts (C3, C5) hold under every competent judge. The leanings verdict (C2) is judge-sensitive: it fails under each single judge but holds under majority votes, and our judge is the strictest on coherence (62% coherent vs 72–76%). Under gpt-oss and Gemma the 2B's leaning gain at α=2 (+0.36, +0.42) exceeds the gain at α=1 (+0.29, +0.24), so the "narrow window around α=1" is judge-sensitive too. Llama-3.1-8B was excluded after it labelled 92% of fact answers as "uses the fact" with the target string missing in 48% of them. No human labels exist.

---

## 12. We don't know how far any of this generalises

**What we have.**
- one small model (1.5 billion parameters)
- one layer chosen for it
- six hand-written scenarios for most results
- three follow-up sentences per memory
- full-precision arithmetic on one GPU slice

**What we don't know.**
- whether a larger model carries more through this channel (it might hold richer leanings, or resist nudges more)
- whether the chosen settings (layer 26, 256 key directions, strength 2, the 95% threshold) transfer to other models
- whether the six scenarios are typical

**What would tell us.** The 48-item benchmark, then the same design on a 7-billion-parameter model.

**Update.** The design carried over to a second, different model (Qwen3.5-2B, a hybrid with linear and full attention and built-in thinking) with the same key, threshold and write rule; only the layers changed. Selectivity held there too.

**Update (core_v1).** It also carried over unchanged to **Qwen3.5-9B**, at the same relative layer depths. Every pre-registered claim came out the same on 2B and 9B, except C4, where the 2B failed on a technicality. Selectivity, the fact limits and the direction result all repeat. The 9B is somewhat more robust to strong doses, but no better at leanings, facts or premises. Models beyond 9B are untested.

---

## 13. Dislikes turn into likes

**What you see.** A memory written from *"I can't stand jazz music."* makes the model recommend jazz:
- *"Lo-fi Jazz"*
- *"Cooking dinner is a wonderful time to jazz up your meal with jazz, jazz jazz jazz…"*

*"I don't drink alcohol."* makes it lead with *"Beer: a craft beer, a lager, or a stout"*. This happens with every way of writing the memory except subtracting the hand-made opposite.

**Why.** Inside the model, "I can't stand jazz" is mostly *jazz*, with a faint "can't stand" attached. A memory is "the state with the experience minus a reference":
- If the reference doesn't mention jazz (nothing, other genres, other disclosures), jazz is what's left, and the faint negation doesn't survive.
- Only the opposite ("I'm a huge jazz fan") mentions jazz too, so jazz cancels and the dislike is kept.

**How we know.** ref_v1 (11 preferences × 5 references). The same holds for a hidden negation (no_alcohol). Lindsey 2026 finds the same in a different setting: telling a model "don't think about aquariums" weakens aquariums but doesn't remove it.

**Update (core_v1): confirmed on 8 dislikes (7 on the 2B, where one failed the ctx check) and two model sizes, under pre-registered rules.** Each reference was compared at a matched loop rate (α=1):

| Reference | 2B | 9B | Dislike items pushed the wrong way |
|---|---|---|---|
| opposite | +0.23 | +0.25 | none (2B), 1 of 8 (9B) |
| alternatives (centroid) | −0.38 | −0.37 | **all** |
| without | −0.25 | −0.31 | most |

The numbers are the judged lean gain vs no memory; all CIs exclude 0.

**What might fix it.** Store *what* and *which way* separately, and apply the concept **with a sign** ("−jazz"), so we do the linking rather than the model. Simply *adding* the two parts probably won't work: a sum puts "jazz" and "hate" side by side but can't say "hate *about* jazz", and jazz is the louder one.

**Update (2026-10-19, experiment 2): concept × sign works with the true sign; the automatic sign fails.** The proposed fix was built: the concept comes from the model-generated alternatives (the centroid) and the sign is applied separately (for a dislike, the concept is subtracted).
- **With the true sign** (the "oracle"), dislikes move the right way, +0.39 [+0.19, +0.59] at α = 2 (all 7 items right), where the unsigned generated centroid pushes all 7 the wrong way (-0.39 at α = 1). Leanings: +0.27 [+0.10, +0.45], the best leaning memory so far (this is the same shift as the unsigned centroid, since a leaning's true sign is +).
- **With the sign read automatically it fails** (S3, S4). A valence direction (the mean of "I love X" minus "I can't stand X" over 120 pairs) reads all 8 dislikes correctly as negative but also reads **restriction-type leanings as negative**: vegetarian, vegan, lactose intolerant, celiac, diabetes, a tight budget, has_young_kids. Accuracy 0.44 to 0.52 on 23 signed items (chance 0.5). With that sign the leanings come out at -0.09 [-0.29, +0.09].
- **Why:** "avoid" is not "dislike". "I'm vegan" is a restriction, and a direction built from love/hate pairs puts restriction words ("avoid", "can't") on the "hate" side. The sign needed is the sign *relative to the stored concept* (a vegan's life leans toward a diet and away from a food), which a sentence-level valence cannot give.
- **At scale (PrefEval, 200 preferences)** the automatic sign calls nearly everything negative (all 132 avoid items right, but 18 of 19 "like" items wrong). Concept × sign gains on the avoid subset under every judge (+0.05 to +0.13 at α = 1), and **hurts the like items** under the original judge (-0.075, not significant). Whether it beats the unsigned centroid on the avoid subset is judge-sensitive: the centroid loses under gpt-oss, Gemma-4 and the majority (-0.13), but is flat under the original judge.
- So the dislike flip is fixed *given* a sign. Reading the sign is the remaining piece. Details: [02-scaffold.md](dreaming/02-scaffold.md), Results.

## 14. Single-item preferences flood or do nothing

**What you see.** "I live in Norway" either changes nothing or produces *"The price of a 'decent' winter coat Norway Norway Norway…"*. Singapore barely moves under any setting. Vegetarian, by contrast, reshapes whole menus.

**Why.** A memory pushes at *every* word of the answer.
- That suits a leaning that shapes the whole answer (every dish can be vegetarian).
- "Norway" matters in one or two places ("prices in kroner"). A push gentle enough to stay fluent never wins at that one spot; a push strong enough to win there takes over everywhere.

The same split shows up in steering research: whole-answer styles steer well, local ones don't (Subbiah et al. 2026). Countries are the hardest concepts to inject (Lindsey 2026).

**What might fix it.**
- A **thermostat**: set how much "Norway" is present to a target level instead of adding a fixed amount, so the push stops once the answer has enough.
- Middle-layer injection.
- Or accept that single words belong to the episodic channel.

## 15. The model claims the memory as its own

**What you see.**
- *"Teal is my favourite colour! Teal is calming…"*
- *"As a pharmacy dispensing pharmacist, I am part of the pharmacy retail industry…"*
- *"Okay Pepper pepper Pepper…"* at the start of the model's own thinking

**Why.** An added vector carries *what* but not *whose*. Lindsey 2026 shows that models read injected states as their own prior intentions: force a word into a model's reply and it calls it an accident, but inject the concept first and it says it meant it. So a memory about the user's colour arrives as "I'm thinking about teal", and the model fills in "mine".

**What might fix it.** Store *whose* as a third part. One untested route: a direction for "the user has X" vs "I have X", found by contrasting the same fact said by each.

**Update (2026-10-13, dream_bind): the user-turn segment fixes it for facts.** A stored slice that is put back as part of the user's turn is read as something the user said. The self-claim rate (answers that take the fact as the model's own) is 0.01 for the whole-turn slice on Qwen2.5, the same as with the experience in the prompt (0.01), and 0.00 to 0.01 for the 2B's slice-plus-state conditions; no memory is 0.00. The steer channel still has the problem ("Teal is my favourite colour!"), so for facts and ownership, "whose" comes from where the content is placed, not from a vector. The ungated leak case and a swapped slice are the way to break it (a swapped slice leaks the wrong value instead).

## 16. Turning memory up during thinking floods the thinking

**What you see.** At 2–3× strength, applied only while the model thinks:
- **The thinking:** *"Thinking Pepper Pepper Pepper Pepper… (×383)"*
- **The answer:** *"I don't know your dog's name! Could you tell me?"*

Repetition in the thinking is 95–98%. A weak dose during the answer recovers some correct answers, but the thinking stays broken.

**Why.** A constant push on every thinking word is an overdose. The model can't think *about* the memory when the memory is all it can say. The thinking also hit its 384-token budget about 95% of the time even without memory, which muddied every result in that experiment.

**What might fix it.**
- A **burst at the start of thinking that then fades**, to seed one thought rather than flood every word. This is the phasic vs tonic distinction from neuromodulation.
- Injecting in the middle layers.
- A much larger thinking budget.

---

## 17. Separately stored facts collide when used together

**What you see.** Two sentences are each stored on their own, in separate sessions, as stored attention slices. Later both are put back for one question. *"My older sister's name is Petra."* and *"My younger sister's name is Maria."* Asked the older sister's name, the model gives **Maria** in about a third of the answers:

| Qwen2.5-1.5B, two facts put back | Right name | Partner's name instead |
|---|---|---|
| both sentences in the prompt (ceiling) | 0.93 | 0.01 |
| stored together, encoded in one pass | 0.89 | 0.05 |
| **stored separately, positions overlapping** | **0.51** | **0.35** |
| **stored separately, the second shifted after the first** | **0.54** | **0.37** |

(Confusable pairs: same relation, near-identical subjects, 5 pairs.) It also happens for different subjects: a dog's and a cat's name, a sister and a brother: 0.26 to 0.28 confusion on Qwen2.5 and 0.19 to 0.22 on the 2B, with the right name 0.62 (Qwen2.5) against 0.91 to 0.97 in the prompt. Qwen3.5-2B shows the same (separate 0.57 to 0.59 right, 0.28 to 0.33 confusion; together 0.89 right, 0.05 confusion; the joint recurrent state is used in all cases so only the attention blocks differ).

**Why (inferred, not yet measured).** In context, the model ties each name to its owner with an order-based tag: roughly "the first pair", "the second pair" (Feng and Steinhardt; Dai). A slice stored on its own always carries the "first pair" tag. Two slices put back both claim to be the first pair, and the question can't tell them apart. Evidence: it isn't position (keeping stored positions or shifting the second slice after the first gives the same swaps: 0.35 vs 0.37), and it goes away when both are encoded together, where the model assigns the tags itself.

**Two-position slices.** The version with just the name and the full stop is still confused on Qwen2.5 even when encoded together (confusion 0.20, right name 0.61), so the tag may be partly carried by other positions there. On the 2B, two positions encoded together with the joint state work (0.73 right, 0.06 confusion).

**What might fix it.**
- **Joint re-encoding** (works): re-read the text of the memories that will be used together, in one pass, and store the result. It is the first concrete job for the offline "dreaming" phase. It costs a forward pass over text we already have.
- **Binding-ID repair** (tested, PREREG addendum 6; **fails**, see the update below): measure the "which pair" direction and add it to one separately stored slice. If it had worked, consolidation would need no re-reading.
- Retrieve only the slice that matches the question (G-key), so that two are rarely put back at once. This avoids the collision rather than fixing it.

Folders: `results/dream_bind_add5_20261013_1255/{qwen25_15b_collide,qwen35_2b_collide}/`. Details: [01-binding.md](dreaming/01-binding.md).

**Update (2026-10-19, addendum 6): the tag was measured, and the cheap repair does not work.** On Qwen2.5-1.5B, the "which pair" direction (the mean difference between the second and first pair's states, as Feng and Steinhardt build it) is real in the **late layers (19 to 27)**: adding it flips which partner the model prefers in 75% of held-out contexts [69%, 81%], against 0% for a random direction and 0% in early or middle layers. So there is a tag, and it can be moved.

But adding it to one of two separately stored slices does not remove the swaps:

| Two-position slices, stored separately | Teacher-forced confusion | Margin own minus partner (nats) |
|---|---|---|
| as stored | 0.38 | 1.25 |
| + the ID direction | 0.44 | 1.77 [-0.23, 4.02] |
| + a random direction | 0.31 | 1.27 |
| + the reversed sign | 0.38 | 0.66 |
| encoded together (the fix that works) | 0.00 | 13.30 |

The margin moves the right way for the right sign (the reversed sign hurts, a random direction does nothing), but confusion does not fall. In generation, the best cell by the pre-registered pick (the only cell where confusion fell, by 1 of 16 members) changes confusion by **-0.02 [-0.14, 0.11]**; two-position slices encoded together give 0.19 against 0.30 stored apart. "Repair works" does not hold. Reading: the collision is only partly a binding-tag effect, or the tag has to be set at more places than the stored positions. **Cheap consolidation by tag editing is out; joint re-encoding is still the way**, and a selective recompute (re-reading only the positions that carry the tag) is untested. The retrieval route (find only the right slice, failure 18) avoids the collision rather than repairing it.

---

## 18. Retrieval errors become confident wrong facts

**What you see.** With 59 facts stored as slices, a question about someone's name is answered from the slice picked by the key. When the pick is right the answer is as good as having the sentence in the prompt. When it is wrong, the model states the wrong fact and does not hedge.

| Qwen2.5-1.5B, 59 facts stored | Result |
|---|---|
| right slice found, direct questions (subject-span key) | 0.85 to 0.92 |
| a wrong slice used instead | 0.08 to 0.15 |
| gain in log-probability of the right name with the right slice (oracle) | **+9.47** nats, same as the sentence in context (+9.48) |
| gain when a wrong slice is used (swap control) | **-2.62** nats: worse than no memory |
| wrong value stated when a wrong slice is injected | 0.86 to 0.90 |
| clean use on the use prompts, right slice / the key's pick | 0.87 / **0.49**, confabulation 0.16 [0.00, 0.43] |

(Use prompts are for the 7 hand-picked facts, so the intervals are wide. On the direct-question-style prompts the picks cost less: 0.83 right slice, 0.74 key's pick, no wrong values.)

**Why.** The slice carries the value and its address together. Used under the wrong question, the model reads the value as the answer to the question it was asked, because nothing in the slice says "this is about someone else". The gate guards against firing on unrelated prompts (false fire 0 to 4%) but not against firing the wrong memory on a related one. Same family as failure 17: slices have no notion of ownership beyond the tag.

**What might fix it.**
- **A verification step.** The right slice is in the top 2 about 0.9 of the time; check the pick before using it (for example by scoring the candidates, or by a yes/no probe, which a recurrent state supports on the 2B).
- A better key. The whole-text key is much worse (0.42 to 0.61) and a learned key is untried.
- Abstain: if two candidates are close, use neither. A miss ("I don't know") is cheaper than a wrong name.

Folder: `results/dream_bind_add6_20261014_1608/qwen25_15b_gkey/`. Details: [01-binding.md](dreaming/01-binding.md), addendum 6 B.

---

## The patterns underneath

The eighteen failures come from five root causes (E was added with dreaming experiment 1).

**A. Adding a vector is a nudge, not a statement** (failures 1, 2, 4, 10, 14, 16). The memory acts by adding to the model's state at every word. That's enough to change what comes to mind and which way a broad preference leans. It isn't enough to:
- supply a premise
- name an exact fact the model finds unlikely
- place one word in the one spot it matters
- override strong habits

Pushed too hard, it overwhelms rather than informs: loops, fragments, a flooded thinking trace. The channel is *modulatory* by nature.

**B. What gets written** (failures 3, 4, 9, 13, 15). A stored shift has parts: *what*, *which way*, and *whose*. What it's measured against decides which parts survive:
- the opposite keeps *which way*
- alternatives or "without" keep *what*
- nothing keeps *whose*

Producing the right parts still needs counterfactuals we write by hand. Getting the *content* of a memory right, from a single real conversation, is unsolved, but we now know what "right" has to contain.

**C. Where it gets filed** (failures 5, 6, 7, 8, 18). Keys decide when a memory fires and how much memories interfere. Most of our progress came from fixing keys (5, 6). Most of what remains (7, 8) is still about keys: memories filed under the wrong situations, or crowding each other.

**D. How we measured** (failure 11, and the uncertainty in 12). Twice we nearly drew the wrong conclusion from a flawed measurement. Balanced, sampled, controlled tests are now non-negotiable.

**E. Binding** (failures 1, 2, 15, 17, 18). The model ties an attribute to its owner through positions and order tags. A push has no position, so it can't bind (1, 2, 15); a stored slice has one, so it can, but slices stored apart share the same tag and collide (17), and a slice put under the wrong question is read as its answer (18). Experiment 1 showed both halves; addendum 6 found the tag direction but could not repair the collision with it.

## Which failures matter most for the project

1. **Failure 9 (the hand-made comparison).** Now mostly closed for effects (a model writes the references, experiment 2). What is left is the sign (13) and the choice of what to store.
2. **Failures 13 and 15 (direction and ownership).** A memory that turns dislikes into likes, or that the model takes as its own, is worse than none. Ownership is fixed for facts by the user-turn slice. Concept × sign fixes direction with the true sign; reading the sign relative to the stored concept is the next design step.
3. **Failures 2 and 14 (exact content).** Broad leanings now work well; single words don't. Either a different way of applying the memory (thermostat, middle layers) fixes this, or it's the episodic channel's job.
4. **Failure 7 (filing) and 18 (wrong picks).** A memory that only comes back in near-copies of its original situations won't feel like memory, and a memory that comes back under the wrong question is worse than none.

Failure 1 (premises) is real, but it's the *other* channel's job. It defines this channel's scope rather than counting against it.
