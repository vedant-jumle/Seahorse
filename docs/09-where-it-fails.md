# Seahorse: where it fails, and why

*A companion to [07-where-we-are.md](07-where-we-are.md) and [08-how-it-works.md](08-how-it-works.md). It collects every way the memory has failed so far, with a real example of each, the best explanation we have, how we know, and what might fix it. Same plain style. Written 2026-10-01.*

---

## The short version

| # | Failure | Status | Root cause, in one line |
|---|---|---|---|
| 1 | It can't give the model a premise to reason with | **Open (a limit of this channel)** | Adding a vector nudges; reasoning needs something the model can *attend to* |
| 2 | The exact fact rarely comes out; the gist does | **Open** | Recall is a blend, and the right answer starts out extremely unlikely |
| 3 | It stores the concept instead of the relation ("peanut" → peanut butter) | **Fixed only with a hand-made opposite** | The plain shift carries the topic as well as the point |
| 4 | Preferences are either faint or crude | **Open** | Clean writing cancels too much; plain writing carries too much |
| 5 | It fired everywhere | **Fixed** | Keys of unrelated situations looked alike, and nothing set a bar |
| 6 | The last memory written won | **Fixed** | Step-by-step writing lets new memories overwrite old ones |
| 7 | It fires only where it was filed | **Open** | A memory is filed under its follow-up sentences, not under the experience |
| 8 | More memories means weaker memories | **Partly open** | All memories share one table and must compromise |
| 9 | It needs a hand-made comparison to write anything | **Open (the biggest gap to real use)** | A "shift" is defined as a difference from a counterfactual we supply |
| 10 | Too much strength breaks the text | **Managed** | An added vector can overpower the model's own state |
| 11 | We fooled ourselves with our measurements | **Fixed** | Unbalanced questions, template artefacts, narrow scores |
| 12 | We don't know how far any of this generalises | **Open** | One small model, a handful of scenarios |

The failures cluster into four root causes, discussed at the end:
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

That second point gives up one of the original thesis's hippocampal properties (reconsolidation). Deciding *when* a new memory should override an old one is a job for the planned modulator.

---

## 7. It fires only where it was filed

**What you see.** The Norway memory stayed completely silent on *"What's a reasonable monthly budget for groceries?"* and *"How much does a cup of coffee usually cost at a cafe?"*. Its match never crossed the threshold. Yet these are exactly the questions where living in Norway matters.

**Why.** A memory's key is the gist of the **follow-up sentences** it was written with, not of the experience. The Norway memory was written alongside "I want to buy a new winter coat", "I'm trying to plan a weekend trip" and "Can you explain what a mutual fund is?". So it's filed under *coats, trips and mutual funds*, and a coffee question doesn't resemble any of them. The fix for failure 5 (a strict threshold) makes this worse: selective sometimes means *too* selective.

**What might fix it.**
- Keys that describe the experience itself (possible once writing no longer needs the hand-made comparison).
- More, and more varied, moments per memory.
- A softer threshold.
- Recall that also considers the *kind* of question ("prices", "shopping").

---

## 8. More memories means weaker memories

**What you see.** Alone, the dog memory yields "Biscuit" 7 times in 300. Stored together with the five others, **once**. Measured more broadly, each memory keeps about half its solo effect when six share the table.

**Why.**
- All memories live in one table and the least-squares solution has to compromise between them, especially where their keys overlap.
- The blend also means a recalled shift carries some of the *other* memories' content.

**What we don't know.** How this curve continues. Six is the most we've tested; the benchmark's 60-fact pool is ready for the real capacity curve.

**What might fix it.** Better-separated keys (so memories overlap less), separate subspaces per memory, or sparse, selective storage (store fewer, more salient moments). The neuromodulation idea was partly motivated by this.

---

## 9. It needs a hand-made comparison to write anything

**What you see.** Nothing in a live conversation. This failure is in the method itself. Every memory was written by comparing *three* versions of a message: with the experience, without it, and with its opposite. A real conversation only ever has the first.

**Why.** We defined "what to remember" as *the difference an experience makes*, and a difference needs something to compare against. We supplied that something by hand. That was fine for asking "can this channel carry memory at all?". It is not how a real memory could work.

**What might fix it.**
- **"Without"** can be automated: re-run the conversation with one earlier sentence left out, and use attention to choose which sentences are worth testing. It costs compute, perhaps in a consolidation pass after the conversation, but needs no human.
- **"Opposite"** is human knowledge and has no automatic equivalent yet. The most promising replacement is the topic baseline from failure 3.
- Alternatively, train a small network to predict the shift from the conversation alone, using our hand-made comparisons as training data.

This is the largest gap between the current system and the thesis's "memory formed from experience".

---

## 10. Too much strength breaks the text *(managed)*

**What you saw.** Doubling the read strength from 2 to 4 broke everything in v0. Even at the normal strength, the crude preference memory produced *"vegetarian, vegetarian and vegetarian…"*.

**Why.** At most positions the added shift is only 20–30% the size of the meaningful part of the model's state. But at some positions, or with a strong memory, it becomes as large as the state itself, and then the memory isn't nudging the model any more; it's overriding it. Mood-like and drug-like dose curves look the same: helpful in a window, harmful beyond it.

**How it's managed.** Strength stays at 2, and the threshold keeps the shift away from positions where it doesn't belong. Norm-preserving reads (rotating the state instead of adding to it) are available if we need more headroom.

---

## 11. We fooled ourselves with our measurements *(fixed)*

This is the least glamorous failure and the most important lesson.

- **The yes/no trap.** All of our early reasoning questions had "No" as the right answer, and the model drifts toward "No" whenever anything in its state changes. Even a neutral placebo sentence does it. So any memory looked like "reasoning progress". The benchmark now balances Yes and No answers and scores them separately.
- **The template trap.** Writing memory on the chat's formatting tokens produced fake effects, including part of that "No" drift. Those tokens are now excluded.
- **The single-pair trap.** Preference strength measured by one pair of phrases suggested big effects that sampled answers don't show (failure 4). Preferences are now judged by rates over many answers.
- **The averaging trap.** "Facts survive when memories are combined" was really just *the last fact written* surviving. The average hid it.
- **Random baselines matter.** When choosing which words to store, picking words at *random* worked as well as our clever choices. Without that control, we'd have credited the clever choice.

**The fix** was a proper benchmark (48 items, 60 facts, balanced questions, a placebo condition for the noise floor) and the habit of checking text, not only scores.

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

---

## The patterns underneath

The twelve failures come from four root causes.

**A. Adding a vector is a nudge, not a statement** (failures 1, 2, 4, 10). The memory acts by adding to the model's state. That's enough to change what comes to mind and which way a preference leans. It isn't enough to supply a premise, to name an exact fact the model finds unlikely, or to override strong habits. And pushed too hard, it overwhelms rather than informs. The channel is *modulatory* by nature.

**B. What gets written** (failures 3, 4, 9). The stored shift is either too much (concept plus relation) or too little (a thin contrastive direction), and producing it needs a counterfactual we write by hand. Getting the *content* of a memory right, from a single real conversation, is unsolved.

**C. Where it gets filed** (failures 5, 6, 7, 8). Keys decide when a memory fires and how much memories interfere. Most of our progress came from fixing keys (5, 6). Most of what remains (7, 8) is still about keys: memories filed under the wrong situations, or crowding each other.

**D. How we measured** (failure 11, and the uncertainty in 12). Twice we nearly drew the wrong conclusion from a flawed measurement. Balanced, sampled, controlled tests are now non-negotiable.

## Which failures matter most for the thesis

1. **Failure 9 (the hand-made comparison).** Without solving it, the system isn't "memory formed from experience". It's the thesis's central claim.
2. **Failure 2 and 4 together (strength and faithfulness).** Selective but faint isn't yet useful. Turning the gist into the fact, and the faint lean into a reliable one, is what would make the memory *matter* in conversation.
3. **Failure 7 (filing).** A memory that only comes back in near-copies of its original situations won't feel like memory.

Failure 1 (premises) is real, but it's the *other* channel's job. It defines this channel's scope rather than counting against it.
