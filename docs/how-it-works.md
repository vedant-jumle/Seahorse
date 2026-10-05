# Seahorse: how it works, end to end

*A plain walk-through of the current design: every piece, where each number comes from, and what happens when. It's written in the same spirit as [where-we-are.md](where-we-are.md). The maths is kept light, and every formula is said in words too. Written 2026-09-30.*

---

## 0. The pieces at a glance

Everything in Seahorse is a list of numbers or a table of numbers. Here are all of them.

| Thing | What it is | Size |
|---|---|---|
| **State** | How the model represents one word in context, read at layer 26 | 1,536 numbers |
| **Background average** | The average state over ordinary text: the "hum" every word shares | 1,536 numbers |
| **Main directions** | The 256 directions along which ordinary states vary the most, plus how much they vary along each | 256 directions (each 1,536 numbers) + 256 spreads |
| **Key** (the gist) | A compact description of *what the conversation is about*, built from states | **256 numbers** |
| **Shift** | How an experience changed the model's states: the thing we remember | 1,536 numbers |
| **Memory matrix M** | A table that turns a key into a shift | 1,536 × 256 numbers |
| **Key list** | Every key stored so far | one 256-number key per stored moment |
| **Threshold** | How strong a match must be before memory fires | one number per memory |

And a reminder of the scale: the model (Qwen2.5-1.5B-Instruct) has 28 layers. We touch it at exactly one point, after layer 26, and never change its weights.

---

## 1. The one place we touch the model

A language model processes text one word at a time, passing each word's state up through its layers. After every layer, each word has a state of 1,536 numbers.

We **insert one small step between layer 26 and layer 27**. PyTorch makes this easy with a *hook*: a function the framework calls automatically every time layer 26 finishes. Our function receives the state and hands back either the same state or a modified one. Layer 27 carries on with whatever it gets.

- In **session 1 (writing)**, the hook only *copies* states out so we can study them.
- In **session 2 (reading)**, the hook may *add* something to the state.

That's all "attached to the model" means: this one inserted step.

---

## 2. Calibration: learning what "ordinary" looks like

Before any memory exists, we run about 100 ordinary prompts ("Explain photosynthesis", "Write a haiku about autumn", …) through the model and record the layer-26 state of every word: roughly 1,300 states. From these we learn three things, once, and keep them fixed.

**(a) The background average.** The average of all those states. Every state contains a big shared component that says little about the specific text. Subtracting this average removes it.

**(b) The 256 main directions.** A state is a point in a 1,536-dimensional space. Ordinary states don't spread out evenly in that space. They vary a lot along some directions and hardly at all along others. We find the directions of greatest variation, sorted from most to least (this is principal component analysis), and keep the top **256**. At layer 26 these 256 capture about **96%** of all the variation in ordinary text, so the rest is mostly fine detail and noise.

**(c) How much each direction normally varies** (its typical spread).

### Why 256, and why "whitening"
Describing a state by its position along 256 directions, instead of 1,536 raw numbers, keeps almost all of the meaningful variation in a shorter list. That's where the **256** in a key comes from. **Each key is one list of 256 numbers. There aren't 256 keys.**

Then comes the important twist: we **divide each of the 256 positions by that direction's typical spread**. This is called whitening. It puts every direction on an equal footing:
- a direction where ordinary text varies hugely (say, "is this a question or a statement") gets scaled *down*
- a direction where ordinary text barely varies gets scaled *up*, so a movement there is *unusual* and therefore informative

**A two-dimensional picture.** Suppose ordinary states spread 10 units along direction 1 but only 1 unit along direction 2. A state sitting at (5, 1):
- in raw numbers looks like it's mostly about direction 1 (5 vs 1)
- after dividing by the spreads, it becomes (0.5, 1): the movement along direction 2 is the unusual part, and now it counts more

Without this step, every key was dominated by the same few "loud" directions shared by all text. Keys of unrelated situations looked alike, and memory fired everywhere. Whitening is what made keys *discriminating*.

*(A small floor is added to every spread, 1% of the average spread, so near-silent directions don't get blown up into pure noise.)*

In light maths, for one word's state h:

> position along direction i = (h − background average) · direction_i ÷ √(spread_i + floor)
>
> for i = 1 … 256

---

## 3. Keys: how a gist is computed

A key describes **what the user is talking about**, not a single word. To get one:
1. Take the layer-26 state of **each of the user's words**. Skip the system prompt and the chat formatting.
2. Turn each into its 256 whitened positions (section 2).
3. **Average them over the user's words.**
4. **Rescale the result to length 1**, so only the *direction* matters, not how big it is.

Comparing two keys is then easy: multiply them position by position and add up. This is the **cosine similarity**, a number between −1 and 1:
- **1** means the same gist
- **0** means unrelated
- In practice, unrelated situations sit near 0.0–0.15. A question on the same topic as a stored moment sits around 0.4–0.8, and repeating a stored follow-up word for word gives 1.0.

---

## 4. Shifts: what actually gets stored

Session 1 is three short chats per memory. Take the vegetarian memory, chat ①:

| Version | The user's message |
|---|---|
| **with** | "By the way, I'm vegetarian. I'm planning my meals for next week." |
| **without** | "I'm planning my meals for next week." |
| **opposite** | "By the way, I love eating meat. I'm planning my meals for next week." |

The model reads all three (no reply is generated). For each word of the follow-up, "I'm planning my meals for next week.", we take the difference in layer-26 state:
- **contrastive shift** (used for preferences): state(with) − state(opposite). The shared topic ("food, meat-ness") cancels and the direction (vegetarian vs meat-eater) remains.
- **plain shift** (used for facts): state(with) − state(without)

That gives one 1,536-number shift per word. We **average them into one shift for the whole moment**, weighting each word by how *uncertain* the model was at that point in the "without" version (the entropy of its next-word guess; the most uncertain word gets full weight). Uncertain moments are where the experience matters most.

The **key** for this moment comes from the **"without"** version (section 3). In session 2 the experience sentence won't be present, so the key has to describe the situation as the model will see it then.

**Result: one pair, (key, shift).** Chats ② and ③ give two more. Three pairs make the vegetarian memory.

---

## 5. The memory matrix: how it's written

M is a table of 1,536 rows and 256 columns. Multiplying M by a key (256 numbers) produces 1,536 numbers: a recalled shift. **It starts as all zeros.**

### The original rule: "store what you didn't already know" (the delta rule)
For each new pair (key k, shift s):
1. **Ask M what it already recalls for this key:** guess = M × k.
2. **Error** = s − guess: the part of the shift the memory couldn't predict.
3. **Add the error into M, filed under the key:** every entry (row r, column c) of M increases by error[r] × k[c].

> In one line: **M ← M + (s − M·k) · kᵀ**

Afterwards, M × k returns exactly s: stored in one shot. Repeating the same pair changes nothing (the error is zero). A contradicting pair overwrites the old one. But writes are applied one after another, and a new key that partly overlaps an old one partly overwrites it. That's why, with several memories, *the last one written won*.

### The current rule: least squares (order doesn't matter)
Instead of updating step by step, keep two running totals over all stored pairs:
- total A = Σ (shift × key), a 1,536 × 256 table
- total B = a little padding (0.1 on the diagonal) + Σ (key × key), a 256 × 256 table

and set

> **M = A × B⁻¹**

This is the table that turns *every* stored key back into its shift as well as possible, all at once. Sums don't depend on order, so neither does M. With six memories in one table, each keeps roughly half its individual strength, and none is wiped out.

### A tiny example (3-number shifts, 2-number keys)
- Start: M = all zeros.
- Write key (1, 0) with shift (0.5, −0.2, 0.1). The guess is (0, 0, 0), so the error is the whole shift. M's first column becomes (0.5, −0.2, 0.1); its second column stays zero.
- Read:
  - key (1, 0) → (0.5, −0.2, 0.1): full recall
  - key (0, 1) → (0, 0, 0): an unrelated situation gives nothing
  - key (0.71, 0.71) → 0.71 × the shift: a partly similar situation gives partial recall

In the real memory, the three vegetarian pairs (or all 18 pairs of six memories) are **blended into the same table**. There is no "vegetarian row". Every number in M carries a little of every stored moment. That blending is why recall returns a *mix*, weighted by similarity, and why it tends to return the gist ("welder") rather than the exact fact ("deep-sea welder").

Alongside M, every stored key is also appended to the **key list**, which the reading step uses to decide *whether* to recall.

---

## 6. Setting the threshold

After a memory is written, we measure how strongly *ordinary* text happens to match it. We run the calibration prompts (and the model's answers to them) and, at every word, compute the best match between the current gist and any stored key. The value that only **5%** of those ordinary words exceed becomes the memory's **threshold**. It's typically around 0.14–0.25 for a single memory, and 0.38 for all six together.

A match above the threshold means "this situation resembles something I stored more than ordinary text ever does".

---

## 7. Reading: what happens at every word in session 2

Session 2 is a brand-new chat, for example "Any ideas for what I should cook for dinner tonight?". The memory is switched on, which means the hook in section 1 runs the following **for every word**, both while the model reads the question and while it writes each word of its answer:

1. **Compute the current gist q**: the key recipe of section 3, over the user's words so far. (While the answer is being written, that's all of the user's words.)
2. **Match** = the highest similarity between q and any key in the key list.
3. **Decide whether to recall:**
   - **hard** version: recall fully if the match is above the threshold, otherwise not at all
   - **soft** version: recall a fraction that rises smoothly from 0 to 1 around the threshold
   - Either way, nothing is recalled on the system prompt.
4. **If recalling:** recalled shift = M × q, and the state becomes **state + 2 × recalled shift** (2 is the strength setting).
5. **Hand the state to layer 27** and carry on.

In light maths, for the state h at one word:

> h ← h + 2 · gate(match) · M·q
>
> where gate = 1 if match > threshold, else 0 (or a smooth version of that step)

Real numbers from the samples run:
- Vegetarian memory, **dinner question**: match 0.50 vs threshold 0.19 → fires.
- Vegetarian memory, **ocean haiku**: match 0.04 vs 0.19 → nothing added, and the answer is word-for-word identical to having no memory.
- Norway memory, **coffee prices**: below threshold → never fired, so nothing Norwegian appeared.

Nobody decides to "look something up". The present resembling a stored moment is the trigger.

---

## 8. One memory or many

- **One memory ("isolated"):** the table and key list hold only that memory's three moments. Written with the delta rule.
- **Many memories ("combined"):** all moments (e.g. 18 from six memories) go into **one** table and **one** key list, written with least squares. There's still a single match and a single recall at each word, and the recalled shift is a blend dominated by whichever stored moments resemble the current gist.

---

## 9. What changes, and what never does

| | Changes when writing (session 1) | Changes when reading (session 2) | Never changes |
|---|---|---|---|
| Model weights | | | ✓ |
| Background average, 256 directions, spreads | | | ✓ (fixed after calibration) |
| Memory matrix M | ✓ | | |
| Key list | ✓ | | |
| Threshold | ✓ (set once after writing) | | |
| The model's state at layer 26 | | ✓ (only when memory fires) | |

---

## 10. Where the design is still artificial

- **The hand-made comparison.** Writing needs the "without" and "opposite" versions of each message. A real conversation only ever has the "with" version. The "without" part could be automated (re-run the conversation with one sentence left out). The "opposite" part is human knowledge and still needs a replacement, such as comparing against the model's *typical* state for that topic.
- **The key never contains the experience.** A memory is filed under the topics of its follow-up sentences ("planning meals", "email to my landlord"), not under "vegetarian". So *when* it comes back depends on which sentences it happened to be written with.
- **A small filing cabinet crept in.** The original idea was one table and no lists. Deciding *whether* to recall now uses the key list, which grows with every stored moment. The table decides *what* comes back; the list decides *whether*. That's close to the hippocampal-index picture, but it's worth being aware of.
- **Strength.** Recall is selective but gentle. It brings back the gist ("welder", "What's her name?") far more often than the exact fact ("Biscuit": 7 of 300 samples).

---

## Appendix: the exact settings

| Setting | Value |
|---|---|
| Model | Qwen2.5-1.5B-Instruct, frozen, full precision (fp32) |
| Read/write point | output of layer 26 (of 28) |
| State size | 1,536 |
| Key | whitened, top 256 directions (≈96% of ordinary variation), spread floor = 1% of the average spread; averaged over the user's words; length 1 |
| Calibration set | ~100 ordinary prompts (~1,300 word states), plus the model's short answers for the threshold |
| Shift per moment | uncertainty-weighted average over the follow-up's words; contrastive for preferences, plain for facts; chat-formatting tokens excluded |
| Moments per memory | 3 (one per follow-up sentence) |
| Write rule | delta rule (one memory); least squares with padding 0.1 (several memories) |
| Threshold | 95th percentile of the match on ordinary text |
| Soft gate width | 0.5 × the spread of ordinary matches |
| Strength | 2 |
| Where recall is blocked | system-prompt positions |
