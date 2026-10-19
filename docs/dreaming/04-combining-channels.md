# Idea: combining the two channels (steer + slice)

*Parked idea, not a spec. Written 2026-10-19 from a discussion with the user ("this is a good thing to try a little later"). Turn it into a spec with a PREREG before running anything.*

---

## Why

Seahorse now has two channels that can carry information, and every run so far used them **one at a time**. The only overlap was a side-by-side comparison in exp 1's teacher-forced pre-pass, never a joint use.

| | Steer (pushed shift; paper 1, exp 2) | Slice (restored cache; exp 1) |
|---|---|---|
| Carries | a disposition: "lean vegetarian" | a bound fact: "dog's name = Pepper" |
| Acts | everywhere, every token, the same push | only where attention fetches it, depending on the question |
| Retrieval | a gate only | must fetch the *right* slice; the weak point (0.49 use with the key at 59 facts) |
| Cost | one small matrix | about 115 KB per fact on Qwen2.5 (the 2B's recurrent state is ~19 MB) |
| Fails at | facts, premises, dislikes without help | combining memories from different chats; automatic retrieval |

Their strengths and weaknesses are close to complementary, which is the reason to try them together.

## Four combinations, most to least informative

**1. The slice for preferences too.**
- **Why it might work:** in context, a dislike stays a dislike, because the model computes the negation while reading. A restored slice of "I can't stand jazz" keeps that computed state, so it should avoid the dislike flip **with no opposite, no alternatives and no sign reader**. Everything exp 2 fought (references, the sign) is a problem of the *push*.
- **Test:** the 29 preference items (and PrefEval) with R-min and R-full slices, against nomem, ctx and the best steer (gen:opposite; cs:oracle as the ceiling).
- **What it decides:** whether the steer is needed for content at all.
  - If R-min (2–3 tokens) carries a preference with its sign, that's a strong result.
  - If only the whole sentence does, it's RAG for preferences, and the steer keeps a job only where the slice can't reach.
- **Open detail:** which tokens a "preference slice" keeps. The value ("jazz") plus the full stop, as for facts? The verb ("can't stand")? Compare a few choices.

**2. Steer for the disposition, slice for the content.**
- **Idea:** one experience, two memories. "I'm vegetarian" goes into the steer as a leaning and into a slice as an explicit fact.
- **Test:** whether the steer keeps answers right when slice retrieval misses (the steer's gate is broad), and whether the slice supplies the premise when asked directly ("Is this burger OK for me?"). Measure the combination against each channel alone, **with automatic retrieval** (see below).
- **Why:** it attacks the slice's biggest weakness, wrong or missed retrieval.

**3. Steer as the "you know this user" signal.**
- **Observation:** on the hybrid 2B, the attention-only slice gave the name in tasks but refused direct questions ("I don't know your dog's name"); the recurrent state fixed that.
- **Idea:** on full-attention models, a small steer toward "I remember things about this user" might play the same role and lift the direct-question rate. Speculative.

**4. Steer as the consolidated layer (the long-term architecture).**
- **Idea:** episodic slices hold specific moments; offline dreaming abstracts what many episodes share ("five dinners without meat") into the steer.
- That is the hippocampus / neocortex split the project started from: slice = episodic memory, steer = semantic or dispositional memory formed from it.
- This is exp 3's abstraction test (2a) with a defined role for each channel.

## Avoid
**Steer and slice on the same fact.** The fact-steer is a late "say Pepper" push (paper 1, C6). On top of a slice it would likely bring back loops ("Pepper Pepper…") and disturb the attention fetch.

## Ground rules for when this runs
- **Automatic retrieval only.** The user's standard (2026-10-19): the hand-written string-match gate is a control and upper bound, never a result. Use the model's key (ideally the fully automatic whole-text key), and add a standard text-retrieval baseline (BM25 or embeddings over the stored sentences) for comparison.
- Start with (1): cheap, since the code exists (`experiments/dream_bind/` for slices, `experiments/dream_refs/` for steer items and PrefEval). Then (2).
- Pre-register as usual; judge with the three-model panel; report each channel alone, both together, nomem and ctx.

## Related
- Exp 1 results: [01-binding.md](01-binding.md)
- Exp 2 results: [02-scaffold.md](02-scaffold.md)
- Consolidation and abstraction: [03-consolidation.md](03-consolidation.md)
- Next steps: HANDOVER §8
