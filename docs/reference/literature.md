# Seahorse: what other work says about our results

*Four recent papers on steering vectors and concept injection, read after the Qwen3.5 experiments (think_v1, ref_v1). For each: what it found, in plain words, and where it touches our results. The older background (complementary learning systems, steering, fast weights, neuromodulation) is in [core-idea.md](core-idea.md) and [where-we-are.md §2](../where-we-are.md#2-what-was-already-out-there). Written 2026-10-07.*

The PDFs are in the parent folder's `papers/`.

---

## The one-paragraph version

Most of what we see is **normal behaviour for steering vectors**, not a Seahorse bug:
- success depends heavily on the concept
- some prompts move the wrong way
- pushing harder trades meaning for incoherence
- one-slot content (names, places) is hard
- vectors pick up junk such as a Yes/No preference

Two of our findings go a step further. First, **what you subtract when writing a memory decides whether it keeps the concept or its direction.** That's a concrete, explainable case of "steering is too simple a tool", and it's why dislikes flipped into likes. Second, **our selectivity gate**: all four papers steer everywhere, all the time.

---

## 1. Tan, Chanin et al. (UCL), "Analysing the Generalisation and Reliability of Steering Vectors" (NeurIPS 2024)

**What they did.** They tested the standard steering method (contrastive activation addition) on 40 behaviours, on Llama-2-7B and Qwen-1.5-14B, prompt by prompt rather than on averages.

**What they found.**
- **Averages hide failure.** For many behaviours, close to half of individual prompts move the *opposite* way to the one intended.
- **Vectors encode junk.** Part of a steering vector often encodes "answer A rather than B" or "say Yes rather than No" instead of the concept.
- **Steerability is mostly a property of the concept,** consistent across both models.
- **Transfer to new settings is decent but imperfect,** and worst when pushing toward behaviour the model doesn't normally show.

**Where it touches us.**
- Our early "reasoning gains" were exactly their Yes/No junk: a general drift toward "No". Balanced yes/no probes are the fix ([metrics.md §7](metrics.md)).
- Some of our preferences barely move under any reference (Singapore). That's consistent with steerability being a property of the concept.
- **What we should copy:** report the share of prompts that move the *wrong* way, not only averages.

## 2. Braun (Tübingen), "Understanding Unreliability of Steering Vectors in Language Models" (2025)

**What they did.** They asked *why* some behaviours steer well and others don't, across 36 behaviours on Llama-2-7B.

**What they found.**
- **Agreement predicts success.** A steering vector is an average of many before/after differences. If those differences point the same way, the average works. If they scatter, they cancel, leaving a short, weak vector.
- **Different recipes give different vectors with similar effects.** Building the vector from different prompt styles gave directions that were often far apart, yet they performed about equally.
- **Conclusion:** many behaviours aren't one straight direction inside the model, so "add one fixed vector" is too simple. Richer methods (projections, not just addition) are needed.

**Where it touches us.**
- Our references also give quite different directions that lean the same way for vegetarian. But for dislikes they lean **opposite** ways, and we can say why (§ "concept vs direction" below). That's a sharper version of his point.
- **Idea it gives us:** use the agreement between a memory's moments as a **confidence** for that memory. Write strongly if they agree, weakly if they scatter. This fits the neuromodulation plan.
- His "richer methods" point to our *thermostat* idea: set the concept to a target level instead of adding to it.

## 3. Subbiah, Hall, McKeown (Columbia), "On the Limits of Steering Vectors for Preference-Aligned Generation" (2026)

**What they did.** They steered 36 writing-style preferences on Qwen2.5-7B and Llama-3.1-8B, judged by another language model.

**What they found.**
- **Whole-answer traits steer well; local ones don't.** Bullet points, a formal tone or step-by-step structure work. Rhyming (it matters only at line ends), screenplay format or a tweet style are weak or turn incoherent.
- **There's always a trade-off between expression and coherence.**
- **Combining vectors weakens each one:** at least 15% loss with two, more with four, and the strength must be retuned per combination.
- **Vectors transfer poorly** from the prompts they were made on to new tasks.

**Where it touches us.**
- **This is our vegetarian vs Norway split.** Steering pushes at every word. A diet shapes the whole answer, so a steady push helps. "Norway" matters in one spot ("prices in kroner"), so the push either never wins or takes over ("Norway Norway…").
- **Their trade-off is our loops.** Same phenomenon, different measure.
- **Combining:** our gate means only matching memories fire, so usually only one is active, which avoids much of their combining loss. Several memories firing on one prompt is the case left to test.
- **What we should copy:** a language-model judge, to separate *recommending* from *mentioning* ("since you don't like jazz…").

## 4. Lindsey (Anthropic), "Emergent Introspective Awareness in Large Language Models" (2026)

**What they did.** They injected concept vectors into Claude models' internal state ("concept injection", the same operation as our memory read), then asked whether the model notices and can name the injected "thought".

**What they found.**
- **Sometimes it notices.** The strongest model (Claude Opus 4.1) notices and names the concept about 20% of the time, with no false alarms. Smaller and base models do much worse.
- **The best layer is about two-thirds deep.** Later layers make the model just *say* the word, sometimes noticing only afterwards.
- **There's a strength window.** Too weak and nothing happens. Too strong and the model is "consumed by the concept", produces garbled text, or loses its identity: injecting "vegetables" gives *"fruits and vegetables are good for me"*.
- **Category matters.** Countries and concrete nouns are hardest; abstract nouns are easiest.
- **Injected states feel like the model's own intentions.** Force a strange word into its reply and it calls it an accident. Inject that concept *before* the word, and it says it meant it, and makes up a reason.
- **"Don't think about X" weakens X but doesn't remove it.**
- **Their concept vector** = the state for "Tell me about {word}" minus the average over many other words.

**Where it touches us.** This is the closest paper to Seahorse.

| Their finding | Our result |
|---|---|
| Concept vector = word minus the average of other words | That's our `centroid` reference: the standard way to get *what* |
| "Don't think about X" still carries X | "I can't stand jazz" still pushes jazz |
| Overdose → consumed by the concept, identity loss ("vegetables are good for me") | "Norway Norway…", *"Teal is my favourite colour!"* |
| Countries hardest | Norway and Singapore never work |
| Injected states read as the model's own intentions | The model claims memories as its own |
| Best layer about two-thirds deep; later = just saying the word | We inject near the very end and get words, puns and loops. **This is what xlayer_v1 tests.** |
| Using an injected thought appears mainly in large models | "No premises" may partly be a 2B limit |

It also suggests that models carry a **"this doesn't fit the context" signal**: their best guess for how injections get noticed. That's the surprise signal the neuromodulation plan wants for deciding *when to write*. We might read it rather than build it.

---

## Concept vs direction: what all four papers point at

Our clearest new result (ref_v1) is that a stored preference has parts, and what you subtract decides which part survives:

| Part | Example | Kept by subtracting |
|---|---|---|
| *what* | jazz | the centroid (other genres), without, other disclosures |
| *which way* | like / dislike | the opposite ("I can't stand jazz" vs "I love jazz") |
| *whose* | the user's, not the assistant's | nothing yet |

- **Keep only *what*,** and a dislike becomes a like, because the model encodes negation faintly (Lindsey sees the same).
- **Keep only *which way*,** and a concept that appears on both sides cancels: the jazz fan becomes "enthusiastic".
- **Simply adding the two** puts "jazz" and "hate" side by side; it can't say "hate *about* jazz". The model reads the louder one, jazz.
- **What should work:** apply the concept with a sign, so we do the binding ([HANDOVER §8](../../HANDOVER.md#8-next-steps-and-ideas-in-rough-priority)).

That's a small step toward the "richer than one added vector" methods Braun calls for.
