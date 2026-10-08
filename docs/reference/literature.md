# Seahorse: what other work says about our results

*Twelve papers that bear directly on our results, read during and after the Qwen3.5 experiments (think_v1, ref_v1, xlayer_v1, core_v1). For each: what it found, in plain words, and where it touches our results. At the end: the theory they give us, and a table of what is prior work and what is ours. The older background (complementary learning systems, steering in general, fast weights, neuromodulation) is in [core-idea.md](core-idea.md) and [where-we-are.md §2](../where-we-are.md#2-what-was-already-out-there). Written 2026-10-07; extended 2026-10-08.*

The PDFs are in the parent folder's `papers/`.

| Group | Papers | Why they matter |
|---|---|---|
| A. How reliable steering is | Tan et al.; Braun; Subbiah et al.; Lindsey | Our failures are mostly normal steering behaviour |
| B. Steering with conditions, context and users | CAST; In-Context Vectors; BiPO; Context Steering (CoS) | The closest prior work to our method and setting |
| C. Where facts live and how they're recalled | Geva et al.; ROME; MEMIT | Why a stored vector can't carry a usable fact |
| D. Negation | Jang et al. | Why dislikes flip into likes |

---

## The short version

- **Most of what we see is normal steering behaviour** (group A):
  - success depends on the concept
  - some prompts move the wrong way
  - pushing harder trades meaning for incoherence
  - one-slot content (names, places) is hard
  - vectors pick up junk such as a Yes/No tilt
- **The pieces of our method exist separately** (group B):
  - gating a steering vector on the context (CAST)
  - turning context into an added vector (In-Context Vectors)
  - personalising by steering (BiPO, CoS)

  None of them stores what a user said in one conversation and brings it back, unprompted, in a later one. That setting is ours.
- **Why facts fail has an explanation in the fact-recall literature** (group C):
  - a fact is fetched by attention, driven by the question
  - the places where facts can be written usably are middle-layer weights at the subject's position, with optimised values

  A fixed added vector has neither.
- **Why dislikes flip has support** (groups B and D):
  - models process negation weakly (Jang; Lindsey)
  - a single context carries its topic, while a contrasting pair isolates the attribute (CoS, in one sentence)

  We show this systematically in a memory, where it turns dislikes into likes.

---

# A. How reliable steering is

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
- **What we copied in core_v1:** we report the share of items moved the *wrong* way, not only averages.

## 2. Braun (Tübingen), "Understanding Unreliability of Steering Vectors in Language Models" (2025)

**What they did.** They asked *why* some behaviours steer well and others don't, across 36 behaviours on Llama-2-7B.

**What they found.**
- **Agreement predicts success.** A steering vector is an average of many before/after differences. If those differences point the same way, the average works. If they scatter, they cancel, leaving a short, weak vector.
- **Different recipes give different vectors with similar effects.** Building the vector from different prompt styles gave directions that were often far apart, yet they performed about equally.
- **Conclusion:** many behaviours aren't one straight direction inside the model, so "add one fixed vector" is too simple. Richer methods (projections, not just addition) are needed.

**Where it touches us.**
- Our references also give quite different directions that lean the same way for vegetarian. But for dislikes they lean **opposite** ways, and we can say why (see "Concept vs direction" below). That's a sharper version of his point.
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
- **Their trade-off is our loops and our coherence collapse.** In core_v1, leanings came through at α = 1. At α = 2, 41–61% of answers were judged incoherent.
- **Combining:** our gate means only matching memories fire, so usually only one is active, which avoids much of their combining loss. Several memories firing on one prompt is the case left to test.
- **What we copied in core_v1:** a language-model judge, to separate *recommending* from *mentioning* ("since you don't like jazz…").

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

**Where it touches us.** This is the closest paper on the *operation*: injecting a concept.

| Their finding | Our result |
|---|---|
| Concept vector = word minus the average of other words | That's our `centroid` reference: the standard way to get *what* |
| "Don't think about X" still carries X | "I can't stand jazz" still pushes jazz |
| Overdose → consumed by the concept, identity loss ("vegetables are good for me") | "Norway Norway…", *"Teal is my favourite colour!"*, *"My name is Omar."* |
| Countries hardest | Norway and Singapore never work |
| Injected states read as the model's own intentions | The model claims memories as its own |
| Best layer about two-thirds deep; later = just saying the word | A fact's stored shift points at the word only in the last layers (xlayer_v1; core_v1 C6 on 2B and 9B). Injected earlier, it fades or garbles. For facts, there's no "thought" version to inject. |
| Using an injected thought appears mainly in large models | No premises at 9B either (core_v1 C4), so size up to 9B doesn't help |

It also suggests that models carry a **"this doesn't fit the context" signal**: their best guess for how injections get noticed. That's the surprise signal the neuromodulation plan wants for deciding *when to write*. We might read it rather than build it.

---

# B. Steering with conditions, context and users

## 5. Lee et al. (UPenn / IBM Research), "Programming Refusal with Conditional Activation Steering" (CAST, ICLR 2025)

**What they did.**
- Add a **condition vector** next to the usual steering ("behaviour") vector.
- At inference, the behaviour vector is added only if the hidden state's cosine with its projection on the condition vector passes a threshold θ.
- Both vectors are extracted with PCA from contrasting prompt sets (e.g. thousands of harmful vs harmless prompts). θ, the layer and the comparison direction are chosen by grid search.

**What they found.**
- Models can be made to refuse **only** certain categories (hate speech, legal advice…) while answering everything else, on seven models.
- Conditions combine with logic ("if hate *or* legal, refuse"), and flipping the comparison gives the complement ("refuse everything *except* health").
- Performance saturates quickly with data.
- Constraining works better for categories that are semantically distinct.

**Where it touches us.**
- **Our gate is this mechanism.** Selectivity is not a new idea, and the write-up must present our gate as CAST-style gating.
- **What differs, and is ours:**
  - the condition is written from **one experience** (the gist of a few follow-up sentences), not from thousands of labelled prompts
  - keys are **whitened**, which is what made unrelated answers token-for-token identical
  - the threshold is calibrated on ordinary text (95th percentile) rather than grid-searched
  - it's **a store of many key → shift pairs**, not one rule

## 6. Liu, Ye, Xing, Zou (Stanford), "In-context Vectors: Making In-Context Learning More Effective and Controllable Through Latent Space Steering" (arXiv 2311.06668, 2024)

**What they did.**
- Turn the demonstrations of a task into one vector: the main direction (first PCA component) of the differences h(target) − h(input) over the demonstration pairs, taken at the last token of each layer.
- At inference, add it at **every layer and every token**, rescaling each state back to its original length.

**What they found.**
- It beats in-context learning and LoRA fine-tuning on detoxification, safety, formality, sentiment transfer and role-play.
- **A single layer does almost nothing;** all layers together are needed.
- Vectors can be added and subtracted: subtracting reverses the task.
- **The key observation:** in-context learning is itself a shift of the model's states, made by attention. The attention output for a query is a mix of the query's own state and the demonstrations' states, with **weights that depend on the query**.

**Where it touches us.**
- "Turn context into an added vector, and it carries style" is known. So our leanings result (C2) is *expected*, not new. It also failed its pre-registered rule.
- "Several layers ≫ one" matches our multi-layer finding.
- **Their observation is the heart of our theory** (below): in context, the shift is recomputed for every query; a stored vector is one fixed average.
- They test tasks and styles, not facts about a user, and not memory across sessions.

## 7. Cao et al. (Penn State), "Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization" (BiPO, arXiv 2406.00045, 2024)

**What they did.**
- They argue that standard steering vectors, built from activation differences on contrasting prompt pairs (CAA), **often don't represent the target behaviour**: the model's own continuation doesn't follow the appended choice.
- So they **optimise** the vector with a DPO-style objective, making the target response more likely and the opposite less likely, in both directions (so −v gives the opposite behaviour).
- Applied at one layer, on every token; scaling it sets intensity.

**What they found.**
- Stronger, smoother control than CAA over personas (power-seeking, wealth-seeking…), truthfulness, hallucination and jailbreaking, with MMLU unchanged.
- Vectors transfer to fine-tuned variants of the model, and adding two vectors steers both behaviours.

**Where it touches us.**
- "Personalised" here means **a behaviour dial chosen by a designer and trained on a preference dataset**, not a memory of what a user said. That leaves our setting open.
- **Their criticism of raw activation differences is a general form of our C5 problem:** a raw difference doesn't hold the content you want. Their fix (optimise against what the model actually generates, in both directions) is the natural *trained* remedy for dislikes. It's future work if training is allowed.

## 8. He, Pandey, Schrum, Dragan (UC Berkeley), "Context Steering: Controllable Personalization at Inference Time" (CoS, ICLR 2025)

**What they did.**
- No activation editing: they work on the output probabilities.
- Run the model twice, with and without the user's context (e.g. "I am a toddler"). The difference is the context's influence; scale it by λ: CoS = LLM(x | C, P) + λ·[LLM(x | C, P) − LLM(x | ∅, P)].
- Also run it in reverse (Bayesian inference over λ or over the context), to measure how strongly a text reflects a context.

**What they found.**
- Controllable personalisation (λ up = more personalised; human ratings agree) and bias reduction.
- Classifying and grading implicit hate speech by inferring its hidden context.
- One of their examples is literally "I am carnivore / I am vegetarian".

**Two sentences that touch our C5 directly.**
1. *"A single context such as 'I am of low STEM proficiency' also indicates STEM is the subject of discussion, thus making all STEM-related generations more likely. Instead, if we contrast it with 'I am of high STEM proficiency', the difference of the two contexts will capture the difference in proficiency."* That's our concept-vs-direction effect, noted informally at the output level.
2. A negative context with a negative λ has a **weaker effect** than the positive one, which they put down to "inverting the context vector in the semantic space does not have clear meanings". That's our sign problem.

**Where it touches us.**
- **C5's core idea isn't new as an observation.** CoS states it in a sentence and uses contrast pairs as a practical fix. We must cite it.
- CoS **needs the context present** at inference (two passes, with the context in the prompt). It *amplifies* context; it doesn't *remember* it. Our setting removes the context entirely.

---

# C. Where facts live and how they're recalled

## 9. Geva, Bastings, Filippova, Globerson (Google DeepMind / Tel Aviv), "Dissecting Recall of Factual Associations in Auto-Regressive Language Models" (arXiv 2304.14767, 2023)

**What they did.** Traced how GPT-2 and GPT-J answer factual prompts ("Beats Music is owned by → Apple"), by blocking attention between positions and projecting states onto the vocabulary.

**What they found.** Recall happens in three steps:
1. **Subject enrichment:** early MLP layers build a representation at the subject's last token that holds many attributes of the subject.
2. **Relation propagation:** the relation moves to the last position.
3. **Attribute extraction:** **attention in the upper layers** reads the subject's representation and pulls out the one attribute the question asks for.

The evidence:
- Blocking attention from the last position to the subject in the middle-upper layers drops the prediction by up to 60%.
- The right attribute is *not* at the top of the subject's representation: its rank there averages about 1,000. Attention picks it out **depending on the question**.
- Many attention heads store subject → attribute mappings in their weights.
- They note that the logit lens is only an approximation, especially in early layers.

**Where it touches us.**
- **The mechanism behind C3, C4 and C6.** A fact is fetched by attention, driven by the question. Our stored fact shift is the *already-extracted output* ("say Pepper"). That's why it exists only in the last layers (C6), and why pushing it at every word floods.
- **Caveat:** they study facts stored in the weights. We assume that a fact said earlier in a *conversation* is also reached by question-driven attention. That's plausible, but an assumption.
- Cite them for the logit lens's limits too.

## 10. Meng, Bau, Andonian, Belinkov (MIT / Northeastern / Technion), "Locating and Editing Factual Associations in GPT" (ROME, NeurIPS 2022)

**What they did.**
- **Causal tracing:** corrupt the subject, then restore one hidden state at a time, and see which ones bring the right answer back.
- Then **edit one fact into the weights** (Rank-One Model Editing). They treat a middle-layer MLP as a **linear associative memory**, keys → values, and insert a new pair with a least-squares update that uses the covariance of the keys.
  - The key is the subject's representation, averaged over random prefixes.
  - The value is **found by optimisation**, so that the model outputs the new fact and keeps the subject's other properties.

**What they found.**
- Two decisive sites: **middle-layer MLPs at the subject's last token** (where the fact is stored), and late attention at the last token (where it's read out).
- ROME edits generalise to paraphrases and stay specific to the subject.
- **Editing late attention instead leads to regurgitation:** the model just repeats the target.
- Weaker editors produce fluency failures ("medicine medicine medicine…").

**Where it touches us.**
- **Our store is the same mathematical object.** Our whitened keys and least-squares writes are ROME's covariance-weighted associative memory, moved from the weights into the activations at inference time.
- **Their late-edit regurgitation and repetition failures match our late-layer floods.**
- **Why ours can't carry a usable fact:** ROME writes at the subject's position, in middle layers, with an *optimised* value. We write an *observed* shift and add it everywhere.

## 11. Meng, Sen Sharma, Andonian, Belinkov, Bau, "Mass-Editing Memory in a Transformer" (MEMIT, ICLR 2023)

**What they did.** Scaled ROME to thousands of facts at once:
- compute a target vector per fact by optimisation
- spread the change over a range of critical MLP layers (layers 3–8 in GPT-J)
- solve one batch least-squares update per layer

**What they found.**
- Thousands of edits (up to 10,000) on GPT-J and GPT-NeoX with good efficacy, generalisation and specificity, where ROME and other methods degrade.
- They count outputs like *"baseball baseball baseball baseball"* as fluency failures.

**Where it touches us.**
- Their batch update is our least-squares (RLS) write. Their term for the pre-existing keys plays the role of our λ prior.
- **The counterpoint for the write-up:** usable facts *can* be stored at scale, but in the weights, with optimisation. That's what our training-free, reversible activation memory gives up.

---

# D. Negation

## 12. Jang, Ye, Seo (KAIST), "Can Large Language Models Truly Understand Prompts? A Case Study with Negated Prompts" (arXiv 2209.12711, 2022)

*CAST cites a 2023 workshop version (Transfer Learning for NLP, PMLR). Cite the published one.*

**What they did.** Gave OPT and GPT-3 models (125M to 175B) nine tasks, each with the instruction also negated ("give an *incorrect* answer").

**What they found.**
- **Larger models do worse** on negated prompts: inverse scaling.
- Averaged over the original and negated versions, accuracy stays flat at about 50%. **The models treat the negated prompt like the original.**
- Instruction tuning doesn't fix it. In-context examples help only sometimes. Fine-tuning helps, but costs accuracy on the original task.
- Their conjecture: pretraining text is skewed toward the positive form, so "not" behaves like noise.

**Where it touches us.**
- Behavioural evidence that negation is processed weakly, which supports our explanation for C5: "I can't stand jazz" ≈ jazz.
- **Two caveats:**
  - Their negation is in the *instruction*; ours is in *a statement about the user*.
  - Their models are from 2022, and they measure outputs, not internal states.
- Our dislike flip holds at both 2B and 9B, so it isn't fixed by scale either.

---

# The theory these give us

## Why a stored shift can carry a leaning but not a fact

**1. In context, the effect of earlier text is recomputed for every question.**
- ICV shows that attention mixes the context's states into each new position with weights that depend on the current query.
- Geva shows that a fact is pulled out by upper-layer attention, driven by the question: the right attribute isn't sitting ready in the subject's representation.

**2. A stored memory replays one fixed average of that effect, at every word.** Seahorse's shift is the pooled difference a moment made, added back unchanged wherever the gate is open.

**3. Only the part that is the same for every question survives the averaging.**
- **A leaning survives.** "Prefers vegetarian" pushes the same way on any food question, so a fixed push helps, within a narrow dose window (core_v1: α = 1 yes, α = 2 incoherent). This matches the steering literature: whole-answer traits steer (Subbiah), and so do styles and personas (ICV, BiPO, CoS).
- **A fact doesn't.** "My dog's name is Pepper" matters at one slot ("your dog's name is ___"). The question-driven retrieval is gone, and what's left is the *already-extracted* output, "say Pepper", which exists only in the last layers (C6; Geva's late extraction; Lindsey's "later layers just say the word"). Pushed at every word, it floods or garbles (C3). ROME's late-layer edits regurgitate and MEMIT scores "baseball baseball" as failure, for the same reason.
- **A premise doesn't either.** Using a fact as a premise means looking it up and combining it with the question, which is attention's job. A fixed push can't do it (C4: no gain at 2B or 9B).

**4. Usable facts live somewhere else.** Either in the weights, at the subject's position in middle layers, with an optimised value (ROME/MEMIT), or in the context, where attention can retrieve them (Seahorse's planned episodic channel). Our store uses ROME/MEMIT's maths in the activations, without optimisation. That keeps it selective and reversible, and gives up usable facts.

## Concept vs direction

A stored preference has parts, and what you subtract when writing it decides which part survives:

| Part | Example | Kept by subtracting |
|---|---|---|
| *what* | jazz | the centroid (other genres), `without`, other disclosures |
| *which way* | like / dislike | the opposite ("I can't stand jazz" vs "I'm a huge jazz fan") |
| *whose* | the user's, not the assistant's | nothing yet |

- **Keep only *what*,** and a dislike becomes a like, because negation is encoded faintly (Jang; Lindsey).
- **Keep only *which way*,** and a concept that appears on both sides cancels: the jazz fan becomes "enthusiastic".
- **Simply adding the two** puts "jazz" and "hate" side by side; it can't say "hate *about* jazz". The model reads the louder one, jazz.

**Pre-registered confirmation (core_v1, 8 dislikes, each reference at its strongest dose with ≤ 10% loops):**

| Reference | Result, 2B | Result, 9B |
|---|---|---|
| opposite | right way, +0.23 | right way, +0.25 |
| centroid | wrong way, −0.38, every item | wrong way, −0.37, every item |
| `without` | wrong way, −0.25 | wrong way, −0.31 |

These are judge-labelled numbers, provisional until the hand audit.

**Prior work and what's ours:**
- CoS notes the effect informally (a single context carries its topic; a contrasting pair isolates the attribute). BiPO argues raw activation differences miss the intended behaviour. Jang and Lindsey show negation is weak.
- **Ours is the systematic, pre-registered demonstration in a stored memory:**
  - the consequence that dislikes flip into likes
  - matched doses
  - two model scales
  - logit-lens evidence of what each reference contains

**What should work:** apply the concept with a sign, so we do the binding ([HANDOVER](../../HANDOVER.md), next steps). A trained alternative is BiPO-style optimisation of the vector against generated answers.

---

# What is prior work and what is ours

| Claim | Prior work | What's ours |
|---|---|---|
| **The setting:** a memory written from one conversation and used, unprompted, in later sessions | ICV, BiPO and CoS steer or personalise, but need the context present (CoS, ICV) or a trained vector from a dataset (BiPO) | A persistent activation memory with no context and no training |
| **The store** | ROME/MEMIT's linear associative memory (least squares with key covariance) | The same maths, in activations instead of weights, with observed instead of optimised values |
| **C1 selectivity** | CAST (conditional steering with a cosine threshold) | Keys written from one experience; whitening; a many-memory store |
| **C2 leanings transfer** | ICV, CAA, BiPO, CoS; the whole-answer-trait finding (Subbiah) | Nothing new. It also failed its pre-registered rule: there's a narrow dose window, and coherence collapses at α = 2 |
| **C3/C4/C6 facts and premises fail** | Explained by Geva (question-driven attention retrieval), ROME/MEMIT (facts in middle-layer weights, optimised values), ICV (the in-context effect is recomputed per query) | The direct demonstration, on 2 scales, with use prompts, balanced yes/no and a per-layer logit lens |
| **C5 concept vs direction** | Noted informally by CoS; BiPO's critique of raw differences; weak negation (Jang, Lindsey) | The systematic demonstration in a memory; dislikes flip; matched doses; two scales |
| **Failure modes** | Overdose, identity loss, countries hardest (Lindsey); regurgitation and repetition (ROME, MEMIT); expression vs coherence (Subbiah) | The same failures, measured with coherence scores and a judge, in a memory setting |
