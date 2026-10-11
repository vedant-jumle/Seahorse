# Experiment 1: dreaming as binding (the subject-position write)

*Can a fact be used if the model's own representation of it is written at the subject's position, rather than pushed at the answer? Exact process. Written 2026-10-11.*

---

## 1. The idea in one paragraph

When the model reads *"My dog's name is Pepper. I need to take my dog to the vet tomorrow."*, the token **dog** in the second sentence carries, at early-middle layers, a representation enriched by the first sentence (Geva's subject enrichment). That representation is the model's *own* binding of "Pepper" to "my dog", made by its own circuits. We harvest it offline (the dreaming pass), store it keyed by the subject, and in a later session add it back **at the subject token** ("my dog") in early-middle layers, where the model's extraction heads can read it when a question arrives. We never compose the binding; we put the model's bound state where its circuits look for it.

The comparison arm reinstates the experience sentence as a short segment of positions attention can attend to (the InfLLM route). Both arms supply a *position*; the current push supplies none.

## 2. What is tested

**Predictions, written before the run (go into `PREREG.md`):**
- **B1.** Subject-position writing yields *clean use* of the fact above the push's ceiling (core_v1: ≤ 0.15) on the use prompts, with the CI above the push.
- **B2.** Balanced yes/no accuracy rises above 0.5 with the CI above 0, for the first time in the project.
- **B3.** The **wrong-position control** (same vector, added at a non-subject token) gives no gain. If it does, the effect is just another push and B1 is not about binding.
- **B4.** Cross-binding stays low: with two facts stored (dog Pepper, cat Jasper), a question about the cat does not produce Pepper more often than with no memory.
- **B5 (comparison).** Segment reinstatement on Qwen2.5-1.5B (all layers full attention) reaches at least the subject write's clean use; on Qwen3.5-2B (six full-attention blocks) it is weaker. Reported, not decisive.
- **B6.** Unrelated prompts remain token-identical to no memory (≥ 0.95), since the injection site is the subject token and the subject does not appear in them.

## 3. Items

The 12 core_v1 facts (`data/bench_v1/facts.yaml` via `experiments/core_v1/items.yaml`). Each gets a new **subject** block in a config (`experiments/dream_bind/config.yaml`):

```yaml
dog_name:
  subject: ["my dog", "your dog", "the dog", "dog"]      # aliases, matched in the user turn and in the model's own words
  harvest_followups: ["I need to take my dog to the vet tomorrow.", "My dog has been really energetic lately."]
```

Rules for the subject list:
- The first alias is canonical; the others are what the subject looks like in the use prompts and in the model's answers ("your dog" when the model refers back to it).
- Some use prompts don't contain the subject literally (*"the place where I grew up"*, *"my native language"*, *"my work's website"*). Add the paraphrase as an alias where it's unambiguous; otherwise leave the prompt as a **no-site** prompt. No-site prompts are reported separately: they measure what happens when there is nowhere to inject (expected: nothing, which is itself informative).
- Only the follow-ups that mention the subject are harvest follow-ups (2 of the 3 per item). The third, generic follow-up is excluded from this experiment's writes.

## 4. The dreaming pass (harvest)

Run on the frozen model, offline, once per item. Three variants, all stored; the primary is **H1**.

**H1: subject delta from the follow-ups (primary).** For each harvest follow-up:
1. Run **with**: chat template + user turn = experience + follow-up. Record the residual at every block at the **last token of the subject span** in the follow-up (Geva: the last subject token is the one that aggregates the subject).
2. Run **without**: the same, with the experience removed.
3. Subject delta at block ℓ: `d_ℓ = h_ℓ(with, subj) − h_ℓ(without, subj)`.
4. One delta per follow-up; average the two (or keep both as separate memory entries under the same key; store both options, report the average as primary).

This is the same two runs core_v1 already does; the only change is *where* the state is read: the subject token, not an entropy-weighted pool over the whole follow-up.

**H2: minimal replay.** Chat template + user turn = the experience sentence alone (*"My dog's name is Pepper."*). Record the residual at the last subject token (**dog** in "My dog's name") at every block, and the corresponding state from a neutral sentence with the same subject (*"I have a dog."*). Delta as above. Tests whether a representation harvested with almost no context is more portable (less orphaned) than H1.

**H3: raw enriched state (no delta).** H2's "with" state alone, not a difference. Added at reduced strength. Tests whether the model's own state is a better thing to reinstate than a difference. Expect it to hijack (core-idea P1); include it because if it works it changes the design.

**Also harvested, for the reinstatement arm (R):** the residual at every block for *every token* of the experience sentence in the minimal replay (H2's run), plus the token ids. On Qwen3.5-2B also the recurrent state of the linear-attention blocks after the segment is not stored (it can't be spliced); the arm is KV-only.

**Checks (fail the job if violated):** the H1 "without" run equals core_v1's; the subject token index is found in every harvest follow-up; the delta norm is finite and within 10× the median residual norm at that block.

## 5. The store and the key

- **Store:** one entry per item: `(key_s, {d_ℓ for ℓ in write layers})`. Delta-rule write into the existing M per layer is fine, but for this experiment the store is a lookup (isolated memories), as in core_v1.
- **Key:** the whitened, pooled key of the **subject span** in the without run (the same `pooled_w256` construction, pooled over the subject tokens only). This files the memory under *the subject*, not the follow-up's situation (fixing failure 7's cause for facts).
- **Gate:** two variants, both run:
  - **G-match:** the subject alias appears in the user text (string match on the detokenised user turn). Fires at every occurrence. Primary.
  - **G-key:** the existing cosine gate with the subject key, threshold at the q0.95 generic match. Reported; tells us whether the key alone would find the subject.
- **Injection site:** the last token of each matched subject span in the prompt; variant **+1** also injects at the following token (Feng & Steinhardt: the binding ID sits on the entity token and the one after). Optionally also at subject tokens the model generates itself ("your dog"); run as variant **gen** with the match applied to generated text in the two-pass scheme.

## 6. Where and how strongly

**Write layers.** Geva: enrichment in early-middle MLPs, extraction by upper attention. For Qwen3.5-2B (24 blocks) sweep single blocks `{4, 6, 8, 10, 12, 14, 16}` and bands `{4–8}, {6–10}, {8–12}, {10–14}`. The deltas are harvested at the same block they are written at (no cross-layer transfer; xlayer_v1 showed that fades). For Qwen2.5-1.5B (28 blocks) scale by depth.

**Dose.** `h_subj ← h_subj + α · d_ℓ`, α ∈ {0.5, 1, 2}. No norm-matching across layers (same layer in and out). Because the injection touches one or two positions rather than every position, higher α may be tolerable; report incoherence per α as always.

**Analytic pre-pass (cheap, before generation).** For every (variant, layer set, α): teacher-forced `Δ log P(target)` and specificity at the measure prefix (*"Your dog's name is"*), and the balanced yes/no margins. Pick the generation grid from this: the best three layer sets per variant, at the two best α. Also run the **logit lens on the subject delta**: it should *not* rank "Pepper" top at early layers (it's an enriched subject, not the word); if it does, we have harvested a late-layer "say Pepper" again.

## 7. Conditions for generation

Per item, per prompt, same random streams across conditions:

| Condition | What |
|---|---|
| `nomem` | nothing |
| `ctx` | experience in the prompt (ceiling) |
| `push@1` | core_v1's memory at blocks 20/21/23, α=1 (the published channel, for reference) |
| `subj:H1@{layers}@{α}` | subject-position write, primary variant, from the analytic pick |
| `subj:H2@…`, `subj:H3@…` | the other harvests |
| `subj:H1+1` | also inject at the token after the subject |
| `subj:H1:gen` | also at generated subject tokens |
| `reinst@{full-attn layers}` | the experience sentence's harvested states spliced as extra K,V entries (recomputed with the frozen W_k, W_v from the stored residuals) at a fixed position before the user turn, inside a user-role span |
| `ctrl:rand` | a random vector with `d_ℓ`'s norm at the subject site |
| `ctrl:swap` | another item's subject delta at this item's subject site |
| `ctrl:wrongpos` | this item's `d_ℓ` at a non-subject token of the same prompt (the token at the same offset from the end) |
| `ctrl:placebo` | the subject delta harvested from a placebo experience (*"By the way, I had a coffee this morning."*) |

**Prompts per item:**
- the direct question
- the two use prompts (with no-site prompts flagged)
- the balanced yes/no probes (greedy, 10 tokens, first-token margins)
- **cross-binding probes** (new): for items with a partner (dog/cat, sister/brother, best friend/boss), the partner's direct question and use prompts with *both* memories stored. Measures: the target of the *wrong* item appearing (should not rise over nomem).
- 50 unrelated prompts, greedy + 2 samples

Generation settings as core_v1: 1 greedy + 4 samples, T=1, top-p 0.95, top-k 20, presence 1.5, cap 120; second seed set for `nomem`, `ctx`, `push@1` and the primary `subj` cell.

## 8. Measures

- **Primary:** judge `clean use` on the use prompts; **balanced yes/no accuracy**.
- Also: `ans_tgt_clean` (regex), mention-without-use, confabulation, self-claim (the owner failure), loop rate, incoherence, NLL, length, `unrel_same`, cross-binding leak, teacher-forced target gain and specificity, the logit lens of the stored deltas per block.
- Judges: the original Qwen3.5-9B plus gpt-oss and Gemma-4 as in the panel; report the majority.

## 9. Decision rules (into PREREG.md)

- **Holds (binding works):** B1 and B2 and B3 and B6. The subject write's clean use CI lies above `push@1`'s *and* above `ctrl:wrongpos`'s; balanced accuracy gain CI > 0; `unrel_same ≥ 0.95`.
- **Push in disguise:** B1 holds but B3 fails (wrong position works as well). Then position isn't what mattered; report as a stronger push, not binding.
- **Orphaned:** nothing beats the push at any layer or α, while the analytic pre-pass shows `Δ log P(target)` rising. The representation is there but not readable out of context. Go to full reinstatement / the trained read path.
- **Reinstatement wins:** `reinst` beats `subj` on Qwen2.5. Then the content channel is attention-based and the subject write is a shortcut that doesn't hold.

## 10. Job plan

- `experiments/dream_bind/run.py --stage harvest | analytic | gen | score | analyze`, `--model qwen35_2b | qwen25_15b`.
- Smoke (`--tiny`: 2 items, one layer set, one α) → harvest+analytic (short) → gen (the picked grid; budget it to ≤ 4 h on a V100: 12 items × ~10 prompts × ~12 conditions × 5 answers ≈ 7,000 generations, about 10 h at core_v1's rate, so **cut the grid to ≤ 6 generation conditions** from the analytic pick) → score on the A100 (judges) → analyze.
- Qwen2.5-1.5B run for the reinstatement arm on `gpu-a100-small`.
- 9B confirmation only for the winning cell.

## 11. What we'd learn even if it fails

- Whether the subject representation at early-middle layers carries the fact at all in a conversation (the analytic pre-pass answers this independently of generation).
- Whether a harvested state is portable across contexts (H2 vs H1).
- Whether position matters (wrongpos control). These three answers decide the content channel's design regardless of the headline.
