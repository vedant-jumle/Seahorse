# Paper 1, arXiv v2: what to change

*Written 2026-10-16 for a fresh session that will make the edits. Paper: "What Survives When Activation Steering Is Used as Memory", source `writeup/paper 1/main.tex` (+ `refs.bib`). It is **public on arXiv**, so this is a v2. The submitted v1 source is `writeup/paper 1/arxiv/arxiv_source.zip`.*

Read first: `HANDOVER.md` (§1, §5 Exp 1), `docs/dreaming/01-binding.md` (Results section), `docs/dreaming/literature-exp1.md`, `docs/reference/literature.md` (groups E and F).

## Ground rules
- **No result in the paper changes.** Every pre-registered claim (C1–C7) and number stays as is. Don't touch §5 Results or its figures.
- **No new headline results.** The cache-slice, collision, hybrid recognition/recall and PrefEval results belong in a separate paper 2. v2 may *point* to follow-up work, without numbers (except stage 0, below).
- **Seahorse is an independent side project.** Never call it thesis or MSc work.
- **Keep the paper's style:** short sentences, claims with CIs, cautious wording.
- **Build:** `latexmk -pdf main.tex`. A review/anonymous variant exists (`main_review.*`, `\REVIEW` defined). Build both if you touch anything conditional. Check `main.log` for undefined references and citations.
- **Commit** `main.tex`, `refs.bib`, any new figure and this file, with the usual trailer. Don't commit `main.pdf` unless it's already tracked; check `git ls-files writeup`.
- **Make a new arXiv source zip** for v2 (same structure as `arxiv/arxiv_source.zip`). Add a v2 comment: "v2: adds direct evidence for the read site of in-context facts (App. X), related work on KV-cache reuse and trained fast-weight memory, and clarifies that the negative result concerns pushed shifts, not activation memory in general."

## 1. MUST: narrow one over-general claim
The paper says activation memory suits dispositions while "facts need weights or retrieval". Follow-up work shows facts *can* live in activations when the model's own cached states are restored at their positions (Exp 1). What fails is the **pushed shift**. Edit:
- **Intro, contributions paragraph** (`main.tex` ≈ L68): "activation memory suits dispositions, while facts need weights or retrieval" → "a pushed activation shift suits dispositions, while facts must be placed where attention can reach them: in context, as restored cached states, or in weights."
- **Conclusion** (≈ L207): the same sentence ("Activation memory is therefore a natural home for dispositions, while facts need weights or retrieval") → the same narrowing.
- **Check the abstract and Fig. 1 caption** (≈ L41–47) for the same generalisation, and narrow it where present. Keep "the memory never serves as a premise", which is correct for this memory.

## 2. SHOULD: stage 0 as direct evidence for P1 (new short appendix + one sentence in §3)
**Why:** §3 (≈ L109–113) explains "facts fail" via Geva et al., which is about facts in *weights*. Stage 0 measures the *in-context* case directly. It was pre-registered (`experiments/dream_bind/PREREG.md`, stage 0) and run on two models.

**What it shows** (numbers from the reports; re-check them there):
- **Setup:** the fact in context ("My dog's name is Pepper. What's my dog's name?"), activation patching between the true-value run and a counter-value run (Pepper vs Rocky), identical token layout. Read-out: share of the logit(Pepper) − logit(Rocky) flip.
- **Patching the question's subject token** ("dog", after the name): **0.00** at every block and band, on Qwen2.5-1.5B and Qwen3.5-2B.
- **Patching the attribute's own position:** **1.00** up to late blocks. Qwen2.5 falls only in the last blocks; the 2B drops sharply at block 19.
- **Attention knockout** (Qwen2.5 only; Qwen3.5's Gated DeltaNet blocks can't be masked per token):
  - blocking answer → subject costs 0.00 in every window
  - blocking answer → attribute costs 0.59 [0.41, 0.81] only in the last window (layers 23–27); the rest flows through relay tokens
- **Exploratory:** the subject mention in the answer prefix ("Your **dog**'s name is") also carries ≤ 0.03.
- **Reading:** the answer reads the value from the position where it was said, late. The subject holds no copy of the value (it may still carry an order-ID pointer, cf. Prakash et al.'s lookbacks), so a vector pushed elsewhere has nothing to imitate.
- **Sources:**
  - `results/dream_bind_stage0_20261011_2321/{qwen25_15b,qwen35_2b}/report.md` (verdicts + per-block tables)
  - `.../*/prefix/report.md` (exploratory)
  - code `experiments/dream_bind/run.py --stage patch|knockout|prefix`
- **Caveats to state:**
  - Different models from the paper's main runs: Qwen2.5-1.5B and Qwen3.5-2B, not the 9B.
  - 7 primary items.
  - Stage 0 belongs to a later pre-registration (cite it as follow-up, run after v1).
- **Edit:**
  - One sentence at the end of §3's first paragraph: "A direct test on in-context facts (Appendix X) finds the same: the answer reads the value from the position where it was stated, and the subject's position holds no copy of it."
  - A short new appendix with a small table: patching subject vs attribute per model, and the knockout line. A per-block curve figure is optional; the data is in `results.jsonl.gz`.

## 3. SHOULD: related work, "Memory in activations and weights" paragraph (≈ L202)
Reviewers will ask "why not store the KV cache?". Add, briefly:
- **Cache and segment reuse as memory:**
  - InfLLM (arXiv 2402.04617), EM-LLM (2407.09450): retrieve stored KV blocks into attention
  - Prompt Cache (2311.04934), CacheBlend (2405.16444), EPIC (2410.15332): reuse cached chunks across contexts/positions
  - Memory³ (2407.01178), KBLaM (2410.10450): knowledge as sparse/explicit KV, trained
  - One sentence: these keep content at *positions attention can reach*, which is the property a pushed shift lacks.
- **Trained fast-weight memory:** FwPKM (Zhao & Jones, arXiv 2601.00671). A delta-rule fast-weight memory read by a gated residual add, which *does* carry facts (needle-in-a-haystack) because the read path is trained with the model. It supports the paper's account: a frozen model was never trained to read a pushed shift.
- **In-context binding:** Feng & Steinhardt (2310.17191, binding IDs), Prakash et al. 2024 (2402.14811, entity tracking) and the lookbacks paper (2505.14685, ICLR 2026): in context, attributes are bound and fetched by position, so content must sit at a position to be bound.
- **BibTeX:** none of these keys exist in `refs.bib` yet. Pull exact metadata from arXiv / the venue (verify titles, authors, venue and year; the lookbacks paper is ICLR 2026, not NeurIPS 2025). Notes on what each shows are in `docs/dreaming/literature-exp1.md` and `docs/reference/literature.md` group F.

## 4. OPTIONAL: point to follow-up work in Limitations / Conclusion (no numbers)
- **Limitations, "Design"** (≈ L216) says the hand-written references are a limitation. Optional addition: "In follow-up experiments (reported separately), references generated by a larger model of the same family reproduce the reference effect."
- **Conclusion's last sentence** (≈ L207) proposes storing the concept and its sign separately. Optional addition: "follow-up experiments find this works when the sign is known, while reading the sign automatically remains open."
- Keep these as pointers. The exp 2 results are exploratory-plus-benchmark and are not reviewed in this paper. Sources if wording needs checking: `results/dream_refs_20261015_1545/qwen35_2b/analysis/report.md` (S2 holds; cs:oracle best; S4 fails on the sign reader).

## 5. CHECK: layout
`HANDOVER.md` flagged, for the workshop cut, an **empty Appendix D header and float placement**. The appendices are A Items, B Judges, C Preliminary, D Additional figures, E Pre-registration, so D is likely "Additional figures" with floats drifting away from its header. Check in v1's PDF and fix if present (e.g. `[h!]`/`\FloatBarrier` from `placeins`, or move the floats under the header).

## Status (2026-10-16)
- Sections 1–5 applied in commit (see git log). Both builds clean. v2 zip at `writeup/paper 1/arxiv/arxiv_source_v2.zip` (git-ignored). Awaiting the user's PDF review before upload.

## Done when
- §1 is applied everywhere the generalisation appears.
- §2's appendix and sentence are in, with numbers re-checked against the reports.
- §3's citations are added with verified BibTeX.
- The build is clean (no undefined refs), and both variants compile.
- A new arXiv source zip and the v2 comment are prepared.
- The user reviews the PDF diff before upload. **Do not upload to arXiv; the user does that.**
