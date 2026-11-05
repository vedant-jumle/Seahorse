# LoCoMo extract for dream_compare (`manifest_locomo10.json`)

- **Source:** `data/locomo10.json` from https://github.com/snap-research/locomo at commit
  `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` (downloaded 2026-11-05; sha256 in the manifest). The raw file and the
  processed split live in `data/external/locomo/` (git-ignored). Only the manifest (counts, hashes, item ids) is here.
- **Licence:** CC BY-NC 4.0 (non-commercial; as released by the authors).
- **Citation:** Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, Yuwei Fang.
  *Evaluating Very Long-Term Conversational Memory of LLM Agents.* ACL 2024. arXiv:2402.17753.
- **Built by:** `python -I experiments/dream_compare/benchmarks/prep_locomo.py [--download]`
  (adapter: `experiments/dream_compare/benchmarks/locomo.py`).

## What is in it

There are 10 conversations and 1,986 questions: 1,540 non-adversarial and 446 adversarial.

**Category codes (verified 2026-11-05):** 1 multi-hop (282), 2 temporal (321), 3 open-domain (96), 4 single-hop (841),
5 adversarial (446). The paper's section 4.1 lists the categories in a different order, so the codes cannot be read off
it. The mapping comes from:
- the official scorer (`task_eval/evaluation.py`: code 1 is scored by sub-answer F1, "multi-hop"; code 5 is
  "adversarial");
- the QA code (`task_eval/gpt_utils.py`: code 2 gets "Use DATE of CONVERSATION...");
- the questions themselves;
- the repo's issues #6 and #29.

How the conversations become memory turns:
- The two speakers are reframed as chat roles: `speaker_a` -> user, `speaker_b` -> assistant. The name stays in the
  text ("Caroline: ..."), because the questions name both people.
- Image turns get their BLIP caption appended: "... [shared an image: <caption>]". 1,226 turns carry one.
- Session dates are kept. Sessions that are listed with a date but have no dialog are skipped.
- Adversarial questions are abstention items with no gold answer. The distractor is kept in `meta.adversarial_answer`.
- A few evidence ids do not name a turn (5 items, for example "D" or "D:11:26"). They are dropped and listed in
  `meta.evidence_dropped`. 5 items have no evidence left.
