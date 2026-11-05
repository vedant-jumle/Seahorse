# LongMemEval extract: the single-session-user items (`ssu70.json`)

- **Source:** LongMemEval, cleaned oracle split, `longmemeval_oracle.json` from
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned (downloaded 2026-11-04; sha256
  `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`). The raw file is kept out of git, in
  `data/external/longmemeval/` (ignored).
- **Licence:** MIT (as released by the authors).
- **Citation:** Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu. *LongMemEval: Benchmarking Chat
  Assistants on Long-Term Interactive Memory.* ICLR 2025. arXiv:2410.10813.
- **Built by:** `python -I experiments/dream_select/prep_data.py data/external/longmemeval/longmemeval_oracle.json data/longmemeval/ssu70.json`

## What is in it

The 70 items with `question_type == "single-session-user"`. Per item: `question_id`, `question`, `answer`, the evidence
turns (the user turns with `has_answer == True`, with their session and turn index), `verbatim` (the answer string
occurs in an evidence turn, case-insensitive), `stored_turn` (the first evidence turn that contains the answer
verbatim, else the first evidence turn; `stored_index` says which) and `usable`.

- **64 usable items**, each with exactly one user evidence turn; **56 of them verbatim**.
- **6 abstention items** (`question_id` ending in `_abs`; the answer is "You did not mention this information ...")
  flag no turn at all. They are kept with `usable: false` and are not used by dream_select.
- **2 items** also flag the assistant reply right after the user's evidence turn (it repeats the fact). Those turns are
  listed under `assistant_flagged` and never stored: the memory is the user's turn.
- No item has 2 user evidence turns.

## dream_compare splits (`manifest_oracle.json`, `manifest_s120.json`)

- **Built by:** `python -I experiments/dream_compare/benchmarks/prep_longmemeval.py [--download]`
  (adapter: `experiments/dream_compare/benchmarks/longmemeval.py`). The processed splits live in
  `data/external/longmemeval/processed/` (git-ignored). The manifests hold counts, hashes and item ids.
- **oracle:** all 500 items of `longmemeval_oracle.json`, 30 of them abstention. The source file is the same as
  dream_select's (sha256 `821a2034...`).
- **s120:** 120 items of `longmemeval_s_cleaned.json` (same repo and revision `98d7416c...`; sha256 `d6f21ea9...`).
  It takes 20 per question type, drawn with `random.Random(7)` over the sorted ids of each type (10 are abstention
  items). Each item keeps its full haystack, about 489k characters on average.
- **Per item:** one context. Sessions are put in time order (the oracle file lists them unordered). Turn ids are
  `<question_id>/<session_id>:<turn index>`. Evidence turns are the `has_answer` turns (user or assistant); evidence
  sessions are `answer_session_ids`.
- **S-file quirks:** 5 empty distractor turns are dropped. 3 sessions repeated inside one haystack get "#2" suffixes.
- **For Stage A:** 182 oracle items have exactly one evidence turn. Counting user turns only (dream_select's rule)
  gives 133, including dream_select's 64 single-session-user items.
