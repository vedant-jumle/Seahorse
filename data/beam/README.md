# BEAM extract for dream_compare (`manifest_100K.json`)

- **Source:** `data/100K-00000-of-00001.parquet` from https://huggingface.co/datasets/Mohammadta/BEAM at revision
  `3205395e897e7318c7b094ef4e6047b9b82dbb03` (downloaded 2026-11-05; sha256 in the manifest). The raw file, a JSON
  dump of its rows (so the cluster, which has no pyarrow, can rebuild the split) and the processed split live in
  `data/external/beam/` (git-ignored).
- **Licence:** CC BY-SA 4.0 (data). The BEAM code (github.com/mohammadtavakoli78/BEAM) is MIT.
- **Citation:** Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani, J Ross Mitchell. *Beyond a
  Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs.* 2025. arXiv:2510.27246.
- **Built by:** `python -I experiments/dream_compare/benchmarks/prep_beam.py [--download] [--rows-json PATH]`
  (adapter: `experiments/dream_compare/benchmarks/beam.py`).

## What is in it

There are 20 conversations of about 100K tokens each. Each has 20 probing questions over 10 types. Summarisation and
instruction-following are dropped (as planned), which leaves 320 questions: 40 each of abstention,
contradiction_resolution, event_ordering, information_extraction, knowledge_update, multi_session_reasoning,
preference_following and temporal_reasoning.

How the data is read:
- `probing_questions` is a Python-literal string in the file. It is parsed with `ast.literal_eval`, which reads
  literals only and executes nothing.
- BEAM appends a marker " ->-> i,j" to some user turns. It is removed from the text and kept in `meta.marker`
  (2,191 turns).
- Answers come from `answer`, `ideal_answer` (contradiction_resolution) or `ideal_response` (abstention).
  preference_following has only a rubric.
- Evidence = `source_chat_ids`; ids that do not exist are dropped. 41 items have no evidence: the 40 abstention items
  and 1 other.
- Scoring: BEAM's rubric judge, averaged over each item's rubric items (`experiments/dream_compare/judges.py`).
