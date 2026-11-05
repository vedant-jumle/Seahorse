# PersonaMem v1 extract for dream_compare (`manifest_32k.json`)

- **Source:** `questions_32k.csv` and `shared_contexts_32k.jsonl` from
  https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1 at revision `a8076d5608c93ba2a28983cd78aa99b01a163ae7`
  (downloaded 2026-11-05; sha256 values in the manifest). The raw and processed files live in
  `data/external/personamem/` (git-ignored).
- **Licence:** MIT.
- **Citation:** Bowen Jiang et al. *Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and
  Personalized Responses at Scale.* 2025. arXiv:2504.14225.
- **Built by:** `python -I experiments/dream_compare/benchmarks/prep_personamem.py [--download]`
  (adapter: `experiments/dream_compare/benchmarks/personamem.py`).

## What is in it

There are 589 multiple-choice questions (4 options each) over 20 personas and 37 shared contexts. The 7 question types
are listed in the manifest.

How the data is read:
- **Sessions:** a new session starts at each system message (the persona prompt). That message is kept as the
  session's first turn, with role "system".
- **Turn text:** the raw "User: " / "Assistant: " prefixes are removed, since the role already carries that
  information. 6,021 turns were affected.
- **Visible context:** each question sees `context[:end_index_in_shared_context]`, as the dataset's README says.
- **Options:** `all_options` is JSON in some rows and a Python-literal list in others. Both are parsed as literals
  only. The "(a) " labels are removed.
- **Evidence:** PersonaMem labels no turns. The evidence session is derived: the one holding the most recent preference
  mention, `distance_to_ref_in_blocks` sessions back from the question. The raw distance fields are kept in `meta`.
- **Scoring:** by option log-probability. There is no judge.
