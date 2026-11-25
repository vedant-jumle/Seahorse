# KVzip and Seahorse Slices (explainer page)

Published (private) artifact: https://claude.ai/artifact/KxVgeK9pxWxZjJbvGZXsTf

What the page covers: how KVzip (Kim et al., arXiv 2505.23416, NeurIPS 2025) scores and keeps cache entries (its three
forms: global budget, per head, shared positions), how Seahorse slices and our selectors work, a side-by-side table and
a worked LongMemEval example on Llama-3.2-3B (item `4fd1909e`), then the results in run order: automatic selection,
selector comparison, head analysis (incl. the exploratory KVzip follow-up), memory-system comparison, and the open 7B
point. Plain names are used on the page; the glossary at the top maps them to the code names once.

Files

- `index.html`: the page (published as is). Its numbers live in the `<script id="data">` block between the
  `DATA-BEGIN` / `DATA-END` markers; the page's script draws every chart and fills every computed sentence from it.
- `extract.py`: pulls every plotted number from the result files and rewrites that data block (and `data.json`).
- `data.json`: the same data, indented, for reading.

Regenerate and republish

    conda run -n torch python docs/explainers/kvzip-vs-slices/extract.py   # needs transformers + the Llama-3.2-3B-Instruct tokenizer

then publish `index.html` to the URL above.

Sources read by `extract.py`

- `results/dream_select_cluster_20261104_0129/<model>/{tf/report.md, report_gen.md}`
- `results/dream_compare_a_20261105_1822/<model>/report_a.md`; example turn and answers from `llama32_3b/{tf,gen}/items/`
- `results/dream_heads_20261111/<model>/{report.md, heads.json, items/}` and `experiments/dream_heads/RESULTS.md` §2
- `results/dream_compare_b_20261105_1915/<model>/report_b.md`

A few numbers are written into the prose by hand (the same values as in those files): the automatic-selection
answer-token figures, the selector-comparison headline numbers, and the memory-system comparison's 7B LongMemEval figures.
