# Blind audit of the core_v1 judge

Qwen3.5-9B labelled the generated answers in core_v1. These two tools let a human label a blind sample and
check whether the judge-based claims (C2, C3, C5) survive the judge's measured error.

Files live in `results/core_v1_20261007_1820/analysis/` (untracked; never commit them):
`audit_sample.jsonl` (200 rows to label), `audit_key.jsonl` (the judge's labels and the condition/model).

## Workflow

1. **Open `tools/audit/label.html`** in a browser (double-click; no server). Load `audit_sample.jsonl` with the file
   picker or drag-and-drop. Do **not** open `audit_key.jsonl`: the page never loads it and shows no condition, model,
   dose or judge label.
2. **Label and export.** One row at a time; instructions for the rubric (same definitions as
   `experiments/core_v1/judge_prompts.yaml`) are on the page. Keys: `1/2/3` answer the highlighted field and move to
   the next, `Enter` or the right arrow goes to the next row, the left arrow goes back, `N` jumps to the next unlabelled
   row, `T` focuses the notes box. Progress autosaves in the browser (keyed by file name), so a reload resumes after
   loading the same file. Click **Export** to download `audit_labelled.jsonl` (it warns if rows are unlabelled).
3. **Compare.** From the repo root:

   ```
   conda run -n torch python tools/audit/compare.py \
       --labelled audit_labelled.jsonl \
       --key results/core_v1_20261007_1820/analysis/audit_key.jsonl \
       --claims results/core_v1_20261007_1820/analysis/claims_all.json \
       --out audit_report.md
   ```

   Standard library + numpy. `--run` (default `results/core_v1_20261007_1820`) points at the folder with the
   `qwen35_*` score outputs used for the robustness check; `--n-boot` (2000) and `--seed` (7) control the bootstrap.

## What the human labels

| rubric | field | values |
|---|---|---|
| fact | `h_uses_fact`, `h_coherent`, `h_self_claim`, `h_wrong_value` | true / false / "unsure" |
| preference | `h_direction` | "toward" / "away" / "neutral" / "unsure" |
| preference | `h_coherent`, `h_self_claim` | true / false / "unsure" |

`h_lexicon_ok` is not asked (it would need the lexicon label, which is hidden); `compare.py` derives lexicon-vs-human
agreement from the key.

## What the report contains

Per rubric and field: n, agreement, Cohen's kappa with a bootstrap 95% CI, the confusion matrix and the judge-vs-human
rates ("unsure" is excluded from kappa and counted). Breakdowns by model and by condition use the key file only.
Claim robustness: sensitivity/specificity (Rogan-Gladen) for the clean-use label behind C3, and the 3x3 lean-class
confusion matrix (inverted) behind C5 and C2, applied to the per-condition numbers recomputed from the saved
generations and judge outputs. Each claim ends with a plain verdict on whether the judge can be trusted for it.
C1, C4 and C6 do not use the judge and are not covered.

Never commit labels made from synthetic or key-derived data; test runs belong in a temp folder.
