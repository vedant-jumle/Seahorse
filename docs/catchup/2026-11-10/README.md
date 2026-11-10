# Catch-up, 10 Nov 2026

What we ran from 22 Oct to 10 Nov 2026, the results, and what they mean next.

**Page:** https://claude.ai/artifact/ShvfnyfP6U5pdjYKMgUhnf (private; "Seahorse Catch-up")

## Files
| file | what |
|---|---|
| `catchup.html` | the published page (built; contains `data.json` inline) |
| `template.html` | the page source; `/*DATA*/{}` is replaced by the data at build time |
| `extract.py` | reads every plotted number from `results/`, writes `data.json`, builds `catchup.html` |
| `data.json` | the extracted numbers |
| `catchup.md` | the page's text as Markdown (no plots) |

Rebuild from the repo root: `python3 docs/catchup/2026-11-10/extract.py` (standard library only). When Stage B finishes,
re-fetch `results/dream_compare_b_20261105_1915/` and rebuild; the Stage B charts pick up any model with a
`verdict_b.json`.

## Read these to resume
1. Memory notes: `dream-select-plan.md` (the history), `training-free-side-fork.md`, `trained-side-fork.md`
2. `catchup.md` here (fastest single read)
3. `/home/vedantjumle/.claude/plans/replicated-wondering-hoare.md` and `experiments/dream_compare/PREREG.md`
4. `results/dream_compare_a_20261105_1822/summary_a.md` + per-model `report_a.md`
5. `results/dream_compare_b_20261105_1915/<model>/report_b.md` (partial: 1.5B, 7B, Llama; fetched 10 Nov)
6. `results/dream_select_cluster_20261104_0129/summary.md`; `results/dream_subj_20261022_1755/report.md`
7. `docs/dreaming/literature-select.md`, `docs/dreaming/literature-sleep.md`, `docs/trained/README.md`
8. `HANDOVER.md` §1, §8, §10 (stale for dream_select / dream_compare)
9. `docs/dreaming/01-binding.md`, addenda 7–9

## Sources used
- `results/dream_subj_20261022_1755/report.md`, `experiments/dream_subj/PREREG.md`
- `results/dream_select_20261104_0032/report.md`, `experiments/dream_select/PREREG.md`
- `results/dream_select_cluster_20261104_0129/` (`summary.*`, `<model>/report_gen.md`, `verdict_gen.json`, `tf/report.md`)
- `results/dream_compare_a_20261105_1822/` (`summary_a.*`, `<model>/report_a.md`, `verdict_a.json`)
- `results/dream_compare_b_20261105_1915/` (`<model>/report_b.md`, `verdict_b.json`, `jobs.txt`, identity checks)
- Cluster run root `/scratch/vvjumle/seahorse_runs/dream_compare_b_20261105_1915`; logs `/scratch/vvjumle/logs/`
