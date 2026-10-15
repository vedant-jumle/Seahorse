# PrefEval: third-party files (unmodified)

From **PrefEval** (Zhao, Hong, Durmus, Liang, Sun, Wei; "Do LLMs Recognize Your Preferences? Evaluating
Personalized Preference Following in LLMs", ICLR 2025, arXiv:2502.09597), by Amazon Science.

Licence: **Creative Commons Attribution-NonCommercial 4.0 International** (CC BY-NC 4.0), the licence of the
GitHub repository; its full text is in [`LICENSE`](LICENSE). Used here for non-commercial research only. The
files are copied verbatim; nothing in them was changed.

| File | Source |
|---|---|
| `check_acknowledge.txt`, `check_violation.txt`, `check_helpful.txt`, `check_hallucination.txt` | github.com/amazon-science/PrefEval, `error_type/`, commit `50795054b5ff5f418d2b768a331d71e480f93331` (2025-04-21) |
| `LICENSE` | the same repository and commit |
| `prefeval_explicit.jsonl` | Hugging Face dataset `siyanzhao/prefeval_explicit`, revision `1b20e0f944d7665f435e24eb8b76f4d45d83c419`, file `data/train-00000-of-00001.parquet` (sha256 `31185e1241fa495f4585ed03bb7adc94dd4b87407246b83be2ada1c0c94787fe`), converted row by row to JSON lines with a `row` index (0..999) added; all 1000 rows match the repository's `benchmark_dataset/explicit_preference/*.json` (topic, preference, question) |

How PrefEval uses the prompts (`generation_task/llm_based_evaluation_errortypes.py` and
`get_preference_following_accuracy_generation_task.py` at the commit above): each prompt is one user turn with
the system prompt "You are a helpful assistant in evaluating an AI assistant's reponse. You should be fair and
strict and follow the user's instruction", temperature 0, at most 100 new tokens (their evaluator: Claude 3
Sonnet). `{question}`, `{preference}`, `{end_generation}` and `{assistant_restatement}` are replaced by plain
string substitution. The answer is read from the `<answer>` tag: violation = "yes" in it, unhelpful = "no" in
the helpfulness answer, acknowledgement = "yes", hallucination = acknowledged and "yes". **Preference-following
accuracy** = not unhelpful and not (inconsistent, hallucinated-preference violation, or preference-unaware
violation), which reduces to: **helpful and not violating**.
