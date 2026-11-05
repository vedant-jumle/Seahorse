# Model configs (test fixtures)

These are copies of `config.json` for the Apache-2.0 models that the local HF cache may lack. The spec test reads the
HF cache first and falls back to these. All were downloaded 2026-11-05 from huggingface.co:

| file | model | sha256 |
|---|---|---|
| `gemma4_26b_a4b.config.json` | google/gemma-4-26B-A4B-it | `ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3` |
| `granite42_30b.config.json` | ibm-granite/granite-4.2-30b | `82c834ac6bf6a2ecbf9801b49c3fb927008df24ea415c35e931b6091cab21277` |
| `qwen25_7b.config.json` | Qwen/Qwen2.5-7B-Instruct | `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c` |
