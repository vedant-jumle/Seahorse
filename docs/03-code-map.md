# 3. Code map: what each file does

```
Seahorse/
├── README.md                     project overview + index of these docs
├── environment.yml               the `seahorse` conda env (python 3.11, torch 2.6 cu126, transformers 4.x)
├── pyproject.toml                makes `seahorse` an installable package (src layout); pytest config
├── src/seahorse/                 the library: memory, hooks, sessions, metrics
│   ├── __init__.py
│   ├── memory.py                 FastWeightMemory: the memory operator M (write / read / key)
│   ├── residual.py               model loading + forward hooks: capture() and inject()
│   ├── sessions.py               chat-template sequence builders, write-token alignment
│   └── metrics.py                teacher-forced log-prob metrics, entropy, greedy decoding
├── tests/                        CPU unit tests (run at the start of every cluster job)
│   ├── test_memory.py            delta-rule properties
│   └── test_hooks.py             hook correctness on a tiny random Qwen2
├── experiments/
│   ├── v0/                       first experiment: can the channel carry memory at all?
│   │   ├── run.py
│   │   ├── scenarios.yaml
│   │   └── generic_prompts.txt   ~100 neutral prompts used to estimate the centring mean μ
│   └── v0_1/                     follow-up: specificity, relations, write positions, samples
│       ├── run.py
│       └── scenarios.yaml
├── slurm/                        DelftBlue job scripts
│   ├── setup_delftblue.sh        one-off env build + model download (login node)
│   ├── v0.slurm                  the job that ran v0 (kept as the record)
│   └── run.slurm                 generic job: EXP=<experiment> sbatch slurm/run.slurm
├── results/                      (untracked) outputs pulled back from the cluster
│   ├── v0_408770/
│   └── v0_1_415909/
└── docs/                         this documentation
```

---

## Library: `src/seahorse/`

### `memory.py`: the memory operator
`FastWeightMemory(d, mu, device=None, dtype=float32, eps=1e-6)`, explained in full in
[02-method.md §2.4](02-method.md#24-the-memory-operator-in-detail-srcseahorsememorypy).

| Member | Does |
|---|---|
| `mu` | Centring vector for this layer (μ_ℓ) |
| `M` | The d×d memory matrix, initialised to zero |
| `key(h)` | `normalize(h − mu)` over the last dim: a unit key for any `[..., d]` residual |
| `predict(k)` | `k @ M.T`: the recalled delta M·k, batched over leading dims |
| `read(h, alpha)` | `h + alpha * predict(key(h))`: the steer the inject hook applies |
| `write(delta, h_key, gate=None, eta=1.0)` | Sequential delta-rule writes over T tokens. Returns `(err_norms, delta_norms)`, the per-token surprise against memory |

### `residual.py`: hooking into the model
| Function | Does |
|---|---|
| `load_model(name, device, dtype=float32)` | Loads the HF causal LM and tokenizer, sets eval mode, freezes all parameters |
| `decoder_layers(model)` | Returns `model.model.layers` (the hook targets) |
| `capture(model, layers)` | Context manager. Registers a forward hook on each listed block and yields a dict `layer → [seq, d]` residual after that block (batch 1). Hooks are removed on exit. |
| `inject(model, layer, memory, alpha)` | Context manager. Registers a hook on block `layer` that replaces its output h with `memory.read(h, alpha)` at every position. Removed on exit. |
| `_hidden`, `_replace` | Handle the decoder layer returning either a tuple or a tensor (varies across transformers versions) |

### `sessions.py`: building the conversations
| Function | Does |
|---|---|
| `chat_ids(tok, user_text)` | Token ids for one user turn plus the assistant header, via the model's chat template |
| `text_ids(tok, text)` | Plain token ids (no special tokens), for prefixes, targets and continuations |
| `common_prefix_len(a, b)` / `common_suffix_len(a, b)` | Shared prefix/suffix length of two id sequences. The prefix finds the template head (skipped when computing μ); the suffix finds the write tokens and the template tail. |
| `write_pair(tok, experience, followup)` | Builds the with/without session-1 sequences and returns `(with_ids, without_ids, n)`, where the last `n` tokens are the shared write tokens |
| `ceiling_ids(tok, experience, probe)` | The probe with the experience in the same user turn (upper bound) |

### `metrics.py`: measurement
| Function | Does |
|---|---|
| `cont_logprobs(model, prompt_ids, cont_ids, device)` | Log-softmax `[C, V]` of the next-token distributions that predict a given continuation |
| `kl(lp_p, lp_q)` | Mean over positions of KL(p ‖ q) |
| `seq_logprob(model, prompt_ids, cont_ids, device)` | Total log P(continuation \| prompt) |
| `entropy(logits)` | Row-wise entropy in nats (the confusion gate) |
| `greedy(model, tok, prompt_ids, n, device)` | Greedy continuation ids (repetition penalty forced to 1.0) |

---

## Tests: `tests/`
| File | Checks |
|---|---|
| `test_memory.py` | Exact one-shot recall; read scaling (α=0 is the identity); orthogonal keys don't interfere; repeating a write gives ~0 error (habituation); a contradiction overwrites; a zero gate writes nothing |
| `test_hooks.py` | On a tiny random `Qwen2ForCausalLM` built from a config (no download): capture equals `hidden_states[ℓ+1]`; α=0 and an empty memory give identical logits; injection changes logits and is removed afterwards; layers before ℓ are unaffected |

Run: `pytest -q tests` (CPU is fine).

---

## Experiments: `experiments/`

### `v0/run.py`: can the channel carry memory at all?
Pipeline, in order:
1. **Load** the model (fp32) and compute **μ_ℓ** over `generic_prompts.txt` (skipping the shared template prefix).
2. **Session 1** (`collect_writes`): for each scenario and follow-up, run with/without, capture the residuals, and keep the deltas, keys and entropy gate for every layer.
3. **References** (`reference_probes`, `reference_unrelated`): ceiling greedy continuations and log-probs, baseline log-probs, and baseline/ceiling targets. Also greedy continuations for the unrelated probes.
4. **Sweep:** modes {isolated, combined} × gates {none, entropy} × layers {6, 10, 14, 17, 20, 23, 26} × α {0, 0.5, 1, 2, 4}. For each, build the memory (`build_memory`), check exact recall, and evaluate probes and leakage with the inject hook on.
5. **Samples:** greedy generations (baseline / memory / ceiling) for each scenario's related probe, at L17, α=1.
6. **Write** `results.jsonl` (one row per probe × config), `leakage.jsonl`, `write_stats.jsonl`, `summary.csv` (grouped by mode, gate, layer, α, type, distance) and `config.json`.

Assertions: α=0 must equal baseline, and the last ungated write must be recalled exactly.

### `v0/scenarios.yaml`
6 scenarios (3 dispositions: vegetarian, peanut allergy, lives in Norway; 3 facts: dog's name
Biscuit, sister's name Ines, deep-sea welder). Each has an `experience`, 3 `followups` (2 on
topic, 1 generic), 3 `probes` (exact / paraphrase / related) and a `measure` (fact
prefix+target, or disposition prefix + a/b contrast). Plus 8 `unrelated_probes`.

### `v0_1/run.py`: the follow-up
Same pipeline, extended:
- **Three runs per follow-up** in session 1 (with / without / counter).
- **`select()`** picks the write position (`all` / `boundary` / `topk` / `pooled`) and baseline (`without` / `contrastive`) for each follow-up.
- **New metrics:** fact foils (specificity) and yes/no relation probes.
- **Sweep:** modes × gate {entropy} × positions × baselines × layers {20, 23, 26} × α {1, 2}.
- **Samples:** 6 variants (positions all/boundary/pooled × both baselines) at the best v0 settings (L23 for dispositions, L26 for facts, α=2, entropy gate), for the related probe and the first relation probe.
- **Outputs:** `results.jsonl` (with `kind` = probe | relation), `write_stats.jsonl`, `summary.csv` (grouped by mode, gate, position, baseline, layer, α, type, distance; relation rows have distance = `relation`), `samples.txt`, `config.json`.

### `v0_1/scenarios.yaml`
The v0 scenarios plus `counter` (a counter-experience per scenario), `foils` (2 per fact) and
`relation_probes` (2 per disposition, each with the consistent answer `a` and the
inconsistent `b`).

---

## Cluster: `slurm/` (DelftBlue)
| File | Does |
|---|---|
| `setup_delftblue.sh` | One-off, on the **login node** (compute nodes have no internet). Creates or updates the `seahorse` conda env in `/scratch/$USER/.conda/envs`, runs `pip install -e .`, pre-downloads the model into `HF_HOME=/scratch/$USER/hf_cache`. (In practice the env was built by hand: an empty conda env plus pip installs, which is faster than the solver.) |
| `v0.slurm` | The exact job that ran v0 (kept for reproducibility) |
| `run.slurm` | Generic job: `EXP=v0_1 sbatch slurm/run.slurm`. Loads modules and the env, sets `HF_HUB_OFFLINE=1`, runs `pytest`, then `experiments/$EXP/run.py --out /scratch/$USER/seahorse_runs/${EXP}_<jobid>`. Extra flags via `EXTRA_ARGS=...` |

Partition: `gpu-a100-small` (one 10GB MIG slice of an A100; ≤2 CPUs per task, ≤8000MB per
CPU, 4h max; allocation is usually instant). v0 needs ~8GB of GPU memory in fp32.

**DelftBlue gotchas found along the way:**
- `srun`/`sbatch` need an explicit `--ntasks=1`.
- `compute-p1` caps memory at 3996MB per CPU.
- Compute nodes have **no internet**.
- `~/.condarc` already points envs and pkgs to scratch, and `PIP_CACHE_DIR` is on scratch.

---

## Running things
```bash
# local unit tests (needs torch + transformers)
pytest -q tests

# on DelftBlue, from /scratch/$USER/Seahorse
git pull
EXP=v0_1 sbatch slurm/run.slurm
tail -f /scratch/$USER/logs/seahorse_<jobid>.out
# outputs: /scratch/$USER/seahorse_runs/v0_1_<jobid>/{summary.csv,samples.txt,results.jsonl,write_stats.jsonl,config.json}
```

Runtime: v0 took 21m53s and v0.1 took 25m13s on one MIG slice.
