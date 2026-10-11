#!/usr/bin/env python
"""dream_bind panel judges: re-judge this run's answers with gpt-oss and Gemma-4 through Ollama, with core_v1's
panel code unchanged (experiments/core_v1/panel_judge.py: the same rubric, filling, parser, judge settings and
resumable output). Only the run layout differs: one model dir, RUN/qwen25_15b (gen/items/*.jsonl.gz and
prep/items.json, written by run.py --stage gen).

  python experiments/dream_bind/panel.py --run RUN --judge gptoss --out RUN/panel --url http://127.0.0.1:PORT \
         [--format schema|json|none] [--limit N] [--workers 16] [--models qwen25_15b]

Output: <out>/<judge>.jsonl (one record per unique prompt; keys "<model>:<answer key>"), read by run.py
--stage analyze, which forms the majority of the original judge + gpt-oss + Gemma-4 (a tie goes to the
original's label).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pj = _load("seahorse_core_v1_panel_judge_db", REPO / "experiments" / "core_v1" / "panel_judge.py")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run root (holds qwen25_15b/)")
    ap.add_argument("--judge", required=True, choices=["gptoss", "gemma4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--format", default="schema", choices=["schema", "json", "none"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--models", nargs="+", default=["qwen25_15b"])
    a = ap.parse_args(argv)
    J = pj.load_prompts()
    tasks = pj.build_tasks(a.run, J, models=tuple(a.models))
    if a.limit:
        tasks = tasks[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not pj.wait_ready(a.url):
        sys.exit(f"Ollama not reachable at {a.url}")
    if tasks:
        try:
            pj.chat(a.url, a.judge, tasks[0]["prompt"], J["fact"]["schema"], pj.SEED)
        except Exception as e:
            print(f"warm-up failed: {e}")
    res = pj.run_tasks(a.url, a.judge, tasks, J, out / f"{a.judge}.jsonl", a.workers, fmt=a.format)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
