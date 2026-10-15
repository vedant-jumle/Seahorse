#!/usr/bin/env python
"""dream_refs panel judges (phase B): re-judge this run's answers with gpt-oss and Gemma-4 through Ollama, with
core_v1's panel code unchanged (experiments/core_v1/panel_judge.py: the same rubric, filling, parser, judge
settings and resumable output). Only the run layout differs: one model dir, RUN/qwen35_2b.

  python experiments/dream_refs/panel.py --run RUN --judge gptoss --out RUN/panel --url http://127.0.0.1:PORT \
         [--format schema|json|none] [--limit N] [--workers 16]

Output: <out>/<judge>.jsonl (one record per unique prompt; keys "qwen35_2b:<answer key>"), read by analyze.py,
which forms the majority of the original judge + gpt-oss + Gemma-4 (a tie goes to the original's label).
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODELS = ("qwen35_2b",)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pj = _load("seahorse_core_v1_panel_judge", REPO / "experiments" / "core_v1" / "panel_judge.py")


def tasks_for(run):
    """Unique judge prompts over this run's answers (core_v1's builder; every answer is judged)."""
    J = pj.load_prompts()
    return J, pj.build_tasks(run, J, models=MODELS)


def exclusive(path):
    """An exclusive, non-blocking flock on <path>.lock for the life of the process: a second job judging into
    the same file exits at once instead of interleaving appends (which tore a line in experiment 1)."""
    import fcntl
    f = open(str(path) + ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another job is writing {path}; refusing to append concurrently")
    f.write(str(__import__("os").getpid()))
    f.flush()
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="dream_refs run root (holds qwen35_2b/)")
    ap.add_argument("--judge", required=True, choices=["gptoss", "gemma4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--format", default="schema", choices=["schema", "json", "none"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--deadline-min", type=float, default=None, help="stop starting new prompts after N minutes")
    a = ap.parse_args(argv)
    J, tasks = tasks_for(a.run)
    if a.limit:
        tasks = tasks[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = exclusive(out / f"{a.judge}.jsonl")   # never two writers on one output file (held until exit)
    if not pj.wait_ready(a.url):
        sys.exit(f"Ollama not reachable at {a.url}")
    if tasks:
        try:
            pj.chat(a.url, a.judge, tasks[0]["prompt"], J["preference"]["schema"], pj.SEED)
        except Exception as e:
            print(f"warm-up failed: {e}")
    res = pj.run_tasks(a.url, a.judge, tasks, J, out / f"{a.judge}.jsonl", a.workers, fmt=a.format,
                       deadline=time.time() + a.deadline_min * 60 if a.deadline_min else None)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
