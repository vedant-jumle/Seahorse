#!/usr/bin/env python
"""dream_bind panel judges: re-judge this run's answers with gpt-oss and Gemma-4 through Ollama, with core_v1's
panel code unchanged (experiments/core_v1/panel_judge.py: the same rubric, filling, parser, judge settings and
resumable output). Only the run layout differs: one model dir, RUN/qwen25_15b (gen/items/*.jsonl.gz and
prep/items.json, written by run.py --stage gen).

  python experiments/dream_bind/panel.py --run RUN --judge gptoss --out RUN/panel --url http://127.0.0.1:PORT [--relink-only] \
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


def relink(tasks, path, judge):
    """Append alias records for answer keys whose prompt hash is already judged under other keys.

    The output is shared by every model's panel job in a run root, and a judged hash is skipped, so an identical
    (prompt, answer) from a later job (often a short direct answer two models both give) kept only the first
    job's keys and its own answers had no panel label (addendum 9: 464 of Qwen2.5's 6,840). An alias record
    copies the judged labels to the missing keys. Returns the number of keys added."""
    p = Path(path)
    if not p.exists():
        return 0
    first, have = {}, set()
    for line in open(p, encoding="utf-8"):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        have.update(rec.get("keys", []))
        if rec.get("ok") and rec["h"] not in first:
            first[rec["h"]] = rec
    added = 0
    with open(p, "a", encoding="utf-8") as f:
        for t in tasks:
            src = first.get(t["h"])
            miss = [k for k in t["keys"] if k not in have]
            if src and miss:
                f.write(json.dumps({**src, "keys": miss, "alias_of": src["keys"][0]}) + "\n")
                added += len(miss)
    print(f"[{judge}] relink: {added} keys given the labels of an identical, already judged prompt")
    return added


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
    ap.add_argument("--relink-only", action="store_true", help="only add alias records for already judged prompts")
    a = ap.parse_args(argv)
    J = pj.load_prompts()
    tasks = pj.build_tasks(a.run, J, models=tuple(a.models))
    if a.limit:
        tasks = tasks[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    relink(tasks, out / f"{a.judge}.jsonl", a.judge)
    if a.relink_only:
        return
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
