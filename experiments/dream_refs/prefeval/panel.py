#!/usr/bin/env python
"""PrefEval panel judges (phase B): gpt-oss and Gemma-4 through Ollama re-judge every answer with PrefEval's
violation and helpfulness rubrics (upstream/check_violation.txt, check_helpful.txt; their system prompt), which
are the two that decide preference-following accuracy. core_v1's panel settings (experiments/core_v1/
panel_judge.py: the models, temperature 0, seed, context, reasoning level, retry once) with PrefEval's output
format: no JSON constraint, the <answer> tag is parsed (bench.parse_answer_tag).

  python experiments/dream_refs/prefeval/panel.py --run RUN --judge gptoss --out RUN/panel --url http://127.0.0.1:PORT

Order: the primary cells first (nomem, ctx and the alpha = 1 cells), then the alpha 0.5 / 2 cells, so a job that
runs out of time has finished the cells the decision rules use. Output: <out>/<judge>.jsonl, one record per unique
prompt {"h", "judge", "rubric", "keys", "raw", "answer": true/false/null, "ok", "attempts", "ms"}; resumable.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pj = _load("seahorse_core_v1_panel_judge", REPO / "experiments" / "core_v1" / "panel_judge.py")
bench = _load("seahorse_prefeval_bench", HERE / "bench.py")
PE_SYSTEM = ("You are a helpful assistant in evaluating an AI assistant's reponse. You should be fair and strict and "
             "follow the user's instruction")
RUBRICS = {"violate": "check_violation.txt", "helpful": "check_helpful.txt"}
NUM_PREDICT = {"gptoss": 1024, "gemma4": 160}


def fill(T, rubric, preference, question, answer):
    return T[rubric].replace("{preference}", preference).replace("{question}", question).replace("{end_generation}", answer)


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_tasks(run, primary_alpha=1):
    d = Path(run) / "qwen35_2b"
    items = json.load(open(d / "prep" / "items.json"))
    T = {k: (HERE / "upstream" / f).read_text() for k, f in RUBRICS.items()}
    by = {}
    for f in sorted((d / "gen" / "items").glob("*.jsonl.gz")):
        for r in read_gz(f):
            if r["sec"] != "gen":
                continue
            ans = r["answer"] if r["answer"].strip() else "(empty reply)"
            pri = 0 if (r["cond"] in ("nomem", "ctx") or r["cond"].endswith(f"@{primary_alpha:g}")) else 1
            for rub in RUBRICS:
                text = fill(T, rub, items[r["item"]]["experience"], r["prompt"], ans)
                h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                t = by.setdefault(h, {"h": h, "rubric": rub, "prompt": text, "keys": [], "pri": pri})
                t["keys"].append(r["key"])
                t["pri"] = min(t["pri"], pri)
    return sorted(by.values(), key=lambda t: (t["pri"], t["h"]))


def chat(url, tag, prompt, seed, num_predict, timeout=900):
    cfg = pj.JUDGES[tag]
    body = {"model": cfg["model"], "stream": False, "keep_alive": "2h",
            "messages": [{"role": "system", "content": PE_SYSTEM}, {"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": seed, "num_ctx": pj.NUM_CTX, "num_predict": num_predict}}
    if cfg["think"] is not None:
        body["think"] = cfg["think"]
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return (out.get("message") or {}).get("content") or "", out


def judge_one(url, tag, task, post=chat):
    t0 = time.time()
    raw, ans, attempts, err, full = "", None, 0, None, {}
    for attempt in range(2):        # retry once with another seed, as core_v1's panel
        attempts += 1
        try:
            raw, full = post(url, tag, task["prompt"], pj.SEED + attempt, NUM_PREDICT.get(tag, 160))
            err = None
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
            raw, err = "", f"{type(e).__name__}: {e}"[:200]
        ans = bench.parse_answer_tag(raw)
        if ans is not None:
            break
    rec = {"h": task["h"], "judge": tag, "rubric": task["rubric"], "keys": task["keys"], "raw": raw[:600],
           "answer": ans, "ok": ans is not None, "attempts": attempts, "ms": int((time.time() - t0) * 1000),
           "tok": [full.get("prompt_eval_count"), full.get("eval_count")]}
    if err:
        rec["err"] = err
    return rec


def run_tasks(url, tag, tasks, out_path, workers=16, post=chat, log=print, every=500, deadline=None):
    done = pj.read_done(out_path)
    todo = [t for t in tasks if t["h"] not in done]
    log(f"[{tag}] {len(tasks)} unique prompts, {len(todo)} to do")
    lock = threading.Lock()
    n = n_bad = n_skip = 0
    t0 = time.time()
    pj._end_with_newline(out_path)

    def work(t):
        if deadline is not None and time.time() > deadline:
            return None
        return judge_one(url, tag, t, post)
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(work, t) for t in todo]):
            rec = fut.result()
            if rec is None:
                n_skip += 1
                continue
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n += 1
                n_bad += not rec["ok"]
                if n % every == 0 or n == len(todo):
                    el = time.time() - t0
                    log(f"[{tag}] {n}/{len(todo)} | {n / el:.2f} prompts/s | invalid {n_bad} | "
                        f"eta {(len(todo) - n) / max(n / el, 1e-9) / 60:.1f} min")
    el = time.time() - t0
    res = {"judge": tag, "done_now": n, "invalid_now": n_bad, "skipped_deadline": n_skip, "seconds": el,
           "per_s": n / el if el > 0 else 0.0}
    json.dump(res, open(str(out_path) + ".stats.json", "w"))
    return res


def exclusive(path):
    """An exclusive, non-blocking flock on <path>.lock for the life of the process: a second job judging into
    the same file exits at once instead of interleaving appends (which tore a line in experiment 1)."""
    import fcntl
    import os
    f = open(str(path) + ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f"another job is writing {path}; refusing to append concurrently")
    f.write(str(os.getpid()))
    f.flush()
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge", required=True, choices=["gptoss", "gemma4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--deadline-min", type=float, default=None)
    a = ap.parse_args(argv)
    tasks = build_tasks(a.run)
    if a.limit:
        tasks = tasks[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = exclusive(out / f"{a.judge}.jsonl")   # never two writers on one output file (held until exit)
    if not pj.wait_ready(a.url):
        sys.exit(f"Ollama not reachable at {a.url}")
    if tasks:
        try:
            chat(a.url, a.judge, tasks[0]["prompt"], pj.SEED, 16)
        except Exception as e:
            print(f"warm-up failed: {e}")
    res = run_tasks(a.url, a.judge, tasks, out / f"{a.judge}.jsonl", a.workers,
                    deadline=time.time() + a.deadline_min * 60 if a.deadline_min else None)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
