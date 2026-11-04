#!/usr/bin/env python
"""dream_select panel judges (PREREG addendum 1): re-judge one model's generated answers with gpt-oss or Gemma-4 through
Ollama, with LongMemEval's own correctness prompt (analyze_gen.lme_prompt) and its yes/no parse (analyze_gen.parse_yesno).
core_v1's panel client is reused for the Ollama calls (panel_judge.chat, wait_ready, read_done, the torn-line guard);
dream_bind's relink (a82244b) for already judged prompts. Standard library + PyYAML + numpy only (no torch).

  python experiments/dream_select/panel.py --model-dir RUN/<model> --judge gptoss --url http://127.0.0.1:PORT \
         [--limit N] [--workers 16]

Output: <model dir>/panel/<judge>.jsonl, one record per UNIQUE prompt of this model's answers (keys = the answer keys),
appended as soon as it is parsed (a resumed job skips completed prompts):
  {"h", "judge", "keys", "raw", "label", "ok", "official", "attempts", "ms"}
An unparsed reply (neither "yes" nor "no") is retried once with another seed, then recorded with ok=false.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pj = _load("dream_select_core_v1_panel_judge", REPO / "experiments" / "core_v1" / "panel_judge.py")
dbp = _load("dream_select_db_panel", REPO / "experiments" / "dream_bind" / "panel.py")
AG = _load("dream_select_analyze_gen_panel", HERE / "analyze_gen.py")

# tokens per reply: gpt-oss's reasoning counts against num_predict (core_v1's 768); Gemma-4 thinks off and answers in a
# word (LongMemEval's max_tokens is 10)
NUM_PREDICT = {"gptoss": 768, "gemma4": 10}
SEED = pj.SEED


def phash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_tasks(model_dir, data=None):
    """Unique prompts of one model dir's answers: [{"h", "prompt", "keys"}] in a fixed order (by hash)."""
    data = data or {x["question_id"]: x for x in json.load(open(AG.DATA))["items"]}
    G, _ = AG.read_gen(model_dir)
    by = {}
    for r in G:
        it = data[r["item"]]
        text = AG.lme_prompt(it["question"], it["answer"], r["answer"])
        h = phash(text)
        by.setdefault(h, {"h": h, "prompt": text, "keys": []})["keys"].append(r["key"])
    return sorted(by.values(), key=lambda t: t["h"])


def judge_one(url, tag, task, post=pj.chat):
    """One prompt: query (temperature 0, no output constraint), parse yes/no; retry ONCE with another seed if unparsed."""
    t0 = time.time()
    raw, p, attempts, err, full = "", None, 0, None, {}
    for attempt in range(2):
        attempts += 1
        try:
            raw, full = post(url, tag, task["prompt"], None, SEED + attempt, num_predict=NUM_PREDICT[tag], fmt="none")
            err = None
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
            raw, err = "", f"{type(e).__name__}: {e}"[:200]
        p = AG.parse_yesno(raw)
        if p["ok"]:
            break
    rec = {"h": task["h"], "judge": tag, "keys": task["keys"], "raw": (raw or "")[:600], **p, "attempts": attempts,
           "ms": int((time.time() - t0) * 1000), "tok": [full.get("prompt_eval_count"), full.get("eval_count")]}
    if err:
        rec["err"] = err
    return rec


def run_tasks(url, tag, tasks, out_path, workers=16, post=pj.chat, log=print, every=200):
    """Judge the tasks not yet in out_path with `workers` concurrent requests; append each record at once."""
    done = pj.read_done(out_path)
    todo = [t for t in tasks if t["h"] not in done]
    log(f"[{tag}] {len(tasks)} unique prompts, {len(done & {t['h'] for t in tasks})} already done, {len(todo)} to do")
    lock = threading.Lock()
    n = n_bad = 0
    t0 = time.time()
    pj._end_with_newline(out_path)
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(judge_one, url, tag, t, post) for t in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n += 1
                n_bad += not rec["ok"]
                if n % every == 0 or n == len(todo):
                    el = time.time() - t0
                    log(f"[{tag}] {n}/{len(todo)} done | {n / el:.2f} prompts/s | unparsed {n_bad} ({n_bad / n:.1%}) | "
                        f"eta {(len(todo) - n) / max(n / el, 1e-9) / 60:.1f} min")
    el = time.time() - t0
    res = {"judge": tag, "done_now": n, "unparsed_now": n_bad, "seconds": el, "per_s": n / el if el > 0 else 0.0,
           "num_predict": NUM_PREDICT[tag], "model": pj.JUDGES[tag]["model"], "think": pj.JUDGES[tag]["think"]}
    json.dump(res, open(str(out_path) + ".stats.json", "w"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="RUN/<model> (holds gen/items/*.jsonl.gz)")
    ap.add_argument("--judge", required=True, choices=sorted(NUM_PREDICT))
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int, default=None, help="the first N unique prompts (smoke)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    tasks = build_tasks(a.model_dir)
    if a.limit:
        tasks = tasks[:a.limit]
    out = Path(a.model_dir) / "panel"
    out.mkdir(parents=True, exist_ok=True)
    dbp.relink(tasks, out / f"{a.judge}.jsonl", a.judge)
    if not pj.wait_ready(a.url):
        sys.exit(f"Ollama not reachable at {a.url}")
    if tasks:   # load the model before the clock starts (not recorded)
        try:
            pj.chat(a.url, a.judge, tasks[0]["prompt"], None, SEED, num_predict=NUM_PREDICT[a.judge], fmt="none")
        except Exception as e:
            print(f"warm-up failed: {e}")
    res = run_tasks(a.url, a.judge, tasks, out / f"{a.judge}.jsonl", a.workers)
    print(json.dumps(res))
    bad = res["unparsed_now"] / max(res["done_now"], 1)
    if res["done_now"] and bad > 0.2:
        sys.exit(f"[{a.judge}] {bad:.0%} of the replies unparsed: check the judge output")


if __name__ == "__main__":
    main()
