#!/usr/bin/env python
"""core_v1 multi-judge robustness check: re-judge the saved core_v1 answers with Ollama judges.

The prompts are the SAME as the original judge's (judge_prompts.yaml, filled exactly as run.judge_text does),
and the output is parsed with the same schema and the same parser (a copy of core.parse_judge, tested equal);
only the judge model changes. Standard library + PyYAML only (no torch), so it runs next to `ollama serve`.

  python panel_judge.py --run <RUN root> --judge gemma4:12b --out <panel dir> [--url http://127.0.0.1:PORT]
         [--subset] [--limit N] [--workers 16] [--smoke]

Output: <out>/<judge-tag>.jsonl, one record per UNIQUE prompt (identical prompts are judged once), appended as
soon as the answer is parsed (a resumed job skips completed prompts):
  {"h": sha1(prompt), "judge": tag, "rubric", "keys": ["qwen35_2b:<key>", ...], "raw", "ok", <fields>,
   "attempts": 1|2, "ms"}
A prompt that fails to parse is retried once (same prompt, a different seed), then recorded with ok=false;
invalid rows are counted in the log and never dropped silently.
"""

import argparse
import gzip
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MODELS = ("qwen35_2b", "qwen35_9b")

# judge tag -> (ollama model, request settings). think: False = reasoning off; gpt-oss only has levels.
JUDGES = {
    "gptoss": {"model": "gpt-oss:latest", "think": "low", "family": "OpenAI", "num_predict": 768},
    "gemma4": {"model": "gemma4:12b", "think": False, "family": "Google", "num_predict": 96},
    "llama31": {"model": "llama3.1:latest", "think": None, "family": "Meta", "num_predict": 96},
    "qwen36": {"model": "qwen3.6:35b-a3b", "think": False, "family": "Qwen (same family as the original judge)",
               "num_predict": 96},
}
SEED = 20261009
NUM_CTX = 4096   # longest prompt ~800 tokens + <= 768 generated (gpt-oss reasoning included)


# ----------------------------------------------------------------------------- prompts (same as run.py)


def render(template, **kw):
    """core.render: replace only the given {keys} (JSON braces in the template stay literal)."""
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def load_prompts(path=None):
    return yaml.safe_load(open(path or HERE / "judge_prompts.yaml"))


def judge_text(J, items, r):
    """run.judge_text: the fact rubric for facts items, else the preference rubric."""
    it = items[r["item"]]
    ans = r["answer"] if r["answer"].strip() else "(empty reply)"
    if it["group"] == "facts":
        return render(J["fact"]["template"], fact=it["experience"], target=it["measure"]["target"].strip(),
                      question=r["prompt"], answer=ans)
    return render(J["preference"]["template"], trait=it["experience"], question=r["prompt"], answer=ans)


def rubric_of(r):
    return "fact" if r["group"] == "facts" else "preference"


def json_schema(spec):
    """judge_prompts.yaml schema -> a JSON schema for Ollama's `format`."""
    props = {}
    for k, v in spec.items():
        props[k] = {"type": "boolean"} if v == "bool" else {"type": "string", "enum": list(v)}
    return {"type": "object", "properties": props, "required": list(spec), "additionalProperties": False}


# ----------------------------------------------------------------------------- parsing (core.parse_judge)

_BOOL = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}


def _coerce(v, spec):
    if spec == "bool":
        if isinstance(v, bool):
            return v
        return _BOOL.get(str(v).strip().strip('"').lower())
    s = str(v).strip().strip('"').lower()
    return s if s in spec else None


def parse_judge(text, schema):
    """Identical to core.parse_judge: first {...} block that json-decodes, else a per-key regex."""
    vals = {}
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in schema):
            vals = {k: _coerce(obj[k], spec) for k, spec in schema.items() if k in obj}
            break
    for k, spec in schema.items():
        if vals.get(k) is None:
            m = re.search(rf'"?{re.escape(k)}"?\s*[:=]\s*"?([A-Za-z]+)', text)
            vals[k] = _coerce(m.group(1), spec) if m else None
    vals["ok"] = all(vals[k] is not None for k in schema)
    return vals


def repair(text):
    """Cheap repair before parsing: drop <think>/<thinking> blocks and code fences."""
    t = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text or "", flags=re.S | re.I)
    t = re.sub(r"```(?:json)?", "", t)
    return t.strip()


def parse_output(text, schema):
    """Parse a judge reply: after repair() (a leaked reasoning block must not be read as the answer), and if
    that fails, the raw text."""
    p = parse_judge(repair(text), schema)
    return p if p["ok"] else (lambda q: q if q["ok"] else p)(parse_judge(text or "", schema))


# ----------------------------------------------------------------------------- data


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rows(run_root, models=MODELS):
    """[(model, items, row)] for every judged answer (sec == 'gen'), in the original file order."""
    out = []
    for m in models:
        d = Path(run_root) / m
        items = json.load(open(d / "prep" / "items.json"))
        for f in sorted((d / "gen" / "items").glob("*.jsonl.gz")):
            for r in read_gz(f):
                if r["sec"] == "gen":
                    out.append((m, items, r))
    return out


def needed(r, claims_cfg):
    """True if the pre-registered C2 / C3 / C5 verdicts use this answer (seed set 1 and one of their conditions).
    claims_cfg: {"c5_alpha": {"opposite", "centroid", "without"}, "c3_doses": [...], "ctrl_alpha": 2}."""
    if r["seedset"] != 1:
        return False
    c, g = r["cond"], r["group"]
    a = claims_cfg["ctrl_alpha"]
    if g == "leanings":      # C2 (and its ctx-rule flags use ctx/nomem)
        return c in ("nomem", "ctx", "mem@1", "mem@2", f"rand@{a:g}", f"swap@{a:g}", f"placebo@{a:g}")
    if g == "facts":         # C3
        return c in ("nomem", "ctx") or c in [f"mem@{d:g}" for d in claims_cfg["c3_doses"]]
    if g == "dislikes":      # C5
        al = claims_cfg["c5_alpha"]
        return c in ("nomem", "ctx", f"mem@{al['opposite']:g}", f"centroid@{al['centroid']:g}",
                     f"without@{al['without']:g}")
    return False


DEFAULT_CLAIMS_CFG = {"c5_alpha": {"opposite": 1, "centroid": 1, "without": 1}, "c3_doses": [0.5, 1, 2, 3],
                      "ctrl_alpha": 2}


def phash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_tasks(run_root, J, subset=False, claims_cfg=None, models=MODELS):
    """Unique prompts to judge: [{"h", "rubric", "prompt", "keys", "pri"}] with the pre-registered subset
    first (pri 0), then the rest (pri 1). With subset=True only pri 0 is returned."""
    claims_cfg = claims_cfg or DEFAULT_CLAIMS_CFG
    by = {}
    for m, items, r in load_rows(run_root, models):
        text = judge_text(J, items, r)
        h = phash(text)
        t = by.setdefault(h, {"h": h, "rubric": rubric_of(r), "prompt": text, "keys": [], "pri": 1})
        t["keys"].append(f"{m}:{r['key']}")
        if needed(r, claims_cfg):
            t["pri"] = 0
    tasks = sorted(by.values(), key=lambda t: (t["pri"], t["h"]))
    return [t for t in tasks if t["pri"] == 0] if subset else tasks


def read_done(path):
    """Hashes already in the output file (a torn last line from a killed job is ignored)."""
    done = set()
    p = Path(path)
    if p.exists():
        for line in open(p, encoding="utf-8"):
            try:
                done.add(json.loads(line)["h"])
            except (ValueError, KeyError):
                continue
    return done


# ----------------------------------------------------------------------------- Ollama client


def chat(url, tag, prompt, schema, seed, num_predict=None, fmt="schema", timeout=900):
    cfg = JUDGES[tag]
    body = {"model": cfg["model"], "stream": False, "keep_alive": "2h",
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": seed, "num_ctx": NUM_CTX,
                        "num_predict": num_predict or cfg["num_predict"]}}
    if cfg["think"] is not None:
        body["think"] = cfg["think"]
    if fmt == "schema":
        body["format"] = json_schema(schema)
    elif fmt == "json":
        body["format"] = "json"
    req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return (out.get("message") or {}).get("content") or "", out


def judge_one(url, tag, task, schema, post=chat, fmt="schema"):
    """One prompt: query, parse, retry ONCE (different seed, no JSON constraint if the first try had one and
    was rejected or empty). Returns the output record. fmt = the first try's output constraint ("schema" = the
    rubric's JSON schema, "json" = any JSON, "none"); the retry uses the schema if the first try had none, else none."""
    t0 = time.time()
    raw, p, attempts, err, full = "", None, 0, None, {}
    for attempt in range(2):
        attempts += 1
        try:
            raw, full = post(url, tag, task["prompt"], schema, SEED + attempt, fmt=(fmt if attempt == 0 else ('schema' if fmt == 'none' else 'none')))
            err = None
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
            raw, err = "", f"{type(e).__name__}: {e}"[:200]
        p = parse_output(raw, schema)
        if p["ok"]:
            break
    rec = {"h": task["h"], "judge": tag, "rubric": task["rubric"], "keys": task["keys"], "raw": raw[:600], **p,
           "fmt": fmt, "attempts": attempts, "ms": int((time.time() - t0) * 1000),
           "tok": [full.get("prompt_eval_count"), full.get("eval_count")]}
    if err:
        rec["err"] = err
    return rec


def _end_with_newline(path):
    """A killed job can leave a torn last line; make sure the next append starts on a fresh line."""
    p = Path(path)
    if p.exists() and p.stat().st_size:
        with open(p, "rb+") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                f.write(b"\n")


def run_tasks(url, tag, tasks, J, out_path, workers=16, post=chat, log=print, every=200, fmt="schema"):
    """Judge `tasks` not yet in out_path with `workers` concurrent requests; append each record at once."""
    done = read_done(out_path)
    todo = [t for t in tasks if t["h"] not in done]
    log(f"[{tag}] {len(tasks)} unique prompts, {len(done & {t['h'] for t in tasks})} already done, {len(todo)} to do")
    schemas = {"fact": J["fact"]["schema"], "preference": J["preference"]["schema"]}
    lock = threading.Lock()
    n = n_bad = 0
    t0 = time.time()
    _end_with_newline(out_path)
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(judge_one, url, tag, t, schemas[t["rubric"]], post, fmt) for t in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n += 1
                n_bad += not rec["ok"]
                if n % every == 0 or n == len(todo):
                    el = time.time() - t0
                    log(f"[{tag}] {n}/{len(todo)} done | {n / el:.2f} prompts/s | invalid {n_bad} ({n_bad / n:.1%}) | "
                        f"eta {(len(todo) - n) / max(n / el, 1e-9) / 60:.1f} min")
    el = time.time() - t0
    return {"judge": tag, "done_now": n, "invalid_now": n_bad, "seconds": el, "per_s": n / el if el > 0 else 0.0}


def wait_ready(url, tries=240, sleep=2.0):
    for _ in range(tries):
        try:
            urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=5).read()
            return True
        except Exception:
            time.sleep(sleep)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="core_v1 run root (holds qwen35_2b/, qwen35_9b/)")
    ap.add_argument("--judge", required=True, choices=sorted(JUDGES))
    ap.add_argument("--out", required=True, help="panel output dir")
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--subset", action="store_true", help="only answers used by the pre-registered C2/C3/C5")
    ap.add_argument("--limit", type=int, default=None, help="first N unique prompts of the priority order")
    ap.add_argument("--stratified", action="store_true", help="with --limit: spread N over both rubrics evenly")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--format", default="schema", choices=["schema", "json", "none"], help="output constraint of the first try")
    ap.add_argument("--file", default=None, help="output file name (default <tag>.jsonl)")
    a = ap.parse_args()
    J = load_prompts()
    tasks = build_tasks(a.run, J, subset=a.subset)
    if a.limit:
        if a.stratified:
            rng = __import__("random").Random(0)
            fact = [t for t in tasks if t["rubric"] == "fact"]
            pref = [t for t in tasks if t["rubric"] != "fact"]
            rng.shuffle(fact), rng.shuffle(pref)
            tasks = fact[:a.limit // 2] + pref[:a.limit - a.limit // 2]
        else:
            tasks = tasks[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not wait_ready(a.url):
        sys.exit(f"Ollama not reachable at {a.url}")
    if tasks:   # load the model before the clock starts (not recorded)
        try:
            chat(a.url, a.judge, tasks[0]["prompt"], J["fact" if tasks[0]["rubric"] == "fact" else "preference"]["schema"], SEED)
        except Exception as e:
            print(f"warm-up failed: {e}")
    res = run_tasks(a.url, a.judge, tasks, J, out / (a.file or f"{a.judge}.jsonl"), a.workers, fmt=a.format)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
