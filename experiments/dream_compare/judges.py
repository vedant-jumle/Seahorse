#!/usr/bin/env python
"""dream_compare judges (Stage 0): each benchmark's official scoring prompt, reproduced verbatim and pinned by hash in
test_dream_compare.py, its parser, and a panel runner on dream_select's Ollama machinery. Standard library only.

  LongMemEval  get_anscheck_prompt of src/evaluation/evaluate_qa.py (github.com/xiaowu0162/LongMemEval; file sha256
               ecce9c4c..., identical at d6dc8b5 (dream_select's pin, 2026-11-04) and at 9e0b455 (fetched 2026-11-05)):
               5 templates (single-session-user / -assistant / multi-session share one; temporal-reasoning;
               knowledge-update; single-session-preference; abstention for "_abs" ids), str.format(question, answer,
               response), max_tokens 10, temperature 0, label = "yes" in reply.lower() (dream_select's parse_yesno)
  LoCoMo J     Mem0's LLM-judge prompt ACCURACY_PROMPT (Chhikara et al. 2025, "Mem0", arXiv:2504.19413; the "J" score):
               mem0ai/mem0 evaluation/metrics/llm_judge.py at aae5989 (2025-07-08, the last commit before the
               evaluation folder moved to mem0ai/memory-benchmarks), file sha256 ac2e242f...; .format(question=,
               gold_answer=, generated_answer=), temperature 0, JSON {"label": "CORRECT" | "WRONG"}; category 5 is
               skipped there (our adversarial items are scored by the abstention rule below instead)
  LoCoMo F1    task_eval/evaluation.py of snap-research/locomo at 3eb6f2c (sha256 8e3be5d5...): normalize_answer,
               f1_score (Porter-stemmed token F1; code 3 answers cut at ';'), f1 (code 1: comma-split sub-answers),
               code 5: 1 if "no information available" or "not mentioned" is in the lower-cased output
  BEAM         unified_llm_judge_base_prompt of src/prompts.py (github.com/mohammadtavakoli78/BEAM at b2da22e, file
               sha256 81fe51ab...), filled per rubric item with str.replace("<rubric_item>", ...) and
               ("<llm_response>", ...) as compute_metrics.py does; JSON {"score": 1.0 | 0.5 | 0.0, "reason"}; the
               item's score = the mean over its rubric items (compute_metrics.evaluate_*). Event ordering's extra
               Kendall-tau score (an LLM alignment step) is not reproduced: its llm_judge_score is
  PersonaMem   no judge: the option with the highest log-probability (sum over its tokens; the mean per token is
               reported beside), correct iff it is the labelled option

Panel: the original judge (Qwen3.5-9B, dream_select's stage_judge) + gpt-oss + Gemma-4 (core_v1 panel_judge.chat via
slurm/ollama_local.sh). Primary label = the majority, a tie goes to the original judge (dream_bind bind.majority, as
dream_select). For a Gemma subject model, panel_label(..., subject_family="Google") also gives the majority without the
same-family judge (Gemma-4): the original + gpt-oss, a tie going to the original.

  python experiments/dream_compare/judges.py --tasks TASKS.jsonl --judge gptoss --url http://127.0.0.1:PORT \
         --out OUT.jsonl [--limit N] [--workers 16]
TASKS.jsonl: {"h", "kind" (lme | locomo_j | beam), "prompt", "keys"}; output one record per unique prompt (as
dream_select's panel.py).
"""

import argparse
import hashlib
import importlib.util
import json
import re
import string
import sys
import threading
import time
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AG = _load("dream_compare_ds_analyze_gen", REPO / "experiments" / "dream_select" / "analyze_gen.py")
PORTER = _load("dream_compare_porter", HERE / "porter_nltk.py").PorterStemmer()

# ----------------------------------------------------------------------------------------------- LongMemEval

LME_TEMPLATES = {
    'default': 'I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.',
    'temporal-reasoning': "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    'knowledge-update': 'I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.',
    'single-session-preference': "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    'abstention': 'I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.',
}
LME_TASK = {"single-session-user": "default", "single-session-assistant": "default", "multi-session": "default",
            "temporal-reasoning": "temporal-reasoning", "knowledge-update": "knowledge-update",
            "single-session-preference": "single-session-preference"}
LME_SOURCE = {"repo": "xiaowu0162/LongMemEval", "file": "src/evaluation/evaluate_qa.py",
              "commits": ["d6dc8b50a2d9ac0c99485ea28fa5755c62414c34", "9e0b455f4ef0e2ab8f2e582289761153549043fc"],
              "file_sha256": "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
              "function": "get_anscheck_prompt", "max_tokens": 10, "temperature": 0, "label": "'yes' in reply.lower()"}


def lme_prompt(task, question, answer, response, abstention=False):
    """LongMemEval's correctness prompt for a question type (abstention: the unanswerable-question prompt, whose
    "answer" is the item's explanation); an empty response is given as "(empty reply)" (dream_select)."""
    resp = response if (response or "").strip() else "(empty reply)"
    t = LME_TEMPLATES["abstention" if abstention else LME_TASK[task]]
    return t.format(question, answer, resp)


parse_yesno = AG.parse_yesno

# ----------------------------------------------------------------------------------------------- LoCoMo

J_PROMPT = '\nYour task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:\n    (1) a question (posed by one user to another user), \n    (2) a ’gold’ (ground truth) answer, \n    (3) a generated answer\nwhich you will score as CORRECT/WRONG.\n\nThe point of the question is to ask about something one user should know about the other user based on their prior conversations.\nThe gold answer will usually be a concise and short answer that includes the referenced topic, for example:\nQuestion: Do you remember what I got the last time I went to Hawaii?\nGold answer: A shell necklace\nThe generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. \n\nFor time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it\'s the same date.\n\nNow it\'s time for the real question:\nQuestion: {question}\nGold answer: {gold_answer}\nGenerated answer: {generated_answer}\n\nFirst, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. \nDo NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n\nJust return the label CORRECT or WRONG in a json format with the key as "label".\n'
J_SOURCE = {"repo": "mem0ai/mem0", "file": "evaluation/metrics/llm_judge.py",
            "commit": "aae5989e78a6188b3b047c104d960c9ad0927e75",
            "file_sha256": "ac2e242f6ea0b817ba1b229ff9e5271f6846b7b8a9aefa8a5a5a81ea58df3b3c",
            "paper": "Chhikara et al. 2025, Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory, "
                     "arXiv:2504.19413", "temperature": 0, "response_format": "json_object"}
_J_WORD = re.compile(r"\b(CORRECT|WRONG)\b")


def j_prompt(question, gold, generated):
    gen = generated if (generated or "").strip() else "(empty reply)"
    return J_PROMPT.format(question=question, gold_answer=gold, generated_answer=gen)


def parse_j(text):
    """Mem0's label: the "label" field of the reply's JSON == "CORRECT". Fallback when no JSON parses: exactly one of
    the words CORRECT / WRONG in the reply (its prompt forbids both). {"label", "ok", "via"}."""
    t = AG.repair(text)
    for m in re.finditer(r"\{[^{}]*\}", t, re.S):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        lab = str(d.get("label", "")).strip().upper() if isinstance(d, dict) else ""
        if lab in ("CORRECT", "WRONG"):
            return {"label": lab == "CORRECT", "ok": True, "via": "json"}
    words = set(_J_WORD.findall(t.upper()))
    if len(words) == 1:
        return {"label": words == {"CORRECT"}, "ok": True, "via": "word"}
    return {"label": None, "ok": False, "via": None}


def normalize_answer(s):
    """LoCoMo evaluation.normalize_answer (commas removed; lower, punctuation, articles + 'and', whitespace)."""
    s = s.replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction, ground_truth):
    """LoCoMo evaluation.f1_score: token F1 over Porter-stemmed normalised tokens."""
    p = [PORTER.stem(w) for w in normalize_answer(prediction).split()]
    g = [PORTER.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0
    pr, rc = n / len(p), n / len(g)
    return (2 * pr * rc) / (pr + rc)


def f1_multi(prediction, ground_truth):
    """LoCoMo evaluation.f1 (multi-hop): mean over comma-split gold parts of the best F1 against the comma-split
    prediction parts."""
    ps = [x.strip() for x in prediction.split(",")]
    gs = [x.strip() for x in ground_truth.split(",")]
    vals = [max(f1_score(p, g) for p in ps) for g in gs]
    return sum(vals) / len(vals)


def locomo_f1(code, prediction, answer):
    """The official per-question score of eval_question_answering (code = the category code 1-5)."""
    answer = str(answer)
    if code == 3:
        answer = answer.split(";")[0].strip()
    if code in (2, 3, 4):
        return float(f1_score(prediction, answer))
    if code == 1:
        return float(f1_multi(prediction, answer))
    if code == 5:
        o = prediction.lower()
        return 1.0 if ("no information available" in o or "not mentioned" in o) else 0.0
    raise ValueError(code)


# ----------------------------------------------------------------------------------------------- BEAM

BEAM_PROMPT = '\nYou are an expert evaluator tasked with judging whether the LLM\'s response demonstrates compliance with the specified RUBRIC CRITERION.\n\n## EVALUATION INPUTS\n- RUBRIC CRITERION (what to check): <rubric_item>\n- RESPONSE TO EVALUATE: <llm_response>\n\n## EVALUATION RUBRIC:\nThe rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. \n\n**IMPORTANT**: Pay careful attention to whether the rubric specifies:\n- **Positive requirements** (things the response SHOULD include/do)\n- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")\n\n## RESPONSIVENESS REQUIREMENT\nA compliant response must be **on-topic** and attempt to answer it.\n- If the response does not address the QUESTION, score **0.0** and stop.\n- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.\n\n## SEMANTIC TOLERANCE RULES:\nJudge by meaning, not exact wording.\n- Accept **paraphrases** and **synonyms** that preserve intent.\n- **Case/punctuation/whitespace** differences must be ignored.\n- **Numbers/currencies/dates** may appear in equivalent forms (e.g., “$68,000”, “68k”, “68,000 USD”, or “sixty-eight thousand dollars”). Treat them as equal when numerically equivalent.\n- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.\n\n## STYLE NEUTRALITY (prevents style contamination):\nIgnore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., “itemized list”, “no citations”, “one sentence”).\n- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.\n- Only evaluate format when the rubric **explicitly** mandates it.\n\n## SCORING SCALE:\n- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.\n  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).\n  - Negative: prohibited element **absent** AND response is **responsive**.\n  \n- **0.5 (Partial Compliance)**: Partially complies.\n  - Positive: element present but minor inaccuracies/incomplete execution.\n  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.\n  \n- **0.0 (No Compliance)**: Fails to comply.\n  - Positive: required element missing or incorrect.\n  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.\n\n## EVALUATION INSTRUCTIONS:\n1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).\n\n2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:\n   - **All elements** must be present for full compliance (1.0)\n   - **Some elements** present indicates partial compliance (0.5)\n   - **No elements** present indicates no compliance (0.0)\n   \n3. **Check Compliance**: \n   - For positive requirements: Look for the presence and quality of the required element\n   - For negative constraints: Look for the absence of the prohibited element\n\n4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.\n\n5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.\n\n## OUTPUT FORMAT:\nReturn your evaluation in JSON format with two fields:\n\n{\n   "score": [your score: 1.0, 0.5, or 0.0],\n   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"\n}\n\nNOTE: ONLY output the json object, without any explanation before or after that\n'
BEAM_SOURCE = {"repo": "mohammadtavakoli78/BEAM", "file": "src/prompts.py", "name": "unified_llm_judge_base_prompt",
               "commit": "b2da22eac88bb0874c64665f13457eb99835774a",
               "file_sha256": "81fe51ab2bcd7e8ad218b3a9a969a340c38d31f90ac2d93fa78a3bb890ba157b",
               "scorer": "src/evaluation/compute_metrics.py (sha256 e72bf25d...): mean over rubric items",
               "paper": "Tavakoli et al. 2025, arXiv:2510.27246"}


def beam_prompt(rubric_item, response):
    resp = response if (response or "").strip() else "(empty reply)"
    return BEAM_PROMPT.replace("<rubric_item>", rubric_item).replace("<llm_response>", resp)


_SCORE = re.compile(r'"score"\s*:\s*\[?\s*"?([01](?:\.\d+)?|0?\.5)"?')


def parse_beam(text):
    """BEAM's parse_json_response (fenced or bare JSON object) -> score in {0, 0.5, 1}; the json_repair fallback of
    the original is replaced by a regex on the "score" field. {"score", "ok", "via"}."""
    t = AG.repair(text)
    cands = [t] + [m.group(1) for m in re.finditer(r"(\{.*?\}|\[.*?\])", t, re.S)]
    for c in cands:
        try:
            d = json.loads(c)
        except ValueError:
            continue
        if isinstance(d, dict) and "score" in d:
            try:
                v = float(d["score"][0] if isinstance(d["score"], list) else d["score"])
            except (TypeError, ValueError, IndexError):
                continue
            if v in (0.0, 0.5, 1.0):
                return {"score": v, "ok": True, "via": "json"}
    m = _SCORE.search(t)
    if m:
        v = float(m.group(1))
        if v in (0.0, 0.5, 1.0):
            return {"score": v, "ok": True, "via": "regex"}
    return {"score": None, "ok": False, "via": None}


def beam_item_score(scores):
    """An item's BEAM score: the mean of its rubric items' scores (None = unparsed, left out; all None -> None)."""
    v = [s for s in scores if s is not None]
    return sum(v) / len(v) if v else None


# ----------------------------------------------------------------------------------------------- PersonaMem


def mc_pick(option_logprobs, option_ntok=None):
    """PersonaMem: (index by summed log-prob, index by mean per-token log-prob)."""
    s = max(range(len(option_logprobs)), key=lambda i: option_logprobs[i])
    if option_ntok is None:
        return s, None
    m = max(range(len(option_logprobs)), key=lambda i: option_logprobs[i] / max(option_ntok[i], 1))
    return s, m


# ----------------------------------------------------------------------------------------------- panel

JUDGE_FAMILY = {"orig": "Qwen", "gptoss": "OpenAI", "gemma4": "Google"}
JUDGES = ("orig", "gptoss", "gemma4")
NUM_PREDICT = {("gptoss", "lme"): 768, ("gemma4", "lme"): 10, ("gptoss", "locomo_j"): 768, ("gemma4", "locomo_j"): 200,
               ("gptoss", "beam"): 1024, ("gemma4", "beam"): 512}
PARSERS = {"lme": lambda t: AG.parse_yesno(t), "locomo_j": parse_j, "beam": parse_beam}


def panel_label(votes, subject_family=None, judges=JUDGES):
    """(majority label, majority without the subject model's family) of {judge: label or None}; a tie (or no vote)
    goes to the original judge (bind.majority). The second is None unless a judge shares subject_family."""
    def maj(js):
        v = [votes.get(j) for j in js]
        t = [x for x in v if x is not None]
        tie = votes.get("orig")
        if not t:
            return tie
        k = sum(bool(x) for x in t)
        return tie if 2 * k == len(t) else 2 * k > len(t)
    full = maj(judges)
    excl = [j for j in judges if JUDGE_FAMILY.get(j) != subject_family]
    return full, (maj(excl) if subject_family and len(excl) < len(judges) else None)


def phash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def judge_one(url, tag, task, post=None):
    """One prompt through Ollama (temperature 0, no output constraint), parsed by its kind; one retry with another
    seed if unparsed (dream_select panel.judge_one)."""
    pj = _pj()
    post = post or pj.chat
    t0 = time.time()
    raw, p, attempts, err, full = "", None, 0, None, {}
    for attempt in range(2):
        attempts += 1
        try:
            raw, full = post(url, tag, task["prompt"], None, pj.SEED + attempt,
                             num_predict=NUM_PREDICT[(tag, task["kind"])], fmt="none")
            err = None
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as e:
            raw, err = "", f"{type(e).__name__}: {e}"[:200]
        p = PARSERS[task["kind"]](raw)
        if p["ok"]:
            break
    rec = {"h": task["h"], "judge": tag, "kind": task["kind"], "keys": task["keys"], "raw": (raw or "")[:600], **p,
           "attempts": attempts, "ms": int((time.time() - t0) * 1000),
           "tok": [full.get("prompt_eval_count"), full.get("eval_count")]}
    if err:
        rec["err"] = err
    return rec


_PJ = None


def _pj():
    global _PJ
    if _PJ is None:
        _PJ = _load("dream_compare_core_v1_panel_judge", REPO / "experiments" / "core_v1" / "panel_judge.py")
    return _PJ


def run_tasks(url, tag, tasks, out_path, workers=16, post=None, log=print, every=200):
    """dream_select panel.run_tasks with kind-dependent parsing: append one record per task not yet in out_path."""
    pj = _pj()
    done = pj.read_done(out_path)
    todo = [t for t in tasks if t["h"] not in done]
    log(f"[{tag}] {len(tasks)} unique prompts, {len(done & {t['h'] for t in tasks})} already done, {len(todo)} to do")
    lock = threading.Lock()
    n = n_bad = 0
    t0 = time.time()
    pj._end_with_newline(out_path)
    by_kind = Counter()
    bad_kind = Counter()
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(judge_one, url, tag, t, post) for t in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n += 1
                n_bad += not rec["ok"]
                by_kind[rec["kind"]] += 1
                bad_kind[rec["kind"]] += not rec["ok"]
                if n % every == 0 or n == len(todo):
                    el = time.time() - t0
                    log(f"[{tag}] {n}/{len(todo)} done | {n / el:.2f} prompts/s | unparsed {n_bad} ({n_bad / n:.1%})")
    el = time.time() - t0
    res = {"judge": tag, "done_now": n, "unparsed_now": n_bad, "seconds": el,
           "parse_rate_by_kind": {k: 1 - bad_kind[k] / v for k, v in by_kind.items()},
           "model": pj.JUDGES[tag]["model"], "think": pj.JUDGES[tag]["think"]}
    json.dump(res, open(str(out_path) + ".stats.json", "w"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--judge", required=True, choices=["gptoss", "gemma4"])
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8") if l.strip()]
    if a.limit:
        tasks = tasks[:a.limit]
    pj = _pj()
    if not pj.wait_ready(a.url):
        sys.exit(f"Ollama not reachable at {a.url}")
    if tasks:
        try:
            pj.chat(a.url, a.judge, tasks[0]["prompt"], None, pj.SEED, num_predict=10, fmt="none")
        except Exception as e:
            print(f"warm-up failed: {e}")
    res = run_tasks(a.url, a.judge, tasks, Path(a.out), a.workers)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
