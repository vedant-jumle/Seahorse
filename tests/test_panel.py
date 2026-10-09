"""core_v1 multi-judge panel (experiments/core_v1/{panel_judge,analyze_panel}.py): the prompts are byte-identical
to the original judge's, the parser is the original one (plus repair), majority vote, Fleiss' kappa, and the
resume / retry logic of the client. CPU only, no Ollama."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "core_v1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("panel_judge_t", EXP / "panel_judge.py")
AP = _load("analyze_panel_t", EXP / "analyze_panel.py")
core = _load("core_v1_core_t", EXP / "core.py")
J = P.load_prompts()

ITEMS = {
    "dog": {"id": "dog", "group": "facts", "experience": "My dog is called Pepper.", "measure": {"target": " Pepper "}},
    "veg": {"id": "veg", "group": "dislikes", "experience": "I am vegetarian.", "measure": None},
}
ROWS = [
    {"item": "dog", "prompt": "Write a note about my dog's checkup.", "answer": "Dear vet, my dog Pepper has {a} checkup."},
    {"item": "dog", "prompt": "Hi", "answer": "   "},
    {"item": "veg", "prompt": "What should I cook tonight? {x}", "answer": "Try a lentil curry {with} braces."},
]


def _run_judge_text():
    try:
        return _load("core_v1_run_t", EXP / "run.py").judge_text
    except Exception as e:  # run.py needs its heavy imports
        pytest.skip(f"run.py not importable here: {e}")


def test_prompts_byte_identical_to_original_judge():
    orig = _run_judge_text()
    L = {"by_id": ITEMS}
    for r in ROWS:
        assert P.judge_text(J, ITEMS, r) == orig(J, L, r)
    assert "(empty reply)" in P.judge_text(J, ITEMS, ROWS[1])
    assert '(so the correct value is "Pepper")' in P.judge_text(J, ITEMS, ROWS[0])
    assert "{trait}" not in P.judge_text(J, ITEMS, ROWS[2])


def test_prompts_equal_on_real_run_if_present():
    run = ROOT / "results" / "core_v1_20261007_1820"
    if not (run / "qwen35_2b" / "prep" / "items.json").exists():
        pytest.skip("no local run")
    orig = _run_judge_text()
    rows = P.load_rows(run, ("qwen35_2b",))
    for m, items, r in rows[:: max(len(rows) // 40, 1)]:
        assert P.judge_text(J, items, r) == orig(J, {"by_id": items}, r)


@pytest.mark.parametrize("text", [
    '{"direction": "toward", "coherent": true, "self_claim": false}',
    'Sure!\n```json\n{"direction": "AWAY", "coherent": "true", "self_claim": 0}\n```',
    '{"direction": "toward", "coherent": true',
    'direction: neutral, coherent: no, self_claim = false',
    "I cannot grade this.",
    '{"direction": "sideways", "coherent": true, "self_claim": true}',
    "",
])
def test_parser_matches_original(text):
    for rub in ("preference", "fact"):
        assert P.parse_judge(text, J[rub]["schema"]) == core.parse_judge(text, J[rub]["schema"])


def test_repair_and_parse_output():
    sch = J["fact"]["schema"]
    raw = '<think>maybe {"uses_fact": false}</think>\n```json\n{"uses_fact": true, "coherent": true, "self_claim": false, "wrong_value": false}\n```'
    p = P.parse_output(raw, sch)
    assert p["ok"] and p["uses_fact"] is True and p["wrong_value"] is False
    assert not P.parse_output("no json here", sch)["ok"]
    assert P.json_schema(J["preference"]["schema"])["properties"]["direction"]["enum"] == ["toward", "away", "neutral"]
    assert P.json_schema(sch)["required"] == list(sch)


def test_majority_and_ties():
    assert AP.majority([True, True, False]) is True
    assert AP.majority(["toward", "away"], tie_break="away", default="neutral") == "away"
    assert AP.majority(["toward", "away"], tie_break="neutral", default="neutral") == "neutral"
    assert AP.majority(["toward", "away", "neutral"], default="neutral") == "neutral"
    ok = lambda **kw: {"ok": True, **kw}
    f = ["direction", "coherent", "self_claim"]
    rows = [ok(direction="toward", coherent=True, self_claim=False), ok(direction="toward", coherent=False, self_claim=False),
            {"ok": False}, None, ok(direction="away", coherent=True, self_claim=True)]
    m = AP.majority_row(rows, f, orig=rows[4])   # toward x2 beats away; coherent tie -> orig (True); self_claim False x2
    assert m == {"ok": True, "n_votes": 3, "direction": "toward", "coherent": True, "self_claim": False}
    assert AP.majority_row([{"ok": False}, None], f) == {"ok": False, "n_votes": 0}
    # tie without a valid original -> False / neutral
    m = AP.majority_row([ok(direction="toward", coherent=True, self_claim=True), ok(direction="away", coherent=False, self_claim=False)], f,
                        orig={"ok": False})
    assert m["direction"] == "neutral" and m["coherent"] is False and m["self_claim"] is False


def test_fleiss_known_example():
    # Wikipedia's worked example: 10 items, 14 raters, 5 categories -> kappa = 0.210
    M = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
         [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    assert AP.fleiss_kappa(M) == pytest.approx(0.210, abs=5e-4)
    assert AP.fleiss_from_labels([[True, False, True], [True, False, True], [True, False, True]], [False, True]) == pytest.approx(1.0)
    assert AP.fleiss_from_labels([[True, False, True, False], [False, True, False, True]], [False, True]) == pytest.approx(-1.0)


def _fake_run(tmp_path):
    for m in P.MODELS:
        d = tmp_path / m
        (d / "prep").mkdir(parents=True)
        (d / "gen" / "items").mkdir(parents=True)
        json.dump(ITEMS, open(d / "prep" / "items.json", "w"))
        rows = []
        for i, (cond, seedset) in enumerate([("nomem", 1), ("mem@2", 1), ("mem@3", 1), ("nomem", 2), ("gate_on@2", 1)]):
            rows.append({"sec": "gen", "key": f"v|{cond}|{seedset}|{i}", "item": "veg", "group": "dislikes", "cond": cond,
                         "seedset": seedset, "prompt": "q", "answer": "same answer"})
        rows.append({"sec": "gen", "key": "d|nomem", "item": "dog", "group": "facts", "cond": "mem@3", "seedset": 1,
                     "prompt": "q2", "answer": "Pepper"})
        rows.append({"sec": "gate", "key": "g", "item": "veg", "group": "dislikes", "cond": "nomem", "seedset": 1,
                     "prompt": "q", "answer": "x"})
        with gzip.open(d / "gen" / "items" / "x.jsonl.gz", "wt") as f:
            f.write("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def test_tasks_dedupe_and_subset(tmp_path):
    run = _fake_run(tmp_path)
    tasks = P.build_tasks(run, J)
    # per model: 5 veg rows share one prompt (same answer) -> 1 prompt; + 1 fact prompt; both models share them
    assert len(tasks) == 2 and sum(len(t["keys"]) for t in tasks) == 12
    assert all(t["keys"][0].split(":")[0] in P.MODELS for t in tasks)
    sub = P.build_tasks(run, J, subset=True)
    assert len(sub) == 2 and all(t["pri"] == 0 for t in sub)   # nomem/seed1 dislikes row and mem@3 fact row are used
    assert P.needed({"seedset": 2, "cond": "nomem", "group": "dislikes"}, P.DEFAULT_CLAIMS_CFG) is False
    assert P.needed({"seedset": 1, "cond": "gate_on@2", "group": "dislikes"}, P.DEFAULT_CLAIMS_CFG) is False
    assert P.needed({"seedset": 1, "cond": "centroid@1", "group": "dislikes"}, P.DEFAULT_CLAIMS_CFG) is True
    assert P.needed({"seedset": 1, "cond": "rand@2", "group": "leanings"}, P.DEFAULT_CLAIMS_CFG) is True
    assert P.needed({"seedset": 1, "cond": "mem@3", "group": "one_of_many"}, P.DEFAULT_CLAIMS_CFG) is False


def test_resume_skips_done_and_survives_torn_line(tmp_path):
    run = _fake_run(tmp_path / "run")
    tasks = P.build_tasks(run, J)
    out = tmp_path / "out.jsonl"
    calls = []

    def post(url, tag, prompt, schema, seed, num_predict=None, fmt="schema", timeout=0):
        calls.append(prompt)
        if "uses_fact" in schema:
            return json.dumps({"uses_fact": True, "coherent": True, "self_claim": False, "wrong_value": False}), {}
        return json.dumps({"direction": "toward", "coherent": True, "self_claim": False}), {}

    P.run_tasks("u", "gemma4", tasks[:1], J, out, workers=2, post=post, log=lambda *_: None)
    assert len(calls) == 1 and len(P.read_done(out)) == 1
    with open(out, "a") as f:
        f.write('{"h": "torn')           # a killed job's partial line
    P.run_tasks("u", "gemma4", tasks, J, out, workers=2, post=post, log=lambda *_: None)
    assert len(calls) == 2               # only the missing prompt was sent
    assert len(P.read_done(out)) == 2
    P.run_tasks("u", "gemma4", tasks, J, out, workers=2, post=post, log=lambda *_: None)
    assert len(calls) == 2


def test_retry_once_then_invalid(tmp_path):
    run = _fake_run(tmp_path / "run")
    task = P.build_tasks(run, J)[0]
    seeds = []

    def bad(url, tag, prompt, schema, seed, num_predict=None, fmt="schema", timeout=0):
        seeds.append((seed, fmt))
        return "I refuse", {}

    rec = P.judge_one("u", "llama31", task, J[task["rubric"] if task["rubric"] == "fact" else "preference"]["schema"], bad)
    assert rec["ok"] is False and rec["attempts"] == 2 and len(seeds) == 2 and seeds[0][0] != seeds[1][0]
    assert seeds[0][1] == "schema" and seeds[1][1] == "none"
    calls = []

    def flaky(url, tag, prompt, schema, seed, num_predict=None, fmt="schema", timeout=0):
        calls.append(1)
        return ("oops", {}) if len(calls) == 1 else ('{"direction": "away", "coherent": true, "self_claim": false}', {})

    t = [x for x in P.build_tasks(run, J) if x["rubric"] == "preference"][0]
    rec = P.judge_one("u", "llama31", t, J["preference"]["schema"], flaky)
    assert rec["ok"] and rec["attempts"] == 2 and rec["direction"] == "away"


def test_prevalence_and_summary_shapes():
    S = AP.claim_summary({n: {m: {"pre": {"C2": False, "C3": True, "C5": True},
                                  **{c: {"holds": n != "gemma4" or c != "C5"} for c in ("C2", "C3", "C5")}} for m in AP.MODELS}
                          for n in ("orig", "gemma4")}, ["orig", "gemma4"])
    assert S["C5"]["qwen35_2b"]["differs_from_orig"] == ["gemma4"]
    assert "NOT robust" in AP.verdict_text(S, ["orig", "gemma4"])
