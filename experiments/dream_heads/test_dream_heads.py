"""dream_heads tests (PREREG.md): the per-head attribution sums to the attention output's projection; the probe's
residuals equal dream_bind's capture; coverage and kept masks on toy tensors (GQA); folds, readout layer, licence and
control heads; the statistics; token classes; and a tiny end-to-end run (h-run -> h-analyze -> h-summary) on a random
model with a fake Stage A record built by Stage A's own functions.

  cd SCRATCH && python -m pytest -q --import-mode=importlib -p no:cacheprovider .../experiments/dream_heads/test_dream_heads.py
"""

import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("test_dh_run", HERE / "run.py")
AN = R.AN
SA, M, HM, DB, S, B = R.SA, R.M, R.HM, R.DB, R.S, R.B

_CTX = {}


def ctx(key):
    if key not in _CTX:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(M.SPECS[key]["hf"], local_files_only=True)
        except Exception:
            pytest.skip(f"{key}: tokenizer not in the local cache")
        C = M.tiny_context(key, tok)
        C.multi_device = False
        _CTX[key] = C
    return _CTX[key]


TURN = "My dog's name is Pepper and I adopted him in Denver last spring."
Q = "What is my dog's name?"


# ================================================================== attribution / probe


@pytest.mark.parametrize("key", ["qwen25_15b", "llama32_3b"])
def test_attribution_sums_to_attention_output(key):
    C = ctx(key)
    hi = R.head_info(C)
    torch.manual_seed(0)
    Dhat = R.unit(torch.randn(hi.L, C.d))
    U = R.attr_maps(C, Dhat)
    store = SA.harvest(C, TURN, stats=False)
    ans = DB.text_ids(C.tok, "Pepper")
    for cond, pf in (("nomem", None), ("R-full", store.P)):
        _, _, pr = R.probe(C, store, cond, Q, ans, pf)
        A = R.attributions(pr, U, hi)                                    # [L, Hq, L]
        for l in range(hi.L):
            tot = pr.out[l].cpu() @ Dhat.T                                # [L] projection of the attention output
            b = R.o_bias(C.blocks[l])
            bias = (b.cpu() @ Dhat.T) if b is not None else 0.0
            assert torch.allclose(torch.tensor(A[l].sum(0)) + bias, tot, atol=1e-4, rtol=1e-4)
        # the o_proj input really is the concatenation of the heads (z . W_O^T == attention output)
        l = 0
        W = C.blocks[l].self_attn.o_proj.weight.float()
        b = R.o_bias(C.blocks[l])
        o = pr.z[l] @ W.T + (b if b is not None else 0)
        assert torch.allclose(o, pr.out[l], atol=1e-5)


def test_probe_residuals_equal_dream_bind_capture():
    C = ctx("qwen25_15b")
    ids = DB.chat_ids(C.tok, Q)
    with R.Probe(C, len(ids) - 1, att=False) as pr:
        M.forward(C, ids)
    st, _ = DB.source_run(C, ids)
    for l in range(len(C.blocks)):
        assert torch.allclose(pr.res[l], st[l][-1].float(), atol=1e-6)


def test_probe_measure_equals_stage_a_measure():
    C = ctx("qwen25_15b")
    store = SA.harvest(C, TURN, stats=False)
    ans = DB.text_ids(C.tok, "Pepper")
    for cond, pf in (("nomem", None), ("R-full", store.P)):
        m, lg, _ = R.probe(C, store, cond, Q, ans, pf)
        m2, lg2 = SA.measure(C, store, cond, Q, ans, pf)
        assert abs(m["answer_score"] - m2["answer_score"]) < 1e-6 and abs(m["refusal"] - m2["refusal"]) < 1e-9
        assert float((lg - lg2).abs().max()) < 1e-5


def test_masked_group_gets_no_attention():
    C = ctx("qwen25_15b")
    hi = R.head_info(C)
    store = SA.harvest(C, TURN, stats=False)
    ans = DB.text_ids(C.tok, "Pepper")
    keep = HM.keep_all(C, store.n)
    keep[1][0, store.nh:store.n] = False
    _, _, pr = R.probe(C, store, "slice", Q, ans, store.P, keep)
    a = pr.att[1]
    g = hi.group
    assert float(a[:g, store.nh:store.n].abs().max()) == 0.0
    assert float(a[g:, store.nh:store.n].sum()) > 0


# ================================================================== coverage, masks


def test_kept_mask_and_coverage_gqa():
    content = [3, 4, 5, 6]
    K = AN.kept_mask([4, 6], content, 2, 2)
    assert K.shape == (2, 2, 4) and K[:, :, [1, 3]].all() and not K[:, :, [0, 2]].any()
    Kp = AN.kept_mask({"0": [[3], [5, 6]], "1": [[], [4]]}, content, 2, 2)
    assert Kp[0, 0].tolist() == [True, False, False, False] and Kp[0, 1].tolist() == [False, False, True, True]
    assert not Kp[1, 0].any() and Kp[1, 1].tolist() == [False, True, False, False]
    a = np.zeros((2, 4, 4))                                      # 4 query heads, group 2
    a[0, 0] = [0.1, 0.2, 0.3, 0.4]
    a[0, 3] = [0.5, 0.0, 0.25, 0.25]
    cov = AN.coverage(a, Kp, 2)
    assert np.isclose(cov[0, 0], 0.1) and np.isclose(cov[0, 3], 0.5)
    assert np.isnan(cov[0, 1]) and np.isnan(cov[1, 0])           # no content attention -> NaN
    assert np.isclose(AN.coverage(a, np.ones_like(Kp), 2)[0, 0], 1.0)
    assert AN.head_mean(cov, [(0, 0), (0, 3), (0, 1)]) == pytest.approx(0.3)


def test_token_classes():
    pieces = ["<u>", "My", " dog", "'s", " name", " is", " Pepper", ".", "</u>"]
    cls = AN.token_classes(pieces, [1, 2, 3, 4, 5, 6, 7], "What is my dog's name?", gold_span=[6])
    assert cls == ["other", "qword", "qword", "qword", "other", "gold", "punct"]
    K = np.zeros((1, 1, 7), bool)
    K[0, 0, [5, 6]] = True
    sh = AN.class_shares(K, cls)
    assert sh["gold"] == 0.5 and sh["punct"] == 0.5 and sh["qword"] == 0


# ================================================================== folds, layers, heads


def test_folds_and_tiny():
    ids = [f"q{i:02d}" for i in range(66)]
    F = AN.folds(ids)
    assert sum(v == "A" for v in F.values()) == 33 and AN.folds(list(reversed(ids))) == F
    its = R.load_items()
    assert len(its) == 66 and sum(x["fold"] == "A" for x in its) == 33
    t = R.load_items(tiny=True)
    assert [x["fold"] for x in t] == ["A", "A", "B", "B"]


def test_readout_and_heads():
    assert AN.candidates(28) == list(range(5, 22))
    rng = np.random.default_rng(0)
    pn = rng.normal(size=(10, 28))
    pf = pn + 0.1 * rng.normal(size=(10, 28))
    pf[:, 12] += 5
    pf[:, 25] += 50                                               # outside the candidates
    ls, sep = AN.readout_layer(pf, pn, AN.candidates(28))
    assert ls == 12 and sep[25] > sep[12]
    dA = np.zeros((28, 6))
    dA[20, 1] = 9                                                 # above L*: excluded
    for j, (l, h) in enumerate([(3, 0), (5, 2), (7, 5), (8, 1), (9, 0), (10, 3), (11, 4), (12, 2), (2, 2)]):
        dA[l, h] = 8 - j
    lic = AN.licence_heads(dA, 12)
    assert len(lic) == 8 and (20, 1) not in lic and (2, 2) not in lic and lic[0] == (3, 0)
    ctrl, ov = AN.control_heads(lic, 6, 3)
    assert [l for l, _ in ctrl] == [l for l, _ in lic] and ov == 0
    for (l, h), (lc, hc) in zip(lic, ctrl):
        assert (lc, hc) not in lic and hc // 3 != h // 3
    ctrl2, ov2 = AN.control_heads([(0, 0), (0, 3)], 6, 3)        # both groups hold a licence head -> overlap
    assert ov2 == 2 and all(c not in [(0, 0), (0, 3)] for c in ctrl2)
    assert AN.groups_of([(0, 0), (0, 2), (1, 5)], 3) == {0: [0], 1: [1]}


def test_statistics():
    assert AN.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert AN.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert AN.spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(0.9486833)   # ties: average ranks (scipy's value)
    assert np.isnan(AN.spearman([1, 1, 1], [1, 2, 3]))
    c = AN.boot_mean([1.0, 1.0, 1.0, float("nan")])
    assert c["mean"] == 1 and c["lo"] == 1 and c["n"] == 3
    X = np.zeros((4, 2, 3))
    X[:, 0, 0] = 3.0
    X[:, 1, 2] = 1.0
    X[:, 1, 1] = -2.0
    assert AN.share_fold(X, [(0, 0)]) == pytest.approx(0.75)
    hb = AN.hb_stat({"A": X, "B": X}, {"A": [(0, 0)], "B": [(1, 2)]}, n_boot=50)
    assert hb["folds"] == {"A": pytest.approx(0.75), "B": pytest.approx(0.25)} and hb["pooled"] == pytest.approx(0.5)


# ================================================================== end to end (tiny random model)


def fake_stage_a(C, items, root):
    """A Stage A tf record per item built with Stage A's own functions on random scores (selections at k8 and k4,
    answer scores from Stage A's measure)."""
    d = Path(root) / "tf" / "items"
    d.mkdir(parents=True, exist_ok=True)
    nkv = [l.n_kv for l in C.layout]
    rng = np.random.default_rng(1)
    for it in items:
        st = SA.harvest(C, it["stored_turn"], stats=False)
        content, nh = st.content, st.nh
        span = S.gold_span(C.tok, st.Seg, content, it["answer"]) if it["verbatim"] else None
        ans = DB.text_ids(C.tok, it["answer"])
        sels, rows = {}, []
        ph = {l: rng.random((nkv[l], st.n)) for l in range(len(C.blocks))}
        for b in ("k4", "k8"):
            k = B.budget_k(b, len(content))
            E = B.budget_entries(k, nkv)
            sels[f"KVzip-g|{b}"] = B.global_topk(ph, content, nh, E)
            sels[f"KVzip-ph|{b}"] = B.per_head_topk(ph, content, nh, k)
            sels[f"KVzip-sh|{b}"] = B.shared_topk(ph, content, nh, k)
            for s in ("S-loo", "S-dream"):
                sels[f"{s}|{b}"] = S.topk(rng.random(len(content)).tolist(), content, k)
            sels[f"random|{b}"] = [S.random_k(content, k, (s, it["index"], k)) for s in (0, 1, 2)]
            if span:
                sels[f"gold@k|{b}"] = S.gold_at_k(span, content, k)
        js = lambda x: {str(l): [list(map(int, h)) for h in v] for l, v in x.items()} if isinstance(x, dict) else x
        sels = {k_: js(v) for k_, v in sels.items()}
        for c in ("nomem", "R-full"):
            m, _ = SA.measure(C, st, c, it["question"], ans, st.P if c == "R-full" else None)
            rows.append({"cond": c, "budget": None, "answer_score": m["answer_score"], "refusal": m["refusal"]})
        for key, sel in sels.items():
            s, b = key.split("|")
            if s in AN.PER_HEAD:
                ms = [SA.measure(C, st, "slice", it["question"], ans, st.P, SA.keep_of(C, st, SA.picks_from_json(sel)))[0]]
            else:
                L = sel if s == "random" else [sel]
                ms = [SA.measure(C, st, "slice", it["question"], ans, SA.splice(st, x))[0] for x in L]
            rows.append({"cond": s, "budget": b, "answer_score": float(np.mean([m["answer_score"] for m in ms])),
                         "refusal": float(np.mean([m["refusal"] for m in ms])),
                         "per_seed_answer_score": [m["answer_score"] for m in ms]})
        info = {"question_id": it["question_id"], "n_head": nh, "n_seg": st.ns, "content": list(content),
                "gold_span": span, "answer": it["answer"], "verbatim": it["verbatim"]}
        SA.write_json_gz(d / f"{it['question_id']}.json.gz", {"info": info, "rows": rows, "sels": sels})


def test_end_to_end_tiny(tmp_path):
    C = ctx("qwen25_15b")
    items = R.load_items(tiny=True)
    for it in items:                                              # keep the tiny CPU run short
        it["stored_turn"] = it["stored_turn"][:600]
        it["verbatim"] = it["verbatim"] and it["answer"].lower() in it["stored_turn"].lower()
    sa = tmp_path / "stage_a"
    fake_stage_a(C, items, sa)
    out = tmp_path / "run" / "qwen25_15b"
    A = SimpleNamespace(out=str(out), stage_a=str(sa), tiny=True, max_reserved_gb=None)
    H = R.stage_run(A, C=C, items=items, tol=0.01)
    assert set(H["folds"]) == {"A", "B"}
    assert all(len(H["folds"][f]["licence"]) == min(8, (H["folds"][f]["lstar"] + 1) * H["n_q"]) for f in "AB")
    assert H["checks_summary"]["stage_a_diff_all"] <= 1e-6 and H["checks_summary"]["sh_mask_vs_splice"] <= 1e-4
    recs = AN.load_items(out)
    assert len(recs) == 4
    r = next(iter(recs.values()))
    assert {"nomem", "R-full", "KVzip-g|k8", "KVzip-sh|k4", "R-full-lic", "KVzip-sh+ctrl"} <= set(r["conds"])
    cv = np.asarray(r["conds"]["R-full"]["coverage"], float)
    assert np.allclose(cv[np.isfinite(cv)], 1)
    lic = [tuple(x) for x in H["folds"][r["info"]["fold"]]["licence"]]
    cl = np.asarray(r["conds"]["R-full-lic"]["coverage"], float)
    assert all(not np.isfinite(cl[l, h]) or cl[l, h] == 0 for l, h in lic)
    ml = np.asarray(r["conds"]["R-full-lic"]["mass"], float)
    assert all(ml[l, h] == 0 for l, h in lic)
    # resumable: a second call changes nothing
    R.stage_run(A, C=C, items=items, tol=0.01)
    V = AN.analyze(out, sa, "qwen25_15b")
    assert V["verdict"] in ("head-gated licence", "no single gate", "in between")
    assert (out / "report.md").exists() and (out / "verdict.json").exists()
    assert AN.summary(tmp_path / "run") in ("head-gated licence", "no single gate", "in between")
