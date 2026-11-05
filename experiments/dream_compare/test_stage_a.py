"""dream_compare Stage A tests (PREREG.md, "Stage A: selectors"): the budget arithmetic, every scorer and selector of
baselines.py on toy tensors, the per-head mask identities through stage_a's own builders, the Expected Attention
pieces against the model's real attention, the reimplemented ours-selectors against dream_select's own functions, the
no-question / no-answer rule for selectors, and a tiny end-to-end run (tf -> gen -> judge tasks -> analysis) on random
models. Gemma-4 cases need transformers >= 5.5 (cluster) and are skipped on 4.46.

  cd /tmp/claude-1000/pytest_scratch && python -m pytest -q --import-mode=importlib -p no:cacheprovider \
      /path/Seahorse/experiments/dream_compare/test_stage_a.py
"""

import gzip
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load("test_sa_baselines", HERE / "baselines.py")
SA = _load("test_sa_stage_a", HERE / "stage_a.py")
AA = _load("test_sa_analyze_a", HERE / "analyze_a.py")
M, HM, DSR, DB, S = SA.M, SA.HM, SA.DSR, SA.DB, SA.S

try:
    from transformers.models.gemma4 import modeling_gemma4  # noqa: F401
    HAVE_GEMMA4 = True
except Exception:
    HAVE_GEMMA4 = False

TINY = ["qwen25_15b", "llama32_3b", "gemma4_26b_a4b"]
TURN = ("My dog's name is Pepper and I adopted him in Denver last spring, right after I started my nursing job at "
        "the children's hospital.")
QUESTION = "What is my dog's name?"
DREAMS = {"questions": ["What is my dog's name?", "Where did I adopt my dog?", "Where do I work?"], "fallback": False}


def _tok(hf):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(hf, local_files_only=True)
    except Exception:
        return None


_CTX = {}


def ctx(key):
    if key not in _CTX:
        if M.SPECS[key]["arch"] == "gemma4" and not HAVE_GEMMA4:
            pytest.skip("Gemma 4 needs transformers >= 5.5 (cluster)")
        tok = _tok(M.SPECS[key]["hf"])
        if tok is None:
            pytest.skip(f"{key}: tokenizer not in the local cache")
        _CTX[key] = M.tiny_context(key, tok)
    return _CTX[key]


def md(a, b):
    return float((torch.as_tensor(a).float() - torch.as_tensor(b).float()).abs().max())


# ================================================================== budgets


def test_budget_arithmetic():
    assert [B.budget_k(b, 50) for b in B.BUDGETS] == [2, 4, 8, 5]
    assert B.budget_k("k10", 9) == 2 and B.budget_k("k8", 5) == 5 and B.budget_k("k10", 104) == 10
    for b in B.BUDGETS:                                                 # dream_select's rule, unchanged
        for n in (3, 9, 48, 210):
            assert B.budget_k(b, n) == S.budget_k(b, n)
    want = {"qwen25_15b": 56, "qwen25_7b": 112, "llama32_3b": 224, "gemma4_26b_a4b": 210, "granite42_30b": 512}
    for key, E in want.items():
        nkv = [l.n_kv for l in M.layout(M.SPECS[key])]
        assert B.entries_per_position(nkv) == E == M.kv_entries_per_position(M.SPECS[key])
        assert B.budget_entries(8, nkv) == 8 * E
    assert len(B.SELECTIONS) == 18 and B.ORACLE <= B.PER_HEAD and not (B.ORACLE & set(B.NON_ORACLE_BASELINES))
    assert len(B.NON_ORACLE_BASELINES) == 11 and set(B.H3_METHODS) <= B.PER_HEAD


# ================================================================== scorers on toy tensors


def _toy_w(hq=4, tq=5, tk=9, seed=0):
    r = np.random.default_rng(seed)
    x = r.random((hq, tq, tk))
    return x / x.sum(-1, keepdims=True)


def test_group_reduce_maps_query_head_to_kv_head():
    x = np.arange(8.0)[:, None] * np.ones((8, 3))             # query head h has value h
    assert np.allclose(B.group_reduce(x, 2, "mean")[:, 0], [1.5, 5.5])
    assert np.allclose(B.group_reduce(x, 2, "max")[:, 0], [3, 7])
    assert np.allclose(B.group_reduce(x, 8)[:, 0], np.arange(8))


def test_kvzip_score_is_max_over_group_and_rows():
    w = _toy_w()
    s = B.kvzip_layer(w, 2, 6)
    assert s.shape == (2, 6)
    for kv in range(2):
        for j in range(6):
            assert s[kv, j] == w[2 * kv:2 * kv + 2, :, j].max()


def test_maxpool_matches_torch():
    x = np.random.default_rng(1).random((3, 11))
    ref = torch.nn.functional.max_pool1d(torch.tensor(x)[None], 7, stride=1, padding=3)[0].numpy()
    assert np.allclose(B.maxpool1d(x, 7), ref)


def test_snapkv_vote_window_group_mean_pool():
    w = _toy_w(4, 40, 50)
    s = B.snapkv_layer(w, 2, 12, window=32)
    v = w[:, -32:, :12].sum(1)
    g = np.stack([v[0:2].mean(0), v[2:4].mean(0)])
    ref = torch.nn.functional.max_pool1d(torch.tensor(g)[None], 7, stride=1, padding=3)[0].numpy()
    assert np.allclose(s, ref)
    # a window longer than the rows = all rows
    assert np.allclose(B.snapkv_layer(w[:, :5], 2, 12), B.maxpool1d(B.group_reduce(w[:, :5, :12].sum(1), 2)))


def test_h2o_vatp_value_norms():
    T = 6
    w = np.tril(np.ones((T, T)))
    w = w / w.sum(1, keepdims=True)
    w = np.stack([w, w * 0 + np.eye(T)])                       # 2 query heads, 1 KV head
    h = B.h2o_layer(w, 1)
    assert np.allclose(h[0], 0.5 * (w[0].sum(0) + np.ones(T)))
    V = np.random.default_rng(2).normal(size=(1, T, 4))
    n1, n2 = B.value_norms(V, 1), B.value_norms(V, 2)
    assert np.allclose(n1, np.abs(V).sum(-1)) and np.allclose(n2, np.linalg.norm(V, axis=-1))
    assert np.allclose(B.vatp_layer(h, n1), h * n1)
    rm = B.rows_mean_layer(w, [2, 4], 1, 5)
    assert np.allclose(rm, w[:, [2, 4], :5].mean(1).mean(0, keepdims=True))


def _rope_cos_sin(d, pos, theta=1e4, rot=None):
    """Explicit rotate_half RoPE tables; rot < d: only the first rot / 2 frequencies of each half rotate."""
    rot = rot or d
    inv = 1.0 / theta ** (np.arange(0, d, 2) / d)
    inv[rot // 2:] = 0.0
    f = np.outer(pos, inv)
    emb = np.concatenate([f, f], -1)
    return np.cos(emb), np.sin(emb)


def test_rope_matrix_and_average():
    d = 8
    q = np.random.default_rng(3).normal(size=d)
    cos, sin = _rope_cos_sin(d, np.arange(5, 25), rot=4)
    for p in range(3):
        A = B.rope_matrix(cos[p], sin[p], B.rotate_half)
        assert np.allclose(q @ A, q * cos[p] + B.rotate_half(q) * sin[p])
    Ab = B.avg_rope_matrix(cos, sin)
    direct = np.mean([q * c + B.rotate_half(q) * s for c, s in zip(cos, sin)], 0)
    assert np.allclose(q @ Ab, direct)


def test_expected_attention_against_direct():
    """EA with the covariance off == the mean over future positions of the rotated mean query's logit; with it on, the
    second-order term k' A' Sigma A k / 2 (scaled); softmax over the given columns only; group mean; x ||v||_2."""
    r = np.random.default_rng(4)
    hq, n_kv, d, T = 4, 2, 8, 7
    q = r.normal(size=(hq, 30, d))
    K, V = r.normal(size=(n_kv, T, d)), r.normal(size=(n_kv, T, d))
    mu, cov = B.query_stats(q)
    assert np.allclose(mu, q.mean(1)) and np.allclose(cov[1], np.cov(q[1].T, bias=True))
    cos, sin = _rope_cos_sin(d, np.arange(10, 40))
    A = B.avg_rope_matrix(cos, sin)
    s, cols = 0.3, [2, 3, 4, 5, 6]
    got = B.ea_layer(mu, cov, A, K, V, s, cols, use_cov=False)
    z = np.zeros((hq, T))
    for h in range(hq):
        rq = np.mean([mu[h] * c + B.rotate_half(mu[h]) * sn for c, sn in zip(cos, sin)], 0)
        z[h] = s * K[h // 2] @ rq
    p = np.zeros_like(z)
    e = np.exp(z[:, cols] - z[:, cols].max(1, keepdims=True))
    p[:, cols] = e / e.sum(1, keepdims=True)
    ref = B.group_reduce(p, n_kv) * np.linalg.norm(V, axis=-1)
    assert np.allclose(got, ref) and np.all(got[:, :2] == 0)
    full = B.ea_layer(mu, cov, A, K, V, s, cols)
    for h in range(hq):
        z[h] += 0.5 * s ** 2 * np.array([K[h // 2, j] @ A.T @ cov[h] @ A @ K[h // 2, j] for j in range(T)])
    e = np.exp(z[:, cols] - z[:, cols].max(1, keepdims=True))
    p[:, cols] = e / e.sum(1, keepdims=True)
    assert np.allclose(full, B.group_reduce(p, n_kv) * np.linalg.norm(V, axis=-1))


# ================================================================== selectors


def test_topk_ties_and_per_head():
    content = [1, 2, 3, 4, 5]
    nh = 3
    sc = np.zeros(nh + 7)
    sc[nh + 2], sc[nh + 4], sc[nh + 5] = 5, 1, 1          # tie between 4 and 5 -> the earlier one
    assert B.topk_content(sc, content, nh, 2) == [2, 4]
    sc[nh + 3] = np.nan
    assert 3 not in B.topk_content(sc, content, nh, 4)
    scores = {0: np.stack([sc, -sc]), 1: np.stack([sc])}
    ph = B.per_head_topk(scores, content, nh, 2)
    assert ph[0][0] == [2, 4] and len(ph[0]) == 2 and len(ph[1]) == 1
    B.check_picks(ph, content, 2 * 3, [2, 1])
    with pytest.raises(AssertionError):
        B.check_picks(ph, content, 7)
    with pytest.raises(AssertionError):
        B.check_picks({0: [[0, 2]]}, content, 2)            # 0 is not a content position


def test_global_topk_budget_and_order():
    nh, content = 2, [0, 1, 2, 3]
    S0 = np.array([[0, 0, 9, 1, 1, 1], [0, 0, 8, 7, 6, 0]], dtype=float)
    S1 = np.array([[0, 0, 0, 0, 0, 10]], dtype=float)
    g = B.global_topk({0: S0, 1: S1}, content, nh, 4)            # 10 (l1 h0 c3), 9, 8, 7
    assert g == {0: [[0], [0, 1]], 1: [[3]]} and B.picks_entries(g) == 4
    g5 = B.global_topk({0: S0, 1: S1}, content, nh, 5)           # + 6 (l0 h1 c2)
    assert g5 == {0: [[0], [0, 1, 2]], 1: [[3]]}
    S2 = {0: np.ones((2, 6)), 1: np.ones((1, 6))}                 # all tied: (layer, head, position) order
    assert B.global_topk(S2, content, nh, 5) == {0: [[0, 1, 2, 3], [0]], 1: [[]]}


def test_shared_recent_heavy_streaming():
    nh, content = 1, list(range(1, 9))
    sc = np.zeros(nh + 10)
    sc[nh + 1], sc[nh + 2] = 5, 4
    assert B.recent_heavy(sc, content, nh, 2) == [1, 8]            # 1 recent (8) + 1 heavy (1)
    assert B.recent_heavy(sc, content, nh, 4) == [1, 2, 7, 8]
    assert B.recent_heavy(sc, content, nh, 3) == [1, 2, 8]         # k // 2 = 1 recent, 2 heavy
    sc2 = sc.copy()
    sc2[nh + 8] = 100                                              # a recent position is not counted as heavy
    assert B.recent_heavy(sc2, content, nh, 2) == [1, 8]
    assert B.streaming(content, 4) == [1, 2, 7, 8] and B.streaming(content, 3) == [1, 7, 8]
    sh = B.shared_score({0: np.stack([sc, sc * 0]), 1: np.stack([sc * 3])})
    assert np.allclose(sh, (sc + 3 * sc) / 3)
    assert B.same_in_every_head({0: [[1, 2], [1, 2]], 1: [[1, 2]]}) == [1, 2]
    assert B.same_in_every_head({0: [[1, 2], [1, 3]]}) is None
    assert B.span_share({0: [[1, 2], [3, 4]]}, [2]) == 0.5


def test_gap():
    assert B.gap(-2.0, -10.0, 0.0, 0.5) == 0.8 and B.gap(1.0, 0.0, 0.3, 0.5) is None and B.gap(None, 0, 1, 0.5) is None


# ================================================================== selectors never see the question or the answer


def test_no_selector_takes_the_question_or_answer():
    allowed = {"C", "store", "dreams"}
    for name, fn in SA.SELECTOR_FNS.items():
        ps = list(inspect.signature(fn).parameters)
        if name in SA.ORACLE_FNS:
            assert ps == ["C", "store", "oracle_question"], name
            continue
        assert set(ps) <= allowed and ps[:2] == ["C", "store"], (name, ps)
        assert not (set(ps) & S.FORBIDDEN_ARGS)
    # every baselines.py function: no question / answer / item / label argument (span_share is a measure on the gold
    # span, information only, never a selector)
    for n, f in inspect.getmembers(B, inspect.isfunction):
        if n == "span_share":
            continue
        assert not (set(inspect.signature(f).parameters) & (S.FORBIDDEN_ARGS - {"q", "a"})), n
    # in tf_item, the only code before the measures that reads the question is the flagged oracle call
    src = inspect.getsource(SA.tf_item)
    pre = src.split("# --- gold reference")[0]
    assert pre.count('it["question"]') == 1 and "sel_snapkv_oracle(C, store, it[\"question\"])" in pre
    assert 'it["answer"]' not in pre


# ================================================================== torch side on tiny random models


@pytest.mark.parametrize("key", TINY)
@torch.no_grad()
def test_avg_rope_and_prerope_queries_match_real_attention(key):
    """prerope_queries + the RoPE matrix at a single position + the layer's own scaling reproduce the model's eager
    attention row exactly (validates Expected Attention's ingredients per layer type, partial rotation incl.); the
    averaged matrix == the mean of the per-position matrices (rope_shift of the identity)."""
    C = ctx(key)
    st = SA.harvest(C, TURN)
    ids = torch.tensor(st.H + st.Seg)
    got = {}
    with SA.Tap(C, lambda l, w: got.__setitem__(l, w[0].float().clone()), inputs=True) as tap:
        M.harvest_kv(C, ids, attn=SA.NEED_OUTPUT_ATTN)
    t = st.n - 1
    for l in range(len(C.blocks)):
        A_t = SA.avg_rope(C, l, t, n=1)
        q = SA.prerope_queries(C, l, tap.inputs[l][t:t + 1]).detach().float().numpy()[:, 0]     # [Hq, d]
        K = st.P.kv[l][0][0].float().numpy()                                                    # [n_kv, T, d]
        G = q.shape[0] // K.shape[0]
        z = SA.attn_scaling(C, l) * np.einsum("hd,htd->ht", q @ A_t, np.repeat(K, G, 0))
        lo = 0 if C.layout[l].window is None else max(0, t - C.layout[l].window + 1)
        z = z[:, lo:t + 1]
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        assert np.abs(p - got[l][:, t, lo:t + 1].numpy()).max() < 2e-4, (key, l)
        A_avg = SA.avg_rope(C, l, 5, n=4)
        E = torch.eye(C.layout[l].head_dim)[None, None]
        ref = np.mean([M.rope_shift(C, E, p_, l)[0, 0].numpy() for p_ in range(5, 9)], 0)
        assert np.abs(A_avg - ref).max() < 1e-5


@pytest.mark.parametrize("key", ["qwen25_15b", "llama32_3b"])
def test_ours_reimplemented_equal_dream_select(key):
    """S-attn, S-dream, S-recon and S-loo through the adapter == dream_select's own functions (same model, same
    dreams): the fresh items and the 2-GPU models use these."""
    C = ctx(key)
    with torch.no_grad():
        st = SA.harvest(C, TURN)
        ds = DSR.harvest(C, TURN)
        assert (st.nh, st.ns, st.content) == (ds.nh, ds.ns, ds.content)
        assert np.allclose(SA.sel_attn(C, st), DSR.sel_attn(C, ds), equal_nan=True, rtol=1e-4, atol=1e-7)
        prim, _ = DSR.sel_dream(C, ds, DREAMS) if C.n == 28 else _ds_dream_prim(C, ds)
        assert np.allclose(SA.sel_dream(C, st, DREAMS), prim, rtol=1e-4, atol=1e-7)
        assert np.allclose(SA.sel_recon(C, st), DSR.sel_recon(C, ds), rtol=1e-4, atol=1e-7)
        a, oa = SA.sel_loo(C, st, DREAMS)
        b, ob = DSR.sel_loo(C, ds, DREAMS)
        assert [x["answer"] for x in oa] == [x["answer"] for x in ob]
        assert np.allclose(a, b, rtol=1e-3, atol=1e-4)


def _ds_dream_prim(C, ds):
    prim = []
    cols = [ds.nh + c for c in ds.content]
    for q in DREAMS["questions"]:
        Mx, qc, last = DSR._dq_attention(C, ds, q)
        prim.append(S.query_mean(Mx.mean(0), qc + [last], cols))
    return S.dream_combine(prim), None


@pytest.mark.parametrize("key", TINY)
def test_per_head_masks_through_stage_a(key):
    """keep_of / measure / gen: keep-all == R-full; the same picks in every head == the splice (teacher-forced and
    sampled decoding); a per-head selection of KVzip-g keeps exactly its entries."""
    C = ctx(key)
    with torch.no_grad():
        st = SA.harvest(C, TURN)
        ans = DB.text_ids(C.tok, "Pepper")
        m0, l0 = SA.measure(C, st, "R-full", QUESTION, ans, st.P)
        m1, l1 = SA.measure(C, st, "slice", QUESTION, ans, st.P, HM.keep_all(C, st.n))
        assert md(l0, l1) <= 1e-5
        sel = st.content[2:5]
        same = {l: [list(sel)] * C.layout[l].n_kv for l in range(len(C.blocks))}
        assert B.same_in_every_head(same) == list(sel)
        _, lm = SA.measure(C, st, "slice", QUESTION, ans, st.P, SA.keep_of(C, st, same))
        _, ls = SA.measure(C, st, "slice", QUESTION, ans, SA.splice(st, sel))
        assert md(lm, ls) <= 1e-4
        qt = DB.qt_ids(C, QUESTION)
        rows, streams = [(True, None), (False, (1, 1)), (False, (1, 2))], {1: (5, 3)}
        a = SA.gen(C, qt, rows, prefix=st.P, max_new=6, streams=streams, keep=SA.keep_of(C, st, same), n_stored=st.n)
        b = SA.gen(C, qt, rows, prefix=SA.splice(st, sel), max_new=6, streams=streams)
        assert [x["ids"] for x in a] == [x["ids"] for x in b]
        kz = SA.sel_kvzip(C, st)
        nkv = SA.n_kv_list(C)
        g = B.global_topk(kz, st.content, st.nh, B.budget_entries(4, nkv))
        keep = SA.keep_of(C, st, g)
        assert HM.stored_entries(keep) == B.budget_entries(4, nkv) + st.nh * sum(nkv)


def test_gen_batch_multi_device_path_equals_dream_bind():
    """models.gen_batch on a context flagged multi_device takes the masked loop (logits moved to the input device);
    token for token equal to dream_bind's loop."""
    C = ctx("qwen25_15b")
    st = SA.harvest(C, TURN, stats=False)
    qt = DB.qt_ids(C, QUESTION)
    rows, streams = [(True, None), (False, (1, 1))], {1: (9, 2)}
    with torch.no_grad():
        a = M.gen_batch(C, qt, rows, prefix=st.P, max_new=8, streams=streams)
        C.multi_device = True
        try:
            b = M.gen_batch(C, qt, rows, prefix=st.P, max_new=8, streams=streams)
        finally:
            C.multi_device = False
    assert [x["ids"] for x in a] == [x["ids"] for x in b]


@pytest.mark.parametrize("key", TINY)
def test_drop_forward_equals_physical_drop(key):
    """S-loo's batched leave-one-out pass == a physical splice without that slot, row by row (Gemma: position masks)."""
    C = ctx(key)
    with torch.no_grad():
        st = SA.harvest(C, TURN, stats=False)
        ids = torch.cat([DB.qt_ids(C, QUESTION), DB.text_ids(C.tok, "Pepper")])
        drops = [st.nh + st.content[0], st.nh + st.content[3]]
        hb = SA.drop_forward(C, st.P, drops, ids)
        for b, j in enumerate(drops):
            h, _ = M.forward(C, ids, M.index_prefix(st.P, [i for i in range(st.n) if i != j]))
            assert md(M.logits(C, hb[b]), M.logits(C, h)) <= 1e-4


# ================================================================== items and the end-to-end tiny run


def _have_data():
    return (REPO / "data" / "external" / "longmemeval" / "processed" / "oracle.json").exists()


@pytest.mark.skipif(not _have_data(), reason="LongMemEval oracle not prepared (data/external)")
def test_load_items():
    items = SA.load_items()
    assert len(items) == 133 and sum(x["ssu"] for x in items) == 64 and sum(x["primary"] for x in items) == 106
    assert sum(x["abstention"] for x in items) == 8 and sum(x["verbatim"] for x in items) == 70
    t = SA.load_items(tiny=True)
    assert [x["ssu"] for x in t] == [True, True, False, False] and all(x["primary"] for x in t)
    assert SA.part_of(items, 1, 3) == [x for x in items if x["index"] % 3 == 1]
    assert sum(len(SA.part_of(items, p, 3)) for p in range(3)) == 133


def _fake_items():
    turns = [TURN, "I finally finished my first marathon in Boston, it took me four hours and ten minutes."]
    qs = [(QUESTION, "Pepper"), ("Where did I run my first marathon?", "Boston")]
    out = []
    for n, (t, (q, a)) in enumerate(zip(turns, qs)):
        out.append({"question_id": f"fake{n}", "question": q, "answer": a, "category": "single-session-user",
                    "abstention": False, "stored_turn": t, "evidence_turn": f"e{n}", "verbatim": True, "primary": True,
                    "ssu": n == 0, "index": n})
    return out


def test_end_to_end_tiny(tmp_path, monkeypatch):
    """a-tf (one fresh item, one 'reused' item) -> a-gen -> a-judge-tasks -> fake votes -> a-analyze -> a-summary, on
    the tiny random Qwen2 (the dreamer is fixed: a random model writes no questions)."""
    key = "qwen25_15b"
    C = ctx(key)
    items = _fake_items()
    monkeypatch.setattr(SA, "load_items", lambda tiny=False: items)
    monkeypatch.setattr(SA, "dream", lambda C_, store: dict(DREAMS, attempts=[], retried=False))
    monkeypatch.setattr(SA, "setup", lambda A: C)
    with torch.no_grad():
        st = SA.harvest(C, items[0]["stored_turn"])
        loo, owns = SA.sel_loo(C, st, DREAMS)
        reuse = {"fake0": {"n_head": st.nh, "n_seg": st.ns, "content": st.content, "turn": items[0]["stored_turn"],
                           "dreams": DREAMS, "own_answers": owns,
                           "scores": {"S-attn": SA.sel_attn(C, st), "S-dream": SA.sel_dream(C, st, DREAMS),
                                      "S-recon": SA.sel_recon(C, st), "S-loo": loo}}}
    monkeypatch.setattr(SA, "load_reuse", lambda root, model: reuse)
    d = tmp_path / key
    A = SimpleNamespace(out=str(d), tiny=False, part=0, nparts=1, reuse="x", model=key, random_model=True,
                        max_reserved_gb=None, device="cpu", gb_per_gpu=66)
    with torch.no_grad():
        SA.stage_tf(A)
    recs = [SA.read_json_gz(d / "tf" / "items" / f"fake{n}.json.gz") for n in range(2)]
    nkv = SA.n_kv_list(C)
    E = B.entries_per_position(nkv)
    for r in recs:
        conds = {(x["cond"], x["budget"]) for x in r["rows"]}
        for b in B.BUDGETS:
            for c in B.SELECTIONS:
                if c == "S-loo" and not r["info"]["loo_run"]:
                    continue
                assert (c, b) in conds, (c, b)
        for x in r["rows"]:
            if x["budget"] in B.BUDGETS:
                assert x["entries"] == x["k"] * E, x["cond"]
        assert r["info"]["identity_rfull_tfull"] <= SA.TF_TOL and r["info"]["identity_keep_all"] <= SA.TF_TOL
    assert recs[0]["info"]["reused"] and not recs[1]["info"]["reused"]
    assert max(v["rel"] for v in recs[0]["info"]["reuse_check"].values()) < 1e-6
    with torch.no_grad():
        SA.stage_gen(A)
    G, I = SA.read_gen(d)
    names = {r["cname"] for r in G}
    assert {"nomem", "R-full", "gold", "KVzip-g@k8", "EA-ph@k4", "S-loo@k8", "random@k4"} <= names
    assert len(G) == len(I["fake0"]["conds"]) * 5 + len(I["fake1"]["conds"]) * 5
    monkeypatch.setattr(SA, "load_items", lambda tiny=False: items)
    SA.stage_judge_tasks(A)
    tasks = [json.loads(l) for l in open(d / "judge" / "tasks.jsonl")]
    assert sum(len(t["keys"]) for t in tasks) == len(G) and all(t["kind"] == "lme" for t in tasks)
    assert "Correct Answer: Pepper" in next(t["prompt"] for t in tasks if t["item"] == "fake0")
    # fake votes: R-full and gold right, everything else alternating
    rng = np.random.default_rng(0)
    for j in AA.JUDGES:
        with open(d / "judge" / f"{j}.jsonl", "w") as f:
            for t in tasks:
                lab = bool(rng.random() < 0.5) or "R-full" in t["keys"][0]
                f.write(json.dumps({"h": t["h"], "judge": j, "keys": t["keys"], "label": lab, "ok": True}) + "\n")
    v = AA.analyze(d, key)
    assert v["labels_complete"] and v["rules"]["H3"]["per_method"].keys() == set(B.H3_METHODS)
    assert v["rules"]["H4"]["best_baseline"] in B.NON_ORACLE_BASELINES
    assert (d / "report_a.md").read_text().startswith("# dream_compare Stage A")
    assert AA.summary(tmp_path, [key, "llama32_3b"]) in ("HOLDS", "does not hold")
    assert (tmp_path / "summary_a.md").exists()


def test_run_py_dispatches_stage_a(tmp_path, monkeypatch):
    RUN = _load("test_sa_run", HERE / "run.py")
    seen = {}
    fake = SimpleNamespace(main=lambda av: seen.setdefault("av", av))
    monkeypatch.setattr(RUN, "_load", lambda name, path: fake if path.name == "stage_a.py" else None)
    RUN.main(["--model", "qwen25_15b", "--stage", "a-judge-tasks", "--out", str(tmp_path)])
    assert seen["av"][3] == "a-judge-tasks"
    src = (HERE / "submit_stage_a.sh").read_text()
    for s in ("1214590", "1214592", "1214593", "1214594", "a-tf", "a-gen", "a-judge-tasks", "a-analyze", "a-summary"):
        assert s in src, s
