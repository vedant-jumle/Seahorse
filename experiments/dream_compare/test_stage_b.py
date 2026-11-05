"""dream_compare Stage B tests (PREREG.md "Stage B"): the paradigm machinery on tiny random models (Qwen2, Llama; Gemma 4
where transformers has it) and small hand-made memories; MemArt's block scoring on toy tensors; storage accounting;
BM25 integration; the judges' routing per benchmark and category; the B1 / B4 statistics.

Run: python -m pytest -q --import-mode=importlib -p no:cacheprovider experiments/dream_compare/test_stage_b.py
"""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("dream_compare_paradigms", HERE / "paradigms.py")
SB = _load("dream_compare_stage_b", HERE / "stage_b.py")
AB = _load("dream_compare_analyze_b", HERE / "analyze_b.py")
M, MA, J, R = P.M, P.MA, P.J, P.R
SCH = _load("test_b_schema", HERE / "benchmarks" / "schema.py")

try:
    from transformers.models.gemma4 import modeling_gemma4  # noqa: F401
    HAVE_GEMMA4 = True
except Exception:
    HAVE_GEMMA4 = False

TOL = 1e-4
ROPE_TOL = 5e-5
TINY = ["qwen25_15b", "llama32_3b", "gemma4_26b_a4b"]
_CTX = {}


def ctx(key, attn="sdpa"):
    if (key, attn) not in _CTX:
        if M.SPECS[key]["arch"] == "gemma4" and not HAVE_GEMMA4:
            pytest.skip("no Gemma 4 classes in this transformers (4.x); run on the cluster (5.18)")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(M.SPECS[key]["hf"], local_files_only=True)
        except Exception:
            pytest.skip(f"{key} tokenizer not in the local cache")
        _CTX[(key, attn)] = M.tiny_context(key, tok, attn=attn)
    return _CTX[(key, attn)]


def md(a, b):
    return (a.float() - b.float()).abs().max().item()


# ----------------------------------------------------------------------------------------------- a hand-made bench

S1 = [("user", "I adopted a beagle named Pepper last spring in Denver."),
      ("assistant", "Congratulations on Pepper! Beagles are lovely and curious dogs."),
      ("user", "I also started a pottery class at the community center on Tuesdays.")]
S2 = [("user", "My sister Ana moved to Lisbon for a job at a bakery."),
      ("assistant", "Lisbon is a beautiful city with great pastries."),
      ("user", "I switched my running shoes to a lighter pair for the marathon.")]


def toy_bench(name="longmemeval", sessions=(S1, S2), dates=("2023/05/01 (Mon) 10:00", "2023/06/02 (Fri) 11:00")):
    turns, sess = [], []
    for si, (s, d) in enumerate(zip(sessions, dates)):
        ts = [SCH.Turn(f"c:{si}:{i}", r, t, d) for i, (r, t) in enumerate(s)]
        sess.append(SCH.Session(f"c:S{si}", d, ts))
        turns += ts
    c = SCH.Context("c", sess, {"speaker_a": "Ann", "speaker_b": "Bob"})
    its = [SCH.Item("q1", name, "c", "What is my dog's name?", ["Pepper"], "single-session-user",
                    evidence_turns=["c:0:0"], question_date="2023/07/01 (Sat) 09:00"),
           SCH.Item("q2", name, "c", "Where did my sister move?", ["Lisbon"], "single-session-user",
                    evidence_turns=["c:1:0"])]
    return SCH.Bench(name, "toy", {"c": c}, its)


def toy_memory(C, sessions=(S1, S2)):
    b = toy_bench(sessions=sessions)
    return b, P.build_memory(C, "longmemeval", b.contexts["c"])


# ----------------------------------------------------------------------------------------------- rendering, layout


def test_render_frames():
    t = SCH.Turn("x", "assistant", "  Caroline: hi  ", "8 May, 2023")
    assert P.render_turn("locomo", t) == ("user", "[8 May, 2023] Caroline: hi")
    assert P.render_turn("longmemeval", t) == ("assistant", "[8 May, 2023] Caroline: hi")
    assert P.render_turn("personamem", SCH.Turn("y", "system", "Persona: Kanoa")) == ("user", "Persona: Kanoa")
    long = SCH.Turn("z", "user", "a" * (P.TURN_MAX_CHARS + 50))
    assert P.render_turn("beam", long)[1] == "a" * P.TURN_MAX_CHARS + " [...]"
    b = toy_bench("locomo")
    q = P.question_text("locomo", b.items[0], b.contexts["c"])
    assert q.startswith("I'm reading a log of conversations between Ann and Bob.") and P.NOT_MENTIONED in q
    assert P.question_text("longmemeval", b.items[0]).startswith("Today is 2023/07/01 (Sat) 09:00. ")
    assert P.question_text("beam", b.items[1]) == b.items[1].question
    assert P.is_refusal("Not mentioned in the conversation.") and P.is_refusal("I don't remember that.")
    assert not P.is_refusal("Pepper")


def test_conditions_and_parse():
    assert P.parse_cond("P4loo-10") == ("P4", "loo", 10) and P.parse_cond("P5s-3") == ("P5", "s", 3)
    assert P.parse_cond("P0") == ("P0", None, None)
    assert P.conds_for("personamem") == AB.CONDS_PM and P.conds_for("beam") == AB.CONDS_GEN


@pytest.mark.parametrize("key", TINY)
def test_memory_layout_and_question_ids(key):
    C = ctx(key)
    b, mem = toy_memory(C)
    assert len(mem.turns) == 6 and len(mem.segments) == 2
    full = M.DB().render_ids(C.tok, [{"role": r, "content": P.render_turn("longmemeval", t)[1]}
                                     for s in b.contexts["c"].sessions for t in s.turns for r in [t.role]])[1]
    # the memory's text layout of every turn == the rendered conversation (minus the generation prompt)
    ids, starts, qs = P.layout_text(mem, list(range(6)))
    Q = P.question_ids(C, "What is my dog's name?")
    assert ids + Q == M.DB().render_ids(C.tok, [{"role": t.role, "content": t.content} for t in mem.turns]
                                        + [{"role": "user", "content": "What is my dog's name?"}])[1]
    assert starts[0] == mem.nh and qs == mem.N
    assert full[:mem.nh] == mem.H


# ----------------------------------------------------------------------------------------------- store, P3, P4


@pytest.mark.parametrize("key", TINY)
def test_p3_all_turns_equals_full_context_text(key):
    """P3 with every turn of a one-session memory retrieved == the full context as text (P1): logits at the question
    rows and greedy tokens; on two sessions the token layout is identical and only the cross-session context
    differs."""
    C = ctx(key)
    _, mem = toy_memory(C, sessions=(S1,))
    st = P.Store(C, mem)
    Q = P.question_ids(C, "What is my dog's name?")
    with torch.no_grad():
        Pf = P.p3_prefix(C, st, list(range(len(mem.turns))))
        ids, _, qs = P.layout_text(mem, list(range(len(mem.turns))))
        assert Pf.start == qs and torch.equal(Pf.pos, torch.arange(qs))
        h, _ = M.forward(C, torch.tensor(Q), Pf)
        ht, _ = M.forward(C, torch.tensor(ids + Q))
        assert md(M.logits(C, h), M.logits(C, ht[-len(Q):])) <= TOL
        a = M.gen_batch(C, torch.tensor(Q), [(True, None)], prefix=Pf, max_new=8)[0]["ids"]
        b = M.gen_batch(C, torch.tensor(ids + Q), [(True, None)], max_new=8)[0]["ids"]
        assert a == b
        # P1 (chunked prefill of the text) == the text
        P1 = P.prefill_text(C, ids, chunk=7)
        h1, _ = M.forward(C, torch.tensor(Q), P1)
        assert md(M.logits(C, h1), M.logits(C, ht[-len(Q):])) <= TOL
        # two sessions: the second session's first turn sits at its P2 position after the first session's last turn
        _, mem2 = toy_memory(C)
        st2 = P.Store(C, mem2)
        sel = [1, 3]
        P3 = P.p3_prefix(C, st2, sel)
        ids2, starts, qs2 = P.layout_text(mem2, sel)
        assert P3.start == qs2 and P3.n == len(ids2)
        assert P3.pos.tolist() == list(range(qs2))
        # the second session's turns as stored != as text after the first session (context differs), shapes equal
        assert all(K.shape[2] == len(ids2) for K, _ in P3.kv)


@pytest.mark.parametrize("key", TINY)
def test_rope_rotate_and_compose_order(key):
    C = ctx(key)
    _, mem = toy_memory(C)
    st = P.Store(C, mem)
    with torch.no_grad():
        for l in range(len(C.layout)):
            K = st.K[l][None, :, :20].clone()
            assert md(P.rope_rotate(C, K, torch.full((20,), 9), l), M.rope_shift(C, K, 9, l)) <= ROPE_TOL
            back = P.rope_rotate(C, P.rope_rotate(C, K, -st.pos[:20], l), st.pos[:20], l)
            assert md(back, K) <= ROPE_TOL
        # Store.compose == models.compose_prefix on the same stored entries (one device)
        whole = M.Prefix([(K[None], V[None]) for K, V in zip(st.K, st.V)], st.pos.clone(), int(st.pos.max()) + 1)
        t = mem.turns[3]
        parts = [(list(range(mem.nh)), 0), (list(range(t.g0, t.g0 + t.n)), mem.nh - int(st.pos[t.g0]))]
        a = st.compose(C, parts, mem.nh + t.n)
        b = M.compose_prefix(C, [("w", ix, s) for ix, s in parts], {"w": whole}, mem.nh + t.n)
        assert torch.equal(a.pos, b.pos) and all(md(x[0], y[0]) == 0 and md(x[1], y[1]) == 0
                                                 for x, y in zip(a.kv, b.kv))
        # out-of-order parts fail
        t1 = mem.turns[1]
        with pytest.raises(AssertionError):
            st.compose(C, [(list(range(t1.g0, t1.g0 + t1.n)), 50), (list(range(mem.nh)), 0)], 200)
        # P4 keeps the head + the picks at their P3 slots and positions; the question starts where P3's does
        sel = [0, 4]
        P3 = P.p3_prefix(C, st, sel)
        picks = {0: [1, 3], 4: [2]}
        keep = P.p4_keep(mem, sel, picks)
        assert keep == list(range(mem.nh)) + [mem.nh + 1, mem.nh + 3, mem.nh + mem.turns[0].n + 2]
        P4 = M.index_prefix(P3, keep)
        assert P4.start == P3.start and P4.pos.tolist() == P3.pos[keep].tolist()


# ----------------------------------------------------------------------------------------------- ragged batches


@pytest.mark.parametrize("key", TINY)
def test_ragged_equals_per_row(key):
    """Rows with different prefixes and prompt lengths in one batch == each row alone (greedy tokens; teacher-forced
    log-probs)."""
    C = ctx(key)
    _, mem = toy_memory(C)
    st = P.Store(C, mem)
    Q = P.question_ids(C, "What is my dog's name?")
    Q2 = P.question_ids(C, "Where does my sister live now, and why?")
    with torch.no_grad():
        P3 = P.p3_prefix(C, st, [0, 4])
        P4 = M.index_prefix(P3, P.p4_keep(mem, [0, 4], {0: [1, 2, 5], 4: [3]}))
        ids2 = P.layout_text(mem, [1, 2])[0]
        rows = [dict(prefix=None, ids=list(mem.H) + Q), dict(prefix=P3, ids=Q2), dict(prefix=None, ids=ids2 + Q2),
                dict(prefix=P4, ids=Q)]
        got = P.ragged_gen(C, rows, 10)
        for r, g in zip(rows, got):
            ref = M.gen_batch(C, torch.tensor(r["ids"]), [(True, None)], prefix=r["prefix"], max_new=10)[0]["ids"]
            assert g["ids"] == ref
        opt = M.DB().text_ids(C.tok, "Pepper the beagle").tolist()
        sr = [dict(r, ids=r["ids"] + opt, n_score=len(opt)) for r in rows]
        sc = P.ragged_score(C, sr)
        for r, (s, n) in zip(rows, sc):
            ref, _ = M.answer_score(C, r["ids"], opt, r["prefix"])
            assert n == len(opt) and abs(s - ref) <= 1e-3
        # sampled rows: reproducible and independent of the batch
        srows = [dict(prefix=P3, ids=Q, greedy=False, seed=11), dict(prefix=None, ids=list(mem.H) + Q2, greedy=False, seed=5)]
        a = P.ragged_gen(C, srows, 8)
        b = P.ragged_gen(C, srows[:1], 8)
        assert a[0]["ids"] == b[0]["ids"]


# ----------------------------------------------------------------------------------------------- MemArt


def test_memart_blocks_and_scores_toy():
    blocks = MA.make_blocks([(3, 35), (38, 16)], block=16)
    assert blocks == [(3, 16), (19, 16), (35, 3), (38, 16)]
    t2b = MA.token_to_block(blocks, 54)
    assert t2b[:3].tolist() == [-1, -1, -1] and t2b[3] == 0 and t2b[37] == 2 and t2b[53] == 3
    torch.manual_seed(0)
    Hq, Hkv, Tq, d, N = 4, 2, 3, 8, 54
    q = torch.randn(Hq, Tq, d)
    k = torch.randn(Hkv, N, d)
    sc = MA.block_scores(q, k, t2b, len(blocks), Hkv)
    assert sc.shape == (Hq * Tq, 4)
    for h in range(Hq):
        for t in range(Tq):
            for j, (g0, n) in enumerate(blocks):
                ref = max(float(q[h, t] @ k[h // 2, g]) for g in range(g0, g0 + n))
                assert abs(float(sc[h * Tq + t, j]) - ref) < 1e-5
    # RR-Max on a hand example: row 0 ranks blocks [2, 0, 1], row 1 ranks [1, 2, 0]
    s = torch.tensor([[0.5, 0.1, 0.9], [0.0, 0.8, 0.3]])
    rr = MA.rr_max(s, c=60)
    assert torch.allclose(rr, torch.tensor([1 / 62, 1 / 61, 1 / 61], dtype=torch.float64))
    assert MA.select_blocks(rr, 2) == [1, 2]          # tie between 1 and 2 at 1/61: both kept, chronological
    assert MA.select_blocks(torch.tensor([0.3, 0.5, 0.5]), 1) == [1]     # ties -> the earlier block
    cand = torch.tensor([True, False, True])
    rr2 = MA.rr_max(s, cand, c=60)
    assert rr2[1] == 0 and MA.select_blocks(rr2, 5, cand) == [0, 2]
    assert MA.n_blocks_for(0) == 0 and MA.n_blocks_for(16) == 1 and MA.n_blocks_for(17) == 2


@pytest.mark.parametrize("key", TINY)
def test_memart_hook_all_blocks_equals_restored(key):
    """P5 with a budget covering every block of a one-segment memory whose turn tokens fill whole blocks == the
    same tokens restored with P3 at the same (contiguous) positions; with a smaller budget the hook picks k blocks per
    layer, writes them at positions nh + 16 j and masks the pad slots of short blocks."""
    C = ctx(key)
    _, mem = toy_memory(C, sessions=(S1,))
    st = P.Store(C, mem)
    Q = P.question_ids(C, "What is my dog's name?")
    n_tok = mem.N - mem.nh
    nb = MA.n_blocks_for(n_tok)
    with torch.no_grad():
        # reference: P3 of all turns, with the question moved to nh + 16 nb (the pad slots of the last block masked)
        Pf = P.p3_prefix(C, st, list(range(len(mem.turns))))
        ref_pf = M.Prefix(Pf.kv, Pf.pos, mem.nh + MA.BLOCK * nb)
        ref = M.gen_batch(C, torch.tensor(Q), [(True, None)], prefix=ref_pf, max_new=8, force_masks=True)[0]["ids"]
        ph = SB.p5_placeholder(C, st, nb)
        row = dict(prefix=ph, ids=Q, p5=dict(store=st, n_blocks=nb, nh=mem.nh, cand=None), q_len=len(Q))
        got = P.ragged_gen(C, [row], 8)[0]
        assert got["ids"] == ref
        assert all(v == list(range(nb)) for v in got["p5_blocks"].values())
        # a budget of 2 blocks: 2 picks per layer, chronological
        ph2 = SB.p5_placeholder(C, st, 2)
        g2 = P.ragged_gen(C, [dict(prefix=ph2, ids=Q, p5=dict(store=st, n_blocks=2, nh=mem.nh, cand=None),
                                    q_len=len(Q))], 4)[0]
        assert all(len(v) == 2 and v == sorted(v) for v in g2["p5_blocks"].values())


# ----------------------------------------------------------------------------------------------- selectors


def test_device_safe_selectors_equal_dream_select():
    """sel_dream_b / sel_loo_b == dream_select's sel_dream (primary) / sel_loo on one device (tiny Qwen2, eager)."""
    C = ctx("qwen25_15b", "eager")
    D = P.DSR()
    turn = "My dog's name is Pepper and I adopted him in Denver last spring."
    dreams = {"questions": ["What is my dog's name?", "Where did I adopt my dog?"]}
    with torch.no_grad():
        st0 = D.harvest(C, turn)
        ref_d, _ = D.sel_dream(C, st0, dreams)
        ref_l, _ = D.sel_loo(C, st0, dreams)
    P_ = D.DB.turn_pieces(C.tok, turn)
    st = P.turn_store(C, P_.H, P_.Seg, turn)
    assert st.content == st0.content
    got_d = P.sel_dream_b(C, st, dreams)
    got_l, _ = P.sel_loo_b(C, st, dreams)
    assert max(abs(a - b) for a, b in zip(got_d, ref_d)) <= 1e-6
    assert max(abs(a - b) for a, b in zip(got_l, ref_l)) <= 1e-4
    with torch.no_grad():
        dr0 = D.dream(C, st0)
        dr1 = P.dream_b(C, st)
    assert dr1["attempts"][0]["raw"] == dr0["attempts"][0]["raw"]


def test_select_turn_short_and_record():
    C = ctx("qwen25_15b")
    t = SimpleNamespace(piece=None, content="Hi there.", role="user", turn_id="t")
    H = C.head
    t.piece = M.session_pieces(C.tok, [{"role": "user", "content": t.content}])[1][0]
    r = P.select_turn(C, H, t)
    assert r["dream"] == r["loo"] and len(r["dream"]) <= P.K_SLICE and r["dreams"] is None


# ----------------------------------------------------------------------------------------------- storage, retrieval


def test_storage_accounting():
    C = ctx("qwen25_15b")
    _, mem = toy_memory(C)
    s = P.storage(C, mem)
    num = M.kv_numbers_per_position(C.layout)
    text = sum(len(t.content.encode()) for t in mem.turns)
    assert s["text_bytes"] == text and s["n_tokens"] == mem.N
    assert s["P3"] == text + mem.N * num * 4 and s["P2"] == text and s["P0"] == 0
    n4 = mem.nh + sum(min(P.K_SLICE, t.n_content_tok) for t in mem.turns)
    assert s["P4"] == text + n4 * (num * 4 + 4)
    nb = sum(math.ceil(n / 16) for _, n in mem.seg_ranges())
    assert s["P5"] == mem.N * num * 4 + nb * sum(L.n_kv * L.head_dim * 2 for L in C.layout) * 4
    s2 = P.storage(C, mem, upto=3)
    assert s2["n_tokens"] == mem.nh + sum(t.n for t in mem.turns[:3])
    # Gemma: V-only on its K = V layers
    lay = M.layout(M.SPECS["gemma4_26b_a4b"])
    fake = SimpleNamespace(layout=lay)
    g = P.storage(fake, mem)
    assert g["kv_all"] - g["kv_all_vonly"] == mem.N * 5 * 2 * 512 * 4


def test_bm25_integration():
    C = ctx("qwen25_15b")
    b, mem = toy_memory(C)
    it = b.items[1]
    got = P.retrieve(b, it, mem, 2)
    ids = R.retrieve(b, it, 2)
    assert [mem.turns[i].turn_id for i in got] == ids and got == sorted(got)
    assert mem.turns[got[0]].turn_id == "c:1:0" or "c:1:0" in ids
    assert P.recall_of(b, it, mem, got) == 1.0
    # PersonaMem: session-level retrieval returns whole (visible) sessions
    pb = toy_bench("personamem")
    pb.items[1].context_turns = 5
    pmem = P.build_memory(C, "personamem", pb.contexts["c"])
    r = P.retrieve(pb, pb.items[1], pmem, 1)
    assert r == [3, 4]                                   # session 2, cut at the visible end (5 turns)
    r2 = P.retrieve(pb, pb.items[0], pmem, 1)
    assert r2 == [0, 1, 2]


# ----------------------------------------------------------------------------------------------- judges routing


def _rec(bench, cat, abst=False, conds=("P0",), rubric=None, code=None):
    return {"bench": bench, "split": "x", "item": "i1", "category": cat, "abstention": abst, "question": "Q?",
            "answers": ["A"] if not abst else ([] if bench == "locomo" else ["expl"]), "category_code": code,
            "rubric": rubric or [], "n_rubric": len(rubric or []),
            "conds": {c: {"text": "an answer", "refusal": False} for c in conds}}


def test_judge_routing():
    t = SB.tasks_for_item(_rec("longmemeval", "temporal-reasoning"), "m")
    assert len(t) == 1 and t[0][0] == "lme" and "off-by-one" in t[0][1] and t[0][2] == "m|longmemeval_x|i1|P0|0"
    t = SB.tasks_for_item(_rec("longmemeval", "knowledge-update", abst=True), "m")
    assert "unanswerable" in t[0][1]
    t = SB.tasks_for_item(_rec("longmemeval", "single-session-preference"), "m")
    assert "Rubric:" in t[0][1]
    assert SB.tasks_for_item(_rec("locomo", "adversarial", abst=True, code=5), "m") == []
    t = SB.tasks_for_item(_rec("locomo", "temporal", code=2), "m")
    assert t[0][0] == "locomo_j" and "Gold answer: A" in t[0][1]
    t = SB.tasks_for_item(_rec("beam", "event_ordering", rubric=["r1", "r2", "r3"]), "m")
    assert [x[0] for x in t] == ["beam"] * 3 and [x[2][-1] for x in t] == ["0", "1", "2"]
    assert SB.tasks_for_item(dict(_rec("personamem", "x"), conds={"P0": {"correct": True}}), "m") == []
    # analysis routing: LoCoMo adversarial by the official rule, BEAM by the median, LME by the majority
    items = {"a": dict(_rec("locomo", "adversarial", True, code=5), item="a",
                       conds={"P0": {"text": "Not mentioned in the conversation"}, "P2-3": {"text": "Paris"}})}
    prim, _ = AB.correctness(items, {}, "m", "locomo_locomo10")
    assert prim["P0"]["a"] == 1.0 and prim["P2-3"]["a"] == 0.0
    items = {"b": dict(_rec("beam", "x", rubric=["r1", "r2"]), item="b")}
    votes = {"m|beam_100K|b|P0|0": {"orig": 1.0, "gptoss": 0.5, "gemma4": 0.0},
             "m|beam_100K|b|P0|1": {"orig": 1.0, "gptoss": 1.0, "gemma4": None}}
    prim, _ = AB.correctness(items, votes, "m", "beam_100K")
    assert prim["P0"]["b"] == pytest.approx((0.5 + 1.0) / 2)
    items = {"c": dict(_rec("longmemeval", "multi-session"), item="c")}
    votes = {"gemma4_26b_a4b|longmemeval_oracle|c|P0|0": {"orig": False, "gptoss": True, "gemma4": True}}
    prim, xf = AB.correctness(items, votes, "gemma4_26b_a4b", "longmemeval_oracle")
    assert prim["P0"]["c"] == 1.0 and xf["P0"]["c"] == 0.0       # without Gemma: orig vs gptoss tie -> orig
    items = {"d": dict(_rec("personamem", "x"), item="d", conds={"P0": {"correct": True}, "P1": {"correct": False}})}
    prim, _ = AB.correctness(items, {}, "m", "personamem_32k")
    assert prim["P0"]["d"] == 1.0 and prim["P1"]["d"] == 0.0


# ----------------------------------------------------------------------------------------------- statistics


def test_tost_and_pareto():
    x = {i: 1.0 for i in range(400)}
    assert AB.tost(x, dict(x))["verdict"] == "equivalent"
    y = {i: (1.0 if i % 2 else 0.0) for i in range(400)}
    assert AB.tost(x, y)["verdict"].startswith("different (P3 better")
    z = {i: (0.0 if i < 8 else 1.0) for i in range(400)}       # 2% worse, n 400: CI straddles -0.03
    assert AB.tost(z, x)["verdict"] in ("inconclusive", "equivalent", "different (P2 better)")
    assert AB.pooled_tost([(x, dict(x)), (x, dict(x))])["verdict"] == "equivalent"
    pts = {"a": (1, 0.5), "b": (2, 0.4), "c": (3, 0.9), "d": (None, 1.0)}
    assert AB.pareto(pts) == ["a", "c"]


def test_subsets_fixed():
    b = toy_bench()
    for i in range(3, 150):
        b.items.append(SCH.Item(f"q{i}", "longmemeval", "c", "x?", ["y"], "multi-session"))
    a1, a2 = SB.subset_ids(b), SB.subset_ids(b)
    assert a1 == a2 and len(a1) == 100 and a1 == sorted(a1)
    import random
    assert a1 == sorted(random.Random(7).sample(sorted(it.item_id for it in b.items), 100))


def test_jobs_runner_env(monkeypatch):
    """jobs.py runs stage_b.py when DC_RUNNER=stage_b (additive; run.py stays the default)."""
    import os
    JB = _load("test_b_jobs", HERE / "jobs.py")
    monkeypatch.delenv("DC_RUNNER", raising=False)
    assert JB.runner_path() == HERE / "run.py"
    monkeypatch.setenv("DC_RUNNER", "stage_b")
    assert JB.runner_path() == HERE / "stage_b.py"


def test_select_turns_batched_equals_single():
    """select_turns (batched dreams, own answers and leave-one-out rows) == select_turn per turn."""
    C = ctx("qwen25_15b")
    _, mem = toy_memory(C)
    idx = [0, 1, 4]
    with torch.no_grad():
        got = P.select_turns(C, mem.H, [mem.turns[i] for i in idx], loo={0, 2}, chunk=2)
        for j, i in enumerate(idx):
            ref = P.select_turn(C, mem.H, mem.turns[i])
            assert got[j]["dream"] == ref["dream"], i
            assert got[j]["dreams"] == ref["dreams"]
            if j in (0, 2):
                assert got[j]["loo"] == ref["loo"], i
            else:
                assert got[j]["loo"] is None or got[j]["loo"] == got[j]["dream"]


@pytest.mark.parametrize("key", TINY)
def test_chunked_prefill_equals_one_pass(key, monkeypatch):
    """Ragged prompts and the store's session encoding in small chunks == one pass (greedy tokens, scores, keys)."""
    C = ctx(key)
    _, mem = toy_memory(C)
    Q = P.question_ids(C, "What is my dog's name?")
    with torch.no_grad():
        st1 = P.Store(C, mem)
        P3 = P.p3_prefix(C, st1, [0, 4])
        ids2 = P.layout_text(mem, [1, 2])[0]
        opt = M.DB().text_ids(C.tok, "Pepper").tolist()
        rows = [dict(prefix=P3, ids=Q), dict(prefix=None, ids=ids2 + Q)]
        srows = [dict(r, ids=r["ids"] + opt, n_score=len(opt)) for r in rows]
        g1, s1 = P.ragged_gen(C, rows, 6), P.ragged_score(C, srows)
        monkeypatch.setattr(P, "PREFILL_CHUNK", 5)
        st2 = P.Store(C, mem)
        g2, s2 = P.ragged_gen(C, rows, 6), P.ragged_score(C, srows)
    assert [a["ids"] for a in g1] == [b["ids"] for b in g2]
    assert all(abs(a[0] - b[0]) <= 1e-3 for a, b in zip(s1, s2))
    assert all(md(a, b) <= TOL for a, b in zip(st1.K, st2.K))
