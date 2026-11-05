"""dream_compare Stage 0 tests (experiments/dream_compare; PREREG.md "Stage 0"): the model specs against the real
configs, the cache utilities and per-head masks on tiny random models (Qwen2, Llama, and Gemma 4 where the installed
transformers has it: 5.x; skipped on 4.46), the benchmark adapters on small fixtures, the judge prompts (pinned by
hash), BM25. Slow / cluster-only tests are marked `cluster` (deselect with -m "not cluster").

Run: python -m pytest -q --import-mode=importlib -p no:cacheprovider experiments/dream_compare/test_dream_compare.py
"""

import glob
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load("test_dc_models", HERE / "models.py")
HM = _load("test_dc_headmask", HERE / "headmask.py")

try:
    from transformers.models.gemma4 import modeling_gemma4  # noqa: F401  (transformers >= 5.5)
    HAVE_GEMMA4 = True
except Exception:
    HAVE_GEMMA4 = False

TOL = 1e-4
ROPE_TOL = 5e-5     # keys are O(1); 1e-5 flaked at 1.07e-5 on a cluster CPU (job 1214589)


def _tok(name):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return None


def _cached_config(hf):
    hits = sorted(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{hf.replace('/', '--')}/snapshots/*/config.json")))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hits += sorted(glob.glob(f"{hf_home}/hub/models--{hf.replace('/', '--')}/snapshots/*/config.json"))
    return json.load(open(hits[-1])) if hits else None


# ================================================================== specs


@pytest.mark.parametrize("key", sorted(M.SPECS))
def test_spec_matches_real_config(key):
    """Each spec agrees with the model's config.json: the HF cache where present, else the committed copy
    (configs/<key>.config.json, Apache-2.0 models only)."""
    cfg = _cached_config(M.SPECS[key]["hf"])
    p = HERE / "configs" / f"{key}.config.json"
    if cfg is None and p.exists():
        cfg = json.load(open(p))
    if cfg is None:
        pytest.skip(f"no config for {key} in the HF cache")
    assert M.check_spec(key, cfg) == []


def test_gemma4_spec_facts():
    """The facts the plan relies on, spelled out (verified against config.json on 2026-11-05)."""
    s = M.SPECS["gemma4_26b_a4b"]
    L = M.layout(s)
    assert len(L) == 30 and [i for i, l in enumerate(L) if l.kind == M.FULL] == [5, 11, 17, 23, 29]
    sl, gl = L[0], L[5]
    assert (sl.n_kv, sl.head_dim, sl.window, sl.rope_theta, sl.partial, sl.k_eq_v) == (8, 256, 1024, 1e4, 1.0, False)
    assert (gl.n_kv, gl.head_dim, gl.window, gl.rope_theta, gl.partial, gl.k_eq_v, gl.rope_type) == \
        (2, 512, None, 1e6, 0.25, True, "proportional")
    assert gl.rot_dim == 128 and s["softcap"] == 30.0
    assert M.kv_entries_per_position(s) == 25 * 8 + 5 * 2
    assert M.kv_numbers_per_position(s) == 2 * (25 * 8 * 256 + 5 * 2 * 512)
    assert M.kv_numbers_per_position(s, dedup_k_eq_v=True) == 2 * 25 * 8 * 256 + 5 * 2 * 512
    assert M.check_spec("gemma4_26b_a4b", json.load(open(HERE / "configs" / "gemma4_26b_a4b.config.json"))) == []


def test_ds_models_match_dream_select():
    """The three dream_select models load through dream_select's own model blocks (same HF ids, 28 blocks)."""
    D = M.DS()
    import yaml
    cfg = yaml.safe_load(open(D.DB_CFG))
    for key in ("qwen25_15b", "qwen25_7b", "llama32_3b"):
        mc, label = D.model_config(M.SPECS[key]["ds_key"], cfg)
        assert mc["model"] == M.SPECS[key]["hf"] and mc["n_blocks"] == M.SPECS[key]["n_layers"] == D.N_BLOCKS
        assert label == M.SPECS[key]["label"] and mc["eager"] is True
    assert D.model_config("llama32_3b", cfg)[0]["date_string"] == M.SPECS["llama32_3b"]["chat"]["date_string"]


# ================================================================== tiny models


def _ctx(key, attn="eager", n_layers=None, window=8):
    hf = M.SPECS[key]["hf"]
    if M.SPECS[key]["arch"] == "gemma4" and not HAVE_GEMMA4:
        pytest.skip("no Gemma 4 classes in this transformers (4.x); run on the cluster (5.18)")
    tok = _tok(hf)
    if tok is None:
        pytest.skip(f"{hf} tokenizer not in the local cache")
    return M.tiny_context(key, tok, n_layers=n_layers, window=window, attn=attn)


_CACHE = {}


def ctx(key, attn="eager"):
    if (key, attn) not in _CACHE:
        _CACHE[(key, attn)] = _ctx(key, attn)
    return _CACHE[(key, attn)]


TINY = ["qwen25_15b", "llama32_3b", "gemma4_26b_a4b"]
TURN = "My dog's name is Pepper and I adopted him in Denver last spring, right after I started my nursing job."
QUESTION = "What is my dog's name?"


def _turn_and_question(C, turn=TURN, q=QUESTION):
    P = M.DB().turn_pieces(C.tok, turn)
    qt = M.DB().qt_ids(C, q)
    return P, qt


def _maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("key", TINY)
def test_full_stored_turn_equals_text(key):
    """Restoring the whole stored turn == the turn as a prior user turn (teacher-forced logits at every question row,
    and greedy generation), through the adapter (plain cache; Gemma: position masks)."""
    C = ctx(key)
    P, qt = _turn_and_question(C)
    with torch.no_grad():
        store, _, _ = M.harvest_kv(C, P.H + P.Seg)
        assert store.start == len(P.H) + len(P.Seg) and store.n == store.start
        h, _ = M.forward(C, qt, store)
        _, tids = M.DB().render_ids(C.tok, [{"role": "user", "content": TURN}, {"role": "user", "content": QUESTION}])
        assert tids == P.H + P.Seg + qt.tolist()
        ht, _ = M.forward(C, torch.tensor(tids))
        assert _maxdiff(M.logits(C, h), M.logits(C, ht[-len(qt):])) <= TOL
        # the model's own default path (its own cache and masks) agrees with the adapter's pass on plain text
        o = C.body(input_ids=torch.tensor(tids)[None], use_cache=True)
        assert _maxdiff(o.last_hidden_state[0], ht) <= TOL
        a = M.gen_batch(C, qt, [(True, None)], prefix=store, max_new=8)
        b = M.gen_batch(C, torch.tensor(tids), [(True, None)], max_new=8)
        assert a[0]["ids"] == b[0]["ids"]


def _keys_at(C, ids, start):
    P = M.Prefix([], torch.zeros(0, dtype=torch.long), start)
    st, _, _ = M.harvest_kv(C, ids, P if start else None)
    return [K for K, _ in st.kv]


@pytest.mark.parametrize("key", TINY)
def test_rope_shift_equals_recompute_per_layer_type(key):
    """rope_shift(K at 0, 7, layer) == K recomputed at +7, in every layer (Gemma: sliding and full layer types,
    the full layers' proportional RoPE rotating 25% of each head); non-rotated dims are unchanged; shifting back
    (-7) recovers the original; layer=None (dream_bind's rope_shift) agrees on the single-rotary models."""
    C = ctx(key)
    ids = M.DB().chat_ids(C.tok, "My dog's name is Pepper.").tolist()
    with torch.no_grad():
        K0, K7 = _keys_at(C, ids, 0), _keys_at(C, ids, 7)
        kinds = set()
        for l, (a, b) in enumerate(zip(K0, K7)):
            assert not torch.allclose(a, b, atol=1e-4)
            s = M.rope_shift(C, a, 7, l)
            assert _maxdiff(s, b) <= ROPE_TOL, (key, l)
            assert _maxdiff(M.rope_shift(C, s, -7, l), a) <= ROPE_TOL
            L = C.layout[l]
            kinds.add(L.kind)
            if L.rot_dim < L.head_dim:          # partial: dims with zero frequency are not rotated
                half, r = L.head_dim // 2, L.rot_dim // 2
                still = list(range(r, half)) + list(range(half + r, L.head_dim))
                assert _maxdiff(a[..., still], b[..., still]) <= ROPE_TOL  # float noise of earlier layers only
                assert _maxdiff(M.rope_shift(C, a, 7, l)[..., still], a[..., still]) == 0.0
                assert not torch.allclose(a[..., :r], b[..., :r], atol=1e-4)
        if not C.sliding:
            assert _maxdiff(M.rope_shift(C, K0[1], 7), K7[1]) <= ROPE_TOL
        else:
            assert kinds == {M.FULL, M.SLIDE}


@pytest.mark.parametrize("key", TINY)
def test_compose_prefix_and_shift(key):
    """compose_prefix: one part == index_prefix; a part harvested at 0 and shifted by +d == the same tokens harvested
    at d (keys, per layer type); positions are carried and checked increasing / before the next turn."""
    C = ctx(key)
    ids = M.DB().chat_ids(C.tok, "I moved to Denver.").tolist()
    with torch.no_grad():
        A, _, _ = M.harvest_kv(C, ids)
        idx = [0, 1, 4, 5, 6]
        P1 = M.compose_prefix(C, [("A", idx, 0)], {"A": A}, start=A.start)
        P2 = M.index_prefix(A, idx)
        assert torch.equal(P1.pos, P2.pos) and all(torch.equal(x[0], y[0]) for x, y in zip(P1.kv, P2.kv))
        d = 13
        B = M.compose_prefix(C, [("A", list(range(A.n)), d)], {"A": A}, start=A.start + d)
        at_d = _keys_at(C, ids, d)
        assert torch.equal(B.pos, torch.arange(d, d + A.n))
        for l, ((K, V), K2) in enumerate(zip(B.kv, at_d)):
            assert _maxdiff(K, K2) <= ROPE_TOL, l
        with pytest.raises(AssertionError):
            M.compose_prefix(C, [("A", [3, 1], 0)], {"A": A}, start=A.start)


@pytest.mark.parametrize("key", TINY)
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_headmask_keep_all_equals_full(key, attn):
    """Per-head masks that keep everything == the full cache (teacher-forced and greedy decoding)."""
    C = ctx(key, attn)
    P, qt = _turn_and_question(C)
    with torch.no_grad():
        store, _, _ = M.harvest_kv(C, P.H + P.Seg)
        h0, _ = M.forward(C, qt, store)
        with HM.HeadMask(C, HM.keep_all(C, store.n), store.n) as hm:
            h1, _ = M.forward(C, qt, store)
            g1 = M.gen_batch(C, qt, [(True, None)], prefix=store, max_new=6)
        assert hm.calls > 0
        assert _maxdiff(h0, h1) <= 1e-5
        assert g1[0]["ids"] == M.gen_batch(C, qt, [(True, None)], prefix=store, max_new=6)[0]["ids"]


@pytest.mark.parametrize("key", TINY)
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_headmask_shared_equals_physical_splice(key, attn):
    """The same kept slots in every head == the physical splice of those slots at their stored positions
    (teacher-forced logits and multi-step greedy + sampled decoding)."""
    C = ctx(key, attn)
    P, qt = _turn_and_question(C)
    nh = len(P.H)
    with torch.no_grad():
        store, _, _ = M.harvest_kv(C, P.H + P.Seg)
        slots = list(range(nh)) + [nh + 3, nh + 4, nh + 9, nh + 10, nh + 11]
        phys = M.index_prefix(store, slots)
        hp, _ = M.forward(C, qt, phys)
        with HM.HeadMask(C, HM.keep_shared(C, store.n, slots), store.n):
            hm, _ = M.forward(C, qt, store)
            streams = {1: (7, 3)}
            rows = [(True, None), (False, (1, 1)), (False, (1, 2))]
            gm = M.gen_batch(C, qt, rows, prefix=store, max_new=8, streams=streams)
        assert _maxdiff(M.logits(C, hp), M.logits(C, hm)) <= TOL
        gp = M.gen_batch(C, qt, rows, prefix=phys, max_new=8, streams=streams)
        assert [g["ids"] for g in gm] == [g["ids"] for g in gp]


@pytest.mark.parametrize("key", TINY)
def test_headmask_gqa_expansion(key):
    """A slot dropped by KV head 0 only gets exactly 0 attention from the query heads of group 0 and positive
    attention from the other groups (eager attentions)."""
    C = ctx(key)
    P, qt = _turn_and_question(C)
    with torch.no_grad():
        store, _, _ = M.harvest_kv(C, P.H + P.Seg)
        keep = HM.keep_all(C, store.n)
        j = store.n - 2                     # inside the tiny Gemma's window (8) of the first question row
        for l in keep:
            keep[l][0, j] = False
        with HM.HeadMask(C, keep, store.n):
            _, att = M.forward(C, qt, store, attn=True)
        for l, a in enumerate(att):
            hq, hkv, g = HM._n_heads(C.blocks[l].self_attn)
            col = a[0, :, :, j]
            assert float(col[:g].abs().max()) == 0.0
            if hkv > 1:                                   # (Gemma's global layers have one KV head)
                assert float(col[g:, 0].min()) > 0.0      # the first question row (within any window)
        assert HM.stored_entries(keep) == sum(k.numel() for k in keep.values()) - len(keep)


def test_gen_masked_loop_equals_dream_bind():
    """The adapter's masked generation loop (used on sliding models) == dream_bind's gen_batch on a full-attention
    model, token for token (greedy and sampled rows, the same random streams), with and without a prefix."""
    C = ctx("qwen25_15b")
    P, qt = _turn_and_question(C)
    with torch.no_grad():
        store, _, _ = M.harvest_kv(C, P.H + P.Seg)
        rows = [(True, None), (False, (1, 1)), (False, (1, 2)), (False, (1, 3))]
        streams = {1: (11, 4)}
        for pf in (None, M.index_prefix(store, list(range(len(P.H))) + [len(P.H) + 2, len(P.H) + 3])):
            a = M.gen_batch(C, qt, rows, prefix=pf, max_new=10, streams=streams)
            b = M.gen_batch(C, qt, rows, prefix=pf, max_new=10, streams=streams, force_masks=True)
            assert [x["ids"] for x in a] == [x["ids"] for x in b]


def test_gemma4_sliding_window_by_position():
    """Gemma 4 (tiny, window 8): (1) the default cache evicts sliding-layer entries, harvest_kv keeps them all;
    (2) a sparse prefix whose RoPE distances exceed the window: the position masks == a full pass whose question rows
    cannot see the dropped slots (the reference), while transformers' slot-based masks are wrong (documented)."""
    C = ctx("gemma4_26b_a4b")
    assert C.sliding and C.window == 8 and C.softcap == 30.0
    torch.manual_seed(1)
    ids = torch.randint(10, 2000, (25,))
    with torch.no_grad():
        o = C.body(input_ids=ids[None, :20], use_cache=True)
        own = [L.keys.shape[2] for L in o.past_key_values.layers]
        assert min(own) == C.window - 1 and max(own) == 20             # the default cache keeps window - 1
        st, _, _ = M.harvest_kv(C, ids[:20])
        assert all(K.shape[2] == 20 for K, _ in st.kv)                  # nothing evicted
        keep = [0, 1, 2] + list(range(15, 20))
        sp = M.index_prefix(st, keep, start=20)
        h, _ = M.forward(C, ids[20:], sp)
        allp = torch.arange(25)
        R = M.pos_masks(C, allp, allp)
        R = {k: v.clone() for k, v in R.items()}
        for k in R:
            R[k][0, 0, 20:, 3:15] = torch.finfo(torch.float32).min
        ref = C.body(input_ids=ids[None], attention_mask=R).last_hidden_state[0, 20:]
        assert _maxdiff(h, ref) <= TOL
        slot = C.body(input_ids=ids[None, 20:], past_key_values=M.make_cache(sp.kv), position_ids=torch.arange(20, 25)[None],
                      use_cache=True).last_hidden_state[0]
        assert _maxdiff(slot, ref) > 1e-2                               # slot masks: wrong for sparse prefixes


def test_gemma4_softcap_and_k_eq_v():
    """Gemma 4 (tiny): C.lm_head applies the final soft-cap (|logit| <= 30, == the model's own forward); on K = V
    (full) layers the cached V is RMSNorm(k_proj(h)) without scale and K = RoPE(k_norm.weight * V)."""
    C = ctx("gemma4_26b_a4b")
    ids = M.DB().chat_ids(C.tok, "Hello there, how are you?")
    with torch.no_grad():
        full = C.model(input_ids=ids[None]).logits[0]
        h = C.body(input_ids=ids[None]).last_hidden_state[0]
        assert _maxdiff(M.logits(C, h), full) <= 1e-4 and float(full.abs().max()) <= 30.0
        st, _, _ = M.harvest_kv(C, ids)
        for l, L in enumerate(C.layout):
            attn = C.blocks[l].self_attn
            assert (attn.v_proj is None) == L.k_eq_v
            if L.k_eq_v:
                K, V = st.kv[l]
                Kre = V * attn.k_norm.weight[None, None, None, :]
                pos = torch.arange(len(ids))[None]
                cos, sin = M._rotary_call(C, Kre, pos, l)
                assert _maxdiff(M._apply_rope(C, l, Kre, cos, sin), K) <= 1e-4


@pytest.mark.parametrize("key", TINY)
def test_session_pieces(key):
    """Multi-turn session pieces tokenise as the whole conversation; the head is dream_bind's template head."""
    C = ctx(key)
    msgs = [{"role": "user", "content": "I just got back from Lisbon."}, {"role": "assistant", "content": "How was it?"},
            {"role": "user", "content": "Great, the pasteis de nata were amazing."}]
    H, T, Q = M.session_pieces(C.tok, msgs, "Where did I travel?")
    assert H == C.head and len(T) == 3 and all(T)
    with pytest.raises(AssertionError):
        M.session_pieces(C.tok, [{"role": "system", "content": "x"}] + msgs)


@pytest.mark.skipif(not HAVE_GEMMA4, reason="Gemma 4 classes need transformers >= 5.5 (cluster)")
def test_gemma4_text_only_loader(tmp_path):
    """load_device_map on a tiny multimodal Gemma 4 checkpoint (Gemma4ForConditionalGeneration, saved): only the text
    decoder is built (key_mapping model.language_model.* -> model.*), the vision tower is skipped (and reported),
    nothing is missing, the head is tied, and the logits equal the full model's."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
    tc = M.tiny_gemma4_config(300)
    vis = dict(hidden_size=32, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
               head_dim=16, global_head_dim=16, patch_size=4, position_embedding_size=64, pooling_kernel_size=2)
    torch.manual_seed(0)
    full = Gemma4ForConditionalGeneration(Gemma4Config(text_config=tc.to_dict(), vision_config=vis, audio_config=None))
    full = full.float().eval()
    full.save_pretrained(tmp_path)
    m, info = M.load_device_map(dict(M.SPECS["gemma4_26b_a4b"], hf=str(tmp_path)), "cpu", "eager")
    assert type(m).__name__ == "Gemma4ForCausalLM" and "model.vision_tower" in info["skipped_prefixes"]
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    ids = torch.randint(5, 300, (1, 12))
    with torch.no_grad():
        assert _maxdiff(full(input_ids=ids).logits, m(input_ids=ids).logits) <= 1e-4


@pytest.mark.cluster
def test_gemma4_checkpoint_keys_map_to_text_model():
    """Cluster (the real checkpoint index in the HF cache): every model.language_model.* key of
    google/gemma-4-26B-A4B-it maps onto a parameter of the text model built from its text config (meta device, no
    weights), every text parameter is in the checkpoint, and the rest are vision-tower keys."""
    if not HAVE_GEMMA4:
        pytest.skip("transformers 4.x")
    hf = M.SPECS["gemma4_26b_a4b"]["hf"]
    hits = []
    for base in [os.path.expanduser("~/.cache/huggingface"), os.environ.get("HF_HOME", "")]:
        hits += glob.glob(f"{base}/hub/models--{hf.replace('/', '--')}/snapshots/*/model.safetensors.index.json")
    if not hits:
        pytest.skip("no Gemma 4 checkpoint index in the HF cache")
    from transformers import AutoConfig, AutoModelForCausalLM
    keys = set(json.load(open(hits[0]))["weight_map"])
    tc = AutoConfig.from_pretrained(hf, local_files_only=True).text_config
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(tc)
    names = set(m.state_dict())
    mapped = {"model." + k[len("model.language_model."):] for k in keys if k.startswith("model.language_model.")}
    assert mapped <= names and names - mapped <= {"lm_head.weight"}
    rest = {k for k in keys if not k.startswith("model.language_model.")}
    assert all(any(k.startswith(p) for p in M.SKIP_PREFIXES) for k in rest)


# ================================================================== benchmarks (small hand-written fixtures)

SCH = _load("test_dc_schema", HERE / "benchmarks" / "schema.py")
LME = _load("test_dc_lme", HERE / "benchmarks" / "longmemeval.py")
LOC = _load("test_dc_locomo", HERE / "benchmarks" / "locomo.py")
PMB = _load("test_dc_personamem", HERE / "benchmarks" / "personamem.py")
BEA = _load("test_dc_beam", HERE / "benchmarks" / "beam.py")


def _lme_raw():
    s = lambda c, h=False: {"role": "user", "content": c, "has_answer": h}
    a = lambda c: {"role": "assistant", "content": c, "has_answer": False}
    return [
        {"question_id": "q1", "question_type": "single-session-user", "question": "What is my dog's name?", "answer": "Max",
         "question_date": "2023/05/02 (Tue) 10:00", "haystack_dates": ["2023/05/01 (Mon) 18:00", "2023/04/30 (Sun) 09:00"],
         "haystack_session_ids": ["sB", "sA"], "answer_session_ids": ["sB"],
         "haystack_sessions": [[s("I adopted a beagle named Max.", True), a("Congrats!")], [s("Plan a trip?"), a("Sure.")]]},
        {"question_id": "q2_abs", "question_type": "multi-session", "question": "What is my cat's name?",
         "answer": "You did not mention this information.", "question_date": "2023/05/02 (Tue) 10:00",
         "haystack_dates": ["2023/05/01 (Mon) 18:00"], "haystack_session_ids": ["sC"], "answer_session_ids": ["sC"],
         "haystack_sessions": [[s("I like dogs."), a("Nice."), s("   ")]]},
        {"question_id": "q3", "question_type": "temporal-reasoning", "question": "How many days?", "answer": 3,
         "question_date": "2023/05/02 (Tue) 10:00", "haystack_dates": ["2023/05/01 (Mon) 18:00"] * 2,
         "haystack_session_ids": ["sD", "sD"], "answer_session_ids": ["sD"],
         "haystack_sessions": [[s("Day one.", True)], [s("Day one again.")]]}]


def test_longmemeval_adapter():
    b = LME.convert(_lme_raw(), "fixture")
    assert SCH.validate(b) == []
    c = b.contexts["q1"]
    assert [s.session_id for s in c.sessions] == ["q1/sA", "q1/sB"]                 # time order
    it = {x.item_id: x for x in b.items}
    assert it["q1"].evidence_turns == ["q1/sB:0"] and it["q1"].evidence_sessions == ["q1/sB"]
    assert it["q1"].answers == ["Max"] and it["q1"].question_date.startswith("2023/05/02") and not it["q1"].abstention
    assert it["q2_abs"].abstention and it["q2_abs"].evidence_turns == [] and b.meta["empty_turns_dropped"] == 1
    assert it["q3"].answers == ["3"] and [s.session_id for s in b.contexts["q3"].sessions] == ["q3/sD", "q3/sD#2"]
    assert it["q3"].evidence_sessions == ["q3/sD", "q3/sD#2"] and b.meta["repeated_sessions_renamed"] == 1
    assert LME.single_turn_ids(b) == ["q1", "q3"] and LME.date_key("2023/04/10 (Mon) 17:50") == "2023/04/10 17:50"
    rt = SCH.from_json(json.loads(json.dumps(SCH.to_json(b))))
    assert SCH.to_json(rt) == SCH.to_json(b)
    raw = [{"question_id": f"{t}{i}", "question_type": t} for t in LME.TYPES for i in range(30)]
    ids = LME.stratified_ids(raw, per_type=20, seed=7)
    assert len(ids) == 120 and ids == LME.stratified_ids(raw, per_type=20, seed=7)
    assert all(sum(x.startswith(t) for x in ids) == 20 for t in LME.TYPES)


def test_longmemeval_s120_manifest():
    """The committed manifest of the S sample: 120 items, 20 per type, the pinned rule and seed."""
    m = json.load(open(REPO / "data" / "longmemeval" / "manifest_s120.json"))
    assert m["counts"]["items"] == 120 and set(m["counts"]["by_category"].values()) == {20}
    assert m["sample"]["seed"] == 7 and m["sample"]["per_type"] == 20 and len(m["item_ids"]) == 120
    assert m["raw"]["sha256"] == LME.SHA256["s"]


def _locomo_raw():
    return [{"sample_id": "conv-x", "conversation": {
        "speaker_a": "Ann", "speaker_b": "Bob", "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [{"speaker": "Ann", "dia_id": "D1:1", "text": "I painted a sunrise."},
                      {"speaker": "Bob", "dia_id": "D1:2", "text": "Lovely!", "img_url": ["u"], "blip_caption": "a photo of a dog"}],
        "session_2_date_time": "9:00 am on 9 May, 2023", "session_3_date_time": "never happened"},
        "qa": [{"question": "What did Ann paint?", "answer": "A sunrise", "evidence": ["D1:1"], "category": 4},
               {"question": "When?", "answer": 2023, "evidence": ["D1:1; D1:2", "D9:9"], "category": 2},
               {"question": "What did Bob paint?", "adversarial_answer": "a sunrise", "evidence": ["D1:1"], "category": 5}]}]


def test_locomo_adapter_and_category_map():
    assert LOC.CATEGORIES == {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
    b = LOC.convert(_locomo_raw())
    assert SCH.validate(b) == []
    t = b.contexts["conv-x"].turns()
    assert [x.role for x in t] == ["user", "assistant"] and t[0].text == "Ann: I painted a sunrise."
    assert t[1].text == "Bob: Lovely! [shared an image: a photo of a dog]" and t[1].date == "1:56 pm on 8 May, 2023"
    assert len(b.contexts["conv-x"].sessions) == 1                                   # date-only sessions skipped
    q = b.items
    assert q[0].category == "single-hop" and q[0].category_code == 4 and q[0].answers == ["A sunrise"]
    assert q[1].evidence_turns == ["conv-x:D1:1", "conv-x:D1:2"] and q[1].meta["evidence_dropped"] == ["D9:9"]
    assert q[1].answers == ["2023"] and q[1].evidence_sessions == ["conv-x:S1"]
    assert q[2].abstention and q[2].answers == [] and q[2].meta["adversarial_answer"] == "a sunrise"


def _personamem_raw():
    ctx = {"abcdef1234567890": [
        {"role": "system", "content": "Persona: Kai."}, {"role": "user", "content": "User: I love jazz."},
        {"role": "assistant", "content": "Assistant: Nice."}, {"role": "system", "content": "Persona: Kai."},
        {"role": "user", "content": "User: Now I prefer techno."}, {"role": "assistant", "content": "Assistant: Cool."},
        {"role": "user", "content": "User: Later message."}]}
    head = ("persona_id,question_id,question_type,topic,context_length_in_tokens,context_length_in_letters,"
            "distance_to_ref_in_blocks,distance_to_ref_in_tokens,num_irrelevant_tokens,distance_to_ref_proportion_in_context,"
            "user_question_or_message,correct_answer,all_options,shared_context_id,end_index_in_shared_context\n")
    rows = ('0,qa,track_full_preference_evolution,music,100,500,1,20,0,20%,What music now?,(b),'
            '"[""(a) Jazz"", ""(b) Techno""]",abcdef1234567890,6\n'
            "0,qb,recall_user_shared_facts,music,100,500,2,80,0,80%,What did I love?,(a),"
            "\"['(a) Jazz', '(b) Rock']\",abcdef1234567890,6\n")
    return head + rows, "\n".join(json.dumps({k: v}) for k, v in ctx.items())


def test_personamem_adapter():
    q, c = _personamem_raw()
    b = PMB.convert(q, c)
    assert SCH.validate(b) == []
    ctx = b.contexts["abcdef1234567890"]
    assert [len(s.turns) for s in ctx.sessions] == [3, 4] and ctx.sessions[0].turns[0].role == "system"
    assert ctx.sessions[0].turns[1].text == "I love jazz." and b.meta["role_prefix_stripped"] == 5
    a, bb = b.items
    assert a.options == ["Jazz", "Techno"] and a.correct_option == 1 and a.answers == ["Techno"] and a.context_turns == 6
    assert a.evidence_sessions == ["abcdef123456:s1"] and bb.evidence_sessions == ["abcdef123456:s0"]
    assert bb.options == ["Jazz", "Rock"] and bb.correct_option == 0                 # the Python-literal options row
    assert [t.turn_id for t in b.item_turns(a)][-1] == "abcdef123456:5"              # context[:end_index]


def _beam_rows():
    chat = [[{"content": "Plan sprints. ->-> 1,1", "id": 0, "index": "1,1", "question_type": "main_question",
              "role": "user", "time_anchor": "March-15-2024"},
             {"content": "Sure.", "id": 1, "index": None, "question_type": None, "role": "assistant", "time_anchor": None}],
            [{"content": "Sprint 1 ends March 29.", "id": 2, "index": "2,1", "question_type": "main_question",
              "role": "user", "time_anchor": "April-05-2024"},
             {"content": "Noted.", "id": 3, "index": None, "question_type": None, "role": "assistant", "time_anchor": None}]]
    pq = {"information_extraction": [{"question": "When does sprint 1 end?", "answer": "March 29",
                                      "source_chat_ids": [2], "rubric": ["LLM response should state: March 29"]}],
          "knowledge_update": [{"question": "Q?", "answer": "A", "source_chat_ids": {"original_info": [0], "updated_info": [2, 99]},
                                "rubric": ["r1", "r2"]}],
          "abstention": [{"question": "What is my cat?", "ideal_response": "No information.", "rubric": ["no info"]}],
          "preference_following": [{"question": "Suggest tools?", "source_chat_ids": [0], "rubric": ["light tools"]}],
          "summarization": [{"question": "Summarise", "rubric": ["x"]}],
          "instruction_following": [{"question": "Follow", "rubric": ["x"]}]}
    return [{"conversation_id": "7", "conversation_seed": {"category": "Coding", "title": "T"}, "chat": chat,
             "probing_questions": repr(pq)}]


def test_beam_adapter():
    b = BEA.convert(_beam_rows())
    assert SCH.validate(b) == []
    t = b.contexts["beam7"].turns()
    assert t[0].text == "Plan sprints." and t[0].meta["marker"] == "1,1" and t[0].date == "March-15-2024"
    cats = [it.category for it in b.items]
    assert "summarization" not in cats and "instruction_following" not in cats and len(b.items) == 4
    it = {x.category: x for x in b.items}
    assert it["information_extraction"].evidence_turns == ["beam7:2"] and it["information_extraction"].answers == ["March 29"]
    assert it["knowledge_update"].evidence_turns == ["beam7:0", "beam7:2"]
    assert it["knowledge_update"].meta["evidence_dropped"] == [99]
    assert it["abstention"].abstention and it["abstention"].answers == ["No information."]
    assert it["preference_following"].answers == [] and it["preference_following"].rubric == ["light tools"]


def test_schema_validate_catches_problems():
    b = LOC.convert(_locomo_raw())
    b.items[0].evidence_turns.append("conv-x:D7:7")
    b.items[1].options, b.items[1].correct_option = ["a"], 3
    b.contexts["conv-x"].sessions[0].turns[0].role = "narrator"
    bad = SCH.validate(b)
    assert any("D7:7" in x for x in bad) and any("correct_option" in x for x in bad) and any("narrator" in x for x in bad)


@pytest.mark.parametrize("name,split,n", [("longmemeval", "oracle", 500), ("longmemeval", "s120", 120),
                                          ("locomo", "locomo10", 1986), ("personamem", "32k", 589), ("beam", "100K", 320)])
def test_processed_splits_match_manifests(name, split, n):
    """Where the processed splits exist (after the prep scripts): sha256 == the committed manifest, schema-valid."""
    p = REPO / "data" / "external" / name / "processed" / f"{split}.json"
    if not p.exists():
        pytest.skip(f"{p} not built")
    m = json.load(open(REPO / "data" / name / f"manifest_{split}.json"))
    assert SCH.sha256(p) == m["processed_sha256"] and m["counts"]["items"] == n
    b = SCH.load(p)
    assert len(b.items) == n and [it.item_id for it in b.items] == m["item_ids"]


# ================================================================== judges

JDG = _load("test_dc_judges", HERE / "judges.py")


def test_judge_prompt_hashes():
    """Every official prompt, byte for byte (sha256 of the strings extracted with ast from: LongMemEval
    evaluate_qa.py at 9e0b455 (file sha256 ecce9c4c..., = dream_select's d6dc8b5); Mem0 llm_judge.py at aae5989;
    BEAM src/prompts.py at b2da22e)."""
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()
    want = {"default": "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc",
            "temporal-reasoning": "8d33a5fdd83afeeb4592454a965eab43d1fcb2dedc042d1d3892f4254be6c273",
            "knowledge-update": "183a9b3a6197ec620940f610cdc1207201ec98c1113dd633ea685cfc322fafac",
            "single-session-preference": "741ee3bcbea7ff5e8ed359acef61d2f8ded3de021bbcff6ee13de455f2e2aa9b",
            "abstention": "5c0b365a1e1d06db36377c735432b56e122ca3c428f89faf61d43a0d5a7e050b"}
    assert {k: h(v) for k, v in JDG.LME_TEMPLATES.items()} == want
    assert JDG.LME_TEMPLATES["default"] == JDG.AG.LME_TEMPLATE                     # dream_select's pinned template
    assert h(JDG.J_PROMPT) == "62395dd312a631dfd9355026a0b69cc936018274c3198b6365b5c2a5c9bca9e0"
    assert h(JDG.BEAM_PROMPT) == "593373c642a288a7b590577d8a8fc92c3f9a2b70e2f64ad6e59a040a6c56b7f5"
    assert set(JDG.LME_TASK) == set(LME.TYPES)


def test_judge_prompt_filling():
    p = JDG.lme_prompt("knowledge-update", "Where do I live?", "Paris", "You moved to Paris.")
    assert "updated answer is the required answer.\n\nQuestion: Where do I live?\n\nCorrect Answer: Paris" in p
    a = JDG.lme_prompt("multi-session", "Cat?", "You did not mention it.", "", abstention=True)
    assert a.startswith("I will give you an unanswerable question") and "Model Response: (empty reply)" in a
    j = JDG.j_prompt("What did I get?", "A shell necklace", "A necklace made of shells.")
    assert "Question: What did I get?\nGold answer: A shell necklace\nGenerated answer: A necklace made of shells." in j
    bm = JDG.beam_prompt("LLM response should state: March 29", "It ends on March 29.")
    assert "- RUBRIC CRITERION (what to check): LLM response should state: March 29\n- RESPONSE TO EVALUATE: It ends" in bm
    assert "<rubric_item>" not in bm and "<llm_response>" not in bm


@pytest.mark.parametrize("text,label,ok", [
    ('{"label": "CORRECT"}', True, True), ('Reasoning... ```json\n{"label": "WRONG"}\n```', False, True),
    ("The answer is right. CORRECT", True, True), ("CORRECT or WRONG", None, False), ("", None, False),
    ('<think>hmm WRONG</think>{"label": "correct"}', True, True)])
def test_parse_j(text, label, ok):
    r = JDG.parse_j(text)
    assert r["label"] == label and r["ok"] == ok


@pytest.mark.parametrize("text,score,ok", [
    ('{"score": 1.0, "reason": "x"}', 1.0, True), ('```json\n{"score": 0.5, "reason": "partly"}\n```', 0.5, True),
    ('{"score": "0.0", "reason": "no"}', 0.0, True), ('{"score": [1.0], "reason": "y"}', 1.0, True),
    ('{"score": 0.7}', None, False), ("score: high", None, False),
    ('Here: {"score": 1.0, "reason": "has a } brace"}', 1.0, True)])
def test_parse_beam(text, score, ok):
    r = JDG.parse_beam(text)
    assert r["score"] == score and r["ok"] == ok


def test_locomo_f1_official():
    """LoCoMo's scorer on hand-checked cases (normalisation removes punctuation, articles and 'and'; Porter stems)."""
    assert JDG.normalize_answer("The Sunrise, and a Dog!") == "sunrise dog"
    assert JDG.locomo_f1(4, "Adoption agencies", "adoption agency") == 1.0           # stems: agenc
    assert abs(JDG.locomo_f1(4, "She researched adoption agencies", "Adoption agencies") - 2 / 3) < 1e-12
    assert JDG.locomo_f1(1, "Summer Sounds", "Summer Sounds, Matt Patterson") == 0.5
    assert JDG.locomo_f1(3, "psychology", "Psychology; counseling") == 1.0
    assert JDG.locomo_f1(5, "That is not mentioned in the conversation.", "x") == 1.0
    assert JDG.locomo_f1(2, "", "7 May 2023") == 0.0


def test_porter_matches_nltk():
    """The vendored stemmer == nltk's PorterStemmer() on every word of LoCoMo's answers (where nltk is installed)."""
    try:
        from nltk.stem import PorterStemmer
    except ImportError:
        pytest.skip("nltk not installed (the cluster env): the vendored copy is the reference there")
    words = set()
    p = REPO / "data" / "external" / "locomo" / "processed" / "locomo10.json"
    if p.exists():
        for it in SCH.load(p).items:
            for a in it.answers:
                words.update(JDG.normalize_answer(a).split())
    words.update("running adoption agencies generously caresses ponies relational happiness was the hopping".split())
    ref = PorterStemmer()
    assert all(JDG.PORTER.stem(w) == ref.stem(w) for w in words)


def test_panel_label_and_same_family_exclusion():
    v = {"orig": True, "gptoss": False, "gemma4": False}
    assert JDG.panel_label(v) == (False, None)
    assert JDG.panel_label(v, subject_family="Google") == (False, True)              # orig vs gptoss tie -> orig
    assert JDG.panel_label({"orig": False, "gptoss": True, "gemma4": True}, "Google") == (True, False)
    assert JDG.panel_label({"orig": None, "gptoss": True, "gemma4": None}) == (True, None)
    assert JDG.panel_label({"orig": None, "gptoss": None, "gemma4": None}) == (None, None)
    assert JDG.mc_pick([-6.0, -1.0, -2.0], [3, 1, 1]) == (1, 1) and JDG.mc_pick([-6.0, -4.0], [6, 2]) == (1, 0)


def test_judge_panel_runner(tmp_path):
    """judges.run_tasks with a fake Ollama: kinds parsed by their own parser, one retry on an unparsed reply, records
    appended, parse rates per kind."""
    replies = {"lme": ["Yes"], "locomo_j": ["hmm", '{"label": "WRONG"}'], "beam": ['{"score": 0.5, "reason": "r"}']}
    calls = {}

    def post(url, tag, prompt, schema, seed, num_predict=None, fmt=None):
        k = prompt.split("|")[0]
        i = calls.get(prompt, 0)
        calls[prompt] = i + 1
        return replies[k][min(i, len(replies[k]) - 1)], {}
    tasks = [{"h": JDG.phash(f"{k}|x"), "kind": k, "prompt": f"{k}|x", "keys": [k]} for k in replies]
    res = JDG.run_tasks("http://x", "gptoss", tasks, tmp_path / "g.jsonl", workers=2, post=post, log=lambda *_: None)
    recs = {json.loads(l)["kind"]: json.loads(l) for l in open(tmp_path / "g.jsonl")}
    assert recs["lme"]["label"] is True and recs["beam"]["score"] == 0.5
    assert recs["locomo_j"]["label"] is False and recs["locomo_j"]["attempts"] == 2
    assert res["parse_rate_by_kind"] == {"lme": 1.0, "locomo_j": 1.0, "beam": 1.0}
    again = JDG.run_tasks("http://x", "gptoss", tasks, tmp_path / "g.jsonl", workers=2, post=post, log=lambda *_: None)
    assert again["done_now"] == 0


# ================================================================== retrieval

RET = _load("test_dc_retrieval", HERE / "retrieval.py")


def test_bm25_hand_computed():
    """Okapi BM25 (k1 1.5, b 0.75, Lucene idf) against a hand computation on three documents."""
    docs = ["cat sat mat", "dog dog park", "cat dog"]
    bm = RET.BM25(docs, stem=False)
    N, avg = 3, 8 / 3
    idf = lambda n: math.log(1 + (N - n + 0.5) / (n + 0.5))
    tf = lambda f, L: f * 2.5 / (f + 1.5 * (1 - 0.75 + 0.75 * L / avg))
    s = bm.scores("dog")
    assert abs(s[1] - idf(2) * tf(2, 3)) < 1e-12 and s[0] == 0.0 and abs(s[2] - idf(2) * tf(1, 2)) < 1e-12
    s2 = bm.scores("cat park")
    assert abs(s2[1] - idf(1) * tf(1, 3)) < 1e-12 and abs(s2[0] - idf(2) * tf(1, 3)) < 1e-12
    assert bm.top("dog", 2) == [1, 2] and bm.top("zebra", 2) == [0, 1]                 # ties -> turn order
    assert RET.tokenize("The Dogs were RUNNING!") == ["dog", "run"]


def test_bm25_retrieve_and_recall():
    b = LOC.convert(_locomo_raw())
    it = b.items[0]
    got = RET.retrieve(b, it, 1)
    assert got == ["conv-x:D1:1"] and RET.item_recall(b, it, got) == 1.0
    assert RET.recall(["a", "b"], ["b", "c"]) == 0.5 and RET.recall(["a"], []) is None
    q, c = _personamem_raw()
    pb = PMB.convert(q, c)
    r = RET.retrieve(pb, pb.items[0], 1)
    assert RET.item_recall(pb, pb.items[0], r) in (0.0, 1.0)                            # session-level evidence
    tab = RET.recall_table(b, ms=(1, 2))
    assert set(tab) == {1, 2} and "all" in tab[1]


def test_jobs_specs_and_run_cli():
    JB = _load("test_dc_jobs", HERE / "jobs.py")
    assert JB.parse_specs("smoke|/x/q15|;judge-tasks|/x/j|--tasks-from /x/s") == \
        [("smoke", "/x/q15", []), ("judge-tasks", "/x/j", ["--tasks-from", "/x/s"])]
    src = (HERE / "run.py").read_text()
    for st in ("smoke", "judge-tasks", "judge-local", "judge-summary"):
        assert f'"{st}"' in src


@pytest.mark.parametrize("key", ["gemma4_26b_a4b", "granite42_30b"])
def test_thinking_off_templates(key):
    """Gemma 4 and Granite 4.2 (thinking on by default): the wrapped template ends with the thinking-off tail; the
    turn pieces and session pieces tokenise as the whole conversation."""
    tok = _tok(M.SPECS[key]["hf"])
    if tok is None:
        pytest.skip("tokenizer not in the local cache")
    if not tok.chat_template:
        pytest.skip("this transformers does not read chat_template.jinja (4.x)")
    if not getattr(tok, "_dc_thinking_off", False):
        M.thinking_off(tok, M.SPECS[key]["chat"]["thinking"])
        tok._dc_thinking_off = True
    s = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    assert s.endswith(M.THINKING_TAILS[M.SPECS[key]["chat"]["thinking"]])
    P = M.DB().turn_pieces(tok, "My dog's name is Pepper.")
    H, T, Q = M.session_pieces(tok, [{"role": "user", "content": "I went to Lisbon."},
                                     {"role": "assistant", "content": "Nice!"}], "Where did I go?")
    assert H == P.H and len(T) == 2
