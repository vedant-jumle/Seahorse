"""dream_bind stage 0 (experiments/dream_bind/{bind,run}.py): layout assertion, question-subject span
finding, the residual-patching and attention-knockout identities on a tiny random Qwen2 (eager, CPU), the
bootstrap CI and the pre-registered verdict rules, and the config's items / aliases."""

import importlib.util
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dream_bind"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bind = _load("test_dream_bind_bind", EXP / "bind.py")
run = _load("test_dream_bind_run", EXP / "run.py")
CFG = yaml.safe_load(open(EXP / "config.yaml"))
FACTS = {f["id"]: f for f in yaml.safe_load(open(ROOT / "data" / "bench_v1" / "facts.yaml"))["facts"]}


class WordTok:
    """A toy tokenizer with BPE-like leading-space tokens (" dog", "'s", "?")."""

    def __init__(self):
        self.vocab = {}

    def encode(self, text):
        ids = []
        for t in re.findall(r"\s?\w+|'\w+|\s?[^\w\s']|\s", text):
            ids.append(self.vocab.setdefault(t, len(self.vocab)))
        return ids

    def decode(self, ids, **_):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[int(i)] for i in ids)


# ------------------------------------------------------------------ layout + spans


def test_attribute_position():
    assert bind.attribute_position([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    with pytest.raises(AssertionError):
        bind.attribute_position([1, 2, 3], [1, 2, 3, 4])      # lengths differ
    with pytest.raises(AssertionError):
        bind.attribute_position([1, 2, 3, 4], [1, 7, 9, 4])   # two positions differ
    with pytest.raises(AssertionError):
        bind.attribute_position([1, 2, 3], [1, 2, 3])          # none differ


def test_find_alias_and_span_in_question_only():
    tok = WordTok()
    user = "My dog's name is Pepper. What's my dog's name?"
    text = "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\nYour dog's name is"
    ids = tok.encode(text)
    spans, dec = bind.char_spans(tok, ids)
    assert dec == text and spans[-1][1] == len(text)
    lay = bind.question_layout(tok, ids, user, "My dog's name is Pepper.", ["my dog", "your dog", "dog"])
    toks = [tok.decode([ids[i]]) for i in lay["subj"]]
    assert toks == [" my", " dog"] and lay["alias"] == "my dog"
    assert tok.decode([ids[lay["q0"]]]) == " What"
    pepper = ids.index(tok.vocab[" Pepper"])
    assert pepper < lay["q0"] <= min(lay["subj"])          # never the experience's own "My dog"
    # multi-token subject: the last token is "friend"
    user2 = "My best friend's name is Omar. What's my best friend's name?"
    ids2 = tok.encode("<s>" + user2 + "</s>Your best friend's name is")
    lay2 = bind.question_layout(tok, ids2, user2, "My best friend's name is Omar.", ["my best friend", "best friend"])
    assert [tok.decode([ids2[i]]) for i in lay2["subj"]] == [" my", " best", " friend"]
    # whole words only, first alias in list order, case-insensitive
    assert bind.find_alias("What's my doggy's name?", ["dog"]) is None
    assert bind.find_alias("Is MY DOG ok?", ["my dog", "dog"])[0] == "my dog"
    assert bind.find_alias("Where did I grow up?", ["my hometown", "the place where I grew up"]) is None
    lay3 = bind.question_layout(tok, tok.encode("I grew up in Leeds. Where did I grow up?"),
                                "I grew up in Leeds. Where did I grow up?", "I grew up in Leeds.", ["my hometown"])
    assert lay3["subj"] == [] and lay3["alias"] is None


def test_cells_and_windows():
    assert bind.windows(8, 5) == [(0, [0, 1, 2, 3, 4]), (1, [1, 2, 3, 4, 5]), (2, [2, 3, 4, 5, 6]), (3, [3, 4, 5, 6, 7])]
    c = bind.cells(28, 5)
    assert len(c) == 28 + 24 and c[0] == ("block", 0, [0]) and c[-1] == ("band", 23, [23, 24, 25, 26, 27])


# ---------------------------------------------------------- hooks on a tiny model


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).eval()
    assert model.config._attn_implementation == "eager"
    src = torch.tensor([1, 2, 10, 11, 12, 13, 20, 3, 4, 5])
    tgt = src.clone()
    tgt[3] = 40                                               # the "attribute" position differs
    C = SimpleNamespace(model=model, body=model.model, lm_head=model.lm_head, blocks=model.model.layers, n=6, mb=4,
                        dev="cpu")
    return C, src, tgt


def test_patch_identities(tiny):
    C, src, tgt = tiny
    T = len(src)
    with torch.no_grad():
        st, lg_src = run.source_run(C, src)
        lg_tgt = run.last_logits(C, tgt[None])[0]
        rows = [{}] + [{"patch": [([l], list(range(T)))]} for l in range(C.n)] + [{"patch": [([2], [8])]}]
        lg = run.run_rows(C, tgt, rows, st)                   # 8 rows in batches of 4
    assert torch.allclose(lg[0], lg_tgt, atol=1e-5)           # nothing patched == target run
    for l in range(C.n):                                      # every position at any block == source run
        assert torch.allclose(lg[1 + l], lg_src, atol=1e-5)
    assert not torch.allclose(lg_src, lg_tgt, atol=1e-3)      # the two runs differ ...
    assert not torch.allclose(lg[-1], lg_tgt, atol=1e-6)      # ... and a one-position patch moves the target run
    # the attribute is the only differing token and positions before it are identical: once its state is
    # the source's after block 0, its states after blocks 1, 2 are the source's too, so the band [0, 1, 2]
    # equals the single block 0 (but not the source run: later positions read its block-0 K/V from the target)
    with torch.no_grad():
        band = run.run_rows(C, tgt, [{"patch": [([0, 1, 2], [3])]}, {"patch": [([0], [3])]}], st)
    assert torch.allclose(band[0], band[1], atol=1e-5)
    assert not torch.allclose(band[1], lg_src, atol=1e-4)


def test_knockout_identities(tiny):
    C, src, _ = tiny
    T, ans = len(src), len(src) - 1
    allL = list(range(C.n))
    with torch.no_grad():
        plain = run.last_logits(C, src[None])[0]
        rows = [{"hook_all": True},                                        # empty knockout
                {"block": [(allL, [ans], list(range(ans)))]},               # attend to itself only
                {"block": [([1, 2, 3, 4, 5], [ans], [3])]},                 # one key, a window
                {"block": [([0], [ans], [])]}]                              # empty key list
        lg = run.run_rows(C, src, rows)
        alone = run.last_logits(C, src[-1:][None])[0]
    assert torch.allclose(lg[0], plain, atol=1e-5)
    assert torch.allclose(lg[1], alone, atol=1e-5)                          # the mask edit takes effect
    assert not torch.allclose(lg[2], plain, atol=1e-6)
    assert torch.allclose(lg[3], plain, atol=1e-5)
    # rows of one batch do not leak into each other
    with torch.no_grad():
        solo = run.run_rows(C, src, [{"block": [([1, 2, 3, 4, 5], [ans], [3])]}])
    assert torch.allclose(solo[0], lg[2], atol=1e-5)


def test_block_mask_float_and_bool():
    m = torch.zeros(1, 1, 4, 4)
    out = bind.block_mask(m, [(1, [3], [0, 2])], 2)
    assert out.shape == (2, 1, 4, 4) and out[0].abs().sum() == 0
    assert out[1, 0, 3, 0] == torch.finfo(torch.float32).min and out[1, 0, 3, 1] == 0
    b = torch.ones(2, 1, 4, 4, dtype=torch.bool)
    ob = bind.block_mask(b, [(0, [2, 3], [1])], 2)
    assert not ob[0, 0, 2, 1] and not ob[0, 0, 3, 1] and ob[1].all() and b.all()   # input untouched
    with pytest.raises(AssertionError):
        bind.block_mask(None, [], 1)


# ------------------------------------------------------------- statistics + rules


def test_share_and_ci():
    assert bind.share(1.0, 0.0, 4.0) == 0.25 and math.isnan(bind.share(1.0, 2.0, 2.0))
    s = bind.ci({"a": 0.5, "b": 0.5, "c": 0.5})
    assert (s["mean"], s["lo"], s["hi"], s["n"]) == (0.5, 0.5, 0.5, 3)
    s1, s2 = bind.ci({"a": 0.1, "b": 0.9, "c": 0.4}, 2000, 7), bind.ci({"a": 0.1, "b": 0.9, "c": 0.4}, 2000, 7)
    assert s1 == s2 and s1["lo"] < s1["mean"] < s1["hi"]           # seeded
    assert bind.ci({"a": float("nan"), "b": 1.0})["n"] == 1        # nan dropped


def _c(mean, lo=None, hi=None):
    return {"mean": mean, "lo": mean - 0.05 if lo is None else lo, "hi": mean + 0.05 if hi is None else hi, "n": 7}


def test_verdict_rules():
    subj_hi = {"block 3": _c(0.7, 0.4, 0.9), "block 4": _c(0.2)}
    subj_lo = {"block 3": _c(0.1), "block 4": _c(0.2)}
    attr_hi = {"block 1": _c(0.9)}
    attr_mid = {"block 1": _c(0.4)}
    # Qwen3.5-2B (patch only)
    assert bind.verdict(subj_hi, attr_hi)["label"] == "subject"
    assert bind.verdict(subj_lo, attr_hi)["label"] == "pointer-suspected"
    assert bind.verdict(subj_lo, attr_mid)["label"] == "in between"
    assert bind.verdict({"block 3": _c(0.6, 0.2, 0.9)}, attr_hi)["label"] == "in between"   # CI does not exclude 25%
    assert bind.verdict({"block 3": _c(0.2, 0.1, 0.55)}, attr_hi)["label"] == "in between"  # CI does not exclude 50%
    # Qwen2.5 (with knockout)
    ko_s, ko_a_lo, ko_a_hi = {"window 2": _c(0.6)}, {"window 2": _c(0.1)}, {"window 2": _c(0.7)}
    assert bind.verdict(subj_hi, attr_hi, ko_s, ko_a_lo)["label"] == "subject"
    assert bind.verdict(subj_hi, attr_hi, ko_s, ko_a_hi)["label"] == "in between"   # attribute knockout too costly
    assert bind.verdict(subj_lo, attr_hi, {"window 2": _c(0.1)}, ko_a_hi)["label"] == "pointer-suspected"
    assert bind.verdict(subj_lo, attr_hi, {"window 2": _c(0.1)}, ko_a_lo)["label"] == "in between"
    # sufficiency safeguard: subject alone >= 50% (CI excludes 25%) blocks the pointer flag
    suff = {"block 3": _c(0.7, 0.5, 0.9)}
    zero = {"mean": 0.02, "lo": -0.01, "hi": 0.05, "n": 7}
    v = bind.verdict(subj_lo, attr_hi, {"window 2": _c(0.1)}, ko_a_hi, suff=suff, suff_net={"block 3": _c(0.1)},
                     suff_sanity=zero)
    assert v["label"] == "in between" and v["suff_readout"] == "raw share" and "outvoted" in v["reasons"][0]
    # sanity not ~0 -> the net share is read (here too small to block the flag)
    v = bind.verdict(subj_lo, attr_hi, {"window 2": _c(0.1)}, ko_a_hi, suff=suff, suff_net={"block 3": _c(0.1)},
                     suff_sanity={"mean": 0.4, "lo": 0.3, "hi": 0.5, "n": 7})
    assert v["label"] == "pointer-suspected" and v["suff_readout"].startswith("net")
    assert bind.verdict(subj_hi, attr_hi, n_flagged=3)["label"] == "undetermined"
    assert bind.verdict(subj_hi, attr_hi, n_flagged=2)["label"] == "subject"
    nan = {"block 3": {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}}
    assert bind.verdict(nan, attr_hi)["label"] == "undetermined"


# ------------------------------------------------------------------------- config


def test_config_items_and_aliases():
    assert CFG["primary"] == ["dog_name", "cat_name", "sister_name", "brother_name", "grandmother_name",
                              "best_friend_name", "boss_name"]
    assert CFG["secondary"] == ["job", "hometown", "first_language", "university", "favourite_colour"]
    assert CFG["subjects"]["favourite_colour"]["no_site"] and CFG["dtype"] == "fp32"
    assert CFG["seeds"]["bootstrap"] == 7 and CFG["n_boot"] == 2000 and CFG["band"] == 5
    for iid in CFG["primary"]:                         # every primary question names its subject
        q = FACTS[iid]["probes"]["related"]
        assert bind.find_alias(q, CFG["subjects"][iid]["subject"]) is not None, iid
    for iid in CFG["primary"] + CFG["secondary"]:      # harvest follow-ups are the item's own, not the generic one
        assert set(CFG["subjects"][iid]["harvest_followups"]) <= set(FACTS[iid]["followups"][:2])
    items = run.load_items(CFG)
    assert [it["id"] for it in items] == CFG["primary"] + CFG["secondary"]
    job = next(it for it in items if it["id"] == "job")
    assert job["counter"] == "I work as a plumber." and job["counter_overridden"]
    assert " plumber" not in FACTS["job"]["measure"]["foils"]


def _qwen25_tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception:
        return None


def test_layouts_with_the_real_tokenizer():
    tok = _qwen25_tok()
    if tok is None:
        pytest.skip("Qwen2.5 tokenizer not in the local cache")
    C = SimpleNamespace(tok=tok)
    for it in run.load_items(CFG):
        P = run.build(C, it)                                   # asserts the layout
        if it["tier"] == "primary":
            assert P.subj and P.q0 > P.a
        assert P.info["subj_toks"][-1:] in ([], [" dog"], [" cat"], [" sister"], [" brother"], [" grandmother"],
                                            [" friend"], [" boss"], [" language"], [" university"], [" colour"])


# ======================================================== harvest + analytic (pure helpers)


def test_find_all_aliases_and_subject_sites():
    occ = bind.find_all_aliases("My dog likes my dog's toy and the dog bed.", ["my dog", "dog"])
    assert [(s, e) for s, e in occ] == [(0, 6), (13, 19), (34, 37)]          # "dog" inside "my dog" skipped
    tok = WordTok()
    user = "Is my dog called Pepper? Answer yes or no."
    ids = tok.encode("<s>user\n" + user + "</s>assistant\nYour dog's name is")
    S = bind.subject_sites(tok, ids, user, ["my dog", "your dog", "dog"], prefix="Your dog's name is")
    d = lambda xs: [tok.decode([ids[i]]) for i in xs]
    assert [d(s) for s in S["spans"]] == [[" my", " dog"]] and d(S["main"]) == [" dog"]
    assert d(S["plus1"]) == [" dog", " called"]
    assert [d(s) for s in S["prefix_spans"]] == [["\nYour", " dog"]] and d(S["gen"]) == [" dog", " dog"]
    assert d(S["wrongpos"]) == ["."]                                          # last non-subject user token
    assert not set(S["wrongpos"]) & set(S["plus1"])
    # user_from: the experience's own mention is skipped (H1 / H2 harvest)
    exp, fu = "My dog's name is Pepper.", "My dog is lovely."
    ids2 = tok.encode("<s>user\n" + exp + " " + fu + "</s>")
    S2 = bind.subject_sites(tok, ids2, exp + " " + fu, ["my dog"], user_from=len(exp) + 1)
    assert len(S2["main"]) == 1 and S2["main"][0] > ids2.index(tok.vocab[" Pepper"])
    assert bind.subject_sites(tok, ids2, exp + " " + fu, [])["main"] == []


def test_adding_hook(tiny):
    C, src, _ = tiny
    v = torch.randn(32)
    with torch.no_grad():
        st, lg = run.source_run(C, src)
        zero = run.run_rows(C, src, [{"add": [([2], [5], {2: torch.zeros(32)})]}])
        added = run.run_rows(C, src, [{"add": [([2], [5], {2: v})]}, {}])
        st2 = {2: st[2].clone()}
        st2[2][5] += v
        patched = run.run_rows(C, src, [{"patch": [([2], [5])]}], st2)
    assert torch.allclose(zero[0], lg, atol=1e-5)
    assert torch.allclose(added[0], patched[0], atol=1e-5)                  # add == patch with h + v
    assert torch.allclose(added[1], lg, atol=1e-5) and not torch.allclose(added[0], lg, atol=1e-4)


def test_norm_match_and_layer_sets():
    v, ref = torch.randn(4, 8), torch.randn(4, 8)
    m = bind.norm_match(v, ref)
    assert torch.allclose(m.norm(dim=-1), ref.norm(dim=-1), atol=1e-5)
    assert torch.allclose(torch.nn.functional.normalize(m, dim=-1), torch.nn.functional.normalize(v, dim=-1), atol=1e-5)
    core = _load("test_dream_bind_core_v1", ROOT / "experiments" / "core_v1" / "core.py")
    A = CFG["analytic"]
    sets = [[l] for l in A["singles"]] + [list(range(a, b + 1)) for a, b in A["bands"]]
    assert bind.map_sets(sets, 24, 24, core.map_layers) == sets                 # identity on the 2B
    q = bind.map_sets(sets, 24, 28, core.map_layers)
    assert [s[0] for s in q[:7]] == core.map_layers(A["singles"], 24, 28)
    assert all(len(s) >= 5 and s == list(range(s[0], s[-1] + 1)) for s in q[7:])
    assert bind.lset_label([4]) == "b4" and bind.lset_label([4, 5, 6]) == "b4-6"


def test_whitening_and_keys():
    torch.manual_seed(0)
    X = torch.randn(4000, 6) @ torch.randn(6, 6)
    X = X - X.mean(0)
    W = bind.whitening(X, r=6, eps=0.0)
    Y = X @ W
    assert torch.allclose(Y.T @ Y / len(Y), torch.eye(6), atol=1e-3)
    H, mu = torch.randn(5, 6), torch.zeros(6)
    user = torch.tensor([False, True, True, False, True])
    k = bind.pooled_keys(H, user, mu, W)
    want = torch.nn.functional.normalize(((H[1] + H[2]) / 2) @ W, dim=-1)
    assert torch.allclose(k[2], want, atol=1e-5) and torch.allclose(k[3], want, atol=1e-5)
    sk = bind.subject_key(H[1:3], mu, W)
    assert torch.allclose(sk, want, atol=1e-5)
    gk = torch.nn.functional.normalize(torch.randn(100, 6), dim=-1)
    thr = bind.gate_threshold(gk, sk[None], 0.95)
    assert abs((gk @ sk > thr).float().mean().item() - 0.05) <= 0.02


def test_ranks_pick_and_no_gain():
    assert bind.ranks([3.0, 1.0, 3.0, float("nan")]) == [1.5, 3.0, 1.5, 4.0]
    cells = []
    for v, off in (("H1", -1.0), ("H2", 0.5), ("H3", 1.0)):
        for i, ls in enumerate(["b4", "b6", "b8", "b10", "b4-8"]):
            for a in (0.5, 1, 2):
                cells.append({"variant": v, "lset": ls, "alpha": a, "dspec": off + 0.1 * i + 0.01 * a,
                              "dbal": off + 0.05 * i, "dlp": off + i})
    P = bind.analytic_pick(cells)
    assert len(P["pool"]) == 3 * 3 * 2 and len(P["picks"]) == 8
    for v in ("H1", "H2", "H3"):
        pv = [c for c in P["pool"] if c["variant"] == v]
        assert len({c["lset"] for c in pv}) == 3 and all(sum(c["lset"] == ls for c in pv) == 2 for ls in {c["lset"] for c in pv})
    assert {c["lset"] for c in P["pool"] if c["variant"] == "H2"} == {"b8", "b10", "b4-8"}  # the best three sets
    assert any(c["variant"] == "H1" for c in P["picks"])                       # H1 kept although it scores worst
    assert bind.no_gain({"mean": 0.3, "lo": -0.1, "hi": 0.7, "n": 7})
    assert not bind.no_gain({"mean": 0.3, "lo": 0.05, "hi": 0.7, "n": 7})
    assert bind.no_gain({"mean": -0.3, "lo": -0.5, "hi": -0.1, "n": 7}) and bind.no_gain(None)


def test_analytic_values():
    r = lambda pk, site, v0, v1, cons=None, var="H1": {"item": "a", "tier": "primary", "pk": pk, "consistent": cons,
                                                       "variant": None if site in ("nomem", "ctx") else var,
                                                       "lset": None if site in ("nomem", "ctx") else "b4",
                                                       "alpha": None if site in ("nomem", "ctx") else 1, "site": site,
                                                       "v0": v0, "v1": v1, "no_site": False}
    rows = [r("m", "nomem", -5.0, -3.0), r("m", "main", -2.0, -3.5),
            r("y0", "nomem", -1.0, -0.5, "Yes"), r("y0", "main", -0.4, -1.2, "Yes"),
            r("y1", "nomem", -2.0, -0.2, "No"), r("y1", "main", -1.0, -0.5, "No")]
    out = run.analytic_values(rows)
    o = out[("H1", "b4", 1, "main")]["a"]
    assert o["dlp"] == pytest.approx(3.0) and o["dspec"] == pytest.approx(1.5 - (-2.0))
    # Yes probe margin: (v0 - v1): nomem -0.5 -> main 0.8 (+1.3); No probe margin (v1 - v0): 1.8 -> 0.5 (-1.3)
    assert o["dmY"] == pytest.approx(1.3) and o["dmN"] == pytest.approx(-1.3) and o["dbal"] == pytest.approx(0.0)
    assert o["bal_acc"] == pytest.approx(1.0) and o["dacc"] == pytest.approx(0.5)
    assert out[("H1", "b4", 1, "gen")]["a"]["dbal"] == pytest.approx(0.0)       # gen on the probes = main


def test_config_harvest_blocks():
    for iid in CFG["primary"] + CFG["secondary"]:
        sub = CFG["subjects"][iid]
        h2 = sub["h2"]
        assert bind.find_alias(h2["mention"], sub["subject"]) is not None, iid   # the second mention names the subject
        if iid in CFG["primary"]:
            for fu in sub["harvest_followups"]:
                assert bind.find_alias(fu, sub["subject"]) is not None, (iid, fu)


# ================================================================ arm R: splice and gating


def test_splice_keep():
    keep, start = bind.splice_keep("R-full", 5, 8, 6)
    assert keep == list(range(13)) and start == 13
    keep, start = bind.splice_keep("R-min", 5, 8, 6)
    assert keep == [0, 1, 2, 3, 4, 11, 12] and start == 13                    # head + attribute + next token
    keep, start = bind.splice_keep("R-orphan", 5, 8, 6, n_unrel=10)
    assert keep == list(range(5)) + list(range(15, 23)) and start == 23        # the exchange is dropped
    with pytest.raises(AssertionError):
        bind.splice_keep("R-min", 5, 8, 7)                                     # no token after the attribute
    assert bind.gate_open("Write a poem about my dog.", ["my dog"]) and not bind.gate_open("Hello", ["my dog"])


def test_majority_yn_and_rules():
    assert bind.majority([True, True, False]) is True and bind.majority([True, None, False], tie=False) is False
    assert bind.majority([None, None], tie=True) is True and bind.majority([False, False, True]) is False
    assert bind.balanced_yn([("Yes", "yes"), ("Yes", "no"), ("No", "no"), ("No", None)]) == pytest.approx(0.5)
    c = lambda m, lo, hi: {"mean": m, "lo": lo, "hi": hi, "n": 7}
    base = {"subj": c(0.4, 0.2, 0.6), "subj_m_wrong": c(0.3, 0.1, 0.5), "bal_gain_subj": c(0.2, 0.05, 0.3),
            "bal_subj": c(0.7, 0.6, 0.8), "wrong_gain": c(0.0, -0.1, 0.1), "leak_gain": c(0.0, -0.05, 0.05),
            "unrel_same": {"subj1": 1.0, "R-full": 0.97}, "reinst_m_subj": c(-0.1, -0.3, 0.1), "analytic_gain": True}
    V = bind.gen_rules(base)
    assert V["labels"] == ["holds (binding works)"] and all(V["B"][k] for k in ("B1", "B2", "B3", "B4", "B6"))
    assert not V["B"]["B5_qwen25"]
    V = bind.gen_rules({**base, "wrong_gain": c(0.3, 0.1, 0.5)})
    assert "push in disguise" in V["labels"] and not V["B"]["B3"]
    V = bind.gen_rules({**base, "subj": c(0.1, 0.0, 0.2), "subj_all": [c(0.1, 0.0, 0.2)], "reinst_m_subj": c(0.4, 0.2, 0.6)})
    assert "orphaned" in V["labels"] and "reinstatement wins" in V["labels"]
    assert not bind.gen_rules({**base, "unrel_same": {"R-full": 0.9}})["B"]["B6"]
    nan = c(float("nan"), float("nan"), float("nan"))
    assert bind.gen_rules({**base, "subj": nan})["labels"][0].startswith("undetermined")


@pytest.fixture(scope="module")
def tiny_real_tok():
    tok = _qwen25_tok()
    if tok is None:
        pytest.skip("Qwen2.5 tokenizer not in the local cache")
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).eval()
    C = SimpleNamespace(model=model, body=model.model, lm_head=model.lm_head, blocks=model.model.layers, n=3, d=32,
                        dev="cpu", tok=tok, cfg=CFG, mc=CFG["models"]["qwen25_15b"], A=SimpleNamespace(tiny=True, random_model="qwen2"))
    run.gen_context(C, CFG)
    return C


def test_turn_pieces_and_R_full_equals_prior_turn(tiny_real_tok):
    C = tiny_real_tok
    it = next(i for i in run.load_items(CFG) if i["id"] == "dog_name")
    P = run.turn_pieces(C.tok, it["experience"], CFG["gen"]["orphan_exchange"])
    assert C.tok.decode(P.Seg).startswith("<|im_start|>user\nMy dog's name is Pepper.") and len(P.U) > 10
    with torch.no_grad():
        R, info = run.harvest_R(C, it, CFG["gen"]["orphan_exchange"])
        assert info["min_tokens"] == [" Pepper", "."] and info["kept"]["R-min"] == info["n_head"] + 2
        assert info["starts"]["R-orphan"] == info["starts"]["R-full"] + info["n_unrel"]
        q = it["question"]
        _, prior = run.render_ids(C.tok, [{"role": "user", "content": it["experience"]}, {"role": "user", "content": q}])
        a = run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=R["R-full"], max_new=12)
        b = run.gen_batch(C, torch.tensor(prior), [(True, None)], max_new=12)
        assert a[0]["ids"] == b[0]["ids"]                                       # R-full == sentence as a prior turn
        m = run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=R["R-min"], max_new=12)
        o = run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=R["R-orphan"], max_new=12)
        assert len(m[0]["ids"]) > 0 and len(o[0]["ids"]) > 0
        # gate closed (no alias in the prompt): no segment, the row is nomem's by construction
        M = SimpleNamespace(add={}, R={"R-full": R["R-full"]})
        streams = {1: (7, 3), 2: (8, 3)}
        kw = dict(S=2, cap=8, streams=streams)
        res, gi = run.run_prompt(C, CFG, it, M, ["nomem", "R-full"], "How do airplanes stay in the air?", **kw)
        assert not gi["gate"] and all(o is None for _, _, o in res["R-full"])
        res, gi = run.run_prompt(C, CFG, it, M, ["nomem", "R-full"], q, **kw)
        assert gi["gate"] and all(o is not None for _, _, o in res["R-full"])
        # the same random streams in every condition: a sampled nomem row equals a solo run of that row
        solo = run.gen_batch(C, run.chat_ids(C.tok, q), [(False, (1, 2))], max_new=8, streams=streams)
        assert next(o for s, j, o in res["nomem"] if (s, j) == (1, 2))["ids"] == solo[0]["ids"]


# ================================================= arm R on the hybrid (Qwen3.5: DeltaNet + full attention)


@pytest.fixture(scope="module")
def tiny_hybrid():
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3_5 import modeling_qwen3_5  # noqa: F401 (transformers >= 5)
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)
        tc = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True).text_config
    except Exception as e:  # transformers 4.x (no qwen3_5) or no cached Qwen3.5 config / tokenizer
        pytest.skip(f"no Qwen3.5 classes / files: {type(e).__name__}")
    tc.hidden_size, tc.intermediate_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 64, 128, 2, 1, 64
    tc.linear_num_key_heads = tc.linear_num_value_heads = 2
    tc.linear_key_head_dim = tc.linear_value_head_dim = 32
    tc.num_hidden_layers, tc.layer_types = 8, list(tc.layer_types[:8])
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(tc).float().eval()
    run.thinking_off(tok)
    C = SimpleNamespace(model=model, body=model.model, lm_head=model.lm_head, blocks=model.model.layers, n=8, d=64,
                        dev="cpu", tok=tok, cfg=CFG, mc=CFG["models"]["qwen35_2b"], A=SimpleNamespace(tiny=True, random_model="qwen2"))
    run.gen_context(C, CFG)
    return C


def test_hybrid_reinstatement(tiny_hybrid):
    C = tiny_hybrid
    assert run.is_hybrid(C) and C.head == []          # Qwen3.5's template has no system block
    it = next(i for i in run.load_items(CFG) if i["id"] == "dog_name")
    q = it["question"]
    with torch.no_grad():
        R, info = run.harvest_R(C, it, CFG["gen"]["orphan_exchange"])
        nh, ns = info["n_head"], info["n_seg"]
        L = info["layers"]
        assert [k for k, _ in L["R-full-state"]] == ["lin", "lin", "lin", "attn"] * 2
        assert all(f for k, f in L["R-full-state"] if k == "lin")                       # DeltaNet saw the segment
        for v, want in (("R-full-kv6", nh + ns), ("R-min-kv6", nh + 2), ("R-orphan-kv6", nh + ns)):
            assert all(n == want for k, n in L[v] if k == "attn") and not any(f for k, f in L[v] if k == "lin")
        assert info["starts"]["R-orphan-kv6"] == info["starts"]["R-full-kv6"] + info["n_unrel"]
        # R-full-state == the sentence as a prior user turn (greedy, token-identical)
        _, prior = run.render_ids(C.tok, [{"role": "user", "content": it["experience"]}, {"role": "user", "content": q}])
        a = run.gen_batch(C, run.qt_ids(C, q), [(True, None), (True, None)], prefix=R["R-full-state"], max_new=10)
        b = run.gen_batch(C, torch.tensor(prior), [(True, None)], max_new=10)
        assert a[0]["ids"] == b[0]["ids"] == a[1]["ids"]
        # the stored cache is not modified by generation (a second run gives the same tokens)
        a2 = run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=R["R-full-state"], max_new=10)
        assert a2[0]["ids"] == a[0]["ids"]
        # kv6 with an empty head: the DeltaNet blocks start fresh; with every attention entry dropped it is nomem
        nm = run.gen_batch(C, run.chat_ids(C.tok, q), [(True, None)], max_new=10)
        empty = (bind.kv6_cache(R["R-full-state"][0], run.run_cache(C, []), []), 0)
        e = run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=empty, max_new=10)
        assert e[0]["ids"] == nm[0]["ids"]
        for v in ("R-full-kv6", "R-min-kv6", "R-orphan-kv6"):
            assert len(run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=R[v], max_new=6)[0]["ids"]) > 0
        # gating and the swap condition name
        M = run.item_memory(C, CFG, it, None, None, None, None, R, R)
        assert set(M.R) >= {"R-full-state", "R-full-kv6", "R-min-kv6", "R-orphan-kv6", "R-swap-kv6", "R-min-state",
                            "R-state-only", "R-orphan-state"} and not M.add
        res, gi = run.run_prompt(C, CFG, it, M, ["nomem", "R-full-kv6"], "How do airplanes stay in the air?", S=1, cap=6,
                                 streams={1: (7, 2), 2: (8, 2)})
        assert not gi["gate"] and all(o is None for _, _, o in res["R-full-kv6"])


def test_hybrid_state_variants(tiny_hybrid):
    """Addendum 4: R-min-state (2 attention entries + the full DeltaNet state), R-state-only (no segment entries +
    the full DeltaNet state), R-orphan-state (2 entries + the state after the unrelated exchange + experience)."""
    C = tiny_hybrid
    assert bind.splice_keep("R-state-only", 3, 8, 5) == ([0, 1, 2], 11)
    assert bind.splice_keep("R-orphan-min", 3, 8, 5, n_unrel=10) == ([0, 1, 2, 18, 19], 21)
    it = next(i for i in run.load_items(CFG) if i["id"] == "dog_name")
    q = it["question"]
    with torch.no_grad():
        R, info = run.harvest_R(C, it, CFG["gen"]["orphan_exchange"])
        nh, ns, L = info["n_head"], info["n_seg"], info["layers"]
        for v, want in (("R-min-state", nh + 2), ("R-state-only", nh), ("R-orphan-state", nh + 2)):
            assert all(n == want for k, n in L[v] if k == "attn"), (v, L[v])
            assert all(f for k, f in L[v] if k == "lin")                               # DeltaNet saw the sentence
        assert info["lin_equal_full"]["R-min-state"] and info["lin_equal_full"]["R-state-only"]
        assert not info["lin_equal_full"]["R-min-kv6"] and not info["lin_equal_full"]["R-orphan-state"]
        assert info["lin_equal_orphan"]["R-orphan-state"]
        assert info["starts"]["R-state-only"] == info["starts"]["R-full-state"] == nh + ns
        assert info["starts"]["R-orphan-state"] == nh + info["n_unrel"] + ns
        # the DeltaNet part alone: R-min-state with its 2 entries == R-full-state's DeltaNet + 2 entries; state-only
        # differs from R-full-state only by the missing attention entries
        for v in ("R-min-state", "R-state-only", "R-orphan-state"):
            assert len(run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=R[v], max_new=6)[0]["ids"]) > 0
        C.A.gen_set = "state"
        try:
            run.gen_context(C, CFG)
            assert C.gc.primary == CFG["models"]["qwen35_2b"]["gen_conds_state"]["primary"] and C.gc.leak == "R-min-state"
            M = run.item_memory(C, CFG, it, None, None, None, None, R, R)
            assert "R-swap-min-state" in M.R and M.R["R-swap-min-state"] is R["R-min-state"]
            res, gi = run.run_prompt(C, CFG, it, M, ["nomem", "R-min-state", "R-state-only"], "Tell me a joke.", S=1, cap=6,
                                     streams={1: (7, 2), 2: (8, 2)})
            assert not gi["gate"] and all(o is None for c in ("R-min-state", "R-state-only") for _, _, o in res[c])
            res, gi = run.run_prompt(C, CFG, it, M, ["nomem", "R-min-state"], q, S=1, cap=6, streams={1: (7, 2), 2: (8, 2)})
            assert gi["gate"] and all(o is not None for _, _, o in res["R-min-state"])
        finally:
            C.A.gen_set = None
            run.gen_context(C, CFG)


# ================================================= addendum 5: text controls and the two-fact collision


def test_pair_plan_positions():
    P = bind.pair_plan(nh=3, nA=8, nB=10, aA=5, aB=6)
    assert P["R2-joint"] == ([("J", list(range(21)), 0)], 21)
    assert P["R2-sep-overlap"] == ([("A", list(range(11)), 0), ("B", list(range(3, 13)), 0)], 13)
    assert P["R2-sep-shift"] == ([("A", list(range(11)), 0), ("B", list(range(3, 13)), 8)], 21)
    assert P["R2-min-joint"] == ([("J", [0, 1, 2, 8, 9, 17, 18], 0)], 21)
    assert P["R2-min-sep-overlap"] == ([("A", [0, 1, 2, 8, 9], 0), ("B", [9, 10], 0)], 13)
    assert P["R2-min-sep-shift"] == ([("A", [0, 1, 2, 8, 9], 0), ("B", [9, 10], 8)], 21)
    names = bind.hybrid_collide_names(P)
    assert names["R2-joint-state"] == ("R2-joint", "J") and names["R2-sep-shift-jstate"] == ("R2-sep-shift", "J")
    assert names["R2-joint-kv6"] == ("R2-joint", "H") and names["R2-sep-shift-kv6"] == ("R2-sep-shift", "H")
    # addendum 7: the joint-state-only condition, R2-seq (on top of A's full slice), R2-seq-min, R2-blend15
    assert P["R2-state-only"] == ([("J", [0, 1, 2], 0)], 21)
    assert P["R2-seq"] == ([("Qf", list(range(21)), 0)], 21)
    assert P["R2-seq-min"] == ([("Qm", [0, 1, 2, 3, 4, 11, 12], 0)], 21)       # head, A's 2 positions, B's value + next
    assert P["R2-blend15"] == ([("A", list(range(11)), 0), ("Bb", list(range(3, 13)), 0)], 21)
    assert names["R2-state-only-J"] == ("R2-state-only", "J") and names["R2-seq-min-jstate"] == ("R2-seq-min", "J")
    assert len(names) == 12
    assert bind.blend_select(torch.tensor([0.1, 5.0, 0.2, 3.0, 0.0]), 0.15) == [1]
    assert bind.blend_select(torch.tensor([0.1, 5.0, 0.2, 3.0, 0.0]), 0.4) == [1, 3]


def _keys_at(C, ids, start, layer):
    pos = torch.arange(start, start + len(ids))[None]
    pkv = C.body(input_ids=torch.tensor(ids)[None], position_ids=pos, use_cache=True).past_key_values
    return run._attn_kv(pkv, layer)[0] if hasattr(pkv, "layers") else bind.kv_list(pkv)[layer][0]


def test_rope_shift_equals_recompute(tiny_real_tok):
    C = tiny_real_tok
    ids = run.chat_ids(C.tok, "My dog's name is Pepper.").tolist()
    with torch.no_grad():
        K0, K7 = _keys_at(C, ids, 0, 1), _keys_at(C, ids, 7, 1)
        assert not torch.allclose(K0, K7, atol=1e-4)
        assert torch.allclose(run.rope_shift(C, K0, 7), K7, atol=1e-5)


def test_rope_shift_hybrid(tiny_hybrid):
    C = tiny_hybrid
    ids = run.chat_ids(C.tok, "My dog's name is Pepper.").tolist()
    l = next(i for i, b in enumerate(C.blocks) if hasattr(b, "self_attn"))
    with torch.no_grad():
        K0, K9 = _keys_at(C, ids, 0, l), _keys_at(C, ids, 9, l)
        assert not torch.allclose(K0, K9, atol=1e-4)
        assert torch.allclose(run.rope_shift(C, K0, 9), K9, atol=1e-5)   # partial rotary + text mrope


def _collide_pair(cfg, pid):
    return next(p for p in run.collide_pairs(cfg) if p[0] == pid)


def test_collide_pairs_config():
    pairs = run.collide_pairs(CFG)
    assert [k for _, k, _, _ in pairs].count("confusable") == 5 and [k for _, k, _, _ in pairs].count("different") == 3
    for pid, kind, A, B in pairs:
        assert A["partner_value"] == B["value"] and B["partner_value"] == A["value"] and A["value"] != B["value"]
        for M in (A, B):
            assert len(M["use"]) == 2 and len(M["verify"]) == 2 and M["value"].startswith(" ")
            assert M["value"].strip() in M["experience"] and M["value"].strip() not in " ".join(M["use"])
    tok = _qwen25_tok()
    if tok is not None:
        for _, _, A, B in pairs:
            for M in (A, B):
                assert len(tok(M["value"], add_special_tokens=False).input_ids) == 1, M["value"]


def test_collide_harvest_and_t2(tiny_real_tok):
    C = tiny_real_tok
    C.mc = {**CFG["models"]["qwen25_15b"]}
    _, _, A, B = _collide_pair(CFG, "sisters")
    with torch.no_grad():
        R, info = run.harvest_pair(C, A, B)
        nh, nA, nB = info["n_head"], info["n_A"], info["n_B"]
        n = lambda c: R[c][0][0][0].shape[2]
        assert n("R2-joint") == n("R2-sep-shift") == n("R2-sep-overlap") == nh + nA + nB
        assert n("R2-min-joint") == n("R2-min-sep-shift") == nh + 4
        assert info["starts"]["R2-sep-overlap"] == nh + max(nA, nB) and info["starts"]["R2-sep-shift"] == nh + nA + nB
        # the head + A part of a separate harvest is the joint harvest's (same computation); B's part differs
        J, Sh = R["R2-joint"][0][1][0], R["R2-sep-shift"][0][1][0]
        assert torch.allclose(J[:, :, :nh + nA], Sh[:, :, :nh + nA], atol=1e-5)
        assert not torch.allclose(J[:, :, nh + nA:], Sh[:, :, nh + nA:], atol=1e-4)
        # R2-joint == both sentences as separate prior user turns (T2): token-identical greedy
        q = A["question"]
        _, t2 = run.render_ids(C.tok, [{"role": "user", "content": A["experience"]}, {"role": "user", "content": B["experience"]},
                                       {"role": "user", "content": q}])
        a = run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=R["R2-joint"], max_new=10)
        b = run.gen_batch(C, torch.tensor(t2), [(True, None)], max_new=10)
        assert a[0]["ids"] == b[0]["ids"]
        # forced gates: the collision conditions splice whether or not the prompt names a subject
        o = run.gen_batch(C, run.qt_ids(C, "Tell me a joke."), [(True, None)], prefix=R["R2-sep-shift"], max_new=4)
        assert len(o[0]["ids"]) > 0
        assert torch.allclose(run.tf_last(C, run.qt_ids(C, q), R["R2-joint"]),
                              run.tf_last(C, torch.tensor(t2)), atol=1e-4)


def test_collide_harvest_hybrid(tiny_hybrid):
    C = tiny_hybrid
    _, _, A, B = _collide_pair(CFG, "dogs")
    with torch.no_grad():
        R, info = run.harvest_pair(C, A, B)
        assert set(R) == {"R2-joint-state", "R2-sep-overlap-jstate", "R2-sep-shift-jstate", "R2-min-joint-jstate",
                          "R2-min-sep-overlap-jstate", "R2-min-sep-shift-jstate", "R2-joint-kv6", "R2-sep-shift-kv6",
                          "R2-state-only-J", "R2-seq-jstate", "R2-seq-min-jstate", "R2-blend15-jstate"}
        assert info["reencode_single_fact_maxdiff"] <= 1e-4                       # re-encode on the head == B's harvest
        assert all(n == 0 for k, n in bind.cache_summary(R["R2-state-only-J"][0]) if k == "attn")
        J = R["R2-joint-state"][0]
        for c in R:
            lin_ok = bind.linear_states_equal(R[c][0], J)
            assert lin_ok == c.endswith(("-jstate", "-state", "-J")), c    # joint DeltaNet state, except the kv6 probes
        assert not any(f for k, f in bind.cache_summary(R["R2-joint-kv6"][0]) if k == "lin")
        q = B["question"]
        _, t2 = run.render_ids(C.tok, [{"role": "user", "content": A["experience"]}, {"role": "user", "content": B["experience"]},
                                       {"role": "user", "content": q}])
        a = run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=R["R2-joint-state"], max_new=10)
        b = run.gen_batch(C, torch.tensor(t2), [(True, None)], max_new=10)
        assert a[0]["ids"] == b[0]["ids"]
        assert torch.allclose(run.tf_last(C, run.qt_ids(C, q), R["R2-seq-jstate"]),
                              run.tf_last(C, run.qt_ids(C, q), R["R2-joint-state"]), atol=1e-4)   # R2-seq == R2-joint


# ================================================= addendum 6: binding-ID repair and G-key capacity


def test_kv_from_resid_and_edit_identity(tiny_real_tok):
    C = tiny_real_tok
    ids = run._turns(C, "My dog's name is Pepper.", "My cat's name is Jasper.")
    with torch.no_grad():
        H, kv = run.block_inputs(C, ids)
        assert H.shape[0] == C.n and H.shape[1] == len(ids)
        P = bind.ea_positions(C.tok, ids, "My cat's name is Jasper.", "Jasper")
        for l in range(C.n):
            k, v = run.kv_from_resid(C, l, H[l][P["attr"]], P["attr"])
            ii = torch.tensor(P["attr"])
            assert torch.allclose(k, kv[l][0][:, :, ii], atol=1e-5) and torch.allclose(v, kv[l][1][:, :, ii], atol=1e-5)
        zero = run.edit_kv(C, kv, H, [(P["attr"], {l: torch.zeros(2, C.d) for l in range(C.n)})], list(range(C.n)))
        assert all(torch.allclose(a[0], b[0], atol=1e-5) for a, b in zip(zero, kv))
        some = run.edit_kv(C, kv, H, [(P["attr"], {1: torch.ones(2, C.d)})], [1])
        assert torch.equal(some[0][0], kv[0][0]) and not torch.allclose(some[1][0], kv[1][0])
        other = [i for i in range(len(ids)) if i not in P["attr"]]
        assert torch.equal(some[1][0][:, :, other], kv[1][0][:, :, other])          # every other position frozen


def test_ea_positions_real_tokenizer():
    tok = _qwen25_tok()
    if tok is None:
        pytest.skip("Qwen2.5 tokenizer not in the local cache")
    C = SimpleNamespace(tok=tok, head=run.turn_pieces(tok, "x").H)
    d = lambda ids, xs: [tok.decode([int(ids[i])]) for i in xs]
    for _, _, A, B in run.collide_pairs(CFG):
        for M in (A, B):
            ids = run._turns(C, M["experience"])
            P = bind.ea_positions(tok, ids, M["experience"], M["value"])
            assert d(ids, P["attr"]) == [M["value"], "."], (M["id"], d(ids, P["attr"]))
            ent = d(ids, P["entity"])
            assert ent[1] in ("'s", " is", " was"), (M["id"], ent)
    ids = run._turns(C, "My mother-in-law's name is Petra.")
    P = bind.ea_positions(tok, ids, "My mother-in-law's name is Petra.", "Petra")
    assert d(ids, P["entity"])[1] == "'s" and d(ids, P["attr"]) == [" Petra", "."]


def test_order_pc_and_templates():
    torch.manual_seed(0)
    X = torch.randn(200, 16)
    order = torch.tensor([0, 1] * 100)
    X[:, 3] += 4 * order.float()                      # the order lives on axis 3
    u, r = bind.order_pc(X, order)
    assert abs(u[3].item()) > 0.9 and r > 0.8
    from seahorse.bench import load_bench
    pool = load_bench(ROOT / "data" / "bench_v1")["pool"]
    T = bind.id_templates(pool, 50, 11)
    assert len(T) == 50 and all(s1 != s2 and v1 != v2 for s1, v1, s2, v2 in T) and T == bind.id_templates(pool, 50, 11)


def test_gkey_facts_and_retrieve():
    F = run.gkey_facts(CFG)
    assert len(F) == 59 and len({f["id"] for f in F}) == 59
    assert [f["tier"] for f in F[:12]] == ["primary"] * 7 + ["secondary"] * 5 and all(f["tier"] == "pool" for f in F[12:])
    assert all(f["swap"] in {x["id"] for x in F} and f["swap"] != f["id"] for f in F)
    assert all(len(f["harvest_followups"]) == 2 for f in F if f["tier"] == "pool")
    q = torch.tensor([1.0, 0.0])
    sk = [("a", torch.tensor([[0.9, 0.1], [0.0, 1.0]])), ("b", torch.tensor([[0.5, 0.5]]))]
    assert run.retrieve(q, sk, 0.6) == ("a", ["a", "b"], pytest.approx(0.9))
    assert run.retrieve(q, sk, 0.95)[0] is None


def test_hybrid_state_decomposition(tiny_hybrid):
    """Addendum 6b: R-state-only-rec / -conv / R-min-state-rec restore exactly the named DeltaNet part; resetting
    both parts (and no attention entries) reproduces nomem."""
    C = tiny_hybrid
    it = next(i for i in run.load_items(CFG) if i["id"] == "sister_name")
    q = it["question"]
    with torch.no_grad():
        R, info = run.harvest_R(C, it, CFG["gen"]["orphan_exchange"])
        rs = info["restored"]
        assert rs["R-state-only"] == rs["R-min-state"] == {"rec": True, "conv": True}
        assert rs["R-state-only-rec"] == rs["R-min-state-rec"] == {"rec": True, "conv": False}
        assert rs["R-state-only-conv"] == {"rec": False, "conv": True}
        assert rs["R-min-kv6"] == {"rec": False, "conv": False}
        nh, L = info["n_head"], info["layers"]
        assert all(n == nh for k, n in L["R-state-only-rec"] if k == "attn")
        assert all(n == nh + 2 for k, n in L["R-min-state-rec"] if k == "attn")
        full, head = R["R-full-state"][0], run.run_cache(C, [])
        none = bind.split_linear(bind.mix_cache(full, list(range(nh)), full), head, "head", "head")
        nm = run.gen_batch(C, run.chat_ids(C.tok, q), [(True, None)], max_new=10)
        e = run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=(none, R["R-state-only"][1]), max_new=10)
        assert e[0]["ids"] == nm[0]["ids"]                                   # fresh = no previous tokens
        assert R["R-full-state"][0] is full and bind.linear_states_equal(R["R-full-state"][0], full)   # source untouched
        C.A.gen_set = "decomp"
        try:
            run.gen_context(C, CFG)
            rows = run.decomp_analytic(C, [it], {it["id"]: R}, ["nomem", "R-state-only-rec", "R-state-only-conv"])
            assert [r["cond"] for r in rows] == ["nomem", "R-state-only-rec", "R-state-only-conv"]
            assert rows[0]["dlp"] == 0 and all(0 <= r["yn_bal_acc"] <= 1 for r in rows)
        finally:
            C.A.gen_set = None
            run.gen_context(C, CFG)


# ================================================= addendum 7: template port, lures, pool facts, directions, G


def _tok(name, llama=False):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        return None
    return run.date_pinned(tok, CFG["models"]["llama32_3b"]["date_string"]) if llama else tok


def test_turn_pieces_template_generic():
    """turn_pieces derives head and segment from the template: token-identical to the old ChatML pieces on Qwen2.5,
    and Llama-3.2's header turns (with the pinned date) on Llama."""
    tq = _qwen25_tok()
    if tq is None:
        pytest.skip("Qwen2.5 tokenizer not in the local cache")
    ex = CFG["gen"]["orphan_exchange"]
    P = run.turn_pieces(tq, "My dog's name is Pepper.", ex)
    assert tq.decode(P.Seg) == "<|im_start|>user\nMy dog's name is Pepper.<|im_end|>\n"
    s_nm = tq.apply_chat_template([{"role": "user", "content": "q"}], tokenize=False, add_generation_prompt=True)
    assert tq.decode(P.H) == s_nm[:s_nm.index("<|im_start|>user")]                     # the old ChatML head
    assert tq.decode(P.U).startswith("<|im_start|>user\nHow do I keep fresh basil") and "<|im_start|>assistant" in tq.decode(P.U)
    tl = _tok("meta-llama/Llama-3.2-3B-Instruct", llama=True)
    if tl is None:
        pytest.skip("Llama-3.2 tokenizer not in the local cache")
    P = run.turn_pieces(tl, "My dog's name is Pepper.", ex)
    assert tl.decode(P.Seg) == "<|start_header_id|>user<|end_header_id|>\n\nMy dog's name is Pepper.<|eot_id|>"
    assert "Today Date: 26 Jul 2024" in tl.decode(P.H) and tl.decode(P.H).startswith("<|begin_of_text|>")
    a = tl.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    assert a == tl.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)


@pytest.fixture(scope="module")
def tiny_llama():
    tok = _tok("meta-llama/Llama-3.2-3B-Instruct", llama=True)
    if tok is None:
        pytest.skip("Llama-3.2 tokenizer not in the local cache")
    model = run.random_llama(tok, 3, 0, "cpu")
    C = SimpleNamespace(model=model, body=model.model, lm_head=model.lm_head, blocks=model.model.layers, n=3, d=64,
                        dev="cpu", tok=tok, cfg=CFG, mc=CFG["models"]["llama32_3b"], tol=0.01,
                        A=SimpleNamespace(tiny=True, random_model="llama", model="llama32_3b"))
    run.gen_context(C, CFG)
    return C


def test_llama_R_full_equals_prior_turn_and_eos(tiny_llama):
    C = tiny_llama
    assert C.tok.convert_tokens_to_ids("<|eot_id|>") in C.eos_list
    it = next(i for i in run.load_items(CFG) if i["id"] == "sister_name")
    q = it["question"]
    with torch.no_grad():
        R, info = run.harvest_R(C, it, CFG["gen"]["orphan_exchange"])
        assert info["min_tokens"] == [" Petra", "."] and info["kept"]["R-min"] == info["n_head"] + 2
        _, prior = run.render_ids(C.tok, [{"role": "user", "content": it["experience"]}, {"role": "user", "content": q}])
        a = run.gen_batch(C, run.qt_ids(C, q), [(True, None)], prefix=R["R-full"], max_new=10)
        b = run.gen_batch(C, torch.tensor(prior), [(True, None)], max_new=10)
        assert a[0]["ids"] == b[0]["ids"]
        pre = run.text_ids(C.tok, it["prefix"])
        assert torch.allclose(run.tf_last(C, *run.cond_ids(C, it, "R-full", q, pre, R)),
                              run.tf_last(C, *run.cond_ids(C, it, "T-full", q, pre, R)), atol=1e-4)


def test_lure_config_and_probes():
    items = {it["id"]: it for it in run.load_items(CFG)}
    assert set(CFG["lures"]) == set(CFG["primary"])
    toks = [t for t in (_qwen25_tok(), _tok("Qwen/Qwen3.5-2B"), _tok("meta-llama/Llama-3.2-3B-Instruct", llama=True)) if t]
    for iid, L in CFG["lures"].items():
        it = items[iid]
        vals = [v for k in run.LURE_KINDS for v in L[k]]
        assert all(len(L[k]) == 2 for k in run.LURE_KINDS) and len(set(vals)) == 6
        assert not {v.strip() for v in vals} & ({f.strip() for f in it["foils"]} | {it["target"].strip()})
        for tok in toks:
            P = run.lure_probes(CFG, it, tok)                                        # asserts single tokens
            kinds = [k for _, _, k, _ in P]
            assert len(P) == 16 and kinds.count("yes") == 2 and kinds.count("unrelated") == 2
            assert all(kinds.count(k) == 4 for k in run.LURE_KINDS)
            assert all(c == "No" for _, c, k, _ in P if k != "yes") and all(t.endswith("Answer yes or no.") for t, *_ in P)
    if toks:
        bad = {**CFG, "lures": {**CFG["lures"], "dog_name": {**CFG["lures"]["dog_name"], "form": [" Max", " Penny"]}}}
        with pytest.raises(AssertionError):
            run.lure_probes(bad, items["dog_name"], toks[0])                       # " Max" is one of the dog's foils


def test_entities_disjoint_from_items():
    E = yaml.safe_load(open(ROOT / CFG["entities"]))
    assert len(E["famous"]) == len(set(E["famous"])) == 100 and len(E["made_up"]) == len(set(E["made_up"])) == 100
    pool = yaml.safe_load(open(ROOT / "data" / "bench_v1" / "fact_pool.yaml"))
    names = {w for l in pool["names"].values() for w in l} | {v for s in pool["slots"] for v in s.get("values", [])}
    for f in FACTS.values():
        names |= {f["measure"]["target"].strip(), *[x.strip() for x in f["measure"]["foils"]], f["counter"].rstrip(".").split()[-1]}
    names |= {v.strip() for L in CFG["lures"].values() for k in L.values() for v in k}
    low = {n.lower() for n in names}
    for e in E["famous"] + E["made_up"]:
        assert not {w.lower() for w in re.split(r"[\s-]+", e)} & low, e


def test_pool_gen_items():
    F = run.pool_gen_items(CFG)
    assert len(F) == 52 and len({f["id"] for f in F}) == 52 and all(f["tier"] == "pool" for f in F)
    kinds = [f["kind"] for f in F]
    assert kinds[:5] == ["attribute"] * 5 and kinds.count("person") == 30 and kinds.count("pet") == 8
    for f in F:
        assert len(f["use"]) == 1 and f["target"].strip() not in f["use"][0], f["id"]
        assert bind.gate_open(f["use"][0], f["aliases"]), (f["id"], f["use"][0])                 # G-match opens on the use prompt
        assert f["json"]["group"] == "facts" and len(f["relation_probes"]) == 4
    assert {f["id"] for f in F[5:]} == {f["id"] for f in run.gkey_facts(CFG) if f["tier"] == "pool"}


def test_s0_pool_items_layouts():
    tok = _qwen25_tok()
    if tok is None:
        pytest.skip("Qwen2.5 tokenizer not in the local cache")
    items, info = run.s0_pool_items(CFG, tok)
    prim = [it for it in items if it["tier"] == "primary"]
    assert len(prim) == 38 and len(items) + len(info["excluded"]) == 47
    C = SimpleNamespace(tok=tok)
    for it in items:
        P = run.build(C, it)                                   # the stage-0 layout asserts (one differing token)
        if it["tier"] == "primary":
            assert P.subj and P.q0 > P.a
    for iid, (old, new) in info["swaps"].items():
        assert len(tok(f" {new}", add_special_tokens=False).input_ids) == 1 and len(tok(f" {old}", add_special_tokens=False).input_ids) > 1


def test_diff_means_and_told_pick():
    torch.manual_seed(0)
    X0 = torch.randn(50, 3, 8)
    off = torch.zeros(3, 8)
    off[:, 2] = 3.0
    D, sep = run.diff_means(X0 + off + 0.1 * torch.randn(50, 3, 8), X0)
    assert torch.allclose(D, off, atol=0.1) and all(s > 2 for s in sep)
    D2, sep2 = run.diff_means(torch.randn(50, 3, 8), torch.randn(50, 3, 8))
    assert all(abs(s) < 1.0 for s in sep2)
    cells = [{"set": "early", "alpha": 1, "score": 3.0, "dlp": -2.0}, {"set": "mid", "alpha": 0.5, "score": 2.0, "dlp": 0.1},
             {"set": "mid", "alpha": 1, "score": 2.0, "dlp": 0.0}]
    p = run.told_pick(cells, -1.0)
    assert (p["set"], p["alpha"]) == ("mid", 0.5)                     # the constraint drops early; tie -> smaller alpha
    p = run.told_pick([dict(c, dlp=-5.0) for c in cells], -1.0)
    assert (p["set"], p["alpha"]) == ("early", 1) and p["rule"].startswith("none")
    assert run.thirds(24) == {"early": list(range(8)), "mid": list(range(8, 16)), "late": list(range(16, 24))}


def test_r2_seq_and_blend(tiny_real_tok):
    """R2-seq: re-encoding B on top of A's stored full slice == R2-joint; on top of the head only == B's separate harvest.
    R2-seq-min keeps head + 4 entries; R2-blend15 replaces ceil(15%) of B's positions with the re-encoded values."""
    C = tiny_real_tok
    C.mc = {**CFG["models"]["qwen25_15b"]}
    C.tol = 0.01
    _, _, A, B = _collide_pair(CFG, "friends")
    with torch.no_grad():
        R, info = run.harvest_pair(C, A, B)
        nh, nA, nB = info["n_head"], info["n_A"], info["n_B"]
        assert info["reencode_single_fact_maxdiff"] <= 1e-4
        q = B["question"]
        assert torch.allclose(run.tf_last(C, run.qt_ids(C, q), R["R2-seq"]), run.tf_last(C, run.qt_ids(C, q), R["R2-joint"]), atol=1e-4)
        assert R["R2-seq-min"][0][0][0].shape[2] == nh + 4 and R["R2-seq-min"][1] == nh + nA + nB
        assert len(info["blend_positions"]) == math.ceil(0.15 * nB)
        # blend with every position replaced == the joint slice; with none == R2-sep-shift
        PA, PB = run.turn_pieces(C.tok, A["experience"]), run.turn_pieces(C.tok, B["experience"])
        cB, cJ = run.run_cache(C, PB.H + PB.Seg), run.run_cache(C, PA.H + PA.Seg + PB.Seg)
        Qf = run.reencode(C, run.run_cache(C, PA.H + PA.Seg), nh + nA, PB.Seg)
        full, sel = run.blend_source(C, cB, Qf, nh, nA, nB, frac=1.0)
        J = bind.kv_list(cJ)
        assert sel == list(range(nB)) and all(torch.allclose(f[0][:, :, nh:], j[0][:, :, nh + nA:], atol=1e-4) for f, j in zip(full, J))
        # R2-seq-min differs from R2-min-sep-shift only in B's 2 entries (A's are the same stored ones)
        sm, ms = R["R2-seq-min"][0], R["R2-min-sep-shift"][0]
        assert all(torch.allclose(a[0][:, :, :nh + 2], b[0][:, :, :nh + 2], atol=1e-5) for a, b in zip(sm, ms))
        assert not all(torch.allclose(a[0][:, :, nh + 2:], b[0][:, :, nh + 2:], atol=1e-4) for a, b in zip(sm[1:], ms[1:]))


def test_probe_stage_and_told(tiny_real_tok):
    """The probe stage on one item (identity check inside), and the told specs (rand: same norms; neg: minus)."""
    C = tiny_real_tok
    C.mc = {**CFG["models"]["qwen25_15b"], "probe_conds": ["nomem", "R-full", "T-min"]}
    C.A.model = "qwen25_15b"
    it = next(i for i in run.load_items(CFG) if i["id"] == "dog_name")
    with torch.no_grad():
        rows, checks, _ = run.stage_probe(C, [it], Path("/nonexistent/probe"))
        R, _ = run.harvest_R(C, it, CFG["gen"]["orphan_exchange"])
    assert checks["identity"][0]["max_abs_logit_diff"] <= 1e-4
    assert sum(r["sec"] == "yn" for r in rows) == 16 * 3 and sum(r["sec"] == "tf" for r in rows) == 3
    D = {"pick": {"layers": [0, 2], "alpha": 2.0}, "told": torch.randn(3, C.d)}
    S = run.told_specs(C, ["nomem", "R-min+told", "R-min+told:rand", "R-min+told:neg", "nomem+told"], R, D, 1)
    assert set(S) == {"R-min+told", "R-min+told:rand", "R-min+told:neg", "nomem+told"} and S["nomem+told"][0] is None
    assert S["R-min+told"][0] is R["R-min"]
    for l in (0, 2):
        assert torch.allclose(S["R-min+told"][1][l], 2.0 * D["told"][l]) and torch.allclose(S["R-min+told:neg"][1][l], -2.0 * D["told"][l])
        assert torch.allclose(S["R-min+told:rand"][1][l].norm(), S["R-min+told"][1][l].norm(), atol=1e-4)


def test_probe_identity_hybrid(tiny_hybrid):
    """Addendum 7 (F): on the hybrid, R-full-state teacher-forced == the sentence as a prior turn (T-full)."""
    C = tiny_hybrid
    it = next(i for i in run.load_items(CFG) if i["id"] == "boss_name")
    with torch.no_grad():
        R, _ = run.harvest_R(C, it, CFG["gen"]["orphan_exchange"])
        pre = run.text_ids(C.tok, it["prefix"])
        a = run.tf_last(C, *run.cond_ids(C, it, "R-full-state", it["question"], pre, R))
        b = run.tf_last(C, *run.cond_ids(C, it, "T-full", it["question"], pre, R))
    assert torch.allclose(a, b, atol=1e-4)


# ================================================= addendum 8: tag at storage, min+subj, pool probe / split, comply, wordings


ADD8 = _load("test_dream_bind_add8_data", EXP / "add8_data.py")


def _y(name):
    return yaml.safe_load(open(ROOT / "data" / "bench_v1" / name))


def test_add8_plan_identities():
    """With 0 dropped tokens the tagged plans flatten to the R2-sep-shift / R2-min-sep-shift plans; bothtagged is symmetric;
    min+subj keeps head + subject span + value + full stop of each member at the joint / shifted positions."""
    nh, nA, nB, aA, aB, h = 3, 9, 9, 6, 6, 2
    sA, sB = [3, 4], [3, 4]
    P = bind.pair_plan(nh, nA, nB, aA, aB)
    Z = bind.add8_plan(nh, nA, nB, aA, aB, sA, sB, h, 0, 0, 0)
    ren = lambda parts, *ab: [(dict(ab).get(s, s), i, sh) for s, i, sh in bind.flat_parts(parts)]
    assert ren(Z["R2-sep-tagged"][0], ("Bt", "B")) == bind.flat_parts(P["R2-sep-shift"][0])
    assert Z["R2-sep-tagged"][0] == [("A", list(range(nh + nA)), 0), ("Bt", list(range(nh, nh + nB)), nA)]
    assert ren(Z["R2-sep-bothtagged"][0], ("At", "A"), ("Bt", "B")) == bind.flat_parts(P["R2-sep-shift"][0])
    assert ren(Z["R2-sep-placebo"][0], ("Bp", "B")) == bind.flat_parts(P["R2-sep-shift"][0])
    assert ren(Z["R2-min-sep-tagged"][0], ("Bt", "B")) == bind.flat_parts(P["R2-min-sep-shift"][0])
    assert ren(Z["R2-min-sep-bothtagged"][0], ("At", "A"), ("Bt", "B")) == bind.flat_parts(P["R2-min-sep-shift"][0])
    assert all(Z[k][1] == P["R2-sep-shift"][1] == nh + nA + nB for k in Z)
    # dropped tokens: B's kept entries come from after the dummy, re-rotated onto B's joint positions
    T = bind.add8_plan(nh, nA, nB, aA, aB, sA, sB, h, 5, 7, 4)
    parts = T["R2-sep-tagged"][0]
    assert parts[0] == ("A", list(range(nh + nA)), 0)
    assert parts[1] == ("Bt", [nh, nh + 1], nA) and parts[2] == ("Bt", list(range(nh + h + 7, nh + nB + 7)), nA - 7)
    final = [i + sh for _, idx, sh in parts[1:] for i in idx]
    assert final == list(range(nh + nA, nh + nA + nB))                      # contiguous after A, like R2-sep-shift
    assert T["R2-min-sep-tagged"][0][-1] == ("Bt", [nh + aB + 7, nh + aB + 8], nA - 7)
    # bothtagged: symmetric in A and B (same dropped count, same layouts): B's parts = A's parts moved by nA
    S = bind.add8_plan(nh, nA, nB, aA, aB, sA, sB, h, 6, 6, 4)
    fa = [(i, sh) for s, i, sh in bind.flat_parts(S["R2-sep-bothtagged"][0]) if s == "At"]
    fb = [(i, sh) for s, i, sh in bind.flat_parts(S["R2-sep-bothtagged"][0]) if s == "Bt"]
    assert [i for i, _ in fa] == [i for i, _ in fb] and all(sb == sa + nA for (_, sa), (_, sb) in zip(fa, fb))
    assert {s for s, *_ in bind.flat_parts(S["R2-sep-bothtagged"][0])} == {"A", "At", "Bt"}   # head from A (identical everywhere)
    # min+subj
    assert T["R2-minsubj-joint"] == ([("J", [0, 1, 2, 6, 7, 9, 10, 15, 16, 18, 19], 0)], 21)
    assert bind.flat_parts(T["R2-minsubj-sep-shift"][0]) == (
        [("A", i, 0) for i in [0, 1, 2, 6, 7, 9, 10]] + [("B", i, nA) for i in [6, 7, 9, 10]])
    with pytest.raises(AssertionError):
        bind.minsubj([5, 6], 6)                                               # the subject span may not hold the value
    assert bind.merge_parts([("A", [0], 0), ("A", [3], 0), ("B", [1], 2), ("B", [], 5)]) == [("A", [0, 3], 0), ("B", [1], 2)]


def test_subject_phrase_and_counter_value():
    want = {"My younger sister's name is Maria.": "younger sister", "My first dog was called Pepper.": "first dog",
            "My friend from work is called Omar.": "friend from work", "My son's teacher is called Hugo.": "son's teacher",
            "My grandmother on my mother's side is called Esther.": "grandmother on my mother's side",
            "My boss's name is Gordon.": "boss", "My mother-in-law's name is Hannah.": "mother-in-law",
            "My guinea pig's name is Sunny.": "guinea pig"}
    for s_, w in want.items():
        a, b = bind.subject_phrase(s_)
        assert s_[a:b] == w
    for _, _, A, B in run.collide_pairs(CFG) + run.pool_pairs(CFG):
        for M in (A, B):
            a, b = bind.subject_phrase(M["experience"])
            assert M["value"].strip() not in M["experience"][a:b] and b > a
    assert bind.counter_value("I grew up in Leeds.", "I grew up in Lyon.") == "Lyon"
    assert bind.counter_value("I work as a pharmacist.", "I work as a plumber.") == "plumber"
    assert bind.counter_value("My dog's name is Pepper.", "My dog's name is Rocky.") == "Rocky"
    assert bind.dummy_for(0, "A", [{"slot": "x", "name": "Y"}, {"slot": "z", "name": "W"}]) == "My x's name is Y."
    assert bind.dummy_for(3, "B", [{"slot": "s", "name": str(i)} for i in range(8)]) == "My s's name is 7."


def test_pool_split_deterministic_disjoint():
    F = run.pool_gen_items(CFG)
    facts = [(f["id"], f["kind"]) for f in F]
    S = _y("pool_split.yaml")
    A, B = bind.pool_split(facts, S["seed"])
    assert (A, B) == bind.pool_split(facts, S["seed"]) and (A, B) == (S["A"], S["B"])
    assert not set(A) & set(B) and set(A) | set(B) == {i for i, _ in facts} and len(A) == len(B) == 26
    kind = dict(facts)
    for k, n in (("person", 15), ("pet", 4), ("attribute", 7)):
        assert sum(kind[i] == k for i in A) == sum(kind[i] == k for i in B) == n
    assert bind.pool_split(facts, S["seed"] + 1) != (A, B)
    assert run.load_split(CFG) == (A, B)
    half = run.pool_half_items(CFG, "B")
    assert [f["id"] for f in half] == B
    for f in half:                                                             # the swap partner: same kind, same half
        assert f["swap"] in B and f["swap"] != f["id"] and kind[f["swap"]] == f["kind"]


def test_form_and_binding_lure_draw():
    one = lambda v: v not in ("Penny",)
    vals = ["Max", "Penny", "Piper", "Pepper", "Pip", "Rocky", "Pumpkin", "Peaches"]
    a = bind.form_lures("Pepper", vals, ["Max", "Pip"], one)
    assert a == bind.form_lures("Pepper", vals, ["Max", "Pip"], one) == ["Peaches", "Piper"]   # shared prefix "Pe" first
    assert "Pepper" not in a and "Penny" not in a and len(bind.form_lures("Rex", ["Rex", "Ann"], [], lambda v: True)) == 1
    facts = [("female", "Ann"), ("male", "Bob"), ("female", "Cat"), ("attribute", "teal"), ("female", "Dee"), ("female", "Ann")]
    assert bind.binding_lures(0, facts, [], lambda v: True) == ["Cat", "Dee"]
    assert bind.binding_lures(4, facts, ["Cat"], lambda v: True) == ["Ann"]     # cyclic; Ann (the target of 5) once
    assert bind.binding_lures(3, facts, [], lambda v: True) == []
    # the written file obeys the rules
    L = _y("pool_lures.yaml")["facts"]
    F = run.pool_gen_items(CFG)
    assert [f["id"] for f in L] == [f["id"] for f in F]
    for f, it in zip(L, F):
        ex = {f["target"], f["counter"], *f["foils"]}
        assert f["target"] == it["target"].strip() and not ex & set(f["form"] + f["binding"])
        assert len(set(f["form"] + f["binding"])) == len(f["form"]) + len(f["binding"])
        assert (f["kind"] == "attribute") or (len(f["form"]) == 2 and len(f["binding"]) == 2)
        if f["kind"] == "attribute":
            assert f["binding"] == []
    assert sum(f["kind"] in ("person", "pet") for f in L) == 38


def _all_toks():
    t = [_qwen25_tok(), _tok("Qwen/Qwen3.5-2B"), _tok("meta-llama/Llama-3.2-3B-Instruct")]
    return t if all(t) else None


@pytest.mark.slow
def test_add8_data_files_match_generator():
    """The generated data files equal a fresh deterministic rebuild (parsed, so YAML formatting cannot matter)."""
    if _all_toks() is None:
        pytest.skip("the three tokenizers are not all in the local cache")
    for name, text in ADD8.build_all(CFG).items():
        assert yaml.safe_load(text) == _y(name), name


def test_dummy_pool_pair_and_comply_files():
    D = _y("dummy_pairs.yaml")["dummies"]
    assert D == ADD8.DUMMIES and len(D) == 8
    names = ADD8.test_names(CFG)
    pp = run.pool_pairs(CFG)
    pool = _y("fact_pool.yaml")
    pslots = {s["slot"] for s in pool["slots"]}
    for d in D:
        assert d["name"].lower() not in names and d["slot"] not in pslots
    assert [k for _, k, _, _ in pp].count("confusable") == 17 and [k for _, k, _, _ in pp].count("different") == 17
    vals = [M["value"] for _, _, A, B in pp for M in (A, B)]
    ids = [M["id"] for _, _, A, B in pp for M in (A, B)]
    assert len(set(vals)) == len(vals) == 68 and len(set(ids)) == 68
    assert not {v.strip().lower() for v in vals} & {d["name"].lower() for d in D}
    for _, kind, A, B in pp:
        assert A["partner_value"] == B["value"] and len(A["use"]) == 1 and len(A["verify"]) == 2
        if kind == "different":
            assert bind.subject_phrase(A["experience"]) != bind.subject_phrase(B["experience"]) or A["experience"] != B["experience"]
    toks = [t for t in (_qwen25_tok(), _tok("Qwen/Qwen3.5-2B"), _tok("meta-llama/Llama-3.2-3B-Instruct", llama=True)) if t]
    for t in toks:
        for v in vals + [f" {d['name']}" for d in D]:
            assert len(t(v, add_special_tokens=False).input_ids) == 1, v
    # comply pairs: >= 60, distinct, no "my" in the general question, disjoint from every test item
    C_ = _y("comply_pairs.yaml")["pairs"]
    assert len(C_) >= 60 and len({c["id"] for c in C_}) == len(C_)
    allv = names | {v.strip().lower() for v in vals} | {d["name"].lower() for d in D}
    slots = pslots | {"best friend", "friend", "teacher", "grandmother", "work", "school", "job", "hometown", "university"}
    for _, _, A, B in run.collide_pairs(CFG):
        for M in (A, B):
            a, b = bind.subject_phrase(M["experience"])
            slots.add(M["experience"][a:b])
    for c in C_:
        assert re.search(r"\b(my|I)\b", c["personal"]) and not re.search(r"\b(my|I)\b", c["general"], re.I), c
        for q in (c["personal"], c["general"]):
            assert not {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", q)} & allv, q
            assert not [s for s in slots if re.search(rf"(?<![\w-]){re.escape(s)}(?![\w-])", q, re.I)], q


def test_comply_regex_and_wordings():
    R = run.COMPLY_REFUSAL_RE
    for t in ("I don't have access to your personal information.", "I'm sorry, but I can't know that.",
              "As an AI, I have no way of knowing your blood type.", "I'm not able to determine that.",
              "Could you tell me more?", "Only you know that."):
        assert R.search(t), t
    for t in ("The most common blood type is O positive.", "Ottawa is the capital of Canada."):
        assert not R.search(t), t
    assert run.told_wording("Is my dog called Pepper? Answer yes or no.", "Based on what I told you earlier, ") == \
        "Based on what I told you earlier, is my dog called Pepper? Answer yes or no."
    assert run.told_wording("Do I work as a pilot? Answer yes or no.", "X, ") == "X, do I work as a pilot? Answer yes or no."
    with pytest.raises(AssertionError):
        run.told_wording("Is my dog called Pepper?", "X")
    tok = _qwen25_tok()
    if tok is None:
        pytest.skip("Qwen2.5 tokenizer not in the local cache")
    it = next(i for i in run.load_items(CFG) if i["id"] == "dog_name")
    P = run.choice_probes(CFG, it, tok, CFG["lures"]["dog_name"], "pet")
    assert len(P) == 8 * 2 * 2 and all(t.endswith("Answer with the name only.") for _, t, *_ in P)
    assert P[0][1] == "Is my dog called Pepper or Max? Answer with the name only." and P[1][1].startswith("Is my dog called Max or Pepper?")
    assert all(x[0] is None or x[0][0] != x[0][1] for x in P) and {k for _, _, k, _, _ in P} == {"unrelated", "form", "assoc", "binding"}
    pf = next(f for f in run.pool_gen_items(CFG) if f["id"] == "pool_instrument")
    PL = {f["id"]: f for f in _y("pool_lures.yaml")["facts"]}
    Pc = run.choice_probes(CFG, pf, tok, PL["pool_instrument"], "attribute")
    assert all(t.endswith("Answer with one word only.") for _, t, *_ in Pc) and len(Pc) == 4 * 2 * 2
    assert run.verify_templates(CFG, pf) == ["Do I play the {v}?", "Is the {v} my instrument?"]
    pp = next(f for f in run.pool_gen_items(CFG) if f["id"] == "pool_mother_in_law")
    assert run.verify_templates(CFG, pp) == ["Is my mother-in-law's name {v}?", "Is {v} the name of my mother-in-law?"]
    probes = run.pool_probes(CFG, pp, tok, PL["pool_mother_in_law"])
    kinds = [k for _, _, k, _ in probes]
    assert len(probes) == 12 and kinds.count("form") == kinds.count("binding") == 4 and kinds.count("yes") == 2
    bad = {**PL["pool_mother_in_law"], "form": [PL["pool_mother_in_law"]["counter"], "Anna"]}
    with pytest.raises(AssertionError):
        run.pool_probes(CFG, pp, tok, bad)                                       # the counter is never a lure


def test_add8_rules():
    c = lambda m, lo, hi: {"mean": m, "lo": lo, "hi": hi, "n": 10}
    nan = c(float("nan"), float("nan"), float("nan"))
    assert bind.tag_rule(c(-0.2, -0.35, -0.05), c(0.1, 0.0, 0.2), c(0.05, 0.0, 0.12), c(0.0, -0.1, 0.1))
    assert not bind.tag_rule(c(-0.2, -0.35, 0.01), c(0.1, 0.0, 0.2), c(0.05, 0.0, 0.12), c(0.0, -0.1, 0.1))
    assert not bind.tag_rule(c(-0.2, -0.35, -0.05), c(0.3, 0.2, 0.4), c(0.05, 0.0, 0.12), c(0.0, -0.1, 0.1))   # no overlap
    assert not bind.tag_rule(c(-0.2, -0.35, -0.05), c(0.1, 0.0, 0.2), c(0.05, 0.0, 0.12), c(-0.2, -0.3, -0.1))
    assert bind.tag_rule(nan, c(0.1, 0.0, 0.2), c(0.05, 0.0, 0.12), c(0.0, -0.1, 0.1)) is None
    assert bind.entity_rule(c(-0.1, -0.2, -0.01), c(0.05, 0.0, 0.1), c(0.03, 0.0, 0.08))
    assert not bind.entity_rule(c(-0.1, -0.2, 0.01), c(0.05, 0.0, 0.1), c(0.03, 0.0, 0.08))
    assert bind.comply_rule(c(0.3, 0.1, 0.5), c(0.8, 0.7, 0.9), c(0.5, 0.3, 0.6)) == "told is specific"
    assert bind.comply_rule(c(0.05, -0.05, 0.15), c(0.8, 0.7, 0.9), c(0.75, 0.6, 0.9)) == "generic comply"
    assert bind.comply_rule(c(0.2, -0.05, 0.4), c(0.8, 0.7, 0.9), c(0.6, 0.4, 0.8)) == "in between"
    assert bind.comply_rule(nan, nan, nan) == "n/a"
    assert bind.recognition_rule(c(0.1, 0.02, 0.2), c(0.9, 0.8, 1.0)) == "gist familiarity"
    assert bind.recognition_rule(c(-0.02, -0.08, 0.04), c(0.9, 0.8, 1.0)) == "item recognition"
    assert bind.recognition_rule(c(-0.02, -0.08, 0.04), c(0.6, 0.45, 0.7)) == "in between"
    assert bind.verification_rule(c(0.95, 0.9, 1.0), c(0.9, 0.8, 1.0)) and not bind.verification_rule(c(0.95, 0.9, 1.0), c(0.85, 0.8, 1.0))


def test_rope_shift_negative(tiny_real_tok):
    C = tiny_real_tok
    ids = run.chat_ids(C.tok, "My dog's name is Pepper.").tolist()
    with torch.no_grad():
        K9, K4 = _keys_at(C, ids, 9, 1), _keys_at(C, ids, 4, 1)
        assert torch.allclose(run.rope_shift(C, K9, -5), K4, atol=1e-5)       # tagged slices move back over the dummy


def test_tagged_turn_layout(tiny_real_tok):
    C = tiny_real_tok
    exp, pre = "My younger sister's name is Maria.", "My mechanic's name is Keith."
    T = run.tagged_turn(C, exp, pre)
    P0 = run.turn_pieces(C.tok, exp)
    assert T.dropped == pre and C.tok.decode(T.Seg[T.h:T.h + T.d]) == pre and T.n_diff <= 1
    kept = T.Seg[:T.h] + T.Seg[T.h + T.d:]
    assert len(kept) == len(P0.Seg) and C.tok.decode(kept[T.h:]).strip() == C.tok.decode(P0.Seg[T.h:]).strip()
    assert C.tok.decode(T.Seg[:T.h]) == "<|im_start|>user\n"
    E = run.tagged_turn(C, exp, "")
    assert E.d == 0 and E.Seg == P0.Seg and E.n_diff == 0
    Pl = run.tagged_turn(C, exp, CFG["gen"]["placebo"])
    assert Pl.dropped == CFG["gen"]["placebo"] and Pl.Seg[Pl.h + Pl.d:] == T.Seg[T.h + T.d:]
    s = run.subject_indices(C, P0.Seg, exp)
    assert [C.tok.decode([P0.Seg[i]]) for i in s] == [" younger", " sister"]
    tl = _tok("meta-llama/Llama-3.2-3B-Instruct", llama=True)
    if tl is not None:
        Cl = SimpleNamespace(tok=tl)
        Tl = run.tagged_turn(Cl, exp, pre)
        assert Tl.dropped == pre and Tl.n_diff <= 1
        assert [tl.decode([run.turn_pieces(tl, exp).Seg[i]]) for i in run.subject_indices(Cl, run.turn_pieces(tl, exp).Seg, exp)] == \
            [" younger", " sister"]


def _add8_prefixes(k=1):
    D = _y("dummy_pairs.yaml")["dummies"]
    return {"A": bind.dummy_for(k, "A", D), "B": bind.dummy_for(k, "B", D), "placebo": CFG["gen"]["placebo"]}


def test_collide_add8_harvest(tiny_real_tok):
    C = tiny_real_tok
    C.mc = {**CFG["models"]["qwen25_15b"]}
    _, _, A, B = _collide_pair(CFG, "sisters")
    conds = CFG["models"]["qwen25_15b"]["collide_conds_add8"]
    with torch.no_grad():
        R, info = run.harvest_pair(C, A, B, add8=_add8_prefixes(), need=set(conds) - {"ctx2"})
        nh, nA, nB = info["n_head"], info["n_A"], info["n_B"]
        assert set(conds) - {"ctx2"} <= set(R) and "R2-sep-tagged@0" in R and "R2-seq" in R and "R2-blend15" not in R
        n = lambda c: R[c][0][0][0].shape[2]
        for c in ("R2-sep-tagged", "R2-sep-bothtagged", "R2-sep-placebo"):
            assert n(c) == nh + nA + nB and R[c][1] == nh + nA + nB
        assert n("R2-min-sep-tagged") == n("R2-min-sep-placebo") == nh + 4
        i8 = info["add8"]
        assert n("R2-minsubj-joint") == n("R2-minsubj-sep-shift") == nh + 4 + len(i8["subj_A"]) + len(i8["subj_B"])
        assert i8["subj_A"] == [" older", " sister"] and i8["dropped"]["B0"] == 0 and i8["dropped"]["Bt"] > 0
        q = B["question"]
        qt = run.qt_ids(C, q)
        assert torch.allclose(run.tf_last(C, qt, R["R2-sep-tagged@0"]), run.tf_last(C, qt, R["R2-sep-shift"]), atol=1e-5)
        assert not torch.allclose(run.tf_last(C, qt, R["R2-sep-tagged"]), run.tf_last(C, qt, R["R2-sep-shift"]), atol=1e-4)
        # A's part of R2-sep-tagged is A's own harvest (as in R2-sep-shift); B's entries differ (stored after the dummy)
        T, S = R["R2-sep-tagged"][0][1][0], R["R2-sep-shift"][0][1][0]
        assert torch.allclose(T[:, :, :nh + nA], S[:, :, :nh + nA], atol=1e-6)
        # the tagged B entries == B's after-dummy harvest, re-rotated: recompute its keys at the joint positions
        PB = run.turn_pieces(C.tok, B["experience"])
        tB = run.tagged_turn(C, B["experience"], _add8_prefixes()["B"])
        kv = bind.kv_list(run.run_cache(C, PB.H + tB.Seg))
        idx = list(range(nh + tB.h + tB.d, nh + len(tB.Seg)))
        want = run.rope_shift(C, kv[1][0][:, :, idx], nA - tB.d)
        assert torch.allclose(T[:, :, nh + nA + tB.h:], want, atol=1e-5)
    full, _ = run.harvest_pair(C, A, B)
    assert "R2-sep-tagged" not in full and len(full) == 10                     # without add8: the addendum-5/7 plans only


def test_collide_add8_hybrid(tiny_hybrid):
    C = tiny_hybrid
    _, _, A, B = _collide_pair(CFG, "dogs")
    conds = CFG["models"]["qwen35_2b"]["collide_conds_add8"]
    with torch.no_grad():
        R, info = run.harvest_pair(C, A, B, add8=_add8_prefixes(2), need=set(conds) - {"ctx2"})
        assert set(conds) - {"ctx2"} <= set(R) and "R2-sep-tagged@0-jstate" in R
        J = R["R2-joint-state"][0]
        for c in R:
            assert bind.linear_states_equal(R[c][0], J), c                        # the joint DeltaNet state everywhere
        q = A["question"]
        qt = run.qt_ids(C, q)
        assert torch.allclose(run.tf_last(C, qt, R["R2-sep-tagged@0-jstate"]), run.tf_last(C, qt, R["R2-sep-shift-jstate"]), atol=1e-5)
        nh, nA, nB = info["n_head"], info["n_A"], info["n_B"]
        att = [k for k, _ in bind.cache_summary(J)]
        for c, want in (("R2-sep-bothtagged-jstate", nh + nA + nB), ("R2-min-sep-placebo-jstate", nh + 4)):
            assert all(n == want for k, n in bind.cache_summary(R[c][0]) if k == "attn"), c
        assert "lin" in att


def test_told_specs_comply(tiny_real_tok):
    C = tiny_real_tok
    it = next(i for i in run.load_items(CFG) if i["id"] == "cat_name")
    with torch.no_grad():
        R, _ = run.harvest_R(C, it, CFG["gen"]["orphan_exchange"])
    D = {"pick": {"layers": [0, 2], "alpha": 2.0}, "told": torch.randn(3, C.d),
         "pick_comply": {"layers": [1], "alpha": 0.5}, "comply": torch.randn(3, C.d)}
    conds = ["nomem", "R-min+told", "R-min+told:rand", "R-min+comply", "R-min+comply:rand"]
    S = run.told_specs(C, conds, R, D, 4321)
    assert set(S) == set(conds[1:]) and S["R-min+comply"][0] is R["R-min"]
    assert set(S["R-min+comply"][1]) == {1} and torch.allclose(S["R-min+comply"][1][1], 0.5 * D["comply"][1])
    assert torch.allclose(S["R-min+comply:rand"][1][1].norm(), S["R-min+comply"][1][1].norm(), atol=1e-4)
    S7 = run.told_specs(C, conds[:3], R, {k: D[k] for k in ("pick", "told")}, 4321)
    assert all(torch.equal(S7[c][1][l], S[c][1][l]) for c in S7 for l in S7[c][1])  # the told arms are unchanged


@pytest.mark.slow
def test_probe_add8_and_directions8(tiny_real_tok, tmp_path):
    """The probe stage with the part-6 wordings (primary item and pool facts), and the add8 directions stage (half A only)."""
    C = tiny_real_tok
    C.mc = {**CFG["models"]["qwen25_15b"], "probe_conds": ["nomem", "T-full"]}
    C.A.model, C.A.probe_set = "qwen25_15b", "add8"
    it = next(i for i in run.load_items(CFG) if i["id"] == "dog_name")
    try:
        with torch.no_grad():
            rows, checks, _ = run.stage_probe(C, [it], Path("/nonexistent/probe"))
            sec = lambda s: sum(r["sec"] == s for r in rows)
            assert sec("yn") == sec("yn_told") == 16 * 2 and sec("choice") == 32 * 2 - 4 * len(checks["choice_skipped"]) * 0
            assert all(r["answer"] is not None for r in rows if r["sec"] == "yn_told")
            assert all(r["prompt"].startswith("Based on what I told you earlier, ") for r in rows if r["sec"] == "yn_told")
            C.A.items = "pool"
            rows_p, checks_p, _ = run.stage_probe(C, [it], Path("/nonexistent/probe"))
            ids = {r["item"] for r in rows_p}
            assert ids == {f["id"] for f in run.pool_gen_items(CFG, tiny=True)} and all(r["answer"] is None for r in rows_p if r["sec"] == "yn_told")
            assert {r["kind"] for r in rows_p if r["sec"] == "yn"} <= {"yes", "unrelated", "form", "binding"}
            C.A.items, C.A.probe_set, C.A.dir_set = "main", None, "add8"
            C.mc = {**CFG["models"]["qwen25_15b"]}
            out = tmp_path / "qwen25_15b" / "directions"
            rows_d, checks_d, _ = run.stage_directions(C, [it], out)
        D = torch.load(tmp_path / "qwen25_15b_dirs8" / "dirs.pt")
        A_, B_ = run.load_split(CFG)
        assert set(D["facts"]) <= set(A_) and not set(D["facts"]) & set(B_) and len(D["cos"]) == C.n
        assert D["told"].shape == D["comply"].shape == (C.n, C.d) and D["pick"]["set"] in ("early", "mid", "late")
        assert len(rows_d) == 4 and all(r["sec"] == "comply" for r in rows_d)
        run.write_jsonl_gz(tmp_path / "qwen25_15b_dirs8" / "results.jsonl.gz", rows_d)
        run.analyze_dirs8(CFG, tmp_path / "qwen25_15b_dirs8", "qwen25_15b")
        assert (tmp_path / "qwen25_15b_dirs8" / "analysis" / "report.md").exists()
    finally:
        C.A.items, C.A.probe_set, C.A.dir_set = "main", None, None
        C.mc = {**CFG["models"]["qwen25_15b"]}
