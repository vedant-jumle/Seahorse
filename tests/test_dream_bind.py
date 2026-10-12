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
        assert set(M.R) == {"R-full-state", "R-full-kv6", "R-min-kv6", "R-orphan-kv6", "R-swap-kv6", "R-min-state",
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
