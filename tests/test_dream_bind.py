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
