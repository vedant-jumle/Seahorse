"""xlayer_v1 (experiments/xlayer_v1/xl.py): the cross-layer dose, the two-pass gate and injection field
(on a tiny random Qwen2, CPU), the Stage-A pick rule and the Stage-B text metrics; plus the config."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from transformers import Qwen2Config, Qwen2ForCausalLM

from seahorse.bench import load_bench
from seahorse.residual import capture, inject

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("xlayer_v1_xl", ROOT / "experiments" / "xlayer_v1" / "xl.py")
xl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xl)


# ------------------------------------------------------------------------- dose


def test_median_norm_and_dose_ratio():
    X = torch.zeros(5, 3)
    X[:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 50.0])
    assert xl.median_norm(X) == 3.0
    assert xl.median_norm([X[:2], X[2:]]) == 3.0  # a list of chunks = their concatenation
    med = {8: 10.0, 16: 20.0, 23: 40.0}
    assert xl.dose_ratio(med, 8, 23) == pytest.approx(0.25)  # a late shift injected early is scaled down
    assert xl.dose_ratio(med, 23, 16) == pytest.approx(2.0)
    assert xl.dose_ratio(med, 16, 16) == 1.0


# ------------------------------------------------------------------ gate + field


def test_cross_field_gate_head_and_scale():
    torch.manual_seed(0)
    T, r, d = 6, 4, 5
    keys = F.normalize(torch.randn(T, r), dim=-1)
    Kst = keys[[2]]  # the stored key = position 2's key: match 1 there
    M = torch.randn(d, r)
    cats = torch.tensor([0, 0, 1, 1, 2, 3])
    f, g, match = xl.cross_field(keys, Kst, M, cats, 0.5, 2.0)
    want = ((keys @ Kst.T).max(-1).values > 0.5) & (cats != 0)
    assert torch.equal(g.bool(), want) and g[2] == 1
    assert torch.allclose(f, 2.0 * want[:, None].float() * (keys @ M.T))
    keys2 = keys.clone()
    keys2[0] = Kst[0]  # a perfect match on the template head never fires
    _, g2, m2 = xl.cross_field(keys2, Kst, M, cats, 0.5, 1.0)
    assert m2[0] > 0.99 and g2[0] == 0


def test_extend_field_repeats_last_row():
    f = torch.arange(6.0).view(3, 2)
    assert xl.extend_field(f, 3) is f
    e = xl.extend_field(f, 5)
    assert e.shape == (5, 2) and torch.equal(e[:3], f) and torch.equal(e[3], f[2]) and torch.equal(e[4], f[2])
    with pytest.raises(AssertionError):
        xl.extend_field(f, 2)
    fs = xl.add_fields({}, 7, f)
    xl.add_fields(fs, 7, f)
    assert torch.equal(fs[7], 2 * f)


# --------------------------------------------- two-pass read on a tiny model


def pooled_keys(h, user):
    """diag_keys.KeySpace pooled key with mu = 0 and no whitening: running mean over user positions."""
    u = user.to(h.dtype)[:, None]
    return F.normalize((h * u).cumsum(0) / u.cumsum(0).clamp_min(1.0), dim=-1, eps=1e-6)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    model = Qwen2ForCausalLM(cfg).eval()
    ids = torch.tensor([1, 2] + [10, 11, 12, 13, 14, 15] + [3, 4] + [20, 21, 22])  # head, user, tail, continuation
    cats = torch.tensor([0] * 2 + [1] * 6 + [2] * 2 + [3] * 3)
    with torch.no_grad(), capture(model, [1, 2]) as st:
        model(ids[None])
    k = pooled_keys(st[2], cats == 1)
    mem = {"Kst": k[[7]], "M": torch.randn(32, 32) * 0.5}  # stored key = the pooled key over all user text
    m_user = (k[cats == 1] @ mem["Kst"].T).squeeze(1)
    thr = (1.0 + m_user.min().item()) / 2  # open after the user text (match 1), shut at the first user token
    return model, ids, cats, mem, thr


def clean_field(model, ids, cats, mem, thr, R, scale):
    """Pass 1 (no memory): keys at layer R -> the field."""
    with torch.no_grad(), capture(model, [R]) as st:
        model(ids[None])
    return xl.cross_field(pooled_keys(st[R], cats == 1), mem["Kst"], mem["M"], cats, thr, scale)


class OnePass:
    """The old design: keys read inside the hook at R from the stream the hook sees, injected at R."""

    def __init__(self, cats, mem, thr, scale):
        self.cats, self.mem, self.thr, self.scale = cats, mem, thr, scale

    def read(self, h, _a):
        f, _, _ = xl.cross_field(pooled_keys(h[0], self.cats == 1), self.mem["Kst"], self.mem["M"], self.cats, self.thr,
                                 self.scale)
        return h + f[None]


def test_two_pass_equals_one_pass_when_W_equals_R(tiny):
    model, ids, cats, mem, thr = tiny
    f, g, _ = clean_field(model, ids, cats, mem, thr, 2, 2.0)
    assert 0 < g.sum() < (cats != 0).sum()
    with torch.no_grad():
        with inject(model, 2, xl.RowField(f[None]), 1.0):
            two = model(ids[None]).logits
        with inject(model, 2, OnePass(cats, mem, thr, 2.0), 1.0):
            one = model(ids[None]).logits
        base = model(ids[None]).logits
    assert torch.allclose(two, one, atol=1e-5)
    assert not torch.allclose(two, base, atol=1e-3)


def test_two_pass_gate_ignores_the_injection_below_R(tiny):
    model, ids, cats, mem, thr = tiny
    R, W = 2, 1
    f1, g1, m1 = clean_field(model, ids, cats, mem, thr, R, 0.5)
    f8, g8, m8 = clean_field(model, ids, cats, mem, thr, R, 8.0)
    assert torch.equal(g1, g8) and torch.equal(m1, m8) and torch.allclose(f8, 16.0 * f1)  # gate fixed, dose linear
    with torch.no_grad(), inject(model, W, xl.RowField(f8[None]), 1.0), capture(model, [R]) as st2:
        model(ids[None])
    # the injection at W < R changes the layer-R stream: keys read there would depend on the injection ...
    k_inj = pooled_keys(st2[R], cats == 1)
    m_inj = (k_inj @ mem["Kst"].T).max(-1).values
    assert not torch.allclose(m_inj, m8, atol=1e-4)
    # ... while the two-pass field is a function of the clean pass only (rerunning pass 1 is unchanged)
    f8b, g8b, _ = clean_field(model, ids, cats, mem, thr, R, 8.0)
    assert torch.equal(f8b, f8) and torch.equal(g8b, g8)


def test_field_is_causal_and_constant_after_user_text(tiny):
    model, ids, cats, mem, thr = tiny
    f, g, _ = clean_field(model, ids, cats, mem, thr, 2, 1.0)
    for t in range(4, len(ids) + 1):  # pass 1 on a prefix = the prefix of pass 1 on the whole sequence
        ft, gt, _ = clean_field(model, ids[:t], cats[:t], mem, thr, 2, 1.0)
        assert torch.allclose(ft, f[:t], atol=1e-5) and torch.equal(gt, g[:t])
    after = f[cats >= 2]  # tail + continuation: no new user text -> the same key, gate and recall
    assert torch.allclose(after, after[:1].expand_as(after), atol=1e-6)


def test_kv_cached_injection_equals_full_sequence(tiny):
    model, ids, cats, mem, thr = tiny
    P, W = 10, 1  # prompt = head + user + tail
    fp, _, _ = clean_field(model, ids[:P], cats[:P], mem, thr, 2, 4.0)
    with torch.no_grad():
        with inject(model, W, xl.RowField(xl.extend_field(fp, len(ids))[None]), 1.0):
            full = model(ids[None]).logits[0]
        xi = xl.XInject(W, fp)
        xi.begin(ids[:P])
        steps = []
        with inject(model, W, xi, 1.0):
            o = model(ids[None, :P], use_cache=True)
            steps.append(o.logits[0, -1])
            past = o.past_key_values
            for t in range(P, len(ids)):
                o = model(ids[None, t:t + 1], past_key_values=past, use_cache=True)
                past = o.past_key_values
                steps.append(o.logits[0, -1])
    assert torch.allclose(torch.stack(steps), full[P - 1:], atol=1e-4)
    with pytest.raises(AssertionError):
        xl.XInject(W, fp).begin(ids[:P - 1])


# ------------------------------------------------------------------ pick rule


def test_pick_cells_rule():
    def c(R, W, a, dspec, drel, dmg):
        return {"R": R, "W": W, "alpha": a, "dspec": dspec, "drel_bal": drel, "dmg": dmg}
    cells = [c(23, 16, 2.0, 9.0, 9.0, 0.0),  # best, but fixed (already in Stage B)
             c(23, 8, 2.0, 3.0, 1.0, 0.01), c(23, 8, 4.0, 4.0, 1.5, 0.02),  # the same (R, W) twice
             c(16, 8, 1.0, 2.0, 0.5, 0.01), c(12, 8, 1.0, -1.0, 2.0, 0.0),  # negative dspec: not eligible
             c(20, 12, 0.5, 1.0, 0.1, 0.05)]
    picks, ranking = xl.pick_cells(cells, {(23, 16, 2.0)}, n=2)
    assert all((r["R"], r["W"], r["alpha"]) != (23, 16, 2.0) for r in ranking)
    assert all(r["dspec"] > 0 for r in ranking)
    assert [(p["R"], p["W"], p["alpha"]) for p in picks] == [(23, 8, 4.0), (16, 8, 1.0)]  # distinct (R, W)
    assert ranking[0]["score"] <= ranking[-1]["score"]
    # fewer than n eligible -> every non-fixed cell competes
    picks, _ = xl.pick_cells([c(1, 1, 1.0, -1.0, 0.0, 0.0), c(2, 1, 1.0, -2.0, 0.0, 0.0)], set(), n=2)
    assert len(picks) == 2
    assert xl.ranks([3.0, 1.0, 3.0, float("nan")]) == [1.5, 3.0, 1.5, 4.0]


# ------------------------------------------------------------------ text metrics


def test_clean_hit_separates_floods_from_clean_hits():
    ok = "Your sister's name is Petra, and she would love a scarf."
    assert xl.clean_hit(ok, list(range(20)), " Petra") == (True, False, True)
    flood = list(range(4)) * 6  # "Petra Petra Petra ...": repeated 4-grams
    assert xl.clean_hit("Petra Petra Petra Petra Petra", flood, "Petra") == (True, True, False)
    assert xl.clean_hit("Your sister is lovely.", list(range(20)), "Petra") == (False, False, False)
    assert xl.has_word("Pepper's checkup", "Pepper") and not xl.has_word("Peppers", "Pepper")
    assert not xl.has_word("teal-blue", "teal") and xl.has_word("TEAL!", "teal")
    assert xl.rep_rate([1, 2, 3]) == 0.0


def test_self_attr_rough_regex():
    yes = ["My dog's name is Pepper.", "I am a pharmacist.", "I'm a pharmacist at the local chemist.",
           "my favourite colour is teal", "Pepper is my dog.", "I grew up in Leeds.", "I love teal."]
    no = ["Your dog's name is Pepper.", "You work as a pharmacist.", "I think your favourite colour is teal.",
          "Pepper. My notes say otherwise.", "Your dog Pepper needs a checkup.", "Is your sister called Petra?"]
    for s in yes:
        assert xl.self_attr(s, _target(s)), s
    for s in no:
        assert not xl.self_attr(s, _target(s)), s


def _target(s):
    for t in ("Pepper", "pharmacist", "teal", "Leeds", "Petra"):
        if t.lower() in s.lower():
            return t
    raise ValueError(s)


def test_parse_yn_and_balanced_accuracy():
    assert xl.parse_yn("Yes, your dog is called Pepper.") == "yes"
    assert xl.parse_yn("I don't know. No idea.") == "no"
    assert xl.parse_yn("Nothing here, nobody.") is None
    always_no = [("Yes", "no"), ("Yes", "no"), ("No", "no"), ("No", "no")]
    assert xl.balanced_acc(always_no) == {"accY": 0.0, "accN": 1.0, "bal": 0.5}
    unparsed = [("Yes", None), ("Yes", "yes"), ("No", None), ("No", "no")]
    assert xl.balanced_acc(unparsed)["bal"] == 0.5
    assert math.isnan(xl.balanced_acc([("Yes", "yes")])["bal"])
    assert xl.nanmean([1.0, float("nan"), None, True]) == 1.0


# ------------------------------------------------------------------------ config


def test_config_items_are_bench_facts_with_clean_use_prompts():
    cfg = yaml.safe_load((ROOT / "experiments" / "xlayer_v1" / "config.yaml").read_text())
    by = {it["id"]: it for it in load_bench(ROOT / "data" / "bench_v1")["scenarios"]}
    ids = [c["id"] for c in cfg["items"]]
    assert 8 <= len(ids) <= 12 and len(set(ids)) == len(ids)
    assert {"dog_name", "sister_name", "job", "favourite_colour"} <= set(ids)
    for c in cfg["items"]:
        b = by[c["id"]]
        assert b["type"] == "fact" and b["source"] == "hand"
        assert len(c["use"]) == 2 and len(set(c["use"])) == 2
        tgt = b["measure"]["target"].strip()
        assert not any(xl.has_word(u, tgt) for u in c["use"]), c["id"]  # the fact is never given away
        assert sorted(r["consistent"] for r in b["relation_probes"]) == ["No", "No", "Yes", "Yes"]
