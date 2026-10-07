"""core_v1 (experiments/core_v1/{core,items}.py, seahorse.stats): the layer-mapping rule, the conditions and
controls (dose, rand norm, swap through the item's own gate, gate_on), batched == unbatched KV-cached
generation on a tiny random Qwen2 (CPU), the bootstrap CI, the judge-output parser and the items file."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from transformers import Qwen2Config, Qwen2ForCausalLM

from seahorse.stats import boot_ci, macro_mean, paired_diff, wrong_way

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "core_v1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


core = _load("core_v1_core", EXP / "core.py")
itm = _load("core_v1_items", EXP / "items.py")

CFG = yaml.safe_load(open(EXP / "config_qwen35_2b.yaml"))


# --------------------------------------------------------------- layer mapping


def test_layer_mapping_rule():
    assert core.map_layers([20, 21, 23], 24, 24) == [20, 21, 23]  # identity on the 2B
    assert core.map_layers([20, 21, 23], 24, 32) == [27, 28, 31]  # Qwen3.5-9B
    assert core.map_layers_alt([20, 21, 23], 24, 32) == [27, 28, 31]  # the other convention agrees here
    assert core.map_layers([23], 24, 48) == [47] and core.map_layers([0], 24, 48) == [1]
    cfg9 = yaml.safe_load(open(EXP / "config_qwen35_9b.yaml"))
    assert cfg9["layers"] == core.map_layers(cfg9["reference_layers"], 24, cfg9["n_blocks"])
    assert CFG["layers"] == core.map_layers(CFG["reference_layers"], 24, CFG["n_blocks"])
    # Qwen3.5 layer types: full attention every 4th block (3, 7, ...); the 9B set includes the last one (31)
    types9 = ["full" if (l + 1) % 4 == 0 else "linear" for l in range(32)]
    assert [types9[l] for l in cfg9["layers"]] == ["full", "linear", "full"]


# ------------------------------------------------------------------ conditions


def test_conditions_and_rows():
    cs = core.conditions(CFG, mech=False)
    ids = [c.cid for c in cs]
    assert ids == ["nomem", "ctx", "mem@0.5", "mem@1", "mem@2", "mem@3", "rand@2", "swap@2", "placebo@2", "gate_on@2"]
    assert {c.cid for c in cs if c.seed2} == {"nomem", "ctx", "mem@1", "mem@2"}
    cm = core.conditions(CFG, mech=True)
    assert [c.cid for c in cm[len(cs):]] == ["centroid@0.5", "centroid@1", "centroid@2", "without@0.5", "without@1",
                                            "without@2"]
    assert all(c.alpha == 2 for c in cs if c.kind in ("rand", "swap", "placebo", "gate_on"))
    assert [core.gated(c) for c in cs] == [False, False, True, True, True, True, True, True, True, False]
    rows = core.row_specs([c for c in cm if c.kind != "ctx"], CFG["samples"])
    assert len(rows) == 15 * 5 + 3 * 4  # 15 non-ctx conditions x (greedy + 4) + seed-2 samples of 3 conditions
    assert all(j > 0 for c, s, j in rows if s == 2)  # seed set 2: samples only
    assert sum(1 for c, s, j in rows if j == 0) == 15


def test_seed_of_is_stable():
    assert core.seed_of("core_v1", "abc", 0) == core.seed_of("core_v1", "abc", 0)
    assert core.seed_of("core_v1", "abc", 0) != core.seed_of("core_v1", "abc", 1)


# ---------------------------------------------------------- fields + controls


def delta_build(shifts, keys):
    """diag_keys.build(rule="delta") for ungated pooled chunks: M += (s - M k) k^T, in order."""
    M = torch.zeros(shifts.shape[1], keys.shape[1])
    for s, k in zip(shifts, keys):
        M += torch.outer(s - M @ k, k)
    return M


@pytest.fixture
def mem_setup():
    torch.manual_seed(0)
    T, r, d = 7, 6, 5
    keys = F.normalize(torch.randn(T, r), dim=-1)
    cats = torch.tensor([0, 0, 1, 1, 1, 2, 3])
    Kst = F.normalize(torch.randn(3, r), dim=-1)
    Kst[2] = keys[4]  # the item's last stored key = position 4's read key
    shifts_own, shifts_partner = torch.randn(3, d), torch.randn(3, d)
    return keys, cats, Kst, shifts_own, shifts_partner


def test_dose_scales_the_field_and_respects_the_gate(mem_setup):
    keys, cats, Kst, s_own, _ = mem_setup
    M = delta_build(s_own, Kst)
    assert torch.allclose(M @ Kst[2], s_own[2], atol=1e-5)  # exact recall of the last write
    match, g = core.read_layer(keys, Kst, 0.9, cats)
    assert g[4] == 1 and g[0] == 0 and g[1] == 0  # match 1 at position 4; never on the template head
    rec = keys @ M.T
    f1 = core.cond_field("mem", 1.0, g, cats, rec)
    f2 = core.cond_field("mem", 2.0, g, cats, rec)
    assert torch.allclose(f2, 2 * f1) and torch.equal(f1[g == 0], torch.zeros_like(f1[g == 0]))
    assert torch.allclose(f1[4], s_own[2], atol=1e-5)  # the raw shift, no rescaling


def test_rand_control_matches_the_recall_norm(mem_setup):
    keys, cats, Kst, s_own, _ = mem_setup
    M = delta_build(s_own, Kst)
    _, g = core.read_layer(keys, Kst, 0.9, cats)
    rec = keys @ M.T
    u = core.rand_direction(rec.shape[1], 123)
    assert abs(u.norm().item() - 1) < 1e-6 and torch.equal(u, core.rand_direction(rec.shape[1], 123))
    fr = core.cond_field("rand", 2.0, g, cats, rec, u)
    fm = core.cond_field("mem", 2.0, g, cats, rec)
    assert torch.allclose(fr.norm(dim=-1), fm.norm(dim=-1), atol=1e-6)  # same norm at every position
    assert torch.equal(fr[g == 0], torch.zeros_like(fr[g == 0]))  # through the item's own gate
    open_rows = fr[g == 1]
    assert torch.allclose(F.normalize(open_rows, dim=-1), u.expand_as(open_rows), atol=1e-6)


def test_swap_uses_own_keys_and_gate(mem_setup):
    keys, cats, Kst, s_own, s_partner = mem_setup
    M_swap = delta_build(s_partner, Kst)  # the partner's shifts written under THIS item's keys
    match, g = core.read_layer(keys, Kst, 0.9, cats)  # the gate comes from the item's own stored keys
    f = core.cond_field("swap", 2.0, g, cats, keys @ M_swap.T)
    assert torch.allclose(f[4], 2 * s_partner[2], atol=1e-5)  # right timing, the partner's content
    assert torch.equal(f[g == 0], torch.zeros_like(f[g == 0]))
    # the item's memory and its swap memory share keys -> identical match, gate and threshold
    assert torch.equal(core.read_layer(keys, Kst, 0.9, cats)[1], g)


def test_gate_on_ignores_the_threshold_but_not_the_head(mem_setup):
    keys, cats, Kst, s_own, _ = mem_setup
    M = delta_build(s_own, Kst)
    rec = keys @ M.T
    _, g = core.read_layer(keys, Kst, 1.5, cats)  # threshold above any cosine: the gate never opens
    assert g.sum() == 0
    f = core.cond_field("gate_on", 2.0, g, cats, rec)
    assert torch.equal(f[cats == 0], torch.zeros_like(f[cats == 0]))
    assert torch.allclose(f[cats != 0], 2 * rec[cats != 0])


def test_stack_fields_zero_rows():
    f = {1: torch.ones(3, 4)}
    out = core.stack_fields([None, f, None], [1, 2], 3, 4, "cpu")
    assert out[1].shape == (3, 3, 4) and out[1][0].abs().sum() == 0 and torch.equal(out[1][1], f[1])
    assert out[2].abs().sum() == 0
    assert core.stack_fields([None, None], [1], 3, 4, "cpu") == {}


# ---------------------------------------------------- batched generation (tiny)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    model = Qwen2ForCausalLM(cfg).eval()
    prompt = torch.tensor([1, 2, 10, 11, 12, 13, 3, 4])
    torch.manual_seed(1)
    field = {1: 0.8 * torch.randn(len(prompt), 32), 2: 0.8 * torch.randn(len(prompt), 32)}
    return model, prompt, field


def gb(model, prompt, rows, fields, n_new=10, streams=None):
    sampler = core.qwen_sampler(1.0, 20, 0.95, 1.5)
    with torch.no_grad():
        return core.generate_batch(model, model.model, model.lm_head, prompt, rows, max_new=n_new, eos=[63], vocab=64,
                                   sampler=sampler, streams=streams or {1: (7, 3), 2: (8, 3)}, fields=fields)


def test_batched_equals_unbatched(tiny):
    model, prompt, field = tiny
    T = len(prompt)
    rows = [(True, None), (False, (1, 1)), (False, (1, 2)), (False, (2, 1))]
    a = gb(model, prompt, rows, {})  # condition A: no memory
    b = gb(model, prompt, rows, core.stack_fields([field] * 4, [1, 2], T, 32, "cpu"))  # condition B: a field
    both = gb(model, prompt, rows + rows, core.stack_fields([None] * 4 + [field] * 4, [1, 2], T, 32, "cpu"))
    assert [o["ids"] for o in both[:4]] == [o["ids"] for o in a]
    assert [o["ids"] for o in both[4:]] == [o["ids"] for o in b]
    assert [o["ids"] for o in a] != [o["ids"] for o in b]  # the field matters
    # one row alone sees the same random numbers as inside the batch (streams do not depend on the batch)
    solo = gb(model, prompt, [(False, (1, 2))], core.stack_fields([field], [1, 2], T, 32, "cpu"))
    assert solo[0]["ids"] == b[2]["ids"]


def test_cached_greedy_equals_full_recompute(tiny):
    """KV-cached generation with the pass-2 field == re-running the whole sequence every step, where every
    position after the prompt gets the last prompt position's field (xl.XInject semantics)."""
    model, prompt, field = tiny
    out = gb(model, prompt, [(True, None)], core.stack_fields([field], [1, 2], len(prompt), 32, "cpu"), n_new=8)
    seq = prompt.tolist()
    for _ in range(8):
        T = len(seq)
        ext = {l: torch.cat([f, f[-1:].expand(T - len(prompt), -1)]) for l, f in field.items()}
        hooks = [core.BatchField(l, ext[l][None]) for l in (1, 2)]
        with torch.no_grad(), core.injecting(model, hooks):
            nxt = model(torch.tensor(seq)[None]).logits[0, -1].argmax().item()
        seq.append(nxt)
        if nxt == 63:
            break
    want = [t for t in seq[len(prompt):] if t != 63]
    assert out[0]["ids"] == want[:len(out[0]["ids"])] and len(out[0]["ids"]) == len(want)


def test_first_step_logprobs_and_nonfinite_guard(tiny):
    model, prompt, field = tiny
    with torch.no_grad():
        o = core.generate_batch(model, model.model, model.lm_head, prompt, [(True, None)], max_new=2, eos=[63], vocab=64,
                                first_ids=[5, 6])
        lp = torch.log_softmax(model(prompt[None]).logits[0, -1], -1)
    assert torch.allclose(torch.tensor(o[0]["first"]), lp[[5, 6]], atol=1e-5)
    bad = {1: torch.full((1, len(prompt), 32), float("nan"))}
    with pytest.raises(FloatingPointError), torch.no_grad():
        core.generate_batch(model, model.model, model.lm_head, prompt, [(True, None)], max_new=2, eos=[63], vocab=64,
                            fields=bad)


# ---------------------------------------------------------------- statistics


def test_bootstrap_ci():
    nested = {"a": {0: 1.0, 1: 3.0}, "b": {0: 5.0}, "c": {0: 0.0, 1: 0.0, 2: 3.0}}
    assert macro_mean(nested) == pytest.approx((2.0 + 5.0 + 1.0) / 3)
    p, lo, hi, n = boot_ci(nested, n_boot=500, seed=3)
    assert n == 3 and p == pytest.approx(8 / 3) and lo <= p <= hi
    assert boot_ci(nested, 500, 3) == (p, lo, hi, n)  # seeded
    const = {i: {0: 0.5, 1: 0.5} for i in "abcd"}
    assert boot_ci(const, 200, 0)[1:3] == (0.5, 0.5)
    assert math.isnan(boot_ci({}, 10, 0)[0])
    a = {"x": {0: 1.0, 1: 2.0}, "y": {0: 3.0, 1: float("nan")}}
    b = {"x": {0: 0.5, 1: 2.5}, "y": {0: 1.0, 1: 1.0}, "z": {0: 1.0}}
    assert paired_diff(a, b) == {"x": {0: 0.5, 1: -0.5}, "y": {0: 2.0}}
    assert wrong_way({"x": {0: 0.5, 1: -1.0}, "y": {0: 2.0}}, +1) == (0.5, 1 / 3)


# ---------------------------------------------------------------------- judge


SCHEMA_P = {"direction": ["toward", "away", "neutral"], "coherent": "bool", "self_claim": "bool"}
SCHEMA_F = {"uses_fact": "bool", "coherent": "bool", "self_claim": "bool", "wrong_value": "bool"}


def test_parse_judge():
    p = core.parse_judge('{"direction": "toward", "coherent": true, "self_claim": false}', SCHEMA_P)
    assert p == {"direction": "toward", "coherent": True, "self_claim": False, "ok": True}
    p = core.parse_judge('Sure.\n```json\n{"direction": "Away", "coherent": "false", "self_claim": "no"}\n```', SCHEMA_P)
    assert p["direction"] == "away" and p["coherent"] is False and p["self_claim"] is False and p["ok"]
    p = core.parse_judge('{"direction": "sideways", "coherent": true, "self_claim": false}', SCHEMA_P)
    assert p["direction"] is None and not p["ok"]
    p = core.parse_judge('"uses_fact": true, "coherent": true, "self_claim": false, "wrong_value": false', SCHEMA_F)
    assert p["ok"] and p["uses_fact"] is True  # no braces: per-key regex fallback
    p = core.parse_judge('{"uses_fact": true, "coherent": true}', SCHEMA_F)
    assert not p["ok"] and p["self_claim"] is None
    assert not core.parse_judge("", SCHEMA_F)["ok"]
    J = yaml.safe_load(open(EXP / "judge_prompts.yaml"))
    t = core.render(J["preference"]["template"], trait="I'm vegetarian.", question="Q?", answer="A.")
    assert "I'm vegetarian." in t and "{trait}" not in t and '{"direction"' in t  # JSON braces stay literal
    assert set(J["fact"]["schema"]) == set(SCHEMA_F) and set(J["preference"]["schema"]) == set(SCHEMA_P)


# ---------------------------------------------------------------------- items


@pytest.fixture(scope="module")
def items():
    return itm.load_items(EXP / "items.yaml", ROOT / "data" / "bench_v1")


def test_items_validate(items):
    res = itm.validate_items(items)
    assert not res["errors"], res["errors"]
    assert res["counts"] == {"leanings": 15, "dislikes": 8, "one_of_many": 6, "facts": 12, "mechanism": 18}
    assert len(items["unrelated"]) == 50


def test_items_facts_are_xlayer_v1s(items):
    x = yaml.safe_load(open(ROOT / "experiments" / "xlayer_v1" / "config.yaml"))
    assert [(c["id"], c["use"]) for c in x["items"]] == [(i, items["by_id"][i]["use"]) for i in items["groups"]["facts"]]


def test_negated_items(items):
    hj, lj = items["by_id"]["hates_jazz"], items["by_id"]["loves_jazz"]
    assert hj["lexicon"]["consistent"] == lj["lexicon"]["inconsistent"]
    assert hj["followups"] == lj["followups"] and hj["related"] == lj["related"]
    assert hj["counter"] == "I'm a huge jazz fan." and hj["experience"] == "I can't stand jazz music."
    assert all(c.startswith("I can't stand") for c in hj["centroid"])
    for i in items["groups"]["dislikes"]:
        it = items["by_id"][i]
        assert sorted(r["a"] for r in it["relation_probes"]) == ["No"] * 3 + ["Yes"] * 3
    mech = {i for i, it in items["by_id"].items() if it["mech"]}
    assert mech == set(items["groups"]["dislikes"]) | set(items["groups"]["one_of_many"]) | {
        "vegetarian", "loves_hiking", "tight_budget", "early_riser"}
