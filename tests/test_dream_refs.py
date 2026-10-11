"""dream_refs (experiment 2; experiments/dream_refs/): the frame validator, the opposite / alternative /
paraphrase / null-value selection, the valence construction, the sign readers and the S3 rule, the S1 rule
(including the indeterminate case), the concept x sign shifts and memory, the loop-matched dose, the S2 / S4
rules, the fixed generation grid, and that the dreamer's prompts never carry a hand-written reference."""

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "dream_refs"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


refs = _load("dream_refs_refs", EXP / "refs.py")


# ------------------------------------------------------------------------------------------- frames


def test_fits_and_frame_validator():
    assert refs.fits("I can't stand {X} music.", "I can't stand rock music.") == "rock"
    assert refs.fits("I can't stand {X} music.", "I can’t  stand Rock music") == "Rock"  # quotes, spaces, final dot
    assert refs.fits("I can't stand {X} music.", "I love rock music.") is None
    assert refs.fits("By the way, I'm {X}.", "By the way, I'm on a keto diet.") == "on a keto diet"
    assert refs.fits("{X} music", "music") is None                                        # an empty filler never fits
    ok = '{"value": "jazz", "frame": "I can\'t stand {X} music.", "key_words": ["jazz"]}'
    f, probs = refs.check_frame(refs.parse_json_obj("Sure:\n" + ok), "I can't stand jazz music.")
    assert probs == [] and f["frame"] == "I can't stand {X} music." and f["key_words"] == ["jazz"]
    bad = [('{"value": "jazz", "frame": "I can\'t stand {X} {X}.", "key_words": ["jazz"]}', "slots"),
           ('{"value": "jazz", "frame": "I hate {X} music.", "key_words": ["jazz"]}', "frame(value) != sentence"),
           ('{"value": "jazz", "frame": "I can\'t stand {X} music.", "key_words": ["rock"]}', "not in the value"),
           ('{"value": "jazz", "frame": "I can\'t stand {X} music.", "key_words": []}', "no key words")]
    for text, why in bad:
        _, probs = refs.check_frame(refs.parse_json_obj(text), "I can't stand jazz music.")
        assert any(why in p for p in probs), (text, probs)
    assert refs.check_frame(None, "x")[1] == ["no JSON object"]
    # greedy invalid -> the first valid sample; nothing valid -> None
    fr, tried = refs.select_frame("not json", ["{}", ok], "I can't stand jazz music.")
    assert fr["source"] == "sample2" and len(tried) == 3
    assert refs.select_frame("x", ["y"], "I can't stand jazz music.")[0] is None


FR = {"value": "jazz", "frame": "I can't stand {X} music.", "key_words": ["jazz"]}
E = "I can't stand jazz music."


def test_negation_and_opposite_rules():
    assert refs.neg_count("I can't stand jazz") == 1 and refs.neg_count("I don't not like it") == 2
    assert refs.neg_count("I never read books; they bore me.") == 1 and refs.neg_count("I love jazz") == 0
    c = refs.check_opposite("I love jazz music.", E, FR)
    assert c["ok"] and c["soft"]["keeps_topic"] and c["soft"]["same_frame"] and not c["soft"]["neg_kept"]
    c = refs.check_opposite("I don't hate jazz music.", E, FR)        # allowed (no NEW negation), but flagged
    assert c["ok"] and c["soft"]["neg_kept"]
    pos = "By the way, I'm vegetarian."
    fr2 = {"value": "vegetarian", "frame": "By the way, I'm {X}.", "key_words": ["vegetarian"]}
    assert not refs.check_opposite("By the way, I'm not vegetarian.", pos, fr2)["hard"]["no_new_negation"]
    assert refs.check_opposite("By the way, I love eating meat.", pos, fr2)["ok"]
    assert not refs.check_opposite(pos, pos, fr2)["ok"]                     # the experience itself
    assert not refs.check_opposite("By the way, I'm vegetarian!", pos, fr2)["hard"]["different_value"] or \
        not refs.check_opposite("By the way, I'm vegetarian!", pos, fr2)["hard"]["not_experience"]
    # the greedy output fails a hard check -> the first sample that passes; nothing passes -> greedy, not ok
    s = refs.select_opposite("By the way, I'm not vegetarian.", ["By the way, I'm not a vegetarian.",
                                                                 "By the way, I eat meat."], pos, fr2)
    assert s["ok"] and s["source"] == "sample2" and s["text"] == "By the way, I eat meat."
    s = refs.select_opposite("By the way, I'm not vegetarian.", [], pos, fr2)
    assert not s["ok"] and s["source"] == "greedy"


def test_alternatives_selection():
    samples = ["1. I can't stand rock music.\n2. I can't stand pop music.\n3. I can't stand jazz music.\n"
               "4. I can't stand smooth jazz music.\n5. I love jazz music.\n6. I can't stand classical music.",
               "1. I can't stand pop music.\n2. I can't stand Rock music.\n3. I hate country music.\n"
               "4. I can't stand pop music.",          # a repeat inside one sample counts once
               "- I can't stand pop music.\n- I can't stand metal music."]
    kept, info = refs.select_alternatives(samples, E, FR, "I love jazz music.", keep=8)
    texts = [a["text"] for a in kept]
    assert texts[:2] == ["I can't stand pop music.", "I can't stand rock music."]   # counts 3, 2 (dedup by value)
    assert [a["count"] for a in kept[:2]] == [3, 2]
    assert "I can't stand jazz music." not in texts and "I can't stand smooth jazz music." not in texts  # key word
    assert "I love jazz music." not in texts                                                      # the opposite
    assert "I hate country music." in texts and kept[-1].get("frame_miss")      # filled from the frame misses, last
    assert info["n_valid"] == 4 and info["n_frame_miss"] == 1
    kept2, _ = refs.select_alternatives(samples, E, FR, "I love jazz music.", keep=3)
    assert [a["text"] for a in kept2] == ["I can't stand pop music.", "I can't stand rock music.",
                                          "I can't stand classical music."]
    # an opposite that fits the frame (one-of-many style) excludes its value
    fr = {"value": "Japanese", "frame": "I love {X} food.", "key_words": ["Japanese"]}
    kept3, _ = refs.select_alternatives(["I love Mexican food.\nI love Thai food."], "I love Japanese food.", fr,
                                        "I love Mexican food.", keep=8)
    assert [a["text"] for a in kept3] == ["I love Thai food."]


def test_paraphrases_and_null_values():
    paras, info = refs.select_paraphrases(["1. I really can't stand jazz.\n2. Jazz is not for me at all.\n"
                                           "3. I can't stand jazz music.", "1. I really can't stand jazz."],
                                          E, ["jazz"], keep=4)
    assert [p["text"] for p in paras][0] == "I really can't stand jazz." and paras[0]["count"] == 2
    assert all(p["text"] != E for p in paras)
    nulls, _ = refs.select_null_values('["tax", "jazz", "rock", "plumbing", "tax"]', ['["gardening"]'], E, FR,
                                       ["rock"], keep=8)
    assert nulls == ["I can't stand tax music.", "I can't stand plumbing music.", "I can't stand gardening music."]
    assert refs.select_null_values('["a"]', [], E, None, [])[0] == []


# ------------------------------------------------------------------------------------------ valence


def test_valence_construction():
    pairs = refs.valence_pairs([{"thing": "board games"}, "rainy days"], refs.FRAMES_DEFAULT)
    assert len(pairs) == 6
    assert pairs[0] == {"topic": "board games", "frame": 0, "like": "I really like board games.",
                        "hate": "I can't stand board games."}
    assert pairs[1]["like"] == "Board games is my favourite." and pairs[1]["hate"] == "Board games is the worst."
    assert pairs[5]["hate"] == "I always avoid rainy days."
    words, phrases = refs.item_value_words(["I can't stand jazz music.", "I live in Norway."], ["seafood"],
                                           [{"consistent": ["tea room"], "inconsistent": ["beach"]}])
    assert {"jazz", "music", "norway", "seafood"} <= words and "stand" not in words and "live" in words
    samples = ["food | pizza\nfood | jazz brunch\nmusic | opera\nweather | rainy days\nfood | carrots\nfood | olives",
               "food | pizza\nplaces | the tea room\nsport | chess\nfood | carrots\nweather | snow"]
    topics, info = refs.select_topics(samples, words, phrases, n=5)
    things = [t["thing"] for t in topics]
    assert "jazz brunch" not in things and "the tea room" not in things      # an item value / a lexicon phrase
    assert things[:3] == ["pizza", "rainy days", "opera"]   # round-robin: food (most counts), weather, music
    assert len(things) == 5 and info["n_excluded"] == 2
    assert refs.parse_topic_lines("1. food | Pizza.\nheader:\ncarrots") == [("food", "Pizza"), ("", "carrots")]


def test_sign_readers_and_s3_rule():
    v = {20: torch.tensor([1.0, 0.0, 0.0])}
    plain_like = {20: torch.tensor([0.5, 1.0, 0.0])}
    plain_hate = {20: torch.tensor([-0.2, 1.0, 0.0])}
    assert refs.read_signs(plain_like, v)[20]["sign"] == 1 and refs.read_signs(plain_hate, v)[20]["sign"] == -1
    assert refs.read_signs(plain_like, v)[20]["cos"] == pytest.approx(0.5 / math.sqrt(1.25))
    from seahorse.stats import boot_ci
    boot = lambda n: boot_ci(n, 500, 7)
    items = [f"i{k}" for k in range(10)]
    true = {i: (1 if k < 7 else -1) for k, i in enumerate(items)}
    pred = dict(true)
    marg = {i: 0.3 * true[i] for i in items}
    r = refs.s3_layer(pred, marg, true, items, boot)
    assert r["accuracy"] == 1.0 and r["margin"] == pytest.approx(0.3) and r["holds"]
    pred2 = dict(true, i0=-1, i1=-1)                         # 0.8 accuracy: fails even with a positive margin
    r2 = refs.s3_layer(pred2, marg, true, items, boot)
    assert r2["accuracy"] == pytest.approx(0.8) and not r2["holds"] and r2["wrong"] == ["i0", "i1"]
    marg3 = {i: (0.5 if k == 0 else -0.01) * true[i] for k, i in enumerate(items)}   # right sign, CI crosses 0
    pred3 = {i: true[i] for i in items}
    r3 = refs.s3_layer(pred3, marg3, true, items, boot)
    assert r3["accuracy"] == 1.0 and r3["margin_lo"] <= 0 and not r3["holds"]
    assert refs.s3_rule({20: r2, 23: r})["holds"] and refs.s3_rule({20: r2, 23: r})["layers_holding"] == [23]
    assert not refs.s3_rule({20: r2, 21: r3})["holds"]
    assert refs.TRUE_SIGN == {"leanings": 1, "one_of_many": 1, "dislikes": -1}


# ----------------------------------------------------------------------------------------------- S1


def test_s1_item_rule_and_indeterminate():
    assert refs.s1_item(0.9, 0.5, 0.6) == "pass"
    assert refs.s1_item(0.85, 0.5, 0.7) == "fail"           # margin 0.15 < 0.2
    assert refs.s1_item(0.75, 0.2, 0.3) == "fail"           # absolute cosine < 0.8
    assert refs.s1_item(0.99, 0.8, 0.1) == "indeterminate"  # a null reaches 0.8: no reference could clear it
    assert refs.s1_item(0.99, 0.1, 0.85) == "indeterminate"
    assert refs.s1_item(0.99, 0.1, None) == "indeterminate"  # null_frame missing (no valid frame)
    assert refs.s1_item(0.99, 0.1, float("nan")) == "indeterminate"
    items = [f"i{k}" for k in range(29)]
    st = {l: {i: "pass" for i in items} for l in (20, 21, 23)}
    assert refs.s1_rule(st)["verdict"] == "holds"
    st[21].update({i: "fail" for i in items[:6]})                       # 23/29 = 0.79 < 0.8 at one layer
    r = refs.s1_rule(st)
    assert r["verdict"] == "fails" and r["per_layer"][21]["verdict"] == "fails"
    st = {l: {i: "pass" for i in items} for l in (20, 21, 23)}
    st[20].update({i: "indeterminate" for i in items[:10]})              # 19 determinate, all pass -> holds
    st[20].update({items[10]: "fail"})                                   # 18/19 still >= 0.8
    r = refs.s1_rule(st)
    assert r["verdict"] == "holds" and r["per_layer"][20]["n_determinate"] == 19
    st[23].update({i: "indeterminate" for i in items[:15]})              # 14 determinate < 15 -> indeterminate
    r = refs.s1_rule(st)
    assert r["per_layer"][23]["verdict"] == "indeterminate" and r["verdict"] == "indeterminate"
    st[21].update({i: "fail" for i in items[:10]})                       # a determinate layer fails -> fails
    assert refs.s1_rule(st)["verdict"] == "fails"
    assert refs.s1_combine({"opposite": "holds", "centroid": "holds"}) == "holds"
    assert refs.s1_combine({"opposite": "holds", "centroid": "indeterminate"}) == "indeterminate"
    assert refs.s1_combine({"opposite": "fails", "centroid": "indeterminate"}) == "fails"


# ------------------------------------------------------------------------------------- concept x sign


def delta_build(shifts, keys):
    M = torch.zeros(shifts.shape[1], keys.shape[1])
    for s, k in zip(shifts, keys):
        M += torch.outer(s - M @ k, k)
    return M


def test_concept_x_sign_fields():
    torch.manual_seed(0)
    layers = (20, 21, 23)
    moments = [{"shifts": {"gen_cent": {l: torch.randn(5) for l in layers}}} for _ in range(3)]
    sign = {20: -1, 21: 1, 23: -1}
    refs.signed_shifts(moments, "gen_cent", "cs_sign", sign)
    refs.signed_shifts(moments, "gen_cent", "cs_oracle", {l: -1 for l in layers})   # a dislike's true sign
    for m in moments:
        for l in layers:
            assert torch.equal(m["shifts"]["cs_sign"][l], sign[l] * m["shifts"]["gen_cent"][l])
            assert torch.equal(m["shifts"]["cs_oracle"][l], -m["shifts"]["gen_cent"][l])
    # the stored memory and the injected field are linear in the shifts: signed memory = sign x concept memory
    keys = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    for l in layers:
        Mc = delta_build(torch.stack([m["shifts"]["gen_cent"][l] for m in moments]), keys)
        Ms = delta_build(torch.stack([m["shifts"]["cs_sign"][l] for m in moments]), keys)
        assert torch.allclose(Ms, sign[l] * Mc, atol=1e-6)
        q = keys[2]
        assert torch.allclose(Ms @ q, sign[l] * moments[2]["shifts"]["gen_cent"][l], atol=1e-5)  # exact recall
    assert refs.mean_shift(moments, "cs_oracle", 20).allclose(-refs.mean_shift(moments, "gen_cent", 20))


def test_loop_matched_dose():
    assert refs.loop_matched({0.5: 0.0, 1: 0.05, 2: 0.4}) == (1, True)
    assert refs.loop_matched({0.5: 0.0, 1: 0.10, 2: 0.10}) == (2, True)     # <= 10% qualifies
    assert refs.loop_matched({0.5: 0.2, 1: 0.5, 2: 0.9}) == (0.5, False)


# ----------------------------------------------------------------------------- the grid and the rules


@pytest.fixture(scope="module")
def run_mod():
    return _load("dream_refs_run", EXP / "run.py")


def test_fixed_generation_grid(run_mod):
    cfg = yaml.safe_load(open(EXP / "config_qwen35_2b.yaml"))
    cs = run_mod.conditions(cfg)
    ids = [c.cid for c in cs]
    want = ["nomem", "ctx"] + [f"{v}@{a:g}" for v in ("hand:opposite", "gen:opposite", "gen:centroid",
                                                       "cs:concept_x_sign", "cs:oracle") for a in (0.5, 1, 2)]
    assert ids == want
    assert all(c.kind == "ref" and run_mod.core.gated(c) for c in cs[2:])
    assert {c.mem for c in cs[2:]} == set(run_mod.MEM_SHIFT) and cs[2].mem == "design"   # hand:opposite = core_v1's
    assert {c.cid for c in cs if c.seed2} == set(cfg["seed2"])
    rows = run_mod.core.row_specs([c for c in cs if c.kind != "ctx"], cfg["samples"])
    assert len(rows) == 16 * 5 + 4 * 4 == cfg["max_batch"]          # one batch per prompt
    prim = [c.cid for c in run_mod.primary(cfg, cs)]
    assert prim == ["nomem", "ctx"] + [f"{v}@1" for v in run_mod.VARIANTS]
    # everything that shapes a memory or a reply is core_v1's
    c1 = yaml.safe_load(open(ROOT / "experiments" / "core_v1" / "config_qwen35_2b.yaml"))
    for k in ("model", "dtype", "layers", "key", "sampler", "samples", "answer_cap", "yn_cap", "unrel_samples", "loop",
              "judge", "generic"):
        assert cfg[k] == c1[k], k
    assert cfg["seeds"]["base"] == c1["seeds"]["base"] and cfg["seeds"]["bootstrap"] == 7
    assert run_mod.cv1.TAG == "core_v1"                          # the same random streams as core_v1


def test_s2_s4_rules():
    an = _load("dream_refs_analyze", EXP / "analyze.py")

    def row(g, v, d, lo, hi):
        return {"group": g, "variant": v, "alpha": 1, "matched": True, "loop": 0.0, "d_j_lean": d, "d_j_lean_lo": lo,
                "d_j_lean_hi": hi, "wrong_items": 0.0}
    good = [row("dislikes", "gen:opposite", 0.2, 0.05, 0.35), row("dislikes", "gen:centroid", -0.3, -0.5, -0.1),
            row("dislikes", "cs:concept_x_sign", 0.15, 0.02, 0.3), row("dislikes", "cs:oracle", 0.2, 0.05, 0.3),
            row("leanings", "gen:opposite", 0.2, 0.05, 0.35), row("leanings", "cs:concept_x_sign", 0.1, -0.05, 0.2)]
    C = an.claims(None, good, "/nonexistent")
    assert C["S2"]["holds"] and C["S4"]["holds"] and C["decisions"]["concept_x_sign_works"]
    bad = [dict(r, d_j_lean_lo=-0.01) if r["variant"] == "cs:concept_x_sign" and r["group"] == "dislikes" else r
           for r in good]
    C = an.claims(None, bad, "/nonexistent")
    assert not C["S4"]["holds"] and "sign reader" in C["S4"]["reading"]
    worse = [dict(r, d_j_lean=0.0) if r["variant"] == "cs:concept_x_sign" and r["group"] == "leanings" else r
             for r in good]
    assert not an.claims(None, worse, "/nonexistent")["S4"]["parts"]["leanings_not_worse"]
    s2bad = [dict(r, d_j_lean_hi=0.01) if r["variant"] == "gen:centroid" else r for r in good]
    assert not an.claims(None, s2bad, "/nonexistent")["S2"]["holds"]


# ------------------------------------------------------------------------------- the dreamer's inputs


def test_dreamer_never_sees_hand_references(tmp_path):
    dr = _load("dream_refs_dream", EXP / "dream.py")
    val = _load("dream_refs_validate", EXP / "validate.py")
    cfg = dr.load_cfg(EXP / "config_dream.yaml")
    inputs = dr.dream_inputs(cfg)
    assert len(inputs) == 29 and all(set(x) == {"id", "group", "experience"} for x in inputs)
    items = val.load_items(yaml.safe_load(open(EXP / "config_dream.yaml")))
    hand = [it["counter"] for it in items.values()] + [a for it in items.values() for a in it["centroid"]]
    hand_lex = [w for it in items.values() for s in ("consistent", "inconsistent") for w in it["lexicon"][s]]
    P = yaml.safe_load(open(EXP / "dream_prompts.yaml"))
    for step, spec in P.items():
        if not isinstance(spec, dict):
            continue
        for k in ("template", "template_no_frame"):
            t = spec.get(k)
            if t:
                assert not any(h in t for h in hand), (step, k)
                assert not any(refs.has_phrase(t, w) for w in hand_lex if len(w) > 3
                               and w.lower() not in ("rainy", "board games")), (step, k)
    # the mock dream (no model) runs through the validator for all 29 items without a structural error
    class A:
        mock, tiny, items, random_model = True, False, None, None
    res = dr.run(cfg, A, dr.MockDreamer(), inputs, tmp_path / "dream")
    out, valence, errors = val.validate(res, items, yaml.safe_load(open(EXP / "config_dream.yaml")))
    assert not errors and len(out) == 29 and all(len(v["alternatives"]) == 8 for v in out.values())
    assert all(v["null_swap"]["partner"] == items[i]["swap"] for i, v in out.items())
    assert len(valence["pairs"]) == 3 * len(valence["topics"])
