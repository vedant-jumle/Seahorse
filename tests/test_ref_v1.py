"""ref_v1 pure helpers (experiments/ref_v1/shifts.py): moment pooling, reference averaging, norm
matching, the disclosure pool and the loop-aware lean. CPU only, no model."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

_P = Path(__file__).resolve().parents[1] / "experiments" / "ref_v1" / "shifts.py"
_spec = importlib.util.spec_from_file_location("ref_v1_shifts", _P)
sh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sh)


def test_moment_weights_exclude_tail_and_sum_to_one():
    ent = torch.tensor([1.0, 3.0, 2.0, 9.0, 9.0])  # last 2 = template tail
    w = sh.moment_weights(ent, tail_len=2)
    assert w.shape == (3,)
    assert torch.allclose(w, torch.tensor([1.0, 3.0, 2.0]) / 6.0)
    with pytest.raises(AssertionError):
        sh.moment_weights(ent[:2], tail_len=2)


def test_reference_averaging_equals_pooled_tokenwise_differences():
    torch.manual_seed(0)
    n, d, K, tail = 7, 5, 4, 2
    w = sh.moment_weights(torch.rand(n) + 0.1, tail)
    h_with = torch.randn(n, d)
    refs = torch.randn(K, n, d)
    # mean over K references, pooled == pooled token-wise (with - mean_k ref_k)
    want = (w[:, None] * (h_with - refs.mean(0))[:n - tail]).sum(0)
    got = sh.reference_shift(sh.pool(h_with, w), [sh.pool(r, w) for r in refs])
    assert torch.allclose(got, want, atol=1e-6)
    assert torch.allclose(sh.reference_shift(sh.pool(h_with, w), torch.stack([sh.pool(r, w) for r in refs])), want,
                          atol=1e-6)
    # one reference run, and a constant reference (the hum mu)
    one = sh.reference_shift(sh.pool(h_with, w), sh.pool(refs[0], w))
    assert torch.allclose(one, (w[:, None] * (h_with - refs[0])[:n - tail]).sum(0), atol=1e-6)
    mu = torch.randn(d)
    assert torch.allclose(sh.reference_shift(sh.pool(h_with, w), mu), sh.pool(h_with - mu, w), atol=1e-6)


def test_norm_matching_keeps_direction():
    torch.manual_seed(1)
    s, base = torch.randn(16) * 40.0, torch.randn(16)
    m = sh.match_norm(s, base.norm())
    assert m.norm().item() == pytest.approx(base.norm().item(), rel=1e-5)
    assert sh.cos(m, s) == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(sh.match_norm(torch.zeros(4), 3.0), torch.zeros(4))  # zero stays zero, no nan


def test_pick_disclosure_excludes_and_is_deterministic():
    cands = [{"id": f"d{i}", "kind": "disposition", "category": "diet" if i < 3 else "likes",
              "topics": ["food"] if i == 5 else [], "experience": f"D{i}."} for i in range(8)]
    cands += [{"id": f"f{i}", "kind": "fact", "category": "pool", "topics": ["music"] if i == 0 else [],
               "experience": f"F{i}."} for i in range(30)]
    item = {"id": "d0", "category": "diet", "topics": ["food"], "exclude_ids": ["d7"]}
    pick = sh.pick_disclosure(cands, item, k=10, k_disp_max=8, seed=0)
    ids = [c["id"] for c in pick]
    assert len(ids) == 10 and len(set(ids)) == 10
    # eligible dispositions: d3, d4, d6 (d0-d2 same category, d5 shares a topic, d7 excluded by id)
    assert sorted(i for i in ids if i.startswith("d")) == ["d3", "d4", "d6"]
    assert ids == [c["id"] for c in sh.pick_disclosure(cands, item, k=10, k_disp_max=8, seed=0)]
    music = {"id": "x", "category": "likes", "topics": ["music"]}
    assert "f0" not in [c["id"] for c in sh.pick_disclosure(cands, music, k=24, k_disp_max=2, seed=0)]
    with pytest.raises(ValueError):
        sh.pick_disclosure(cands, item, k=40, k_disp_max=8, seed=0)


def test_lexicon_tokens_and_lex_gain():
    vocab = {" jazz": [5, 9], " swing": [6], " pop": [7], " big": [8], " big band": [8, 2], " boy band": [8, 3]}
    cons, inc = sh.lexicon_token_ids(vocab.__getitem__, {"consistent": ["jazz", "swing", "big band"],
                                                         "inconsistent": ["pop", "boy band"]})
    assert cons == [5, 6] and inc == [7]  # first subwords; 8 is on both sides -> dropped
    logits = torch.zeros(10)
    logits[5], logits[6], logits[7] = 3.0, 1.0, -1.0
    assert sh.lex_gain(logits, cons, inc) == pytest.approx(3.0)
    assert math.isnan(sh.lex_gain(logits, cons, []))


def test_lean_clean_drops_loops():
    labels = ["consistent", "consistent", "inconsistent", "neutral"]
    loops = [True, False, False, False]
    s = sh.lean_stats(labels, loops)
    assert (s["cons"], s["inc"], s["lean"], s["loop"]) == (0.5, 0.25, 0.25, 0.25)
    assert s["n_clean"] == 3 and s["lean_clean"] == pytest.approx(0.0)
    assert math.isnan(sh.lean_stats(["consistent"], [True])["lean_clean"])
    assert sh.rep_rate([1, 2, 3, 4] * 5) > 0.3 > sh.rep_rate(list(range(20)))
