"""bench_v1 data checks (CPU, tokenizer only). The tokenisation checks are skipped when the
Qwen tokenizer is not in the local HF cache; nothing is downloaded."""

from collections import Counter
from pathlib import Path

import pytest

from seahorse.bench import disposition_rate, load_bench, sample_pool, validate_bench

BENCH = Path(__file__).resolve().parents[1] / "data" / "bench_v1"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def bench():
    return load_bench(BENCH)


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception as e:  # not cached / offline
        pytest.skip(f"tokenizer {MODEL} not cached: {e}")


def test_schema_balance_duplicates(bench):
    res = validate_bench(bench, tok=None)
    assert not res["errors"], "\n".join(res["errors"])
    c = res["counts"]
    assert 30 <= c["items"] <= 50
    assert c["unrelated_probes"] >= 25 and c["pool_slots"] >= 50


def test_tokenisation(bench, tok):
    res = validate_bench(bench, tok=tok)
    assert not res["errors"], "\n".join(res["errors"])


def test_v01_compatible(bench):
    for it in bench["scenarios"]:
        assert {"id", "type", "experience", "counter", "followups", "probes", "measure"} <= set(it)
        assert [p["distance"] for p in it["probes"]] == ["exact", "paraphrase", "related"]
        key = {"a", "b"} if it["type"] == "disposition" else {"target", "foils"}
        assert {"prefix"} | key <= set(it["measure"])
        for rp in it["relation_probes"]:
            assert rp["text"].endswith("Answer yes or no.") and rp["a"] == rp["consistent"]
        assert Counter(rp["a"] for rp in it["relation_probes"]) == {
            "Yes": len(it["relation_probes"]) // 2, "No": len(it["relation_probes"]) // 2}


def test_pool_sampling(bench):
    hand = {it["slot"] for it in bench["scenarios"] if it.get("slot")}
    items = sample_pool(bench["pool"], n=30, seed=3, exclude_slots=hand)
    assert len({it["slot"] for it in items}) == 30 and not hand & {it["slot"] for it in items}
    assert sum(it["group"] == "partner" for it in items) <= 1
    again = sample_pool(bench["pool"], n=30, seed=3, exclude_slots=hand)
    assert [it["experience"] for it in items] == [it["experience"] for it in again]


def test_disposition_rate():
    lex = {"consistent": ["oat milk", "not spicy", "tofu"], "inconsistent": ["milk", "spicy", "beef"]}
    gens = ["Try a latte with oat milk.",        # oat milk beats milk
            "A beef stew with whole milk.",      # 2 inconsistent
            "Make it not spicy, with tofu.",     # not spicy beats spicy
            "Here is a haiku about the sea."]    # neutral
    r = disposition_rate(gens, lex)
    assert (r["consistent"], r["inconsistent"], r["neutral"], r["n"]) == (0.5, 0.25, 0.25, 4)
    assert r["lean"] == pytest.approx(0.25)
