#!/usr/bin/env python
"""LongMemEval -> dream_compare splits (README: data/longmemeval/README.md, section "dream_compare").

  oracle   all 500 items of longmemeval_oracle.json (evidence sessions only)
  s120     120 items of longmemeval_s_cleaned.json: 20 per question type, random.Random(7) over the sorted ids of
           each type (longmemeval.stratified_ids); full ~115k-token haystacks

  python -I experiments/dream_compare/benchmarks/prep_longmemeval.py [--download]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_common as PC  # noqa: E402

LME = PC._load("dc_prep_lme", HERE / "longmemeval.py")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["oracle", "s120"])
    a = ap.parse_args(argv)
    d = PC.EXTERNAL / "longmemeval"
    if "oracle" in a.splits:
        f = d / LME.FILES["oracle"]
        sha = PC.fetch(PC.hf_url(LME.HF_REPO, LME.HF_REV, LME.FILES["oracle"]), f, LME.SHA256["oracle"], a.download)
        b = LME.convert(json.load(open(f, encoding="utf-8")), "oracle")
        PC.write_split(b, "longmemeval", "oracle", {"file": LME.FILES["oracle"], "sha256": sha},
                       {"single_turn_items": len(LME.single_turn_ids(b)),
                        "single_turn_items_user_only": len(LME.single_turn_ids(b, user_only=True))})
    if "s120" in a.splits:
        f = d / LME.FILES["s"]
        sha = PC.fetch(PC.hf_url(LME.HF_REPO, LME.HF_REV, LME.FILES["s"]), f, LME.SHA256["s"], a.download)
        raw = json.load(open(f, encoding="utf-8"))
        ids = LME.stratified_ids(raw)
        keep = set(ids)
        b = LME.convert([x for x in raw if x["question_id"] in keep], "s120")
        b.items.sort(key=lambda it: ids.index(it.item_id))
        n_tok_chars = sum(len(t.text) for c in b.contexts.values() for t in c.turns())
        PC.write_split(b, "longmemeval", "s120", {"file": LME.FILES["s"], "sha256": sha},
                       {"sample": {"per_type": LME.S_SAMPLE_PER_TYPE, "seed": LME.S_SAMPLE_SEED,
                                   "rule": "random.Random(seed).sample over the sorted ids of each type, types in "
                                           "longmemeval.TYPES order"},
                        "history_chars_mean": n_tok_chars / max(len(b.items), 1)})


if __name__ == "__main__":
    main()
