#!/usr/bin/env python
"""LoCoMo -> dream_compare split "locomo10" (README: data/locomo/README.md).

  python -I experiments/dream_compare/benchmarks/prep_locomo.py [--download]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_common as PC  # noqa: E402

LO = PC._load("dc_prep_locomo", HERE / "locomo.py")
SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args(argv)
    f = PC.EXTERNAL / "locomo" / "locomo10.json"
    sha = PC.fetch(LO.RAW_URL, f, SHA256, a.download)
    b = LO.convert(json.load(open(f, encoding="utf-8")))
    n_img = sum(1 for c in b.contexts.values() for t in c.turns() if "caption" in t.meta)
    PC.write_split(b, "locomo", "locomo10", {"file": "data/locomo10.json", "url": LO.RAW_URL, "sha256": sha},
                   {"non_adversarial": sum(not it.abstention for it in b.items),
                    "adversarial": sum(it.abstention for it in b.items), "image_turns_captioned": n_img,
                    "items_with_dropped_evidence": sum(1 for it in b.items if it.meta.get("evidence_dropped")),
                    "items_without_evidence": sum(1 for it in b.items if not it.evidence_turns)})


if __name__ == "__main__":
    main()
