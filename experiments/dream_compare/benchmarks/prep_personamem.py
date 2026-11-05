#!/usr/bin/env python
"""PersonaMem v1, 32k split -> dream_compare split "32k" (README: data/personamem/README.md).

  python -I experiments/dream_compare/benchmarks/prep_personamem.py [--download]
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_common as PC  # noqa: E402

PM = PC._load("dc_prep_personamem", HERE / "personamem.py")
SHA256 = {"questions": "cccd34cf53e0bc4d9536c04cff5ca045156d9a4e227e83327112482840bbc93c",
          "contexts": "217247ebfec9e8442fc53570c795ab69f21aad08745f7de78d9beab51b122d4a"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args(argv)
    d = PC.EXTERNAL / "personamem"
    raw = {}
    for k, fn in PM.FILES.items():
        raw[k] = {"file": fn, "sha256": PC.fetch(PC.hf_url(PM.HF_REPO, PM.HF_REV, fn), d / fn, SHA256[k], a.download)}
    b = PM.convert(open(d / PM.FILES["questions"], encoding="utf-8").read(),
                   open(d / PM.FILES["contexts"], encoding="utf-8").read())
    PC.write_split(b, "personamem", "32k", raw, {"n_personas": len({it.meta["persona_id"] for it in b.items})})


if __name__ == "__main__":
    main()
