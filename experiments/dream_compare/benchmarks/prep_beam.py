#!/usr/bin/env python
"""BEAM, 100K split -> dream_compare split "100K" (README: data/beam/README.md).

The raw file is a parquet; reading it needs pyarrow, which the cluster's seahorse env does not have. Either run this
where pyarrow exists (the local torch env), or pass --rows-json: a JSON dump of the rows that this script writes next to
the parquet when pyarrow is available (data/external/beam/100K.rows.json; its sha256 is in the manifest), so the
cluster rebuilds the split from that dump without pyarrow.

  python -I experiments/dream_compare/benchmarks/prep_beam.py [--download] [--rows-json PATH]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_common as PC  # noqa: E402

BE = PC._load("dc_prep_beam", HERE / "beam.py")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--rows-json", default=None, help="rows dump to use instead of the parquet (no pyarrow needed)")
    a = ap.parse_args(argv)
    d = PC.EXTERNAL / "beam"
    f = d / Path(BE.FILE).name
    sha = PC.fetch(PC.hf_url(BE.HF_REPO, BE.HF_REV, BE.FILE), f, BE.SHA256, a.download)
    dump = Path(a.rows_json) if a.rows_json else d / "100K.rows.json"
    if a.rows_json is None:
        try:
            import pyarrow.parquet as pq
            rows = pq.read_table(f).to_pylist()
            with open(dump, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False)
        except ImportError:
            assert dump.exists(), "no pyarrow: pass --rows-json (a dump written where pyarrow exists)"
            rows = json.load(open(dump, encoding="utf-8"))
    else:
        rows = json.load(open(dump, encoding="utf-8"))
    b = BE.convert(rows)
    PC.write_split(b, "beam", "100K", {"file": BE.FILE, "sha256": sha, "rows_dump": dump.name,
                                       "rows_dump_sha256": PC.sha256(dump)},
                   {"items_without_evidence": sum(1 for it in b.items if not it.evidence_turns),
                    "turn_markers_removed": sum(1 for c in b.contexts.values() for t in c.turns() if "marker" in t.meta)})


if __name__ == "__main__":
    main()
