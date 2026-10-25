#!/usr/bin/env python
"""dream_bind job driver: pytest and several run.py calls in ONE Python process, so that torch / transformers are imported
once per job. (On 2026-10-25 importing them from the conda env on /scratch took ~12 minutes per interpreter; the
addendum-8 smokes started 7 interpreters each and timed out. Infrastructure only: every run.py call is unchanged.)

  SPECS="STAGES|OUT|EXTRA;STAGES|OUT|EXTRA" python experiments/dream_bind/multi.py --model M [--device cuda]
        [--pytest "tests -m 'not slow'"] [--dtype fp32]

Each spec runs run.main(["--model", M, "--stage", *STAGES, "--out", OUT, "--device", D, "--dtype", T, *EXTRA]); between
specs the model is released (gc + cuda.empty_cache) so the fit check of the next call sees only its own memory. A
failing pytest or spec raises (the job fails), exactly as the shell loop did.
"""

import argparse
import gc
import os
import shlex
import sys
import time
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def parse_specs(text):
    """'STAGES|OUT|EXTRA;...' -> [(stages list, out, extra list)] (empty entries skipped)."""
    out = []
    for spec in text.split(";"):
        if not spec.strip():
            continue
        st, o, ex = (spec.split("|") + ["", ""])[:3]
        out.append((st.split(), o.strip(), shlex.split(ex)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--pytest", default=None, help="pytest arguments (shell-split), run first in this process")
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch  # noqa: F401  (the one slow import of the job)
    print(f"[multi] imports in {time.time() - t0:.0f}s", flush=True)
    if a.pytest:
        import pytest
        t1 = time.time()
        rc = pytest.main(shlex.split(a.pytest))
        print(f"[multi] pytest exit {rc} in {time.time() - t1:.0f}s", flush=True)
        if rc != 0:
            sys.exit(f"pytest failed ({rc})")
        print("pytest passed", flush=True)
    spec = importlib.util.spec_from_file_location("dream_bind_run_multi", HERE / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    for st, o, ex in parse_specs(os.environ.get("SPECS", "")):
        argv_ = ["--model", a.model, "--stage", *st, "--out", o, "--device", a.device, "--dtype", a.dtype, *ex]
        print(f"=== run.py {' '.join(argv_)}", flush=True)
        run.main(argv_)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Done: {o}", flush=True)
    print(f"[multi] total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
