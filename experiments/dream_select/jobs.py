#!/usr/bin/env python
"""dream_select job driver (PREREG addendum 1): pytest and several run.py stages in ONE Python process, so that torch /
transformers are imported once per job (as experiments/dream_bind/multi.py; imports from the conda env on /scratch took
~12 min per interpreter on 2026-10-25). Infrastructure only: every run.py call is unchanged.

  SPECS="STAGE|OUT|EXTRA;STAGE|OUT|EXTRA" python experiments/dream_select/jobs.py --model M [--device cuda] \
        [--pytest "-q ..."]

Each spec runs run.main(["--model", M, "--stage", STAGE, "--out", OUT, "--device", D, *EXTRA]); between specs the model is
released (gc + cuda.empty_cache). A failing pytest or spec raises (the job fails).
"""

import argparse
import gc
import importlib.util
import os
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def parse_specs(text):
    """'STAGE|OUT|EXTRA;...' -> [(stage, out, extra list)] (empty entries skipped)."""
    out = []
    for spec in text.split(";"):
        if not spec.strip():
            continue
        st, o, ex = (spec.split("|") + ["", ""])[:3]
        out.append((st.strip(), o.strip(), shlex.split(ex)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pytest", default=None, help="pytest arguments (shell-split), run first in this process")
    a = ap.parse_args(argv)
    t0 = time.time()
    sys.path.insert(0, str(REPO / "src"))
    import torch  # noqa: F401  (the one slow import of the job)
    print(f"[jobs] imports in {time.time() - t0:.0f}s", flush=True)
    if a.pytest:
        import pytest
        t1 = time.time()
        rc = pytest.main(shlex.split(a.pytest))
        print(f"[jobs] pytest exit {rc} in {time.time() - t1:.0f}s", flush=True)
        if rc != 0:
            sys.exit(f"pytest failed ({rc})")
        print("pytest passed", flush=True)
    spec = importlib.util.spec_from_file_location("dream_select_run_jobs", HERE / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    for st, o, ex in parse_specs(os.environ.get("SPECS", "")):
        argv_ = ["--model", a.model, "--stage", st, "--out", o, "--device", a.device, *ex]
        print(f"=== run.py {' '.join(argv_)}", flush=True)
        run.main(argv_)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Done: {st} {o}", flush=True)
    print(f"[jobs] total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
