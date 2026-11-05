#!/usr/bin/env python
"""dream_compare job driver: pytest and several run.py stages in ONE Python process (torch / transformers imported
once per job; imports from the conda env on /scratch are slow), as experiments/dream_select/jobs.py, whose spec parser
is reused.

  SPECS="STAGE|OUT|EXTRA;STAGE|OUT|EXTRA" python experiments/dream_compare/jobs.py --model M [--device cuda] \
        [--pytest "-q ..."]

Each spec runs run.main(["--model", M, "--stage", STAGE, "--out", OUT, "--device", D, *EXTRA]); between specs the model
is released. A failing pytest or spec raises (the job fails).
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DSJ = _load("dream_compare_ds_jobs", REPO / "experiments" / "dream_select" / "jobs.py")
parse_specs = DSJ.parse_specs


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
    run = _load("dream_compare_run_jobs", HERE / "run.py")
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
