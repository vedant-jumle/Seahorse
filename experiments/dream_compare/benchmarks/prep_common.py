#!/usr/bin/env python
"""Shared helpers of the prep_*.py scripts (standard library only): pinned downloads into data/external/<bench>/
(git-ignored; untrusted files, parsed only, never executed), sha256 checks, the processed split file
(data/external/<bench>/processed/<split>.json) and the small committed manifest (data/<bench>/manifest_<split>.json)."""

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
EXTERNAL = REPO / "data" / "external"
DATA = REPO / "data"


def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("dc_prep_schema", HERE / "schema.py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dst, want_sha=None, download=True):
    """dst if present (sha checked), else downloaded from url (when download) to dst.tmp, checked, renamed."""
    dst = Path(dst)
    if not dst.exists():
        if not download:
            sys.exit(f"{dst} missing (run with --download)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        t0 = time.time()
        print(f"downloading {url} -> {dst}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "seahorse-dream-compare-prep"})
        token = os.environ.get("HF_TOKEN")
        if token and "huggingface.co" in url:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        tmp.replace(dst)
        print(f"  {dst.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)
    got = sha256(dst)
    if want_sha:
        assert got == want_sha, f"{dst}: sha256 {got} != pinned {want_sha}"
    return got


def hf_url(repo, rev, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


def write_split(bench, name, split, raw_info, extra=None, item_ids=True):
    """Validate, write data/external/<name>/processed/<split>.json and data/<name>/manifest_<split>.json."""
    bad = S.validate(bench)
    assert not bad, f"{name}/{split}: {len(bad)} schema problems, e.g. {bad[:5]}"
    out = EXTERNAL / name / "processed" / f"{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    S.save(bench, out)
    man = {"bench": name, "split": split, "processed": str(out.relative_to(REPO)), "processed_sha256": sha256(out),
           "raw": raw_info, "counts": bench.counts(), "meta": bench.meta,
           "built": time.strftime("%Y-%m-%d"), "built_by": f"experiments/dream_compare/benchmarks/prep_{name}.py"}
    if extra:
        man.update(extra)
    if item_ids:
        man["item_ids"] = [it.item_id for it in bench.items]
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"manifest_{split}.json", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: man[k] for k in ("bench", "split", "counts", "processed_sha256")}, indent=1))
    return man


def load_split(name, split):
    """The processed Bench of a split (data/external/<name>/processed/<split>.json; run prep_<name>.py first)."""
    p = EXTERNAL / name / "processed" / f"{split}.json"
    assert p.exists(), f"{p} missing: run python -I experiments/dream_compare/benchmarks/prep_{name}.py"
    return S.load(p)
