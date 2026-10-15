#!/usr/bin/env python
"""PrefEval dreaming pass (experiment 2 addendum): the dream_refs dreamer (Qwen3.5-9B, bf16, thinking off) writes,
from each sampled preference sentence ALONE, its frame, an opposite and alternatives with dream_refs' FINAL v3
prompts (../dream_prompts.yaml, unchanged), and neutral follow-ups with prompts.yaml. The dreamer never sees the
test question or the topic label.

  python experiments/dream_refs/prefeval/dream.py --out RUN/dream [--tiny] [--mock] [--random-model qwen2]

Per item: frame (greedy + 12 samples), opposite (greedy + 12), alternatives (12 samples x 12),
follow-ups (12 samples x 4). Same sampler, seeds (seed_of("dream_refs/<step>" | "prefeval/followups", prompt,
seed_base)) and fallbacks as dream_refs/dream.py. Outputs RUN/dream/{raw.jsonl.gz, dream.json, config.json};
validate.py redoes every selection from the raw outputs.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


drd = _load("seahorse_dream_refs_dream", EXP / "dream.py")
refs = drd.refs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--out", required=True)
    p.add_argument("--items", nargs="+", default=None)
    p.add_argument("--random-model", default=None, choices=["qwen2"])
    p.add_argument("--tiny", action="store_true", help="3 items, short caps, 2 samples")
    p.add_argument("--mock", action="store_true", help="no model: canned outputs (only for --tiny pipelines)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def load_cfgs(path):
    cfg = yaml.safe_load(open(path))
    dcfg = drd.load_cfg(REPO / cfg["dream_config"])
    return cfg, dcfg


def dream_inputs(cfg):
    """What the dreamer may see: per sampled item only its id and the preference sentence."""
    S = json.load(open(REPO / cfg["sample"]["file"]))
    return [{"id": it["id"], "experience": it["preference"]} for it in S["items"]]


MOCK_FOLLOWUPS = ["I'm sorting out my plans for the coming weekend.", "My sister is visiting us next month.",
                  "I have a long train journey on Friday.", "I'm reorganising my calendar for the autumn."]


class MockDreamer(drd.MockDreamer):
    def __call__(self, step, prompt, greedy, n, max_new, seed, sentence=None, frame=None, **kw):
        if step == "followups":
            out = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(MOCK_FOLLOWUPS))
            return {"greedy": out if greedy else None, "samples": [out] * n, "capped": [False] * (n + greedy)}
        return super().__call__(step, prompt, greedy, n, max_new, seed, sentence=sentence, frame=frame, **kw)


def run(dcfg, fcfg, A, dreamer, items, out):
    P = yaml.safe_load(open(dcfg["prompts"]))
    P["followups"] = yaml.safe_load(open(REPO / fcfg))["followups"]
    assert yaml.safe_load(open(dcfg["prompts"]))["version"] == 3, "the FINAL v3 reference prompts are required"
    n = dcfg["n_samples"]
    raw, by_item = [], {}
    t0 = time.time()

    def ask(step, item_id, greedy, **fill):
        spec = P[step]
        tmpl = spec["template_no_frame"] if (step == "alternatives" and fill.get("frame") is None) else spec["template"]
        prompt = drd.render(tmpl, **{k: v for k, v in fill.items() if v is not None})
        tag = "prefeval/followups" if step == "followups" else f"dream_refs/{step}"
        seed = drd.seed_of(tag, prompt, dcfg["seed_base"])
        r = dreamer(step, prompt, greedy, n, spec["max_new"] if not A.tiny else min(spec["max_new"], 48), seed,
                    sentence=fill.get("sentence"), frame=fill.get("frame"))
        raw.append({"item": item_id, "step": step, "prompt": prompt, "seed": seed, **r})
        return r

    for k, it in enumerate(items):
        e = it["experience"]
        d = {"id": it["id"], "experience": e}
        r = ask("frame", it["id"], True, sentence=e)
        d["frame_raw"] = r
        fr, tried = refs.select_frame(r["greedy"], r["samples"], e)
        d["frame_used"], d["frame_tried"] = fr, tried
        d["opposite_raw"] = ask("opposite", it["id"], True, sentence=e)
        cat = (fr.get("category") or f'the same kind of thing as "{fr["value"]}"') if fr else None
        d["alternatives_raw"] = ask("alternatives", it["id"], False, sentence=e, frame=fr["frame"] if fr else None,
                                    value=fr["value"] if fr else None, category=cat)
        d["followups_raw"] = ask("followups", it["id"], False, sentence=e)
        by_item[it["id"]] = d
        el = time.time() - t0
        print(f"[pe-dream] {it['id']}: frame {'ok (' + fr['source'] + ')' if fr else 'NONE'} | {el / 60:.1f} min, "
              f"eta {el / (k + 1) * (len(items) - k - 1) / 60:.1f} min", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "raw.jsonl.gz", "wt") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    res = {"mock": bool(A.mock), "tiny": bool(A.tiny), "subset": bool(A.items), "random_model": A.random_model,
           "model": dcfg["model"],
           "items": by_item, "minutes": (time.time() - t0) / 60}
    json.dump(res, open(out / "dream.json", "w"), indent=1)
    return res


def main(argv=None):
    A = parse_args(argv)
    cfg, dcfg = load_cfgs(A.config)
    if A.tiny:
        dcfg["n_samples"] = 2
    items = dream_inputs(cfg)
    if A.items:
        items = [it for it in items if it["id"] in A.items]
    elif A.tiny:
        items = items[:3]
    out = Path(A.out)
    if A.mock:
        dreamer = MockDreamer()
    else:
        dcfg["model_tokenizer_tiny"] = "Qwen/Qwen3.5-2B" if A.random_model else None
        dreamer = drd.Dreamer(dcfg, A)
    res = run(dcfg, cfg["followup_prompts"], A, dreamer, items, out)
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        import subprocess
        env["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        pass
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    json.dump({"cfg": cfg, "dream_cfg": dcfg, "args": vars(A), "env": env, "ref_prompts_sha": sha(dcfg["prompts"]),
               "followup_prompts_sha": sha(REPO / cfg["followup_prompts"]), "n_items": len(items),
               "minutes": res["minutes"]}, open(out / "config.json", "w"), indent=2, default=str)
    print(f"[pe-dream] done: {len(items)} items in {res['minutes']:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
