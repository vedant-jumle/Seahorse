#!/usr/bin/env python
"""dream_refs dreaming pass: the dreamer (Qwen3.5-9B, bf16, thinking off) writes, from each preference item's
EXPERIENCE sentence alone, its frame template, an opposite, alternatives, paraphrases and null-frame values;
and once, the valence topics. docs/dreaming/02-scaffold.md §4-§5; PREREG.md "The dreaming pass".

  python experiments/dream_refs/dream.py --config experiments/dream_refs/config_dream.yaml --out RUN/dream
  python experiments/dream_refs/dream.py ... --tiny --random-model qwen2   # plumbing on a tiny random model (CPU)
  python experiments/dream_refs/dream.py ... --mock                         # no model: canned outputs for --tiny runs

Per item, in order (the later prompts use the dreamer's own earlier outputs, never a hand-written reference):
  frame        greedy + 12 samples; the first valid one (refs.select_frame) is used by the next prompts
  opposite     greedy + 12 samples (the samples are fallbacks if the greedy output fails a hard check)
  alternatives 12 samples at T = 0.7 (each lists 12)
  paraphrases  12 samples at T = 0.7 (each lists 4)
  null_values  greedy + 12 samples (needs a valid frame)
Once: topics, 12 samples at T = 0.7 (each lists 50).

Outputs: <out>/raw.jsonl.gz (every prompt and every output), <out>/dream.json (the same, by item and step, plus
the frame used for the later prompts), <out>/config.json. All selection and validation is redone from the raw
outputs, without a model, by validate.py.
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


refs = _load("seahorse_dream_refs_refs", HERE / "refs.py")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "config_dream.yaml"))
    p.add_argument("--out", required=True)
    p.add_argument("--items", nargs="+", default=None, help="only these items (default: all 29)")
    p.add_argument("--random-model", default=None, choices=["qwen2"], help="tiny random model (CPU plumbing)")
    p.add_argument("--tiny", action="store_true", help="2 items, short caps, 2 samples")
    p.add_argument("--mock", action="store_true", help="no model: canned outputs (only for --tiny pipelines)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def load_cfg(path):
    cfg = yaml.safe_load(open(path))
    for k in ("items", "bench", "prompts"):
        cfg[k] = str((REPO / cfg[k]).resolve())
    return cfg


def dream_inputs(cfg):
    """What the dreamer may see: per preference item only its id, group and EXPERIENCE sentence."""
    itm = _load("seahorse_core_v1_items", REPO / "experiments" / "core_v1" / "items.py")
    L = itm.load_items(cfg["items"], cfg["bench"])
    return [{"id": it["id"], "group": it["group"], "experience": it["experience"]}
            for it in L["items"] if it["group"] in cfg["groups"]]


def render(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def seed_of(tag, text, base):
    import zlib
    return (base * 1000003 + zlib.crc32(f"{tag}|{text}".encode())) % (2 ** 31 - 1)


# ----------------------------------------------------------------------------------------- dreamers


class Dreamer:
    """One prompt -> {greedy, samples}: core_v1's batched KV-cached generation, row 0 greedy, rows 1..n the
    samples (one torch.Generator stream per prompt, seeded from the prompt; the same numbers on any batch)."""

    def __init__(self, cfg, A):
        cv1 = _load("seahorse_core_v1_run", REPO / "experiments" / "core_v1" / "run.py")
        self.core, self.cv1 = cv1.core, cv1
        from seahorse.residual import load_model, text_model
        from seahorse.sessions import chat_ids
        self.chat_ids = chat_ids
        t0 = time.time()
        if A.random_model:
            model, tok = cv1.tv1.load_any(SimpleNamespace(tiny=A.random_model, model=cfg["model_tokenizer_tiny"],
                                                         device=A.device))
        else:
            model, tok = load_model(cfg["model"], device=A.device, dtype=getattr(torch, cfg["dtype"]))
        self.tok = cv1.thinking_off(tok)
        self.model, self.body, self.lm_head = model, text_model(model), model.lm_head
        self.vocab = self.lm_head.out_features
        eos = {tok.eos_token_id} | {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        ge = model.generation_config.eos_token_id
        eos |= set(ge if isinstance(ge, (list, tuple)) else [ge])
        self.eos = sorted(e for e in eos if isinstance(e, int) and e != tok.unk_token_id)
        s = cfg["sampler"]
        self.sampler = self.core.qwen_sampler(s["temp"], s["top_k"], s["top_p"], s["presence"])
        self.dev = A.device
        print(f"[dream] loaded {cfg['model']} ({'random ' + A.random_model if A.random_model else cfg['dtype']}) "
              f"in {time.time() - t0:.0f}s", flush=True)

    def __call__(self, step, prompt, greedy, n, max_new, seed, **_):
        ids = self.chat_ids(self.tok, prompt)
        rows = ([(True, None)] if greedy else []) + [(False, (1, j)) for j in range(1, n + 1)]
        with torch.inference_mode():
            outs = self.core.generate_batch(self.model, self.body, self.lm_head, ids, rows, max_new=max_new,
                                            eos=self.eos, vocab=self.vocab, sampler=self.sampler,
                                            streams={1: (seed, n + 1)} if n else None)
        texts = [self.tok.decode(o["ids"], skip_special_tokens=True).strip() for o in outs]
        capped = [o["capped"] for o in outs]
        return {"greedy": texts[0] if greedy else None, "samples": texts[1:] if greedy else texts,
                "capped": capped}


MOCK_VALUES = ["I like apples", "I like trains", "I like chess", "I like rain", "I like violins", "I like owls",
               "I like maths", "I like sailing", "I like puzzles", "I like Lisbon", "I like ink", "I like kites"]
MOCK_TOPICS = ["board games", "rainy days", "pizza", "opera", "cats", "dogs", "chess", "swimming", "jogging",
               "lemonade", "maths", "sushi bars", "gardening", "camping", "ice hockey", "karaoke", "puzzles",
               "thunderstorms", "the circus", "trains", "ferries", "sailing", "parrots", "owls", "painting",
               "pottery", "knitting", "baking bread", "fishing", "birdwatching", "carrots", "broccoli", "mangoes",
               "tea parties", "zoos", "aquariums", "rollercoasters", "tango", "ballet", "geography", "chemistry",
               "vacuuming", "ironing", "laundry", "cycling to work", "bus rides", "road trips", "snowmen",
               "fireworks", "crosswords"]


class MockDreamer:
    """Canned outputs in the dreamer's formats, for the --tiny pipeline only (validate / prep / lens / gen on a
    tiny random model need well-formed references). Marked "mock" everywhere; run.py refuses them without
    --tiny."""

    def __call__(self, step, prompt, greedy, n, max_new, seed, sentence=None, frame=None, **_):
        e = (sentence or "").rstrip(".")
        if step == "frame":
            kw = max(refs.content_words(e) or {"x"}, key=len)
            words = [w for w in e.split() if refs.stem(w.lower().strip(",.'")) == kw]
            out = json.dumps({"value": e, "frame": "{X}.", "key_words": [words[0].strip(",.") if words else e],
                              "category": "a thing about the user"})
        elif step == "opposite":
            out = "Honestly, " + e[:1].lower() + e[1:] + " - or the reverse."
        elif step == "alternatives":
            out = "\n".join(f"{i + 1}. {v}." for i, v in enumerate(MOCK_VALUES))
        elif step == "paraphrases":
            out = "\n".join(f"{i + 1}. {p}{e[:1].lower() + e[1:]}." for i, p in
                            enumerate(["Honestly, ", "To be honest, ", "Frankly, ", "In fact, "]))
        elif step == "null_values":
            out = json.dumps([f"I collect {x}" for x in ("stamps", "coins", "maps", "shells", "keys", "pens",
                                                         "cards", "rocks")])
        elif step == "topics":
            out = "\n".join(f"misc | {t}" for t in MOCK_TOPICS)
        else:
            raise ValueError(step)
        return {"greedy": out if greedy else None, "samples": [out] * n, "capped": [False] * (n + greedy)}


# ------------------------------------------------------------------------------------------------ run


def run(cfg, A, dreamer, items, out):
    P = yaml.safe_load(open(cfg["prompts"]))
    n = cfg["n_samples"]
    raw, by_item = [], {}
    t0 = time.time()

    def ask(step, item_id, greedy, **fill):
        spec = P[step]
        tmpl = spec["template_no_frame"] if (step == "alternatives" and fill.get("frame") is None) else spec["template"]
        prompt = render(tmpl, **{k: v for k, v in fill.items() if v is not None})
        seed = seed_of(f"dream_refs/{step}", prompt, cfg["seed_base"])
        r = dreamer(step, prompt, greedy, n, spec["max_new"] if not A.tiny else min(spec["max_new"], 48), seed,
                    sentence=fill.get("sentence"), frame=fill.get("frame"))
        rec = {"item": item_id, "step": step, "prompt": prompt, "seed": seed, **r}
        raw.append(rec)
        return r

    for k, it in enumerate(items):
        e = it["experience"]
        d = {"id": it["id"], "group": it["group"], "experience": e}
        r = ask("frame", it["id"], True, sentence=e)
        d["frame_raw"] = r
        fr, tried = refs.select_frame(r["greedy"], r["samples"], e)
        d["frame_used"], d["frame_tried"] = fr, tried
        d["opposite_raw"] = ask("opposite", it["id"], True, sentence=e)
        cat = (fr.get("category") or f'the same kind of thing as "{fr["value"]}"') if fr else None
        d["alternatives_raw"] = ask("alternatives", it["id"], False, sentence=e,
                                    frame=fr["frame"] if fr else None, value=fr["value"] if fr else None,
                                    category=cat)
        d["paraphrases_raw"] = ask("paraphrases", it["id"], False, sentence=e)
        if fr:
            d["null_values_raw"] = ask("null_values", it["id"], True, frame=fr["frame"], value=fr["value"],
                                       category=cat, sentence=e)
        else:
            d["null_values_raw"] = None
        by_item[it["id"]] = d
        el = time.time() - t0
        print(f"[dream] {it['id']}: frame {'ok (' + fr['source'] + ')' if fr else 'NONE'} | {el / 60:.1f} min, "
              f"eta {el / (k + 1) * (len(items) - k - 1) / 60:.1f} min", flush=True)
    topics = ask("topics", None, False)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "raw.jsonl.gz", "wt") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    res = {"mock": bool(A.mock), "tiny": bool(A.tiny), "random_model": A.random_model, "model": cfg["model"],
           "items": by_item, "topics_raw": topics, "minutes": (time.time() - t0) / 60}
    json.dump(res, open(out / "dream.json", "w"), indent=1)
    return res


def main(argv=None):
    A = parse_args(argv)
    cfg = load_cfg(A.config)
    if A.tiny:
        cfg["n_samples"] = 2
    items = dream_inputs(cfg)
    if A.items:
        items = [it for it in items if it["id"] in A.items]
    elif A.tiny and not A.mock:
        items = items[:2]
    out = Path(A.out)
    if A.mock:
        dreamer = MockDreamer()
    else:
        cfg["model_tokenizer_tiny"] = "Qwen/Qwen3.5-2B" if A.random_model else None
        dreamer = Dreamer(cfg, A)
    res = run(cfg, A, dreamer, items, out)
    env = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        import subprocess
        env["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    prompts_sha = hashlib.sha256(Path(cfg["prompts"]).read_bytes()).hexdigest()[:16]
    json.dump({"cfg": cfg, "args": vars(A), "env": env, "prompts_sha": prompts_sha, "n_items": len(items),
               "minutes": res["minutes"]}, open(out / "config.json", "w"), indent=2, default=str)
    print(f"[dream] done: {len(items)} items in {res['minutes']:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
