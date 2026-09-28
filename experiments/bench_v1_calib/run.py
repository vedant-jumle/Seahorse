#!/usr/bin/env python
"""bench_v1 calibration: baseline vs ceiling for every bench measure, with no memory.

Conditions per prompt (all single user turns through the chat template):
  base     the prompt alone
  ceil     experience + prompt (the upper bound a memory should approach)
  placebo  an irrelevant sentence + prompt: the noise floor ("what does any extra
           sentence do?"), placebos[item index % 3]

Measures (one calib.csv row per item x measure instance; d_ceil = ceil - base):
  target        facts: log P(target | probe + prefix), per probe distance
  spec          facts: Δtarget - mean Δfoil (ceil or placebo vs base)
  verification  facts: log P(a) - log P(b) on yes/no verification probes (a = consistent)
  inference     dispositions: the same on the balanced relation probes
  contrast      dispositions: log P(a) - log P(b) after prefix, per contrast x probe
  rate          dispositions: lean = consistent - inconsistent fraction over --samples
                generations at --temperature per ambiguous prompt (lexicon-scored); the
                same seed is used for all three conditions

Flags: a row is `weak` if d_ceil < max(--min-*, p90 of |placebo - base| for that measure),
and `wrong_sign` if d_ceil < -max(p90, 0.1), i.e. beyond the noise in the wrong direction. Relation rows also get `ceil_wrong` (the ceiling answers
inconsistently) or `saturated` (already consistent at baseline, so a memory can't show a
gain). Facts also get `not_top` (a foil beats the target under the ceiling).

Outputs: calib.csv, report.txt (noise floor, per-category summary, Yes vs No, emotional
pairs, flagged items and prompts, pool, unrelated-probe continuations), samples.jsonl
(all sampled generations with labels), config.json.
"""

import argparse
import csv
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers

from seahorse import metrics as mx
from seahorse.bench import (disposition_rate, format_report, label, load_bench, pool_items,
                            validate_bench)
from seahorse.residual import load_model
from seahorse.sessions import chat_ids, text_ids

REPO = Path(__file__).resolve().parents[2]
CONDS = ("base", "ceil", "placebo")
RELATION = ("inference", "verification")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--bench", default=str(REPO / "data" / "bench_v1"))
    p.add_argument("--samples", type=int, default=8, help="generations per ambiguous prompt and condition")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--gen-len", type=int, default=80)
    p.add_argument("--cont-len", type=int, default=40, help="greedy tokens for unrelated probes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-seed", type=int, default=0)
    p.add_argument("--no-pool", action="store_true")
    p.add_argument("--min-target", type=float, default=1.0, help="nats")
    p.add_argument("--min-spec", type=float, default=1.0, help="nats")
    p.add_argument("--min-relation", type=float, default=0.5, help="nats")
    p.add_argument("--min-contrast", type=float, default=0.5, help="nats")
    p.add_argument("--min-rate", type=float, default=0.2, help="lean shift")
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N items per type (and N pool)")
    p.add_argument("--tiny", action="store_true", help="smoke test: tiny random Qwen2, real tokenizer")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(args):
    if not args.tiny:
        return load_model(args.model, device=args.device)
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    return Qwen2ForCausalLM(cfg).to(args.device).eval(), tok


class Scorer:
    def __init__(self, model, tok, args):
        self.model, self.tok, self.args, self.dev = model, tok, args, args.device
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def ids(self, cond, item, placebo, text):
        ctx = {"base": None, "ceil": item["experience"], "placebo": placebo}[cond]
        return chat_ids(self.tok, text if ctx is None else f"{ctx} {text}")

    def t(self, text):
        return text_ids(self.tok, text)

    def lps(self, prompt, cands):
        """Total log-prob of each candidate continuation after `prompt`, one right-padded batch."""
        P = len(prompt)
        L = P + max(len(c) for c in cands)
        batch = torch.full((len(cands), L), self.pad, dtype=torch.long)
        mask = torch.zeros_like(batch)
        for i, c in enumerate(cands):
            s = torch.cat([prompt, c])
            batch[i, :len(s)], mask[i, :len(s)] = s, 1
        logits = self.model(batch.to(self.dev), attention_mask=mask.to(self.dev)).logits.float()
        lp = torch.log_softmax(logits[:, P - 1:L - 1], dim=-1)
        return [lp[i, torch.arange(len(c)), c.to(self.dev)].sum().item() for i, c in enumerate(cands)]

    def diff(self, prompt, a, b):
        la, lb = self.lps(prompt, [self.t(a), self.t(b)])
        return la - lb

    def sample(self, prompt, seed):
        torch.manual_seed(seed)
        x = prompt[None].to(self.dev)
        out = self.model.generate(
            x, attention_mask=torch.ones_like(x), do_sample=True, temperature=self.args.temperature,
            top_k=0, top_p=1.0, repetition_penalty=1.0, num_return_sequences=self.args.samples,
            max_new_tokens=self.args.gen_len, pad_token_id=self.pad)
        return [g.replace("\n", " ").strip()
                for g in self.tok.batch_decode(out[:, x.shape[1]:], skip_special_tokens=True)]


def row(it, measure, detail, v, consistent="", extra=None):
    return {"item": it["id"], "type": it["type"], "category": it["category"],
            "pair": it.get("pair", ""), "phrasing": it.get("phrasing", ""), "measure": measure,
            "detail": detail, "consistent": consistent, "base": v["base"], "ceil": v["ceil"],
            "placebo": v["placebo"], "d_ceil": v["ceil"] - v["base"],
            "d_placebo": v["placebo"] - v["base"], "flags": "", "extra": extra or {}}


def relation_rows(S, it, placebo):
    return [row(it, rp["kind"], rp["question"],
                {c: S.diff(S.ids(c, it, placebo, rp["text"]), rp["a"], rp["b"]) for c in CONDS},
                consistent=rp["consistent"])
            for rp in it["relation_probes"]]


def calib_fact(S, it, placebo):
    m = it["measure"]
    pre = S.t(m["prefix"])
    cands = [S.t(x) for x in [m["target"]] + m["foils"]]
    rows = []
    for pr in it["probes"]:
        v = {c: S.lps(torch.cat([S.ids(c, it, placebo, pr["text"]), pre]), cands) for c in CONDS}
        tgt = {c: v[c][0] for c in CONDS}
        foil = {c: statistics.mean(v[c][1:]) for c in CONDS}
        rows.append(row(it, "target", pr["distance"], tgt, extra={
            **{f"foil_{c}": foil[c] for c in CONDS},
            **{f"{c}_top": v[c][0] > max(v[c][1:]) for c in CONDS}}))
        spec = {"base": 0.0, **{c: (tgt[c] - tgt["base"]) - (foil[c] - foil["base"]) for c in CONDS[1:]}}
        rows.append(row(it, "spec", pr["distance"], spec))
    return rows + relation_rows(S, it, placebo)


def calib_disposition(S, it, placebo, idx, samples):
    rows = []
    for ci, ct in enumerate(it["contrasts"]):
        pre = S.t(ct["prefix"])
        for pr in it["probes"]:
            v = {c: S.diff(torch.cat([S.ids(c, it, placebo, pr["text"]), pre]), ct["a"], ct["b"])
                 for c in CONDS}
            rows.append(row(it, "contrast", f"c{ci} {pr['distance']}", v,
                            extra={"contrast": ci, "a": ct["a"], "b": ct["b"]}))
    rows += relation_rows(S, it, placebo)
    for pi, prompt in enumerate(it["ambiguous"]):
        seed = S.args.seed * 1_000_003 + idx * 101 + pi
        v, ex = {}, {}
        for c in CONDS:
            gens = S.sample(S.ids(c, it, placebo, prompt), seed)
            r = disposition_rate(gens, it["lexicon"])
            v[c], ex[c] = r["lean"], {k: r[k] for k in ("consistent", "inconsistent", "neutral")}
            samples.append({"item": it["id"], "prompt": prompt, "cond": c, "seed": seed,
                            "labels": [label(g, it["lexicon"]) for g in gens], "generations": gens})
        rows.append(row(it, "rate", prompt, v, extra=ex))
    return rows


# ------------------------------------------------------------------------ flagging


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def noise_floor(rows, args):
    mins = {"target": args.min_target, "spec": args.min_spec, "verification": args.min_relation,
            "inference": args.min_relation, "contrast": args.min_contrast, "rate": args.min_rate}
    out = {}
    for m, lo in mins.items():
        d = [abs(r["d_placebo"]) for r in rows if r["measure"] == m and r["category"] != "pool"]
        p90 = quantile(d, 0.9)
        out[m] = {"n": len(d), "median": quantile(d, 0.5), "p90": p90, "min": lo,
                  "threshold": max(lo, p90) if d else lo}
    return out


def neg_margin(noise, m):
    """A shift counts as wrong-signed only beyond the noise floor in the wrong direction."""
    return max(noise[m]["p90"], 0.1)


def flag_rows(rows, noise):
    for r in rows:
        m, f = r["measure"], []
        thr = noise[m]["threshold"]
        if m in RELATION and r["ceil"] <= 0:
            f.append("ceil_wrong")
        if r["d_ceil"] < -neg_margin(noise, m):
            f.append("wrong_sign")
        elif r["d_ceil"] < thr:
            sat = m in RELATION and r["base"] > 0 and r["ceil"] > 0
            f.append("saturated" if sat else "weak")
        if m == "target" and not r["extra"]["ceil_top"]:
            f.append("not_top")
        r["flags"] = ",".join(f)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def item_reasons(rs, noise):
    """Item-level reasons to exclude or rewrite, from its rows."""
    by = defaultdict(list)
    for r in rs:
        by[r["measure"]].append(r)
    out = []
    for m in ("target", "spec", "rate"):
        if by[m]:
            d = mean(r["d_ceil"] for r in by[m])
            if d < noise[m]["threshold"]:
                out.append(f"{m} {'wrong-signed' if d < -neg_margin(noise, m) else 'weak'} (mean Δceil {d:+.2f})")
    rel = [r for r in by["target"] if r["detail"] == "related" and "not_top" in r["flags"]]
    if rel:
        out.append("a foil beats the target under the ceiling (related probe)")
    cons = defaultdict(list)
    for r in by["contrast"]:
        cons[r["extra"]["contrast"]].append(r["d_ceil"])
    for ci, ds in sorted(cons.items()):
        d = mean(ds)
        if d < noise["contrast"]["threshold"]:
            wrong = d < -neg_margin(noise, "contrast")
            out.append(f"contrast c{ci} {'wrong-signed' if wrong else 'weak'} (mean Δceil {d:+.2f})")
    for m in RELATION:
        bad = [r for r in by[m] if "ceil_wrong" in r["flags"] or "wrong_sign" in r["flags"]]
        if bad:
            out.append(f"{m}: {len(bad)}/{len(by[m])} probes where the ceiling answers wrong or moves the "
                       f"wrong way ({', '.join(r['consistent'] for r in bad)})")
    neg = [r for r in by["rate"] if "wrong_sign" in r["flags"]]
    if neg:
        out.append(f"rate: {len(neg)}/{len(by['rate'])} ambiguous prompts shift the wrong way")
    return out


# -------------------------------------------------------------------------- report


def fmt(x):
    return f"{x:+.2f}" if isinstance(x, float) else str(x)


def table(title, header, lines):
    rows = [header] + [[fmt(x) for x in l] for l in lines]
    w = [max(len(r[i]) for r in rows) for i in range(len(header))]
    out = [title] + ["  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join(out) + "\n"


def summary_line(rs, extra_cols=False):
    line = [len(rs), mean(r["base"] for r in rs), mean(r["ceil"] for r in rs),
            mean(r["d_ceil"] for r in rs), mean(r["d_placebo"] for r in rs),
            f"{mean(bool(r['flags']) for r in rs):.2f}"]
    if extra_cols:
        line += [f"{mean(r['base'] > 0 for r in rs):.2f}", f"{mean(r['ceil'] > 0 for r in rs):.2f}"]
    return line


def build_report(rows, noise, unrel, meta):
    hand = [r for r in rows if r["category"] != "pool"]
    pool = [r for r in rows if r["category"] == "pool"]
    parts = [f"bench_v1 calibration  {meta}\n"]
    parts.append(table(
        "NOISE FLOOR: |placebo - base| per measure (hand items); flag threshold = max(min, p90)",
        ["measure", "n", "median", "p90", "min", "threshold"],
        [[m, v["n"], v["median"], v["p90"], v["min"], v["threshold"]] for m, v in noise.items()]))
    hdr = ["type/category", "measure", "consistent", "n", "base", "ceil", "Δceil", "Δplacebo", "flagged"]
    groups = defaultdict(list)
    for r in hand:
        groups[(f"{r['type']}/{r['category']}", r["measure"], r["consistent"])].append(r)
    parts.append(table("PER-CATEGORY SUMMARY (means over rows; flagged = fraction of rows with any flag)",
                       hdr, [[*k, *summary_line(v)] for k, v in sorted(groups.items())]))
    yn = defaultdict(list)
    for r in hand + pool:
        if r["measure"] in RELATION:
            yn[(r["measure"], "pool" if r["category"] == "pool" else "hand", r["consistent"])].append(r)
    parts.append(table(
        "YES vs NO (relation scores = log P(consistent) - log P(inconsistent); correct = score > 0)",
        ["measure", "set", "consistent", "n", "base", "ceil", "Δceil", "Δplacebo", "flagged",
         "base correct", "ceil correct"],
        [[*k, *summary_line(v, True)] for k, v in sorted(yn.items())]))
    pairs = defaultdict(lambda: defaultdict(list))
    for r in hand:
        if r["pair"]:
            pairs[(r["pair"], r["measure"])][r["phrasing"]].append(r["d_ceil"])
    parts.append(table("EMOTIONAL PAIRS: mean Δceil, neutral vs emotional phrasing",
                       ["pair", "measure", "neutral", "emotional", "emo - neu"],
                       [[p, m, mean(v["neutral"]), mean(v["emotional"]),
                         mean(v["emotional"]) - mean(v["neutral"])] for (p, m), v in sorted(pairs.items())]))
    by_item = defaultdict(list)
    for r in hand:
        by_item[r["item"]].append(r)
    flagged = {i: item_reasons(rs, noise) for i, rs in by_item.items()}
    flagged = {i: rs for i, rs in flagged.items() if rs}
    lines = [f"FLAGGED ITEMS ({len(flagged)}/{len(by_item)}): weak or wrong-signed ceiling effects"]
    for i, rs in flagged.items():
        lines.append(f"  {i}")
        lines += [f"      - {x}" for x in rs]
    parts.append("\n".join(lines) + "\n")
    bad = [r for r in hand if r["measure"] in RELATION + ("rate",)
           and ({"ceil_wrong", "wrong_sign"} & set(r["flags"].split(",")))]
    parts.append(table("PROMPTS TO REWRITE (relation probes where the ceiling is wrong / wrong-signed; "
                       "ambiguous prompts with a wrong-signed rate shift)",
                       ["item", "measure", "consistent", "base", "ceil", "flags", "prompt"],
                       [[r["item"], r["measure"], r["consistent"], r["base"], r["ceil"], r["flags"],
                         r["detail"]] for r in bad]))
    if pool:
        pt = defaultdict(list)
        for r in pool:
            pt[r["item"]].append(r)
        weak = [i for i, rs in pt.items() if item_reasons(rs, noise)]
        parts.append(table(f"FACT POOL ({len(pt)} slots, one value each): weak slots: {', '.join(weak) or 'none'}",
                           ["measure", "consistent", "n", "base", "ceil", "Δceil", "Δplacebo", "flagged"],
                           [[m, c, *summary_line([r for r in pool if r["measure"] == m and r["consistent"] == c])]
                            for m, c in sorted({(r["measure"], r["consistent"]) for r in pool})]))
    parts.append("UNRELATED PROBES: baseline greedy continuations\n"
                 + "\n".join(f"  {u['probe']}\n      -> {u['cont']}" for u in unrel) + "\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------- main


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"loading {args.model} on {args.device}{' (tiny random model)' if args.tiny else ''}")
    model, tok = load(args)

    bench = load_bench(args.bench)
    val = validate_bench(bench, tok)
    log("bench validation:\n" + format_report(val))
    if val["errors"]:
        raise SystemExit("bench validation failed")
    items, placebos = bench["scenarios"], bench["placebos"]
    pool = [] if args.no_pool else pool_items(bench["pool"], args.pool_seed)
    if args.limit:
        items = ([it for it in items if it["type"] == "disposition"][:args.limit]
                 + [it for it in items if it["type"] == "fact"][:args.limit])
        pool = pool[:args.limit]
    S = Scorer(model, tok, args)
    rows, samples = [], []

    with torch.inference_mode():
        for idx, it in enumerate(items + pool):
            placebo = placebos[idx % len(placebos)]
            t = time.time()
            if it["type"] == "fact":
                rs = calib_fact(S, it, placebo)
            else:
                rs = calib_disposition(S, it, placebo, idx, samples)
            rows += rs
            log(f"[{idx + 1}/{len(items) + len(pool)}] {it['id']}: {len(rs)} rows, {time.time() - t:.1f}s")
        unrel = []
        for text in bench["unrelated_probes"]:
            ids = chat_ids(tok, text)
            cont = tok.decode(mx.greedy(model, tok, ids, args.cont_len, args.device), skip_special_tokens=True)
            unrel.append({"probe": text, "cont": cont.replace("\n", " ").strip()})
        log(f"unrelated probes: {len(unrel)} greedy continuations")

    noise = noise_floor(rows, args)
    flag_rows(rows, noise)
    fields = list(rows[0].keys())
    with open(out / "calib.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "extra": json.dumps(r["extra"])})
    with open(out / "samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = None
    meta = (f"model={args.model}{' (TINY)' if args.tiny else ''} commit={commit} items={len(items)} "
            f"pool={len(pool)} rows={len(rows)} samples={args.samples}@T={args.temperature}")
    (out / "report.txt").write_text(build_report(rows, noise, unrel, meta))
    json.dump({**vars(args), "commit": commit, "noise_floor": noise, "placebos": placebos,
               "validation": val["counts"], "n_items": len(items), "n_pool": len(pool), "n_rows": len(rows),
               "runtime_s": round(time.time() - t0, 1), "torch": torch.__version__,
               "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              open(out / "config.json", "w"), indent=2)
    log(f"wrote {len(rows)} rows to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
