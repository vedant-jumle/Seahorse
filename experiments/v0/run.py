#!/usr/bin/env python
"""Seahorse v0: write an experience into a fast-weight memory on the residual
stream in session 1, then test recall in a fresh session 2.

Forward passes only; the base model is frozen. See docs/Idea.md for the design.
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers
import yaml

from seahorse import metrics as mx
from seahorse.memory import FastWeightMemory
from seahorse.residual import capture, inject, load_model
from seahorse.sessions import ceiling_ids, chat_ids, common_prefix_len, text_ids, write_pair

HERE = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[6, 10, 14, 17, 20, 23, 26])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--modes", nargs="+", default=["isolated", "combined"], choices=["isolated", "combined"])
    p.add_argument("--gates", nargs="+", default=["none", "entropy"], choices=["none", "entropy"])
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--scenarios", default=str(HERE / "scenarios.yaml"))
    p.add_argument("--generic", default=str(HERE / "generic_prompts.txt"))
    p.add_argument("--cont-len", type=int, default=20, help="tokens of greedy continuation for KL")
    p.add_argument("--sample-layer", type=int, default=17)
    p.add_argument("--sample-alpha", type=float, default=1.0)
    p.add_argument("--sample-len", type=int, default=48)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- mu


def compute_mu(model, tok, prompts, layers, device):
    """Mean residual per layer over generic chat prompts, excluding the shared
    template prefix (system prompt, attention-sink token)."""
    seqs = [chat_ids(tok, p) for p in prompts]
    skip = min(common_prefix_len(seqs[0], s) for s in seqs[1:])
    sums = {l: 0.0 for l in layers}
    count = 0
    for ids in seqs:
        with capture(model, layers) as store:
            model(ids[None].to(device))
        for l in layers:
            sums[l] = sums[l] + store[l][skip:].sum(0)
        count += len(ids) - skip
    return {l: sums[l] / count for l in layers}, skip


# -------------------------------------------------------------------------- write


def collect_writes(model, tok, scen, layers, device):
    """For each follow-up: deltas and keys on the shared suffix, per layer, plus
    the entropy gate from the without-experience run."""
    chunks = []
    for fu in scen["followups"]:
        with_ids, without_ids, n = write_pair(tok, scen["experience"], fu)
        with capture(model, layers) as s_with:
            model(with_ids[None].to(device))
        with capture(model, layers) as s_without:
            out = model(without_ids[None].to(device))
        ent = mx.entropy(out.logits[0, -n:])
        per_layer = {
            l: (s_with[l][-n:] - s_without[l][-n:], s_without[l][-n:]) for l in layers
        }
        chunks.append({
            "followup": fu,
            "n": n,
            "layers": per_layer,
            "gate": ent / ent.max().clamp_min(1e-8),
        })
    return chunks


def build_memory(mu_l, chunk_lists, layer, gate_mode, eta, device):
    """Write every chunk (in order) into a fresh memory for one layer."""
    d = mu_l.shape[0]
    mem = FastWeightMemory(d, mu_l, device=device)
    stats = []
    last = None
    for sid, chunks in chunk_lists:
        for ci, ch in enumerate(chunks):
            delta, hkey = ch["layers"][layer]
            gate = ch["gate"] if gate_mode == "entropy" else None
            err, dn = mem.write(delta, hkey, gate=gate, eta=eta)
            rel_h = (delta.norm(dim=-1) / hkey.norm(dim=-1)).mean().item()
            stats.append({
                "scenario": sid,
                "followup_idx": ci,
                "tokens": ch["n"],
                "mean_rel_error": (err / dn.clamp_min(1e-8)).mean().item(),
                "mean_delta_over_h": rel_h,
            })
            last = (delta[-1], hkey[-1])
    return mem, stats, last


def check_recall(mem, last, eta):
    """With eta=1 the most recent write must be reproduced exactly."""
    if last is None or eta != 1.0:
        return
    delta, hkey = last
    rec = mem.predict(mem.key(hkey))
    rel = ((rec - delta).norm() / delta.norm().clamp_min(1e-8)).item()
    assert rel < 1e-3, f"recall sanity check failed: rel error {rel:.2e}"


# ------------------------------------------------------------------------ measure


def target_score(model, tok, scen, prompt_ids, device):
    meas = scen["measure"]
    ids = torch.cat([prompt_ids, text_ids(tok, meas["prefix"])])
    if scen["type"] == "fact":
        return mx.seq_logprob(model, ids, text_ids(tok, meas["target"]), device)
    a = mx.seq_logprob(model, ids, text_ids(tok, meas["a"]), device)
    b = mx.seq_logprob(model, ids, text_ids(tok, meas["b"]), device)
    return a - b


def reference_probes(model, tok, scen, args):
    """No-memory references for one scenario's probes."""
    refs = []
    for pr in scen["probes"]:
        base = chat_ids(tok, pr["text"])
        ceil = ceiling_ids(tok, scen["experience"], pr["text"])
        cont = mx.greedy(model, tok, ceil, args.cont_len, args.device)
        lp_ceil = mx.cont_logprobs(model, ceil, cont, args.device)
        lp_base = mx.cont_logprobs(model, base, cont, args.device)
        refs.append({
            "probe": pr["text"],
            "distance": pr["distance"],
            "base_ids": base,
            "ceil_ids": ceil,
            "cont": cont,
            "lp_ceil": lp_ceil,
            "kl_base": mx.kl(lp_ceil, lp_base),
            "tgt_base": target_score(model, tok, scen, base, args.device),
            "tgt_ceil": target_score(model, tok, scen, ceil, args.device),
        })
    return refs


def reference_unrelated(model, tok, probes, args):
    refs = []
    for text in probes:
        ids = chat_ids(tok, text)
        cont = mx.greedy(model, tok, ids, args.cont_len, args.device)
        refs.append({"probe": text, "ids": ids, "cont": cont,
                     "lp_base": mx.cont_logprobs(model, ids, cont, args.device)})
    return refs


def eval_probes(model, tok, scen, refs, mem, layer, alpha, device):
    rows = []
    with inject(model, layer, mem, alpha):
        for r in refs:
            lp_mem = mx.cont_logprobs(model, r["base_ids"], r["cont"], device)
            rows.append({
                "scenario": scen["id"],
                "type": scen["type"],
                "distance": r["distance"],
                "probe": r["probe"],
                "kl_base": r["kl_base"],
                "kl_mem": mx.kl(r["lp_ceil"], lp_mem),
                "tgt_base": r["tgt_base"],
                "tgt_ceil": r["tgt_ceil"],
                "tgt_mem": target_score(model, tok, scen, r["base_ids"], device),
            })
    return rows


def eval_leakage(model, unrel, mem, layer, alpha, device):
    out = []
    with inject(model, layer, mem, alpha):
        for u in unrel:
            lp_mem = mx.cont_logprobs(model, u["ids"], u["cont"], device)
            out.append({"probe": u["probe"], "leak": mx.kl(u["lp_base"], lp_mem)})
    return out


def check_alpha_zero(rows, leaks):
    for r in rows:
        assert abs(r["kl_mem"] - r["kl_base"]) < 1e-5, f"alpha=0 changed KL: {r}"
        assert abs(r["tgt_mem"] - r["tgt_base"]) < 1e-3, f"alpha=0 changed target: {r}"
    for lk in leaks:
        assert lk["leak"] < 1e-5, f"alpha=0 leaked: {lk}"


# ------------------------------------------------------------------------ summary


def summarize(rows, leak_rows):
    leak = defaultdict(list)
    for lk in leak_rows:
        leak[(lk["mode"], lk["gate"], lk["layer"], lk["alpha"])].append(lk["leak"])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["mode"], r["gate"], r["layer"], r["alpha"], r["type"], r["distance"])].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        mode, gate, layer, alpha, typ, dist = key
        kl_b = sum(r["kl_base"] for r in rs)
        kl_m = sum(r["kl_mem"] for r in rs)
        d_mem = sum(r["tgt_mem"] - r["tgt_base"] for r in rs) / len(rs)
        d_ceil = sum(r["tgt_ceil"] - r["tgt_base"] for r in rs) / len(rs)
        lk = leak[(mode, gate, layer, alpha)]
        out.append({
            "mode": mode, "gate": gate, "layer": layer, "alpha": alpha,
            "type": typ, "distance": dist, "n": len(rs),
            "gap_closed": 1 - kl_m / kl_b if kl_b > 0 else float("nan"),
            "dtarget_mem": d_mem,
            "dtarget_ceil": d_ceil,
            "target_frac": d_mem / d_ceil if abs(d_ceil) > 1e-6 else float("nan"),
            "leak": sum(lk) / len(lk) if lk else float("nan"),
        })
    return out


def write_samples(model, tok, scenarios, mems, args, path):
    """Greedy generations for each scenario's 'related' probe."""
    with open(path, "w") as f:
        f.write(f"# isolated memory, gate=none, layer={args.sample_layer}, alpha={args.sample_alpha}\n\n")
        for scen in scenarios:
            pr = next(p for p in scen["probes"] if p["distance"] == "related")
            base = chat_ids(tok, pr["text"])
            ceil = ceiling_ids(tok, scen["experience"], pr["text"])
            gen = lambda ids: tok.decode(mx.greedy(model, tok, ids, args.sample_len, args.device),
                                         skip_special_tokens=True)
            with inject(model, args.sample_layer, mems[scen["id"]], args.sample_alpha):
                mem_text = gen(base)
            f.write(f"## {scen['id']} ({scen['type']})\n")
            f.write(f"experience: {scen['experience']}\nprobe: {pr['text']}\n\n")
            f.write(f"[baseline] {gen(base)}\n\n[memory]   {mem_text}\n\n[ceiling]  {gen(ceil)}\n\n")


# --------------------------------------------------------------------------- main


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.scenarios))
    scenarios, unrelated = cfg["scenarios"], cfg["unrelated_probes"]
    generic = [l.strip() for l in open(args.generic) if l.strip()]
    layers = sorted(set(args.layers) | {args.sample_layer})

    log(f"loading {args.model} on {args.device}")
    model, tok = load_model(args.model, device=args.device)

    with torch.inference_mode():
        log(f"computing mu over {len(generic)} generic prompts")
        mu, skip = compute_mu(model, tok, generic, layers, args.device)

        log("session 1: collecting writes")
        writes = {s["id"]: collect_writes(model, tok, s, layers, args.device) for s in scenarios}

        log("references (no memory)")
        refs = {s["id"]: reference_probes(model, tok, s, args) for s in scenarios}
        unrel = reference_unrelated(model, tok, unrelated, args)

        rows, leak_rows, write_rows = [], [], []
        sample_mems = {}
        for mode in args.modes:
            for gate in args.gates:
                for layer in args.layers:
                    if mode == "isolated":
                        groups = [(s["id"], [s]) for s in scenarios]
                    else:
                        groups = [("combined", scenarios)]
                    for gid, members in groups:
                        mem, stats, last = build_memory(
                            mu[layer], [(s["id"], writes[s["id"]]) for s in members],
                            layer, gate, args.eta, args.device)
                        if gate == "none":
                            check_recall(mem, last, args.eta)
                        for st in stats:
                            write_rows.append({"mode": mode, "gate": gate, "layer": layer,
                                               "memory": gid, **st})
                        for alpha in args.alphas:
                            tag = {"mode": mode, "gate": gate, "layer": layer, "alpha": alpha}
                            new_rows = []
                            for s in members:
                                new_rows += eval_probes(model, tok, s, refs[s["id"]], mem,
                                                        layer, alpha, args.device)
                            leaks = eval_leakage(model, unrel, mem, layer, alpha, args.device)
                            if alpha == 0.0:
                                check_alpha_zero(new_rows, leaks)
                            rows += [{**tag, "memory": gid, **r} for r in new_rows]
                            leak_rows += [{**tag, "memory": gid, **lk} for lk in leaks]
                    log(f"done mode={mode} gate={gate} layer={layer}")

        log("samples")
        for s in scenarios:
            mem, _, _ = build_memory(mu[args.sample_layer], [(s["id"], writes[s["id"]])],
                                     args.sample_layer, "none", args.eta, args.device)
            sample_mems[s["id"]] = mem
        write_samples(model, tok, scenarios, sample_mems, args, out_dir / "samples.txt")

    summary = summarize(rows, leak_rows)
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "leakage.jsonl", "w") as f:
        for r in leak_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "write_stats.jsonl", "w") as f:
        for r in write_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    json.dump({
        **vars(args),
        "mu_skip_tokens": skip,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, open(out_dir / "config.json", "w"), indent=2)
    log(f"wrote {len(rows)} probe rows, {len(leak_rows)} leakage rows to {out_dir}")


if __name__ == "__main__":
    main()
