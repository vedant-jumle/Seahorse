#!/usr/bin/env python
"""Seahorse diag_dose: how big is the steer, where does it land, what does combining do?

A measurement pass. The method is unchanged (v0.1 pipeline: mu, session-1 writes with
with/without/counter runs, build_memory). A logging copy of the inject hook computes
k = k(h), the recall r = M k and the steer s = alpha r, logs per-position statistics,
and returns h' = h + s, asserted equal to memory.read(h, alpha).

Per position of every input it logs:
  cat   template head / user text / template tail (end-of-turn + assistant header) /
        continuation; ans = 1 at the last prompt position (the answer position)
  |h|, |h - mu|, |s|, |h'|, s/|h-mu| (sr), |h'|/|h| - 1 (dn)
  mc    max cosine between k(h) and all stored write keys (mct: tail keys, mcc: content keys)
  rf    recall fraction |M k(h)| / mean|delta| of the probe's own scenario at this layer
        (unrelated probes: the memory's scenario, or all scenarios in combined mode)
  cc,lr combined mode only: cos(M_comb k, M_iso k) and |M_comb k| / |M_iso k|, where M_iso is
        the probe's own isolated memory
Write side, per write token: |delta|, |h_without - mu| and their ratio.

Inputs: each scenario's 3 probes + the ceiling's greedy continuation (teacher-forced),
the relation probes (prompt only), the 8 unrelated probes + the baseline's greedy
continuation. Config: gate entropy, position all, baselines without (primary) and
contrastive (secondary), modes isolated/combined, layers 14 17 23 26, alpha 1 2 4.

Outputs: positions.jsonl.gz, summary.csv, write_tokens.csv, report.txt, config.json
"""

import argparse
import csv
import gzip
import importlib.util
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import yaml

from seahorse import metrics as mx
from seahorse.residual import _hidden, _replace, decoder_layers, inject, load_model
from seahorse.sessions import ceiling_ids, chat_ids, common_prefix_len, common_suffix_len

HERE = Path(__file__).resolve().parent
V01_DIR = HERE.parent / "v0_1"
CATS = ["head", "user", "tail", "cont"]
KINDS = ["exact", "paraphrase", "related", "relation", "unrelated"]
STATS = ["h", "hmu", "s", "hp", "sr", "dn", "mc", "mct", "mcc", "rf", "cc", "lr"]


def _load_v01():
    """The v0.1 pipeline (compute_mu, collect_writes, select, build_memory)."""
    spec = importlib.util.spec_from_file_location("seahorse_v0_1_run", V01_DIR / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v01 = _load_v01()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--layers", type=int, nargs="+", default=[14, 17, 23, 26])
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--baselines", nargs="+", default=["without", "contrastive"],
                   choices=["without", "contrastive"])
    p.add_argument("--modes", nargs="+", default=["isolated", "combined"], choices=["isolated", "combined"])
    p.add_argument("--scenarios", default=str(V01_DIR / "scenarios.yaml"))
    p.add_argument("--generic", default=str(v01.GENERIC))
    p.add_argument("--cont-len", type=int, default=20)
    p.add_argument("--tiny", action="store_true",
                   help="smoke test: tiny random Qwen2 with --model's tokenizer")
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
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=max(args.layers) + 2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    model = Qwen2ForCausalLM(cfg).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ------------------------------------------------------------------ logging hook


def steer_stats(h, k, r, s, h_new, mu, ctx):
    """Per-position statistics ([T, d] tensors in, lists out)."""
    r_n = r.norm(dim=-1)
    s_n = s.norm(dim=-1)
    h_n = h.norm(dim=-1)
    hmu_n = (h - mu).norm(dim=-1)
    hp_n = h_new.norm(dim=-1)
    C = k @ ctx["K"].T
    tm = ctx["tail"]
    st = {"h": h_n, "hmu": hmu_n, "s": s_n, "hp": hp_n,
          "sr": s_n / hmu_n.clamp_min(1e-12), "dn": hp_n / h_n.clamp_min(1e-12) - 1,
          "mc": C.max(-1).values, "mct": C[:, tm].max(-1).values, "mcc": C[:, ~tm].max(-1).values,
          "rf": r_n / ctx["den"]}
    if ctx.get("own") is not None:
        r_iso = ctx["own"].predict(k)
        st["cc"] = F.cosine_similarity(r, r_iso, dim=-1)
        st["lr"] = r_n / r_iso.norm(dim=-1).clamp_min(1e-12)
    return {key: v.float().cpu().tolist() for key, v in st.items()}


@contextmanager
def log_inject(model, layer, memory, alpha, ctx, rec):
    """seahorse.residual.inject plus logging. Returns exactly memory.read(h, alpha)."""
    def hook(module, args, out):
        h = _hidden(out)
        assert h.shape[0] == 1, "log_inject expects batch size 1"
        k = memory.key(h)
        r = memory.predict(k)
        s = alpha * r
        h_new = h + s
        ref = memory.read(h, alpha)
        exact = torch.equal(h_new, ref)
        assert exact or torch.allclose(h_new, ref, rtol=1e-6, atol=1e-5), "logging hook != memory.read"
        rec["exact"] = exact
        rec["stats"] = steer_stats(h[0], k[0], r[0], s[0], h_new[0], memory.mu, ctx)
        return _replace(out, h_new)

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield rec
    finally:
        handle.remove()


# ------------------------------------------------------------------------ inputs


def build_inputs(model, tok, scenarios, unrelated, head, tail, args):
    inputs = []

    def add(scen, kind, text, prompt, cont):
        ids = torch.cat([prompt, cont]) if cont is not None else prompt
        p, nh, nt = len(prompt), len(head), len(tail)
        assert torch.equal(prompt[:nh], head) and torch.equal(prompt[p - nt:], tail), f"template mismatch: {text}"
        cat = [3] * len(ids)
        for t in range(p):
            cat[t] = 0 if t < nh else (2 if t >= p - nt else 1)
        inputs.append({"iid": len(inputs), "scenario": scen["id"] if scen else None,
                       "type": scen["type"] if scen else "unrelated", "kind": kind, "text": text,
                       "ids": ids, "tok": ids.tolist(), "cat": cat, "ans": p - 1, "prompt_len": p})

    for s in scenarios:
        for pr in s["probes"]:
            base = chat_ids(tok, pr["text"])
            cont = mx.greedy(model, tok, ceiling_ids(tok, s["experience"], pr["text"]), args.cont_len, args.device)
            add(s, pr["distance"], pr["text"], base, cont)
        for rp in s.get("relation_probes", []):
            add(s, "relation", rp["text"], chat_ids(tok, rp["text"]), None)
    for text in unrelated:
        base = chat_ids(tok, text)
        add(None, "unrelated", text, base, mx.greedy(model, tok, base, args.cont_len, args.device))
    return inputs


def write_keys(mem, writes, sid, layer, baseline, tail_len):
    """Unit write keys [N, d], tail mask [N], |delta| [N], and per-token write-side rows."""
    ks, tm, dn, rows = [], [], [], []
    for ci, ch in enumerate(writes[sid]):
        delta, h_key, g = v01.select(ch, layer, "all", baseline, "entropy", tail_len, 4)
        n = delta.shape[0]
        is_tail = torch.arange(n, device=delta.device) >= n - tail_len
        d_n, hwo_n, hwomu_n = delta.norm(dim=-1), h_key.norm(dim=-1), (h_key - mem.mu).norm(dim=-1)
        ks.append(mem.key(h_key))
        tm.append(is_tail)
        dn.append(d_n)
        for t in range(n):
            rows.append({"baseline": baseline, "layer": layer, "scenario": sid, "followup_idx": ci, "t": t,
                         "is_tail": int(is_tail[t]), "gate": float(g[t]), "delta_norm": float(d_n[t]),
                         "h_without_norm": float(hwo_n[t]), "h_without_mu_norm": float(hwomu_n[t]),
                         "ratio": float(d_n[t] / hwomu_n[t].clamp_min(1e-12))})
    return torch.cat(ks), torch.cat(tm), torch.cat(dn), rows


def r4(x):
    return float(f"{x:.4g}") if math.isfinite(x) else None


# --------------------------------------------------------------------- analysis


class Cols:
    """Column view of the position rows, for masks and medians."""

    def __init__(self, rows):
        self.c = {}
        for k in ["b", "m", "sc", "ty", "kind", "cat", "mem"]:
            self.c[k] = np.array([str(r.get(k)) for r in rows])
        for k in ["L", "a", "ans", "iid"]:
            self.c[k] = np.array([r[k] for r in rows], dtype=float)
        for k in STATS:
            self.c[k] = np.array([np.nan if r.get(k) is None else r[k] for r in rows], dtype=float)

    def mask(self, **kw):
        m = np.ones(len(self.c["b"]), dtype=bool)
        for k, v in kw.items():
            m &= np.isin(self.c[k], list(v)) if isinstance(v, (list, tuple, set)) else (self.c[k] == v)
        return m

    def vals(self, m, key):
        x = self.c[key][m]
        return x[~np.isnan(x)]

    def q(self, m, key, p=50):
        x = self.vals(m, key)
        return float(np.percentile(x, p)) if len(x) else float("nan")

    def energy(self, m):
        return float((self.vals(m, "s") ** 2).sum())


def summarize(cols):
    out = []
    c = cols.c
    combos = sorted(set(zip(c["b"], c["m"], c["L"], c["a"], c["kind"])))
    for b, m, L, a, kind in combos:
        base = cols.mask(b=b, m=m, L=L, a=a, kind=kind)
        e_all = cols.energy(base)
        cats = {cat: base & (c["cat"] == cat) for cat in CATS}
        cats["answer"] = base & (c["ans"] == 1)
        cats["all"] = base
        for cat, mk in cats.items():
            if not mk.any():
                continue
            row = {"baseline": b, "mode": m, "layer": int(L), "alpha": a, "kind": kind, "cat": cat, "n": int(mk.sum())}
            for k in ["h", "hmu", "s", "hp", "sr", "dn", "mc", "mct", "mcc", "rf", "cc", "lr"]:
                row[f"med_{k}"] = cols.q(mk, k)
            row["p90_sr"], row["p90_dn"], row["mean_rf"] = cols.q(mk, "sr", 90), cols.q(mk, "dn", 90), \
                float(np.mean(cols.vals(mk, "rf"))) if len(cols.vals(mk, "rf")) else float("nan")
            row["energy"] = cols.energy(mk)
            row["energy_share"] = row["energy"] / e_all if e_all > 0 else float("nan")
            out.append(row)
    return out


def fmt(x, spec=".3f"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return format(x, spec)


def table(header, rows):
    rows = [[str(v) for v in r] for r in rows]
    header = [str(h) for h in header]
    w = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(v.ljust(w[i]) if i == 0 else v.rjust(w[i]) for i, v in enumerate(r))
    return "\n".join([line(header), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def report(cols, write_rows, args, checks):
    c = cols.c
    L_ = args.layers
    A = args.alphas
    a_mid = 2.0 if 2.0 in A else A[len(A) // 2]
    out = []
    w = out.append
    w("Seahorse diag_dose report")
    w("=" * 25)
    w(f"write: gate=entropy position=all; baselines={args.baselines}; layers={L_}; alphas={A}")
    w("s = alpha M k(h) (the steer); sr = |s|/|h-mu|; dn = |h'|/|h| - 1; mc/mct/mcc = max cos of k(h) to all/tail/content "
      "write keys; rf = |M k(h)| / mean|delta| (own scenario, this layer); cc/lr = cos and length ratio of "
      "M_comb k vs own M_iso k.")
    w(f"checks: logging hook == memory.read exactly on {checks['exact']}/{checks['calls']} calls "
      f"(rest within tol); max |logits(log hook) - logits(library inject)| = {checks['e2e_max']:.2e} "
      f"over {checks['e2e_n']} end-to-end checks.")
    w("Stats at the injection layer that do not involve alpha (|h|, |h-mu|, mc, rf, cc, lr) are identical for every alpha.\n")

    for b in args.baselines:
        w(f"#################### baseline = {b} ####################\n")
        # T1
        for mode in args.modes:
            rows = []
            for L in L_:
                r = [f"L{L}"]
                for a in A:
                    m = cols.mask(b=b, m=mode, L=L, a=a)
                    mnh = m & (c["cat"] != "head")
                    r += [fmt(cols.q(m, "sr")), fmt(cols.q(mnh, "sr")), fmt(cols.q(m, "dn"), "+.4f"),
                          fmt(cols.q(m, "dn", 90), "+.4f")]
                rows.append(r)
            hdr = ["layer"] + [f"{x} a={a:g}" for a in A for x in ("sr", "sr(nohead)", "dn", "dn p90")]
            w(f"T1 [{mode}] median |s|/|h-mu| and median |h'|/|h|-1, all inputs, all positions (sr(nohead) excludes the template head)")
            w(table(hdr, rows))
            w("")
        # T2 by category
        w(f"T2 [isolated, alpha={a_mid:g}] by position category (all inputs pooled)")
        for L in L_:
            rows = []
            for cat in CATS + ["answer"]:
                m = cols.mask(b=b, m="isolated", L=L, a=a_mid)
                m &= (c["ans"] == 1) if cat == "answer" else (c["cat"] == cat)
                rows.append([cat, int(m.sum())] + [fmt(cols.q(m, k), f) for k, f in
                            (("h", ".1f"), ("hmu", ".1f"), ("s", ".2f"), ("sr", ".3f"), ("dn", "+.4f"),
                             ("mc", ".3f"), ("mct", ".3f"), ("mcc", ".3f"), ("rf", ".3f"))])
            w(table([f"L{L} cat", "n", "|h|", "|h-mu|", "|s|", "sr", "dn", "mc", "mct", "mcc", "rf"], rows))
            w("")
        # T3 energy share on unrelated probes
        w("T3 steer-energy share (sum |s|^2) by category on UNRELATED probes (pooled; alpha-independent). "
          "per-input tail share: median [min, max]")
        rows = []
        for mode in args.modes:
            for L in L_:
                m = cols.mask(b=b, m=mode, L=L, a=a_mid, kind="unrelated")
                tot = cols.energy(m)
                shares = [cols.energy(m & (c["cat"] == cat)) / tot for cat in CATS]
                per = []
                for mem in sorted(set(c["mem"][m])):
                    for iid in sorted(set(c["iid"][m & (c["mem"] == mem)])):
                        mi = m & (c["mem"] == mem) & (c["iid"] == iid)
                        per.append(cols.energy(mi & (c["cat"] == "tail")) / cols.energy(mi))
                rows.append([mode, f"L{L}"] + [fmt(x) for x in shares] +
                            [f"{np.median(per):.3f} [{min(per):.3f}, {max(per):.3f}]",
                             fmt(shares[1] + shares[3])])
        w(table(["mode", "layer", "head", "user", "tail", "cont", "tail per input", "user+cont"], rows))
        w("")
        # T4 recall fraction on content (user) positions
        w("T4 recall fraction rf = |M k| / mean|delta| on USER-TEXT positions: median (IQR for related)")
        rows = []
        for mode in args.modes:
            for L in L_:
                r = [mode, f"L{L}"]
                for kind in KINDS:
                    m = cols.mask(b=b, m=mode, L=L, a=a_mid, kind=kind, cat="user")
                    if kind == "related":
                        r.append(f"{cols.q(m, 'rf'):.3f} ({cols.q(m, 'rf', 25):.3f}-{cols.q(m, 'rf', 75):.3f})")
                    else:
                        r.append(fmt(cols.q(m, "rf")))
                m = cols.mask(b=b, m=mode, L=L, a=a_mid, kind="related", cat="cont")
                r.append(fmt(cols.q(m, "rf")))
                rows.append(r)
        w(table(["mode", "layer"] + KINDS + ["related(cont)"], rows))
        w("")
        # T5 combined vs isolated
        if "combined" in args.modes:
            own = ["exact", "paraphrase", "related", "relation"]
            w("T5 combined vs own isolated recall at the same h (own-scenario inputs): median cos(M_comb k, M_iso k) / "
              "median |M_comb k|/|M_iso k|")
            rows = []
            for L in L_:
                r = [f"L{L}"]
                for cat in CATS:
                    m = cols.mask(b=b, m="combined", L=L, a=a_mid, kind=own, cat=cat)
                    r.append(f"{fmt(cols.q(m, 'cc'), '+.3f')} / {fmt(cols.q(m, 'lr'))}")
                mi = cols.mask(b=b, m="isolated", L=L, a=a_mid, kind="unrelated")
                mc_ = cols.mask(b=b, m="combined", L=L, a=a_mid, kind="unrelated")
                r.append(fmt(cols.q(mc_, "s") / cols.q(mi, "s")))
                mi = cols.mask(b=b, m="isolated", L=L, a=a_mid, kind=own)
                mc_ = cols.mask(b=b, m="combined", L=L, a=a_mid, kind=own)
                r.append(fmt(cols.q(mc_, "s") / cols.q(mi, "s")))
                rows.append(r)
            w(table(["layer"] + CATS + ["unrel |s| comb/iso", "own |s| comb/iso"], rows))
            w("")
            w("T5b per scenario, non-head positions of own inputs: median cc / median lr")
            scen = [s for s in dict.fromkeys(c["sc"][cols.mask(b=b, m="combined")]) if s != "None"]
            rows = []
            for s in scen:
                r = [s]
                for L in L_:
                    m = cols.mask(b=b, m="combined", L=L, a=a_mid, sc=s) & (c["cat"] != "head")
                    r.append(f"{fmt(cols.q(m, 'cc'), '+.3f')} / {fmt(cols.q(m, 'lr'))}")
                rows.append(r)
            w(table(["scenario (write order)"] + [f"L{L}" for L in L_], rows))
            w("")
        # T6 answer position
        w(f"T6 ANSWER position (last prompt token) [isolated, alpha={a_mid:g}]: median mc / mct / mcc / rf / sr")
        groups = [("relation (disp)", dict(kind="relation")),
                  ("related (disp)", dict(kind="related", ty="disposition")),
                  ("related (fact)", dict(kind="related", ty="fact")),
                  ("exact (all)", dict(kind="exact")), ("unrelated", dict(kind="unrelated"))]
        for mode in args.modes:
            rows = []
            for name, kw in groups:
                for L in L_:
                    m = cols.mask(b=b, m=mode, L=L, a=a_mid, **kw) & (c["ans"] == 1)
                    rows.append([name, f"L{L}", int(m.sum())] + [fmt(cols.q(m, k)) for k in ("mc", "mct", "mcc", "rf", "sr")])
            w(f"[{mode}]")
            w(table(["inputs", "layer", "n", "mc", "mct", "mcc", "rf", "sr"], rows))
            w("")
        w("T6b same groups, USER-TEXT positions (median mc / rf) [isolated]")
        rows = []
        for name, kw in groups:
            r = [name]
            for L in L_:
                m = cols.mask(b=b, m="isolated", L=L, a=a_mid, cat="user", **kw)
                r.append(f"{fmt(cols.q(m, 'mc'))} / {fmt(cols.q(m, 'rf'))}")
            rows.append(r)
        w(table(["inputs"] + [f"L{L}" for L in L_], rows))
        w("")
        # T7 Q4 match strength by category
        w("T7 match strength by category [isolated]: median mc (mct | mcc), and share of positions with mc > 0.5")
        for L in L_:
            rows = []
            for kind in ["related", "exact", "unrelated", "relation"]:
                for cat in CATS:
                    m = cols.mask(b=b, m="isolated", L=L, a=a_mid, kind=kind, cat=cat)
                    if not m.any():
                        continue
                    v = cols.vals(m, "mc")
                    rows.append([kind, cat, fmt(cols.q(m, "mc")), fmt(cols.q(m, "mct")), fmt(cols.q(m, "mcc")),
                                 fmt(float((v > 0.5).mean())), fmt(cols.q(m, "rf"))])
            w(table([f"L{L} kind", "cat", "mc", "mct", "mcc", "P(mc>0.5)", "rf"], rows))
            w("")
        # T8 write side
        w("T8 write side: median |delta|, |h_without - mu|, |delta|/|h_without - mu| per write token")
        rows = []
        for L in L_:
            for part, flag in (("content", 0), ("tail", 1), ("all", None)):
                rs = [r for r in write_rows if r["baseline"] == b and r["layer"] == L and (flag is None or r["is_tail"] == flag)]
                rows.append([f"L{L}", part, len(rs)] + [fmt(float(np.median([r[k] for r in rs])), f) for k, f in
                                                         (("delta_norm", ".2f"), ("h_without_mu_norm", ".2f"), ("ratio", ".3f"))])
        w(table(["layer", "tokens", "n", "|delta|", "|h_wo-mu|", "ratio"], rows))
        w("")

    # pre-registered predictions, primary config
    b = "without" if "without" in args.baselines else args.baselines[0]
    w(f"#################### PREDICTIONS (baseline={b}, isolated, alpha={a_mid:g}) ####################")
    m = cols.mask(b=b, m="isolated", a=a_mid)
    sr_all, sr_nh = cols.q(m, "sr"), cols.q(m & (c["cat"] != "head"), "sr")
    per_l = ", ".join(f"L{L} {cols.q(m & (c['L'] == L), 'sr'):.3f}" for L in L_)
    w(f"P1 median |s|/|h-mu| in [0.5, 1]: all positions {sr_all:.3f}, non-head {sr_nh:.3f} ({per_l}) -> "
      f"{'SUPPORTED' if 0.5 <= sr_all <= 1 else 'NOT SUPPORTED'} (all positions), "
      f"{'SUPPORTED' if 0.5 <= sr_nh <= 1 else 'NOT SUPPORTED'} (non-head)")
    dn = cols.q(m, "dn")
    per_l = ", ".join(f"L{L} {cols.q(m & (c['L'] == L), 'dn'):+.4f}" for L in L_)
    w(f"P2 median |h'|/|h|-1 < 0.10: {dn:+.4f}, p90 {cols.q(m, 'dn', 90):+.4f} ({per_l}) -> "
      f"{'SUPPORTED' if dn < 0.10 else 'NOT SUPPORTED'}")
    mu_ = m & (c["kind"] == "unrelated")
    tail_share = cols.energy(mu_ & (c["cat"] == "tail")) / cols.energy(mu_)
    head_share = cols.energy(mu_ & (c["cat"] == "head")) / cols.energy(mu_)
    per_l = ", ".join(f"L{L} {cols.energy(mu_ & (c['L'] == L) & (c['cat'] == 'tail')) / cols.energy(mu_ & (c['L'] == L)):.3f}" for L in L_)
    w(f"P3 template-tail share of steer energy on unrelated probes > 0.5: {tail_share:.3f} ({per_l}); "
      f"template-head share {head_share:.3f}; tail+head {tail_share + head_share:.3f} -> "
      f"{'SUPPORTED' if tail_share > 0.5 else 'NOT SUPPORTED'} (tail only), "
      f"{'SUPPORTED' if tail_share + head_share > 0.5 else 'NOT SUPPORTED'} (tail+head)")
    mr = m & (c["kind"] == "related") & (c["cat"] == "user")
    rf = cols.q(mr, "rf")
    per_l = ", ".join(f"L{L} {cols.q(mr & (c['L'] == L), 'rf'):.3f}" for L in L_)
    w(f"P4 median recall fraction on user-text positions of related probes ~0.3 (accept 0.2-0.4): {rf:.3f} "
      f"(IQR {cols.q(mr, 'rf', 25):.3f}-{cols.q(mr, 'rf', 75):.3f}; {per_l}) -> "
      f"{'SUPPORTED' if 0.2 <= rf <= 0.4 else 'NOT SUPPORTED'}")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------- main


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.scenarios))
    scenarios, unrelated = cfg["scenarios"], cfg["unrelated_probes"]
    ids = [s["id"] for s in scenarios]
    generic = [l.strip() for l in open(args.generic) if l.strip()]
    bargs = SimpleNamespace(device=args.device, topk=4, eta=1.0)

    log(f"loading {args.model} on {args.device}{' (tiny random model)' if args.tiny else ''}")
    model, tok = load(args)
    a_ids, b_ids = chat_ids(tok, "alpha"), chat_ids(tok, "beta")
    head_len, tail_len = common_prefix_len(a_ids, b_ids), common_suffix_len(a_ids, b_ids)
    head, tail = a_ids[:head_len], a_ids[len(a_ids) - tail_len:]

    rows, write_rows = [], []
    checks = {"calls": 0, "exact": 0, "e2e_max": 0.0, "e2e_n": 0}
    with torch.inference_mode():
        log(f"mu over {len(generic)} generic prompts, layers {args.layers}")
        mu, skip = v01.compute_mu(model, tok, generic, args.layers, args.device)
        log("session 1: collecting writes (with / without / counter)")
        writes = {s["id"]: v01.collect_writes(model, tok, s, args.layers, args.device) for s in scenarios}
        log("inputs (ceiling / baseline greedy continuations)")
        inputs = build_inputs(model, tok, scenarios, unrelated, head, tail, args)

        for b in args.baselines:
            for L in args.layers:
                iso = {s["id"]: v01.build_memory(mu[L], writes, [s], L, "all", b, "entropy", bargs, tail_len)[0]
                       for s in scenarios}
                comb = v01.build_memory(mu[L], writes, scenarios, L, "all", b, "entropy", bargs, tail_len)[0]
                K, T, D = {}, {}, {}
                for sid in ids:
                    K[sid], T[sid], dn, wr = write_keys(comb, writes, sid, L, b, tail_len)
                    D[sid] = dn.mean().item()
                    write_rows += wr
                d_all = float(np.mean([r["delta_norm"] for r in write_rows if r["baseline"] == b and r["layer"] == L]))
                K_all, T_all = torch.cat([K[s] for s in ids]), torch.cat([T[s] for s in ids])
                for mode in args.modes:
                    if mode == "isolated":
                        jobs = [(sid, iso[sid], {"K": K[sid], "tail": T[sid]},
                                 [i for i in inputs if i["scenario"] in (sid, None)]) for sid in ids]
                    else:
                        jobs = [("combined", comb, {"K": K_all, "tail": T_all}, inputs)]
                    for a in args.alphas:
                        first = True
                        for mem_id, mem, kctx, ins in jobs:
                            for inp in ins:
                                sid = inp["scenario"]
                                den = D[sid] if sid else (D[mem_id] if mode == "isolated" else d_all)
                                own = iso[sid] if (mode == "combined" and sid) else None
                                ctx = {**kctx, "den": den, "own": own}
                                ids_dev = inp["ids"][None].to(args.device)
                                rec = {}
                                with log_inject(model, L, mem, a, ctx, rec):
                                    logits = model(ids_dev).logits
                                checks["calls"] += 1
                                checks["exact"] += int(rec["exact"])
                                if first:  # end-to-end: same logits as the library's inject
                                    with inject(model, L, mem, a):
                                        ref = model(ids_dev).logits
                                    diff = (logits - ref).abs().max().item()
                                    assert diff < 1e-4, f"logging hook changes logits by {diff}"
                                    checks["e2e_max"] = max(checks["e2e_max"], diff)
                                    checks["e2e_n"] += 1
                                    first = False
                                tag = {"b": b, "m": mode, "L": L, "a": a, "mem": mem_id}
                                st = rec["stats"]
                                for t in range(len(inp["tok"])):
                                    row = {**tag, "iid": inp["iid"], "sc": sid, "ty": inp["type"], "kind": inp["kind"],
                                           "t": t, "tok": inp["tok"][t], "cat": CATS[inp["cat"][t]],
                                           "ans": int(t == inp["ans"])}
                                    for k, v in st.items():
                                        row[k] = r4(v[t])
                                    rows.append(row)
                log(f"done baseline={b} layer={L} ({len(rows)} position rows so far)")

    log("summarizing")
    cols = Cols(rows)
    summary = summarize(cols)
    with gzip.open(out_dir / "positions.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    for name, data in (("summary.csv", summary), ("write_tokens.csv", write_rows)):
        with open(out_dir / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            wr.writeheader()
            wr.writerows(data)
    (out_dir / "report.txt").write_text(report(cols, write_rows, args, checks))
    json.dump({**vars(args), "mu_skip_tokens": skip, "head_len": head_len, "tail_len": tail_len,
               "checks": checks, "row_keys": {"b": "baseline", "m": "mode", "L": "layer", "a": "alpha",
                                              "mem": "memory", "iid": "input id", "sc": "probe scenario",
                                              "ty": "type", "t": "position", "tok": "token id"},
               "inputs": [{k: i[k] for k in ("iid", "scenario", "type", "kind", "text", "prompt_len")}
                          | {"n_tokens": len(i["tok"])} for i in inputs],
               "torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              open(out_dir / "config.json", "w"), indent=2)
    log(f"wrote {len(rows)} position rows to {out_dir}")


if __name__ == "__main__":
    main()
