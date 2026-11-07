#!/usr/bin/env python
"""dream_compare Stage 0: why the Gemma-4 smoke (job 1214592) failed its tolerances (diagnostic, GPU, 2 x A100).

The smoke's identity checks passed (full == text 1.1e-4, keep-all 0, shared == splice 1.1e-4, rope_shift ~1e-6) and every
benchmark item gave the same greedy tokens restored vs text, but three logit gaps exceeded 0.01: the sliding check
(contiguous vs the model's own pass 0.0196; sparse vs the masked reference 0.0050 is fine) and the LongMemEval item
(R-full vs text 0.026, a 624 + 53-token memory: inside the 1024 window). Candidate causes: Gemma-4's MoE routing (top-8 of
128 experts per token and layer) flipping on near-ties between two numerically different but equivalent passes; fp32
accumulation order (matmul shapes, 2 GPUs); the soft-cap; or a real bug in the sliding mask / restore path.

What this runs (eager attention, as the smoke and Stage A; part "stageb" switches to SDPA, as Stage B):
  Routing          a record / compare / replay hook on every layer's router: "record" keeps each token's top-8 experts
                   (and the 8th-9th probability margin) of a reference pass; "compare" counts the tokens whose expert
                   SET differs in a second pass; "replay" makes the second pass use the reference's experts (their
                   weights recomputed from its own router probabilities, as the router does) on the tokens that differ.
                   If the gaps are routing flips, replay brings them down to the other models' level (~1e-4).
  smoke            the Stage 0 checks again: identity (run.smoke_identity), the sliding check and the four benchmark
                   items, each restored path also with routing compare / replay; determinism (the same pass twice).
  stagea           Stage A's per-item asserts on its 133 items (stage_a.harvest / measure; R-full vs T-full at the
                   teacher-forced rows, the gen-time last-row check, keep-all vs R-full, the same picks in every head
                   vs the splice) with flips counted and the R-full check replayed: the distribution the Gemma
                   tolerance is chosen from.
  stageb           Stage B's job-failing identity check (stage_b.identity_checks, SDPA) on the first context of every
                   split, in the order the gen jobs and the smoke meet them.
-> <out>/diag.json (+ the log). Run through jobs.py (pytest first):
  DC_RUNNER=diag_gemma_moe SPECS="diag|OUT|" python experiments/dream_compare/jobs.py --model gemma4_26b_a4b ...
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE / "benchmarks"))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Routing:
    """Record / compare / replay of the MoE routing (Gemma 4: layer.router -> (probs, top-k weights, top-k index)).
    Rows are consumed in call order per layer, so a restored path (harvest of the memory, then the question) lines up
    with a one-pass reference over the same tokens."""

    def __init__(self, C):
        import torch
        self.torch = torch
        self.routers = {l: blk.router for l, blk in enumerate(C.blocks) if getattr(blk, "router", None) is not None}
        for l, r in self.routers.items():
            r.forward = self._wrap(l, r, r.forward)
        self.reset()

    def reset(self, mode=None, src=None):
        self.mode, self.src = mode, src
        self.ptr = defaultdict(int)
        self.rec = defaultdict(list)
        self.n_tok = 0
        self.n_flip = 0
        self.flip_layers = []
        self.flip_margins = []

    def recorded(self):
        t = self.torch
        return {l: (t.cat([a for a, _ in v]), t.cat([b for _, b in v])) for l, v in self.rec.items()}

    def _wrap(self, l, router, orig):
        t = self.torch

        def fwd(hs):
            probs, w, idx = orig(hs)
            if self.mode is None:
                return probs, w, idx
            k = idx.shape[-1]
            if self.mode == "record":
                tv = probs.topk(k + 1, dim=-1).values
                self.rec[l].append((idx.detach().cpu(), (tv[:, k - 1] - tv[:, k]).detach().float().cpu()))
                return probs, w, idx
            n = idx.shape[0]
            p = self.ptr[l]
            sidx, smarg = self.src[l]
            assert p + n <= sidx.shape[0], f"layer {l}: {p + n} rows > the reference's {sidx.shape[0]}"
            ridx = sidx[p:p + n].to(idx.device)
            self.ptr[l] = p + n
            diff = (t.sort(idx, -1).values != t.sort(ridx, -1).values).any(-1)
            nd = int(diff.sum())
            self.n_tok += n
            self.n_flip += nd
            if nd:
                self.flip_layers += [l] * nd
                self.flip_margins += smarg[p:p + n][diff.cpu()].tolist()
            if self.mode == "replay" and nd:
                rw = probs.gather(-1, ridx)
                rw = rw / rw.sum(-1, keepdim=True)
                rw = rw * router.per_expert_scale[ridx]
                w = t.where(diff[:, None], rw.to(w.dtype), w)
                idx = t.where(diff[:, None], ridx, idx)
            return probs, w, idx
        return fwd

    def stats(self):
        return {"tokens": self.n_tok, "flipped_tokens": self.n_flip,
                "flip_layers": sorted(set(self.flip_layers)),
                "flip_margin_max": max(self.flip_margins) if self.flip_margins else None,
                "flip_margin_median": (sorted(self.flip_margins)[len(self.flip_margins) // 2]
                                       if self.flip_margins else None)}


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def dist(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "max": max(xs) if xs else None, "p99": pct(xs, 0.99), "p95": pct(xs, 0.95),
            "p90": pct(xs, 0.9), "median": pct(xs, 0.5), "n_over_0.01": sum(x > 0.01 for x in xs),
            "n_over_0.05": sum(x > 0.05 for x in xs), "n_over_0.1": sum(x > 0.1 for x in xs)}


# ----------------------------------------------------------------------------------------------- part: smoke


def pair(rt, ref_fn, alt_fn, maxdiff):
    """ref_fn() -> logits (recorded); alt_fn() -> logits, run with compare and with replay."""
    rt.reset("record")
    a = ref_fn()
    rec = rt.recorded()
    rt.reset("compare", rec)
    b = alt_fn()
    st = rt.stats()
    rt.reset("replay", rec)
    c = alt_fn()
    rt.reset()
    return {"diff": maxdiff(a, b), "diff_replay": maxdiff(a, c), "routing": st}


def part_smoke(C, M, HM, RUN, rt):
    import torch
    DB = M.DB()
    out = {"identity": RUN.smoke_identity(C, M, HM)}
    log(f"identity (as the smoke): {json.dumps(out['identity'])}")
    # sliding (as run.smoke_sliding)
    W = C.window
    text = " ".join(f"Note {i}: the {['red', 'blue', 'green'][i % 3]} box holds item {i * 7 % 101}." for i in range(160))
    ids = DB.chat_ids(C.tok, text)[:W + 130]
    n = len(ids)
    dev = M.input_device(C)
    lg = lambda h: M.logits(C, h)
    with torch.no_grad():
        own = lambda: lg(C.body(input_ids=ids[None].to(dev)).last_hidden_state[0, n - 20:])

        def contiguous():
            st, _, _ = M.harvest_kv(C, ids[:n - 20])
            h, _ = M.forward(C, ids[n - 20:], st)
            return lg(h)
        keep = list(range(3)) + list(range(n - 220, n - 20))
        allp = torch.arange(n)
        Rm = {k: v.clone() for k, v in M.pos_masks(C, allp, allp).items()}
        drop = torch.ones(n, dtype=torch.bool)
        drop[torch.tensor(keep)] = False
        drop[n - 20:] = False
        for k in Rm:
            Rm[k][0, 0, n - 20:, drop.to(Rm[k].device)] = torch.finfo(torch.float32).min
        ref = lambda: lg(C.body(input_ids=ids[None].to(dev), attention_mask=Rm).last_hidden_state[0, n - 20:])

        def sparse():
            st, _, _ = M.harvest_kv(C, ids[:n - 20])
            h, _ = M.forward(C, ids[n - 20:], M.index_prefix(st, keep, start=n - 20))
            return lg(h)
        s = {"n_tokens": n, "window": W}
        s["own_twice"] = DB.maxdiff(own(), own())
        s["contiguous_twice"] = DB.maxdiff(contiguous(), contiguous())
        s["contiguous_vs_own"] = pair(rt, own, contiguous, DB.maxdiff)
        s["sparse_vs_masked_reference"] = pair(rt, ref, sparse, DB.maxdiff)
        # the masked reference with its last 20 rows' window left as is == own (nothing dropped) vs own: the mask path
        s["own_vs_explicit_position_masks"] = DB.maxdiff(
            own(), lg(C.body(input_ids=ids[None].to(dev), attention_mask=M.pos_masks(C, allp, allp)).last_hidden_state[0, n - 20:]))
    out["sliding"] = s
    log(f"sliding: {json.dumps(s)}")
    # the four benchmark items (as run.smoke_bench: R-full vs the memory as text)
    out["benchmarks"] = {}
    for name in RUN.BENCH_SPLITS:
        b = RUN.load_bench(name)
        it, (msgs, ev_i, mem_ids) = RUN.pick_item(b, C.tok, RUN.SMOKE_PICK[name])
        H, T, Q = M.session_pieces(C.tok, msgs, it.question)
        mem = H + [t for x in T for t in x]
        ans = it.answers[0] if it.answers else it.options[it.correct_option]
        a_ids = DB.text_ids(C.tok, ans)
        Qa = torch.cat([torch.tensor(Q), a_ids])
        with torch.no_grad():
            text = lambda: lg(M.forward(C, torch.cat([torch.tensor(mem), Qa]))[0][len(mem):])

            def restored():
                st, _, _ = M.harvest_kv(C, mem)
                return lg(M.forward(C, Qa, st)[0])
            r = {"item": it.item_id, "n_turn_tokens": [len(x) for x in T], "text_twice": DB.maxdiff(text(), text()),
                 "restored_twice": DB.maxdiff(restored(), restored())}
            r.update(pair(rt, text, restored, DB.maxdiff))
        out["benchmarks"][name] = r
        log(f"{name}: {json.dumps(r)}")
    return out


# ----------------------------------------------------------------------------------------------- part: Stage A


def part_stage_a(C, SA, rt, n_items=None):
    import torch
    M, HM, B, DB = SA.M, SA.HM, SA.B, SA.DB
    items = SA.load_items()[:n_items] if n_items else SA.load_items()
    rows = []
    t0 = time.time()
    for i, it in enumerate(items):
        q = it["question"]
        ans = DB.text_ids(C.tok, it["answer"])
        with torch.no_grad():
            st0 = SA.harvest(C, it["stored_turn"], stats=False)
            r = {"item": it["question_id"], "n": st0.n, "n_content": len(st0.content)}
            # (1) R-full vs T-full, teacher-forced rows (stage_a.tf_item's d_id); the restored path harvests INSIDE
            # the compare / replay window (one harvest + one measure per call), so its rows line up with T-full's
            def rfull():
                st = SA.harvest(C, it["stored_turn"], stats=False)
                return SA.measure(C, st, "R-full", q, ans, st.P)[1]
            tf = pair(rt, lambda: SA.measure(C, st0, "T-full", q, ans)[1], rfull, DB.maxdiff)
            r["tf_rfull_vs_tfull"] = tf["diff"]
            r["tf_rfull_vs_tfull_replay"] = tf["diff_replay"]
            r["tf_routing"] = tf["routing"]
            # (2) gen-time check at the last prompt row (stage_a.gen_item's d_id)
            qt = DB.qt_ids(C, q)

            def g_rest():
                st = SA.harvest(C, it["stored_turn"], stats=False)
                return M.logits(C, M.forward(C, qt, st.P)[0][-1:])
            g = pair(rt, lambda: M.logits(C, M.forward(C, SA.prompt_ids(C, st0, "T-full", q))[0][-1:]), g_rest,
                     DB.maxdiff)
            r["gen_rfull_vs_tfull"] = g["diff"]
            r["gen_rfull_vs_tfull_replay"] = g["diff_replay"]
            r["gen_flipped_tokens"] = g["routing"]["flipped_tokens"]
            # (3) keep-all vs R-full (no routing difference expected: the same pass shapes)
            _, lr = SA.measure(C, st0, "R-full", q, ans, st0.P)
            _, la = SA.measure(C, st0, "slice", q, ans, st0.P, HM.keep_all(C, st0.n))
            r["keep_all_vs_rfull"] = DB.maxdiff(la, lr)
            # (4) the same picks in every head vs the splice (StreamingLLM at k2, as tf_item)
            sel = B.streaming(st0.content, B.budget_k("k2", len(st0.content)))
            same = {l: [list(sel)] * C.layout[l].n_kv for l in range(len(C.blocks))}
            sp = pair(rt, lambda: SA.measure(C, st0, "slice", q, ans, st0.P, SA.keep_of(C, st0, same))[1],
                      lambda: SA.measure(C, st0, "slice", q, ans, SA.splice(st0, sel))[1], DB.maxdiff)
            r["same_picks_vs_splice"] = sp["diff"]
            r["same_picks_vs_splice_replay"] = sp["diff_replay"]
            r["same_flipped_tokens"] = sp["routing"]["flipped_tokens"]
        rows.append(r)
        if i % 10 == 0 or tf["diff"] > 0.01:
            el = (time.time() - t0) / 60
            log(f"stage A {i + 1}/{len(items)} {r['item']} n {r['n']}: tf {r['tf_rfull_vs_tfull']:.2e} (replay "
                f"{r['tf_rfull_vs_tfull_replay']:.2e}, flipped {tf['routing']['flipped_tokens']}/"
                f"{tf['routing']['tokens']}), gen {r['gen_rfull_vs_tfull']:.2e}, keep {r['keep_all_vs_rfull']:.1e}, "
                f"same {r['same_picks_vs_splice']:.2e} | {el:.1f} min")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    keys = ["tf_rfull_vs_tfull", "tf_rfull_vs_tfull_replay", "gen_rfull_vs_tfull", "gen_rfull_vs_tfull_replay",
            "keep_all_vs_rfull", "same_picks_vs_splice", "same_picks_vs_splice_replay"]
    summ = {k: dist([r[k] for r in rows]) for k in keys}
    nf = [r for r in rows if r["tf_routing"]["flipped_tokens"] == 0]
    summ["tf_items_without_flips"] = len(nf)
    summ["tf_rfull_vs_tfull_without_flips"] = dist([r["tf_rfull_vs_tfull"] for r in nf])
    summ["tf_rfull_vs_tfull_with_flips"] = dist([r["tf_rfull_vs_tfull"] for r in rows if r not in nf])
    summ["flip_margins_max"] = max((r["tf_routing"]["flip_margin_max"] or 0) for r in rows)
    log(f"stage A summary: {json.dumps(summ)}")
    return {"items": rows, "summary": summ}


# ----------------------------------------------------------------------------------------------- part: Stage B


def part_stage_b(C, SB, rt):
    """stage_b.identity_checks (SDPA) on the first context of every split as a fresh gen job meets it (the first item's
    context in file order) and as the smoke meets it (--limit 1: the subset's first item)."""
    import torch
    Pm = SB.P()
    out = {}
    for name, splits in SB.BENCHES.items():
        for split in splits:
            b = SB.PC.load_split(name, split)
            subset = SB.subset_ids(b) if (name, split) in SB.SUBSET_SPLITS else []
            smoke_first = (sorted(subset) or sorted(it.item_id for it in b.items))[0]
            by_id = {it.item_id: it for it in b.items}
            cands = {"gen": b.items[0].context_id, "smoke": by_id[smoke_first].context_id}
            for tag, cid in cands.items():
                key = f"{name}_{split}|{tag}"
                if tag == "smoke" and cid == cands["gen"]:
                    out[key] = {"same_as": f"{name}_{split}|gen"}
                    continue
                context = b.contexts[cid]
                its = [it for it in b.items if it.context_id == cid]
                if tag == "smoke":
                    its = [by_id[smoke_first]]
                t0 = time.time()
                with torch.no_grad():
                    mem = Pm.build_memory(C, name, context)
                    store = Pm.Store(C, mem, raw_gpu_gb=4.0)
                    Q = Pm.question_ids(C, Pm.question_text(name, its[0], context))
                    ident = SB.identity_checks(C, store, mem, Q)
                del store
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ident.update(context=cid, n_turns=len(mem.turns), n_tokens=mem.N, seconds=round(time.time() - t0, 1))
                out[key] = ident
                log(f"stage B identity {key}: {json.dumps(ident)}")
    return out


# ----------------------------------------------------------------------------------------------- main


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="diag")
    ap.add_argument("--model", default="gemma4_26b_a4b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gb-per-gpu", type=int, default=66)
    ap.add_argument("--parts", default="smoke,stagea,stageb")
    ap.add_argument("--n-items", type=int, default=None, help="Stage A items (default all 133)")
    A = ap.parse_args(argv)
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    SA = _load("dream_compare_stage_a_diag", HERE / "stage_a.py")
    SB = _load("dream_compare_stage_b_diag", HERE / "stage_b.py")
    RUN = _load("dream_compare_run_diag", HERE / "run.py")
    M, HM = SA.M, SA.HM
    t0 = time.time()
    C = M.setup(A.model, A.device, attn="eager", gb_per_gpu=A.gb_per_gpu)
    C.multi_device = len((C.load_info or {}).get("device_map_devices") or []) > 1
    cfg = C.model.config
    res = {"model": A.model, "load": C.load_info, "paris": C.paris, "env": M.DB().env_meta(),
           "experts_implementation": getattr(cfg, "_experts_implementation", None),
           "top_k_experts": getattr(cfg, "top_k_experts", None), "num_experts": getattr(cfg, "num_experts", None),
           "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}}
    log(f"{C.label} loaded in {time.time() - t0:.0f}s; {json.dumps({k: res[k] for k in ('experts_implementation', 'top_k_experts', 'num_experts', 'tf32')})}")
    rt = Routing(C)
    assert len(rt.routers) == len(C.blocks), f"routers on {len(rt.routers)} of {len(C.blocks)} layers"
    parts = A.parts.split(",")

    def save():
        res["minutes"] = (time.time() - t0) / 60
        res["peak_gb"] = M.peak_gb()
        json.dump(res, open(out / "diag.json", "w"), indent=1, default=str)
    if "smoke" in parts:
        res["smoke"] = part_smoke(C, M, HM, RUN, rt)
        save()
    if "stagea" in parts:
        res["stage_a"] = part_stage_a(C, SA, rt, A.n_items)
        save()
    if "stageb" in parts:
        C.model.set_attn_implementation("sdpa")
        assert C.model.config._attn_implementation == "sdpa"
        SB.P().ATTN_GB = 4.0
        res["stage_b"] = part_stage_b(C, SB, rt)
        save()
    save()
    log(f"diag done in {res['minutes']:.1f} min -> {out / 'diag.json'}")


if __name__ == "__main__":
    main()
