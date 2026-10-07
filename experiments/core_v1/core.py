"""core_v1 pure helpers (torch, no model loading): the layer-mapping rule, the condition list, the
per-condition injection fields (doses and controls), the batched two-pass injection hook, batched
KV-cached generation with fixed per-row random streams, and the judge-output parser.
Unit-tested in tests/test_core_v1.py (tiny random Qwen2 on CPU).

Two-pass reading (as xlayer_v1): pass 1 runs the model WITHOUT memory and gives, at every read layer l,
the pooled_w256 read keys k_t, the match m_t = max cos(k_t, stored keys) and the hard gate
g_t = [m_t > thr] * [t not in the template head]. Pass 2 adds, at the output of block l,
    h_t <- h_t + field_t,   field_t = alpha * g_t * M k_t            (the design: mem@alpha)
Every position after the prompt (generated tokens) reuses the last prompt position's field (no new user
text, so the pooled key, the gate and the recall do not change), exactly as xl.XInject.

Controls (all at one alpha, through the item's OWN gate g, which is identical for every memory of an item
because every memory of an item is filed under the same keys):
  rand     alpha * g_t * |M k_t| * u_l       u_l a fixed random unit vector per (item, layer)
  swap     alpha * g_t * M_swap k_t          M_swap: the partner item's shifts written under THIS item's keys
  placebo  alpha * g_t * M_placebo k_t       M_placebo: the placebo experience's plain shifts, this item's keys
  gate_on  alpha * [t not head] * M k_t      the design memory with the threshold removed
"""

import contextlib
import json
import math
import re
import zlib
from dataclasses import dataclass

import torch

from seahorse.residual import inject

# ------------------------------------------------------------------ layer mapping


def map_layers(src_layers, n_src, n_dst):
    """Relative-depth rule: the output of block l of an n_src-block model sits at depth (l + 1) / n_src;
    the block of an n_dst-block model at the same depth is round((l + 1) * n_dst / n_src) - 1 (half up)."""
    return [int(math.floor((l + 1) * n_dst / n_src + 0.5)) - 1 for l in src_layers]


def map_layers_alt(src_layers, n_src, n_dst):
    """The other common convention, depth = l / (n - 1); documented next to map_layers."""
    return [int(math.floor(l * (n_dst - 1) / (n_src - 1) + 0.5)) for l in src_layers]


def layer_types(config):
    tc = getattr(config, "text_config", config)
    lt = getattr(tc, "layer_types", None) or ["full_attention"] * tc.num_hidden_layers
    return ["full" if t == "full_attention" else "linear" for t in lt]


# --------------------------------------------------------------------- conditions


@dataclass(frozen=True)
class Cond:
    cid: str
    kind: str          # nomem | ctx | mem | rand | swap | placebo | gate_on | ref
    mem: str = None    # which stored memory: design | swap | placebo | centroid | without
    alpha: float = 0.0
    seed2: bool = False  # also generated with the second seed set (samples only)
    mech: bool = False


def conditions(cfg, mech):
    """The generated conditions of one item. cfg: doses, control_alpha, mech_doses, seed2 (cids)."""
    s2 = set(cfg["seed2"])
    a = cfg["control_alpha"]
    cs = [Cond("nomem", "nomem", seed2="nomem" in s2), Cond("ctx", "ctx", seed2="ctx" in s2)]
    cs += [Cond(f"mem@{d:g}", "mem", "design", d, seed2=f"mem@{d:g}" in s2) for d in cfg["doses"]]
    cs += [Cond(f"rand@{a:g}", "rand", "design", a), Cond(f"swap@{a:g}", "swap", "swap", a),
           Cond(f"placebo@{a:g}", "placebo", "placebo", a), Cond(f"gate_on@{a:g}", "gate_on", "design", a)]
    if mech:  # opposite@alpha is the design memory, i.e. mem@alpha (aliased in the analysis)
        cs += [Cond(f"{r}@{d:g}", "ref", r, d, mech=True) for r in ("centroid", "without") for d in cfg["mech_doses"]]
    return cs


def gated(c):
    """Conditions whose injection goes through the item's gate (zero when the gate is shut everywhere)."""
    return c.kind in ("mem", "rand", "swap", "placebo", "ref")


def row_specs(conds, n_samples, seed1=1, seed2=2):
    """Generated rows of one prompt: per condition 1 greedy + n_samples samples with seed set 1, and
    n_samples more samples with seed set 2 for the seed2 conditions (the greedy answer does not depend on
    the seed). Returns [(cond, seedset, j)], j = 0 greedy, j >= 1 sample stream j."""
    rows = []
    for c in conds:
        rows += [(c, seed1, j) for j in range(n_samples + 1)]
        if c.seed2:
            rows += [(c, seed2, j) for j in range(1, n_samples + 1)]
    return rows


def seed_of(tag, text, base):
    """samples_v2.seed_of: the per-prompt seed (the same random numbers in every condition)."""
    return (base * 1000003 + zlib.crc32(f"{tag}|{text}".encode())) % (2 ** 31 - 1)


# ------------------------------------------------------------------------ fields


def hard_gate(match, cats, thr):
    """1 where match > thr and the position is not the template head (diag_keys hard read)."""
    return ((match > thr) & (cats != 0)).to(match.dtype)


def read_layer(keys, Kst, thr, cats):
    """Pass-1 read at one layer: match [T] and hard gate [T]."""
    match = (keys @ Kst.T).max(-1).values
    return match, hard_gate(match, cats, thr)


def rand_direction(d, seed):
    g = torch.Generator().manual_seed(int(seed))
    v = torch.randn(d, generator=g, dtype=torch.float64)
    return (v / v.norm()).float()


def cond_field(kind, alpha, g, cats, recall, rand_dir=None):
    """[T, d] field added at one layer. recall = M k_t [T, d] of the memory the condition uses."""
    if kind in ("mem", "swap", "placebo", "ref"):
        return alpha * g[:, None] * recall
    if kind == "rand":
        return alpha * g[:, None] * recall.norm(dim=-1, keepdim=True) * rand_dir.to(recall)[None]
    if kind == "gate_on":
        return alpha * (cats != 0).to(recall.dtype)[:, None] * recall
    raise ValueError(kind)


def memory_chunks(shifts, keys):
    """[n, d] shifts + [n, r] keys -> diag_keys.build chunks (one pooled write per moment)."""
    return [(shifts[j][None], keys[j][None], None) for j in range(shifts.shape[0])]


# ---------------------------------------------------------------- batched hooks


class BatchField:
    """Pass-2 hook at one layer for B rows: the prefill (the whole sequence) gets field [B, T, d]; every
    later one-token step gets the last row field[:, -1]. begin() before each forward / generation."""

    def __init__(self, layer, field):
        self.layer, self.field, self.prefilled = layer, field, False

    def begin(self, ids=None):
        self.prefilled = False

    def read(self, h, _alpha):
        if not self.prefilled:
            assert h.shape[:2] == self.field.shape[:2], (tuple(h.shape), tuple(self.field.shape))
            self.prefilled = True
            return h + self.field.to(h.dtype)
        assert h.shape[1] == 1, "after the prefill, one token per step"
        return h + self.field[:, -1:].to(h.dtype)


@contextlib.contextmanager
def injecting(model, hooks):
    with contextlib.ExitStack() as st:
        for hk in hooks:
            hk.begin()
            st.enter_context(inject(model, hk.layer, hk, 1.0))
        yield


def stack_fields(per_row, layers, T, d, device):
    """per_row: [None | {layer: [T, d]}] -> {layer: [B, T, d]} (zeros for rows without a field), or {} if
    no row has one."""
    if not any(per_row):
        return {}
    out = {l: torch.zeros(len(per_row), T, d, device=device) for l in layers}
    for b, F in enumerate(per_row):
        for l, f in (F or {}).items():
            out[l][b] = f
    return out


# ------------------------------------------------------------ batched generation


def qwen_sampler(T, top_k, top_p, presence):
    """think_v1.qwen_sampler: presence penalty on generated tokens -> top-k -> temperature -> top-p."""
    def f(logits, u, seen):
        l = logits - presence * seen.float() if presence else logits
        v, i = l.topk(top_k, -1)
        p = torch.softmax(v.double() / T, -1)
        p = p * ((p.cumsum(-1) - p) < top_p)
        c = p.cumsum(-1)
        j = torch.searchsorted(c, u * c[:, -1:]).clamp_max(top_k - 1)
        return i.gather(1, j).squeeze(1)
    f.presence = presence > 0
    return f


def generate_batch(model, body, lm_head, prompts, rows, *, max_new, eos, vocab, sampler=None, streams=None,
                   fields=None, first_ids=None, check_finite=True):
    """KV-cached generation of B rows, thinking off.

    prompts  [T] (one prompt for every row) or [B, T] (equal-length prompts, no padding)
    rows     [(greedy, stream)]: greedy rows take the argmax; sample rows take sampler(logits, u) with u from
             stream (s, j): streams[s] = (seed, n_draw) is one torch.Generator seeded with `seed` that draws
             n_draw uniforms per step, and the row uses the j-th (think_v1.generate's layout: B = 1 + samples,
             row 0 greedy, row j uses u[j]). The numbers a row sees depend only on (seed, n_draw, j), never
             on the other rows, so every condition gets the same random numbers.
    fields   {layer: [B, T, d]} pass-2 fields (BatchField), or None
    first_ids  token ids whose first-step log-probs are returned per row (yes/no margins)
    Returns [{"ids", "n_tok", "capped", "first"}] per row.
    """
    dev = lm_head.weight.device
    B = len(rows)
    x = (prompts[None].expand(B, -1) if prompts.dim() == 1 else prompts).to(dev).contiguous()
    assert x.shape[0] == B
    streams = streams or {}
    gens = {}
    for s, (seed, n_draw) in streams.items():
        g = torch.Generator(device=dev)
        g.manual_seed(int(seed))
        gens[s] = (g, n_draw)
    greedy = torch.tensor([bool(r[0]) for r in rows], device=dev)
    assert bool(greedy.all()) or sampler is not None, "sample rows need a sampler"
    by_stream = {}
    for b, (gr, st) in enumerate(rows):
        if not gr:
            assert st is not None and st[0] in gens and 1 <= st[1] < gens[st[0]][1], f"row {b}: bad stream {st}"
            by_stream.setdefault(st[0], ([], []))
            by_stream[st[0]][0].append(b)
            by_stream[st[0]][1].append(st[1])
    by_stream = {s: (torch.tensor(bs, device=dev), torch.tensor(js, device=dev)) for s, (bs, js) in by_stream.items()}
    eos_t = torch.tensor(sorted(eos), device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    seen = torch.zeros(B, vocab, dtype=torch.bool, device=dev) if (sampler is not None and sampler.presence) else None
    ar = torch.arange(B, device=dev)
    hooks = [BatchField(l, F) for l, F in sorted((fields or {}).items())]
    steps, first, past = [], None, None
    with injecting(model, hooks):
        for t in range(max_new):
            o = body(input_ids=x, past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = lm_head(o.last_hidden_state[:, -1]).float()
            if check_finite and not bool(torch.isfinite(logits[~done]).all()):
                raise FloatingPointError(f"non-finite logits at step {t}")
            if t == 0 and first_ids is not None:
                first = torch.log_softmax(logits, -1)[:, first_ids].cpu()
            u = torch.zeros(B, 1, dtype=torch.float64, device=dev)
            draws = {s: torch.rand(n, 1, generator=g, device=dev, dtype=torch.float64) for s, (g, n) in gens.items()}
            for s, (bs, js) in by_stream.items():
                u[bs] = draws[s][js]
            nxt = logits.argmax(-1)
            if sampler is not None and not bool(greedy.all()):
                nxt = torch.where(greedy, nxt, sampler(logits, u, seen))
            active = ~done
            nxt = torch.where(done, eos_t[0], nxt)
            steps.append(nxt)
            if seen is not None:
                seen[ar, nxt] |= active
            done = done | (active & torch.isin(nxt, eos_t))
            if bool(done.all()):
                break
            x = nxt[:, None]
    toks = torch.stack(steps, 1).cpu().tolist()
    eset = set(eos)
    out = []
    for b, row in enumerate(toks):
        k = next((i for i, t in enumerate(row) if t in eset), None)
        ids = row if k is None else row[:k]
        out.append({"ids": ids, "n_tok": len(ids), "capped": k is None,
                    "first": None if first is None else first[b].tolist()})
    return out


# --------------------------------------------------------------- text metrics


def rep_rate(ids, n=4):
    """Share of repeated n-grams (1 - distinct / total); 0 for answers shorter than n."""
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(g)) / len(g) if g else 0.0


# ------------------------------------------------------------------------ judge


def render(template, **kw):
    """str.format with only the given keys replaced (JSON braces in the template stay literal)."""
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


_BOOL = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}


def _coerce(v, spec):
    if spec == "bool":
        if isinstance(v, bool):
            return v
        return _BOOL.get(str(v).strip().strip('"').lower())
    s = str(v).strip().strip('"').lower()
    return s if s in spec else None


def parse_judge(text, schema):
    """Judge output -> {key: value or None, ..., "ok": every key parsed}. schema: {key: "bool" | [options]}.
    Takes the first {...} block that json-decodes; otherwise reads each key with a regex."""
    vals = {}
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in schema):
            vals = {k: _coerce(obj[k], spec) for k, spec in schema.items() if k in obj}
            break
    for k, spec in schema.items():
        if vals.get(k) is None:
            m = re.search(rf'"?{re.escape(k)}"?\s*[:=]\s*"?([A-Za-z]+)', text)
            vals[k] = _coerce(m.group(1), spec) if m else None
    vals["ok"] = all(vals[k] is not None for k in schema)
    return vals
