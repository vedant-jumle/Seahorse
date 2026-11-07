#!/usr/bin/env python
"""dream_compare Stage B, paradigm P5: a MemArt-style KV memory (PREREG.md "Stage B"), reimplemented from the paper
(MemArt, "KVCache-Centric Memory for LLM Agents", ICLR 2026 submission; no public code).

What the paper specifies, and what is done here:
  blocks      the stored KV cache is cut into fixed blocks of BS = 16 tokens (paper: BS 16; 8 vs 16 "largely
              insensitive"). Here: each session's turn tokens (the chat head excluded; it is always kept) are cut
              into consecutive 16-token blocks; the last block of a session may be shorter (padded, see below).
  relevance   I(q, K) = max over the block's keys of q . k (eq. 3 is the AABB upper bound of exactly this; the plan
              asks for exact scoring without the AABB pre-filter). q and k are taken WITHOUT RoPE ("decoupled
              position encoding": blocks are stored without PE; Fig. 3 "Request querys" vs "Historical KVCache W/O
              PE"): the stored post-RoPE keys are rotated back to position 0 per entry; q = the layer's query
              projection (+ q_norm where the model has one), before RoPE.
  aggregation multi-token: normalise each query's scores over all candidate blocks, then aggregate over the query
              tokens (eq. 4). Reciprocal-rank normalisation 1 / (rank + c), c = 60 (Cormack et al. 2009; the paper
              gives no c) and Max aggregation ("RR-Max", the paper's choice for Qwen2.5-7B; Max beat Sum in its
              ablation). Every (query token, query head) pair counts as one query.
  selection   top-k blocks, "the same set of k blocks for all attention heads"; per LAYER (the scoring uses that
              layer's queries and keys inside the forward pass: retrieval happens in the prefill of the request,
              so layer l's queries already attended to the blocks retrieved at layers < l). Ties go to the earlier
              block. The blocks are concatenated in chronological order.
  positions   the retrieved blocks are re-encoded contiguously (eq. 6): after the chat head (positions 0..nh-1),
              block j's tokens get positions nh + 16 j + i; the question follows at nh + 16 k.
  budget      k = the number of blocks matched to a token budget (P5: P3's restored turn tokens; P5s: P4's kept
              entries), ceil(budget / 16).
Departures (PREREG): exact scoring instead of AABB; RR c = 60 chosen here; per-layer selection inferred from the
paper's per-attention Algorithm 1; the request's queries are the question turn's tokens (user header + question +
generation prompt); short last blocks of a session are padded with masked slots so that every layer keeps
exactly 16 k slots at the same positions (pad slots get -inf attention in that layer only).

Toy-tensor functions (block_scores, rr_max, select_blocks) are tested in test_stage_b.py.
"""

import math
import sys

import torch

BLOCK = 16
RR_C = 60.0
SCORE_GB = 0.5          # block_scores' [query rows, memory tokens] matrix is built in chunks above this (memory only)


# ----------------------------------------------------------------------------------------------- blocks


def make_blocks(sessions_tokens, block=BLOCK):
    """sessions_tokens: [(g0, n)] the global [start, start + n) index range of each session's turn tokens (head
    excluded). -> [(g0, n_b)] blocks: consecutive runs of `block` tokens within each session (the last one shorter)."""
    out = []
    for g0, n in sessions_tokens:
        for s in range(0, n, block):
            out.append((g0 + s, min(block, n - s)))
    return out


def token_to_block(blocks, n_tokens):
    """[n_tokens] long: the block of each global token index (-1 for tokens in no block, e.g. the head)."""
    t2b = torch.full((n_tokens,), -1, dtype=torch.long)
    for j, (g0, n) in enumerate(blocks):
        t2b[g0:g0 + n] = j
    return t2b


# ----------------------------------------------------------------------------------------------- scoring


def block_scores(q, k_raw, t2b, n_blocks, n_kv):
    """Exact block relevance I(q, K) = max_{k in block} q . k for every query row.
    q [Hq, Tq, d] (one request, pre-RoPE), k_raw [Hkv, N, d] (pre-RoPE, the memory tokens), t2b [N] block of each
    token (-1 = none). Query head h reads KV head h // (Hq / Hkv) (GQA). -> [Hq * Tq, n_blocks] (rows = (head, token),
    -inf where a block has no token)."""
    Hq, Tq, d = q.shape
    g = Hq // n_kv
    qg = q.reshape(n_kv, g, Tq, d).float()
    kf = k_raw.float()
    keep = t2b >= 0
    N = kf.shape[1]
    # the [rows, N] score matrix in query-token chunks when it would exceed SCORE_GB (the same numbers below it)
    tc = Tq if Hq * Tq * N * 4 <= SCORE_GB * 2 ** 30 else max(1, int(SCORE_GB * 2 ** 30 // (Hq * N * 4)))
    out = torch.full((n_kv, g, Tq, n_blocks), -math.inf, device=q.device, dtype=torch.float32)
    kd = keep.to(q.device)
    bidx = t2b[keep].to(q.device)
    for a in range(0, Tq, tc):
        b = min(Tq, a + tc)
        s = torch.einsum("hgtd,hnd->hgtn", qg[:, :, a:b], kf).reshape(Hq * (b - a), -1)   # [R, N]
        s = s[:, kd]
        o = torch.full((s.shape[0], n_blocks), -math.inf, device=s.device, dtype=s.dtype)
        o = o.scatter_reduce(1, bidx[None].expand(s.shape[0], -1), s, reduce="amax", include_self=True)
        out[:, :, a:b] = o.reshape(n_kv, g, b - a, n_blocks)
        del s, o
    return out.reshape(Hq * Tq, n_blocks)


def rr_max(scores, cand=None, c=RR_C):
    """Reciprocal-rank normalisation per query row over the candidate blocks, then Max over the rows (MemArt's
    RR-Max). scores [R, nb]; cand [nb] bool (None = all). Rank 1 = the highest score of the row; ties broken by
    the earlier block. -> [nb] (0 for non-candidates)."""
    R, nb = scores.shape
    s = scores.clone()
    if cand is not None:
        s[:, ~cand.to(s.device)] = -math.inf
    order = torch.argsort(-s, dim=1, stable=True)                    # [R, nb] block ids by score
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(1, nb + 1, device=s.device)[None].expand(R, -1))
    rr = 1.0 / (rank.to(torch.float64) + c)
    if cand is not None:
        rr[:, ~cand.to(s.device)] = 0.0
    return rr.max(0).values


def select_blocks(agg, k, cand=None):
    """The k blocks with the highest aggregate (ties -> the earlier block), returned in chronological order."""
    nb = agg.shape[0]
    ids = torch.arange(nb, device=agg.device)
    if cand is not None:
        ids = ids[cand.to(agg.device)]
    k = min(int(k), int(ids.numel()))
    a = agg[ids].double()
    order = torch.argsort(-a, stable=True)[:k]
    return sorted(int(x) for x in ids[order].tolist())


def n_blocks_for(tokens, block=BLOCK):
    return int(math.ceil(tokens / block)) if tokens > 0 else 0


# ----------------------------------------------------------------------------------------------- queries


def query_states(attn, hidden, head_dim):
    """The attention module's pre-RoPE queries of hidden [B, T, D] (its input, after the layer norm): q_proj, the
    head split, q_norm where the module has one (Gemma 4, Qwen3) -> [B, Hq, T, head_dim]."""
    B, T, _ = hidden.shape
    q = attn.q_proj(hidden).view(B, T, -1, head_dim)
    if getattr(attn, "q_norm", None) is not None:
        q = attn.q_norm(q)
    return q.transpose(1, 2)


# ----------------------------------------------------------------------------------------------- the hook


class MemArtHook:
    """Per-layer MemArt retrieval inside a batched prefill (paradigms.Ragged). For every P5 row b of the batch:
      rows[b] = dict(store=<paradigms.Store>, n_blocks=k, slot0=<first block slot in the batched cache>,
                     q_cols=(c0, c1) the request's columns in this forward, nh=<head length>,
                     cand=<[nb] bool candidate blocks or None>)
    On the first forward (the prefill) the pre-hook of layer l scores the store's blocks with the row's queries,
    writes the selected blocks' K (re-rotated to their new positions) and V into the row's slots of the cache, and
    records the pad slots of short blocks; on every forward (prefill and decoding) it masks those pad slots in that
    layer. The selections are kept in self.picked[b][l]."""

    def __init__(self, C, rows, kv_of_cache, rope_rotate, slot_width):
        self.C, self.rows = C, rows
        self.kv_of_cache = kv_of_cache            # cache -> [(K, V)] per layer (references)
        self.rope_rotate = rope_rotate            # (C, K, delta [S], layer) -> K rotated per entry
        self.W = slot_width                       # S_max: the stored-slot width of the batch (the mask's first W cols)
        self.handles = []
        self.done = set()
        self.pad = {}                             # layer -> [(b, slot list)]
        self.picked = {b: {} for b in rows}
        self.seconds = 0.0

    def _select(self, l, attn, hs, cache):
        import time
        t0 = time.time()
        L = self.C.layout[l]
        K, V = self.kv_of_cache(cache)[l]
        pads = []
        raw = {}
        for b, r in self.rows.items():
            st = r["store"]
            c0, c1 = r["q_cols"]
            q = query_states(attn, hs[b:b + 1, c0:c1], L.head_dim)[0]           # [Hq, Tq, d]
            if id(st) not in raw:
                raw[id(st)] = st.raw_keys(self.C, l, q.device)                 # [Hkv, N, d]
            kr = raw[id(st)]
            sc = block_scores(q, kr, st.t2b, len(st.blocks), L.n_kv)
            agg = rr_max(sc, r.get("cand"))
            pick = select_blocks(agg, r["n_blocks"], r.get("cand"))
            self.picked[b][l] = pick
            nh, s0 = r["nh"], r["slot0"]
            src, dst, newp = [], [], []
            for j, blk in enumerate(pick):
                g0, n = st.blocks[blk]
                a = s0 + BLOCK * j
                src.append(torch.arange(g0, g0 + n))
                dst.append(torch.arange(a, a + n))
                newp.append(torch.arange(nh + BLOCK * j, nh + BLOCK * j + n))
                if n < BLOCK:
                    pads.append((b, list(range(a + n, a + BLOCK))))
            if src:                       # one gather / rotation / write per row and layer
                src, dst, newp = torch.cat(src), torch.cat(dst), torch.cat(newp)
                Kb = st.K[l][:, src].to(K.device)[None]                       # [1, Hkv, n, d]
                Vb = st.V[l][:, src].to(V.device)
                Kb = self.rope_rotate(self.C, Kb, (newp - st.pos[src]).to(K.device), l)
                K[b, :, dst.to(K.device)] = Kb[0].to(K.dtype)
                V[b, :, dst.to(V.device)] = Vb.to(V.dtype)
            for j in range(len(pick), r["n_blocks"]):          # fewer candidate blocks than the budget
                pads.append((b, list(range(s0 + BLOCK * j, s0 + BLOCK * (j + 1)))))
        self.pad[l] = pads
        self.seconds += time.time() - t0

    def _mask(self, l, m):
        pads = self.pad.get(l)
        if not pads or m is None:
            return m
        m = m.clone()
        if m.dtype == torch.bool:
            for b, slots in pads:
                m[b, :, :, slots] = False
        else:
            mn = torch.finfo(m.dtype).min
            for b, slots in pads:
                m[b, :, :, slots] = mn
        return m

    def _hook(self, l):
        def pre(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
            if l not in self.done:
                with torch.no_grad():
                    self._select(l, module, hs, cache)
                self.done.add(l)
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = self._mask(l, kwargs["attention_mask"])
            return args, kwargs
        return pre

    def __enter__(self):
        for l, blk in enumerate(self.C.blocks):
            self.handles.append(blk.self_attn.register_forward_pre_hook(self._hook(l), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


if __name__ == "__main__":
    sys.exit("memart.py is a library (see paradigms.py / stage_b.py)")
