#!/usr/bin/env python
"""dream_compare Stage 0: per-head masking of a stored prefix (the selector-baseline design's "per-head splice").

The whole stored turn stays in the cache; what each (layer, KV head) may read of it is set by an additive mask that a
forward pre-hook (register_forward_pre_hook(with_kwargs=True)) writes into every attention module's
kwargs["attention_mask"]: the stored slots a KV head drops get finfo.min in the rows of all query heads of its GQA
group (repeat_interleave by the group size), everything after the stored slots (the question and generated tokens) is
untouched. Works for transformers 4.46 (Qwen2 / Llama eager and SDPA classes: mask [B, 1, Tq, >= Tk], sliced to Tk
there) and 5.x (eager_attention_forward / sdpa_attention_forward; float or bool masks; Gemma 4's per-layer-type masks).
A None mask (SDPA's implicit causal mask) is materialised from the cache length first.

  keep = {layer: bool [n_kv_l, n_stored]}      # True = this KV head may read this stored slot
  with HeadMask(C, keep, n_stored):
      ... any forward / generation after a prefix of n_stored slots ...

Tests (test_dream_compare.py): keep-all == the full cache; the same positions in every head == the physical splice;
a masked entry gets exactly 0 attention in its own group only (eager attentions).
"""

import torch


def _n_heads(attn):
    """(query heads, KV heads, group size) of an attention module (transformers 4.46 and 5.x attribute names)."""
    g = getattr(attn, "num_key_value_groups", None)
    hq = getattr(attn, "num_heads", None)
    if hq is None:
        cfg = getattr(attn, "config", None)
        hq = getattr(cfg, "num_attention_heads", None)
    assert g and hq, f"no head counts on {type(attn).__name__}"
    return hq, hq // g, g


def _past_len(kwargs, layer_idx):
    pkv = kwargs.get("past_key_values", kwargs.get("past_key_value"))
    if pkv is None:
        return 0
    try:
        return int(pkv.get_seq_length(layer_idx))
    except TypeError:
        return int(pkv.get_seq_length())


class HeadMask:
    """Context manager: per (layer, KV head) masks over the first n_stored cache slots (see the module docstring).
    keep: {layer index: bool tensor [n_kv, n_stored]}; layers not in keep are unchanged."""

    def __init__(self, C, keep, n_stored):
        self.C, self.n = C, int(n_stored)
        self.keep = {int(l): torch.as_tensor(k, dtype=torch.bool) for l, k in keep.items()}
        for l, k in self.keep.items():
            assert k.dim() == 2 and k.shape[1] == self.n, (l, tuple(k.shape), self.n)
        self.handles = []
        self.calls = 0
        self._dev = {}

    def _keep_q(self, l, attn, device):
        """[Hq, n_stored] bool: each KV head's keep row repeated for the query heads of its group."""
        key = (l, str(device))
        if key not in self._dev:
            hq, hkv, g = _n_heads(attn)
            k = self.keep[l]
            assert k.shape[0] == hkv, f"layer {l}: keep has {k.shape[0]} KV heads, the module {hkv}"
            self._dev[key] = k.repeat_interleave(g, dim=0).to(device)
        return self._dev[key]

    def _hook(self, l):
        def pre(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            B, Tq = hs.shape[0], hs.shape[1]
            m = kwargs.get("attention_mask")
            kq = self._keep_q(l, module, hs.device)               # [Hq, n]
            if m is None:                                         # SDPA's implicit causal mask: build it
                Tk = _past_len(kwargs, getattr(module, "layer_idx", l)) + Tq
                q = torch.arange(Tk - Tq, Tk, device=hs.device)[:, None]
                kpos = torch.arange(Tk, device=hs.device)[None, :]
                m = torch.where(kpos <= q, 0.0, torch.finfo(hs.dtype).min).to(hs.dtype)[None, None]
            assert m.dim() == 4, f"layer {l}: a {m.dim()}D attention mask"
            W = m.shape[-1]
            assert W >= self.n, f"layer {l}: mask width {W} < {self.n} stored slots"
            m = m.expand(B, kq.shape[0], m.shape[2], W)
            if m.dtype == torch.bool:                             # True = attend
                out = m.clone()
                out[..., :self.n] &= kq[None, :, None, :]
            else:
                mn = torch.finfo(m.dtype).min
                out = m.clone()
                out[..., :self.n] = (out[..., :self.n] + torch.where(kq, 0.0, mn).to(m.dtype)[None, :, None, :]
                                     ).clamp(min=mn)
            kwargs["attention_mask"] = out
            self.calls += 1
            return args, kwargs
        return pre

    def __enter__(self):
        for l in self.keep:
            attn = self.C.blocks[l].self_attn
            self.handles.append(attn.register_forward_pre_hook(self._hook(l), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False


def keep_all(C, n_stored, layers=None):
    """keep for every attention layer: everything kept (the identity check)."""
    out = {}
    for l, b in enumerate(C.blocks):
        if layers is not None and l not in layers:
            continue
        if hasattr(b, "self_attn"):
            _, hkv, _ = _n_heads(b.self_attn)
            out[l] = torch.ones(hkv, n_stored, dtype=torch.bool)
    return out


def keep_shared(C, n_stored, slots):
    """keep for every attention layer: the same stored slots in every KV head (== the physical splice of `slots`)."""
    out = keep_all(C, n_stored)
    for l in out:
        out[l][:] = False
        out[l][:, torch.as_tensor(slots, dtype=torch.long)] = True
    return out


def keep_per_head(C, n_stored, picks, always=()):
    """keep from per-(layer, KV head) picks {layer: [[slots of KV head 0], [slots of head 1], ...]} plus `always` (e.g.
    the template head) in every head; layers missing from picks keep only `always`."""
    out = keep_all(C, n_stored)
    al = torch.as_tensor(list(always), dtype=torch.long)
    for l in out:
        out[l][:] = False
        if len(al):
            out[l][:, al] = True
        for h, slots in enumerate(picks.get(l, [])):
            out[l][h, torch.as_tensor(list(slots), dtype=torch.long)] = True
    return out


def stored_entries(keep):
    """Total kept (layer, KV head, slot) entries of a keep dict (Stage A's per-head budget accounting)."""
    return int(sum(int(k.sum()) for k in keep.values()))
