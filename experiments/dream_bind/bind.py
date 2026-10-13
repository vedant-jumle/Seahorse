"""dream_bind pure helpers (torch + numpy, no model loading). Unit-tested in tests/test_dream_bind.py.

Stage 0 asks where an in-context fact lives when the question is answered: at the question's subject
("dog" in "What's my dog's name?") or only at the attribute's own position ("Pepper").

  layout        source (true value) and target (counter value) prompts must have the same length and
                differ at exactly one token (the attribute)
  spans         the question-subject span, found by character offsets in the QUESTION only (never the
                mention inside the experience sentence, which precedes the attribute)
  patching      the residual output of block l at chosen positions, copied from the source run into the
                target run; one row of a batch per (site, block or band)
  blocking      attention from chosen query positions to chosen key positions removed at chosen layers
                (eager attention: the 4D additive mask of the attention module is edited per row)
  shares        patching: (patched - target) / (source - target) of logit(target) - logit(counter);
                knockout: (lp_ctx - lp_knocked) / (lp_ctx - lp_nomem) of log P(target)
  verdict       the pre-registered stage-0 read-out (PREREG.md)
"""

import math
import re
from contextlib import contextmanager

import torch

from seahorse.stats import boot_ci

# ------------------------------------------------------------------ thresholds (PREREG.md)
HI, LO = 0.50, 0.25            # "fact lives at the subject / at its own position" thresholds
SANITY_ZERO = 0.10             # |share| <= this counts as "about 0" (subject-alone sanity cell)
MIN_FLIP = 1.0                 # an item's full flip (logits) / ctx gain (nats) below this is flagged
MAX_FLAGGED = 2                # more flagged primary items than this -> the model's verdict is undetermined
BAND = 5                       # band / knockout window width (blocks)


# ------------------------------------------------------------------------- layout + spans


def attribute_position(src, tgt):
    """src, tgt: 1-D id sequences. Asserts equal length and exactly one differing position; returns it."""
    src, tgt = list(map(int, src)), list(map(int, tgt))
    assert len(src) == len(tgt), f"source / target lengths differ ({len(src)} vs {len(tgt)})"
    diff = [i for i, (a, b) in enumerate(zip(src, tgt)) if a != b]
    assert len(diff) == 1, f"source / target differ at {len(diff)} positions {diff} (want exactly 1)"
    return diff[0]


def char_spans(tok, ids):
    """[(start, end)] character span of every token in the decoded sequence (incremental decode; works for
    any tokenizer whose decode is prefix-stable, e.g. byte-level BPE on ASCII text). Returns (spans, text)."""
    ids = [int(i) for i in ids]
    dec = lambda x: tok.decode(x, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    ends = [len(dec(ids[:i + 1])) for i in range(len(ids))]
    text = dec(ids)
    assert ends[-1] == len(text), "decode is not prefix-stable"
    spans, prev = [], 0
    for e in ends:
        assert e >= prev, "decode is not monotone"
        spans.append((prev, e))
        prev = e
    return spans, text


def find_alias(text, aliases, start=0):
    """The first alias (in list order: the canonical one first) that occurs as whole words in text[start:],
    case-insensitive. Returns (alias, char_start, char_end) or None."""
    for a in aliases:
        m = re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", text[start:], re.I)
        if m:
            return a, start + m.start(), start + m.end()
    return None


def tokens_overlapping(spans, s, e):
    """Indices of the tokens whose character span overlaps [s, e)."""
    return [i for i, (a, b) in enumerate(spans) if a < e and b > s]


def question_layout(tok, ids, user_text, experience, aliases):
    """Locate, in a chat sequence whose user turn is `experience + " " + question`:
    q0         the first token of the question (every position >= q0 is "after the experience sentence")
    subj       the question-subject span (token indices) or [] when no alias occurs in the question
    alias      the alias found (or None)."""
    spans, text = char_spans(tok, ids)
    u = text.find(user_text)
    assert u >= 0 and text.find(user_text, u + 1) < 0, "user text not found exactly once in the sequence"
    assert user_text.startswith(experience + " ")
    qs = u + len(experience) + 1                     # first character of the question
    qe = u + len(user_text)
    q0 = next(i for i, (a, b) in enumerate(spans) if b > qs)
    hit = find_alias(text[:qe], aliases, qs)
    if hit is None:
        return {"q0": q0, "subj": [], "alias": None}
    alias, s, e = hit
    subj = tokens_overlapping(spans, s, e)
    assert subj and min(subj) >= q0, "subject span must lie inside the question"
    return {"q0": q0, "subj": subj, "alias": alias}


# ------------------------------------------------------------------------------ cells


def windows(n_blocks, width=BAND):
    """Sliding windows of `width` consecutive blocks: [(start, [start .. start + width - 1])]."""
    return [(s, list(range(s, s + width))) for s in range(n_blocks - width + 1)]


def cells(n_blocks, width=BAND):
    """Patching cells: every single block, then every band of `width` consecutive blocks."""
    return [("block", l, [l]) for l in range(n_blocks)] + [("band", s, ls) for s, ls in windows(n_blocks, width)]


# ------------------------------------------------------------------------------ hooks


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@contextmanager
def patching(blocks, plan, src):
    """Residual patching at block outputs, per batch row.
    blocks: the decoder-layer ModuleList; plan: {layer: [(row, [positions])]}; src: {layer: [T, d]} source
    states (batch-1 capture). Row r's state at those positions after block `layer` is replaced by src."""
    handles = []
    for l, todo in plan.items():
        if not todo:
            continue

        def hook(module, args, out, l=l, todo=todo):
            h = _hidden(out).clone()
            s = src[l].to(device=h.device, dtype=h.dtype)
            for row, pos in todo:
                p = torch.as_tensor(pos, device=h.device, dtype=torch.long)
                h[row, p] = s[p]
            return _replace(out, h)
        handles.append(blocks[l].register_forward_hook(hook))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def block_mask(mask, todo, batch):
    """Edit a 4D attention mask (additive float, or boolean keep-mask) so that, for every (row, queries,
    keys) in `todo`, those queries cannot attend to those keys. Returns a new [batch, 1, Q, K] tensor."""
    assert mask is not None and mask.dim() == 4, "blocking needs a materialised 4D mask (eager attention)"
    m = mask.expand(batch, *mask.shape[1:]).clone()
    for row, qs, ks in todo:
        qi = torch.as_tensor(qs, device=m.device, dtype=torch.long)
        ki = torch.as_tensor(ks, device=m.device, dtype=torch.long)
        if not len(qi) or not len(ki):
            continue
        if m.dtype == torch.bool:
            m[row, :, qi[:, None], ki[None, :]] = False
        else:
            m[row, :, qi[:, None], ki[None, :]] = torch.finfo(m.dtype).min
    return m


@contextmanager
def blocking(blocks, plan):
    """Attention knockout per batch row: plan {layer: [(row, [query positions], [key positions])]}.
    A forward pre-hook on blocks[layer].self_attn rewrites its `attention_mask` kwarg (needs eager
    attention, where the mask is always a materialised 4D tensor). A layer with an empty list still gets
    the hook (the 'empty knockout' identity check runs through the same code path)."""
    handles = []
    for l, todo in plan.items():
        def pre(module, args, kwargs, todo=todo):
            assert "attention_mask" in kwargs, "self_attn called without an attention_mask kwarg"
            hs = kwargs.get("hidden_states", args[0] if args else None)
            kwargs = dict(kwargs)
            kwargs["attention_mask"] = block_mask(kwargs["attention_mask"], todo, hs.shape[0])
            return args, kwargs
        handles.append(blocks[l].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


# ------------------------------------------------------------------------------ shares


def share(x, lo_ref, hi_ref):
    """(x - lo_ref) / (hi_ref - lo_ref); nan when the denominator is 0."""
    d = hi_ref - lo_ref
    return (x - lo_ref) / d if d != 0 else float("nan")


def ci(per_item, n_boot=2000, seed=7):
    """{item: value} -> (mean, lo, hi, n): the mean over items and its 95% percentile bootstrap CI (items
    resampled; one value per item). nan / None values are dropped."""
    p, lo, hi, n = boot_ci({k: {0: v} for k, v in per_item.items()}, n_boot=n_boot, seed=seed)
    return {"mean": p, "lo": lo, "hi": hi, "n": n}


def curve(per_cell, n_boot=2000, seed=7):
    """{cell label: {item: share}} -> {cell label: ci(...)}, in the input order."""
    return {c: ci(v, n_boot, seed) for c, v in per_cell.items()}


def best(cv):
    """The cell with the highest mean (nan means are skipped): (label, stats) or (None, None)."""
    ok = [(c, s) for c, s in cv.items() if not math.isnan(s["mean"])]
    return max(ok, key=lambda x: x[1]["mean"]) if ok else (None, None)


# ------------------------------------------------------------------------------ verdict


def verdict(subj, attr, ko_subj=None, ko_attr=None, suff=None, suff_net=None, suff_sanity=None, n_flagged=0):
    """The pre-registered stage-0 read-out for one model (PREREG.md). Every argument except n_flagged is a
    curve (cell label -> {mean, lo, hi, n}) over the primary items:
      subj       patching the question-subject's LAST token (blocks and bands)
      attr       patching the attribute position
      ko_subj    knockout answer -> subject span (windows of 5 layers); None on Qwen3.5-2B
      ko_attr    knockout answer -> attribute position
      suff       subject-alone cell (attribute blocked, subject patched), raw share; suff_net the same
                 minus the blocked-no-patch baseline; suff_sanity {mean, ...} of that baseline
    Returns {label, reasons, ...}. Labels: subject | pointer-suspected | in between | undetermined."""
    out = {"reasons": []}
    if n_flagged > MAX_FLAGGED:
        out.update(label="undetermined", reasons=[f"{n_flagged} primary items flagged (> {MAX_FLAGGED})"])
        return out
    sb, sbs = best(subj)
    ab, abs_ = best(attr)
    if sbs is None or abs_ is None:
        out.update(label="undetermined", reasons=["no usable primary item in the patching curves"])
        return out
    out["subj_best"], out["attr_best"] = {"cell": sb, **(sbs or {})}, {"cell": ab, **(abs_ or {})}
    subj_ok = sbs is not None and sbs["mean"] >= HI and sbs["lo"] > LO
    subj_low = bool(subj) and all(s["mean"] < LO and s["hi"] < HI for s in subj.values())
    attr_ok = abs_ is not None and abs_["mean"] >= HI
    has_ko = ko_subj is not None and ko_attr is not None
    if has_ko:
        kb, kbs = best(ko_subj)
        ka, kas = best(ko_attr)
        out["ko_subj_best"], out["ko_attr_best"] = {"cell": kb, **(kbs or {})}, {"cell": ka, **(kas or {})}
        ko_subject = kbs is not None and kbs["mean"] >= HI and kas is not None and kas["mean"] < LO
        ko_pointer = kas is not None and kas["mean"] >= HI
    else:
        ko_subject = ko_pointer = True
    is_subject = subj_ok and ko_subject
    is_pointer = subj_low and attr_ok and ko_pointer
    if suff is not None:
        use_net = suff_sanity is None or math.isnan(suff_sanity["mean"]) or abs(suff_sanity["mean"]) > SANITY_ZERO
        cv = suff_net if use_net else suff
        fb, fbs = best(cv)
        out["suff_readout"] = "net (patched - blocked baseline)" if use_net else "raw share"
        out["suff_best"] = {"cell": fb, **(fbs or {})}
        if is_pointer and fbs is not None and fbs["mean"] >= HI and fbs["lo"] > LO:
            is_pointer = False
            out["reasons"].append("subject alone is sufficient (>= 50%, CI excludes 25%): sufficient but outvoted")
    out["reasons"] += [f"subject patch best {_f(sbs)}", f"attribute patch best {_f(abs_)}"]
    if has_ko:
        out["reasons"].append(f"knockout subject best {_f(kbs)}, attribute best {_f(kas)}")
    if is_subject:
        out["label"] = "subject"
    elif is_pointer:
        out["label"] = "pointer-suspected"
    else:
        out["label"] = "in between"
    out["checks"] = {"subj_ok": subj_ok, "subj_low_everywhere": subj_low, "attr_ok": attr_ok,
                     "ko_subject": ko_subject if has_ko else None, "ko_pointer": ko_pointer if has_ko else None}
    return out


def _f(s):
    if not s:
        return "n/a"
    return f"{s['mean']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]"


# =============================================================== harvest + analytic (spec sections 4-6)


def find_all_aliases(text, aliases, start=0, end=None):
    """Every non-overlapping alias occurrence in text[start:end] (G-match fires at every occurrence).
    Aliases are tried in list order; a later alias inside an earlier match ("dog" inside "my dog") is
    skipped. Returns sorted [(char_start, char_end)]."""
    end = len(text) if end is None else end
    taken = []
    for a in aliases:
        for m in re.finditer(rf"(?<![\w-]){re.escape(a)}(?![\w-])", text[start:end], re.I):
            s, e = start + m.start(), start + m.end()
            if all(e <= s2 or s >= e2 for s2, e2 in taken):
                taken.append((s, e))
    return sorted(taken)


def subject_sites(tok, ids, user_text, aliases, prefix=None, user_from=0):
    """Injection sites in a chat sequence whose user turn is `user_text` (optionally followed by an
    assistant-side teacher-forced `prefix` at the very end):
      spans         token spans of every alias occurrence in user_text[user_from:]
      main          the last token of each span (G-match; primary site)
      plus1         main + the token after each (variant +1)
      prefix_spans  alias spans inside the prefix (the model's own mention, "Your dog")
      gen           main + the last token of each prefix span (variant gen)
      wrongpos      one non-subject user-text token per main site: counted back from the end of the user
                    text, skipping subject-span tokens and +1 tokens (ctrl:wrongpos)
      user_tokens   every token overlapping the user text."""
    spans, text = char_spans(tok, ids)
    u0 = text.find(user_text)
    assert u0 >= 0 and text.find(user_text, u0 + 1) < 0, "user text not found exactly once in the sequence"
    u1 = u0 + len(user_text)
    user_tokens = tokens_overlapping(spans, u0, u1)
    occ = find_all_aliases(text, aliases, u0 + user_from, u1)
    sp = [tokens_overlapping(spans, s, e) for s, e in occ]
    main = [s[-1] for s in sp]
    plus1 = sorted(set(main) | {p + 1 for p in main if p + 1 < len(ids)})
    pspans = []
    if prefix is not None:
        assert text.endswith(prefix), "the sequence does not end with the prefix"
        pspans = [tokens_overlapping(spans, s, e) for s, e in find_all_aliases(text, aliases, len(text) - len(prefix))]
    gen = sorted(set(main) | {s[-1] for s in pspans})
    excl = {i for s in sp for i in s} | set(plus1)
    cand = [i for i in reversed(user_tokens) if i not in excl]
    return {"spans": sp, "main": main, "plus1": plus1, "prefix_spans": pspans, "gen": gen,
            "wrongpos": sorted(cand[:len(main)]), "user_tokens": user_tokens}


@contextmanager
def adding(blocks, plan):
    """Additive injection at block outputs, per batch row: plan {layer: [(row, [positions], vec [d])]};
    h[row, positions] += vec after block `layer`."""
    handles = []
    for l, todo in plan.items():
        if not todo:
            continue

        def hook(module, args, out, todo=todo):
            h = _hidden(out).clone()
            for row, pos, v in todo:
                p = torch.as_tensor(pos, device=h.device, dtype=torch.long)
                h[row, p] = h[row, p] + v.to(device=h.device, dtype=h.dtype)
            return _replace(out, h)
        handles.append(blocks[l].register_forward_hook(hook))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def norm_match(v, ref):
    """Rows of v [n, d] rescaled to the norms of ref [n, d] (H3: the raw state at H2's delta norm)."""
    return v * (ref.norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def map_sets(sets, ref_blocks, n_blocks, map_layers):
    """Layer sets of the 24-block reference model -> this model by core_v1's relative-depth rule. A set
    {a..b} maps to the contiguous range {map(a)..map(b)}; duplicates are dropped (order kept)."""
    out, seen = [], set()
    for s in sets:
        m = map_layers(list(s), ref_blocks, n_blocks)
        m = list(range(min(m), max(m) + 1)) if len(s) > 1 else m
        m = [min(max(x, 0), n_blocks - 1) for x in m]
        if tuple(m) not in seen:
            seen.add(tuple(m))
            out.append(m)
    return out


def lset_label(ls):
    return f"b{ls[0]}" if len(ls) == 1 else f"b{ls[0]}-{ls[-1]}"


# ------------------------------------------------------------------------------ G-key


def whitening(X, r=256, eps=0.01):
    """diag_keys' pooled_w256 whitening: X [N, d] centred -> W [d, r] = U_r / sqrt(ev_r + eps * mean(ev))."""
    X = X.double()
    S = X.T @ X / X.shape[0]
    ev, U = torch.linalg.eigh(S)
    ev, U = ev.flip(0).clamp_min(0), U.flip(1)
    r = min(r, X.shape[1])
    return (U[:, :r] / (ev[:r] + eps * ev.mean()).sqrt()).float()


def pooled_keys(H, user, mu, W):
    """diag_keys.KeySpace(pooled) keys: the running mean over user-text positions of (h - mu) W,
    normalised (zero before the first user token). H [T, d], user [T] bool -> [T, r]."""
    f = (H.float() - mu) @ W
    u = user.to(f.dtype)[:, None]
    f = (f * u).cumsum(0) / u.cumsum(0).clamp_min(1.0)
    return torch.nn.functional.normalize(f, dim=-1, eps=1e-6)


def subject_key(Hs, mu, W):
    """The stored key: the whitened mean of the subject-span states [n, d] -> [r] (unit)."""
    return torch.nn.functional.normalize(((Hs.float().mean(0) - mu) @ W)[None], dim=-1, eps=1e-6)[0]


def gate_threshold(gen_keys, stored, q=0.95):
    """q-quantile over generic positions of the best cosine to the stored keys [m, r]."""
    return torch.quantile((gen_keys @ stored.T).max(-1).values.float(), q).item()


# --------------------------------------------------------------------------- pick rule


def ranks(vals, higher=True):
    """1 = best; ties (and nan, the worst) share their mean rank (xlayer_v1 / think_v1 convention)."""
    k = [(1, 0.0) if (v is None or (isinstance(v, float) and math.isnan(v))) else (0, -v if higher else v)
         for v in vals]
    order = sorted(range(len(vals)), key=lambda i: k[i])
    out, i = [0.0] * len(vals), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and k[order[j + 1]] == k[order[i]]:
            j += 1
        for t in range(i, j + 1):
            out[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return out


def _scored(cells):
    rs, rb = ranks([c["dspec"] for c in cells]), ranks([c["dbal"] for c in cells])
    sc = [dict(c, score=a + b) for c, a, b in zip(cells, rs, rb)]
    return sorted(sc, key=lambda c: (c["score"], -(c["dlp"] if not math.isnan(c["dlp"]) else -1e9)))


PICK_RULE = ("PREREG analytic-pick rule: per variant, eligible cells have dspec > 0 (if fewer than 3 qualify: every "
             "cell of the variant); score = rank(dspec) + rank(dbal) (higher better; ties -> higher dlp); the best "
             "3 layer sets (each by its best-scoring alpha) are kept, each at its 2 best-scoring alphas. The pool is "
             "re-scored jointly with the same formula and taken best-first up to 8 cells, always including the "
             "pool's best H1 cell.")


def analytic_pick(cells, n_sets=3, n_alpha=2, n_max=8, must="H1"):
    """cells: [{variant, lset, alpha, dspec, dbal, dlp}] (macro means over the primary items) -> {pool, picks}."""
    pool = []
    for v in sorted({c["variant"] for c in cells}):
        cand = [c for c in cells if c["variant"] == v]
        elig = [c for c in cand if not math.isnan(c["dspec"]) and c["dspec"] > 0]
        if len(elig) < 3:
            elig = cand
        sc = _scored(elig)
        sets = []
        for c in sc:
            if c["lset"] not in sets:
                sets.append(c["lset"])
        for ls in sets[:n_sets]:
            pool += [c for c in sc if c["lset"] == ls][:n_alpha]
    pool = _scored([{k: v for k, v in c.items() if k != "score"} for c in pool])
    picks = pool[:n_max]
    if must and not any(c["variant"] == must for c in picks):
        best_must = next((c for c in pool if c["variant"] == must), None)
        if best_must is not None:
            picks = picks[:n_max - 1] + [best_must]
    return {"pool": pool, "picks": picks, "rule": PICK_RULE}


# =============================================================== arm R: gated reinstatement (KV splice)


def kv_list(pkv):
    """A model's past_key_values (transformers 4.x legacy tuple / DynamicCache.key_cache, or 5.x .layers) ->
    [(K, V)] per layer, each [B, kv_heads, S, head_dim] (K already rotated at its harvest positions)."""
    if hasattr(pkv, "layers"):
        return [(L.keys, L.values) for L in pkv.layers]
    if hasattr(pkv, "key_cache"):
        return list(zip(pkv.key_cache, pkv.value_cache))
    return [(k, v) for k, v in pkv]


def make_cache(kv, B=1):
    """[(K, V)] per layer ([1, h, S, d]) -> a DynamicCache holding them for B rows (empty if kv is empty)."""
    from transformers import DynamicCache
    cache = DynamicCache()
    for l, (k, v) in enumerate(kv):
        cache.update(k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous(), l)
    return cache


def splice_keep(variant, n_head, n_seg, attr, n_unrel=0):
    """Which harvested cache entries a reinstatement keeps, and the RoPE position the question turn starts at.
    The harvest sequence is [head][unrelated exchange (R-orphan only)][experience user turn]; `attr` is the
    attribute's index inside the experience turn.
      R-full    head + the whole experience turn (= the sentence as a prior user turn); question at n_head + n_seg
      R-min     head + the attribute and the next token only, at their stored positions; question as R-full
      R-orphan  head + the whole experience turn harvested AFTER the unrelated exchange (which is dropped);
                the stored positions are the harvest ones; question continues after them
    Returns (keep indices into the harvest sequence, question start position)."""
    head = list(range(n_head))
    if variant == "R-full":
        return head + list(range(n_head, n_head + n_seg)), n_head + n_seg
    if variant == "R-min":
        assert 0 <= attr and attr + 1 < n_seg
        return head + [n_head + attr, n_head + attr + 1], n_head + n_seg
    if variant == "R-orphan":
        s0 = n_head + n_unrel
        return head + list(range(s0, s0 + n_seg)), s0 + n_seg
    if variant == "R-orphan-min":   # R-min's two entries, harvested after the unrelated exchange (dropped)
        s0 = n_head + n_unrel
        assert 0 <= attr and attr + 1 < n_seg
        return head + [s0 + attr, s0 + attr + 1], s0 + n_seg
    if variant == "R-state-only":   # no segment entries; the question keeps the positions it has after the segment
        return head, n_head + n_seg
    raise ValueError(variant)


def is_linear_layer(L):
    """A cache layer of a linear-attention (Gated DeltaNet) block: conv + recurrent states, no K,V."""
    return hasattr(L, "conv_states") and hasattr(L, "recurrent_states")


def expand_cache(cache, B):
    """A deep copy of a batch-1 cache object (transformers 5 DynamicCache, hybrid or not) expanded to B rows.
    The stored cache is never modified (decoding updates conv states in place)."""
    import copy
    c = copy.deepcopy(cache)
    for L in c.layers:
        if is_linear_layer(L):
            for d in (L.conv_states, L.recurrent_states):
                for i, t in d.items():
                    if t is not None:
                        d[i] = t.expand(B, *t.shape[1:]).contiguous()
        elif getattr(L, "is_initialized", False):
            L.keys = L.keys.expand(B, -1, -1, -1).contiguous()
            L.values = L.values.expand(B, -1, -1, -1).contiguous()
    return c


def mix_cache(attn_src, keep, lin_src):
    """A new hybrid cache: the attention blocks hold the entries `keep` of `attn_src` (at their stored RoPE
    positions); every linear-attention (DeltaNet) block holds `lin_src`'s state. Neither source is modified."""
    import copy
    c = copy.deepcopy(attn_src)
    for i, L in enumerate(c.layers):
        if is_linear_layer(L):
            c.layers[i] = copy.deepcopy(lin_src.layers[i])
        elif getattr(L, "is_initialized", False):
            ki = torch.as_tensor(keep, device=L.keys.device, dtype=torch.long)
            L.keys = L.keys[:, :, ki].contiguous()
            L.values = L.values[:, :, ki].contiguous()
    return c


def kv6_cache(src, head, keep):
    """Hybrid reinstatement 'kv6': the attention blocks keep only the entries `keep` of the stored cache `src`
    (at their stored RoPE positions); every linear-attention (DeltaNet) block gets the state of `head` (the run
    WITHOUT the experience turn), so the segment is invisible to them. Returns a new cache (src is unchanged)."""
    return mix_cache(src, keep, head)


def split_linear(c, head, rec_from, conv_from):
    """Addendum 6b: in cache `c` (whose DeltaNet layers hold the after-experience state), reset the recurrent state
    and / or the conv buffer of every DeltaNet layer to the head-only ("fresh") value: `head`'s state when it has
    one, else zeros (= no previous tokens). rec_from / conv_from: "full" (keep) or "head" (reset). In place."""
    for i, L in enumerate(c.layers):
        if not is_linear_layer(L):
            continue
        H = head.layers[i]
        for which, d, hd in (("rec", L.recurrent_states, H.recurrent_states), ("conv", L.conv_states, H.conv_states)):
            if (rec_from if which == "rec" else conv_from) != "head":
                continue
            for j, t in d.items():
                if t is not None:
                    d[j] = hd[j].clone() if hd.get(j) is not None else torch.zeros_like(t)
    return c


def linear_states_equal(a, b):
    """True if every linear-attention layer of caches a and b holds identical conv and recurrent states."""
    for La, Lb in zip(a.layers, b.layers):
        if is_linear_layer(La) != is_linear_layer(Lb):
            return False
        if is_linear_layer(La):
            for da, db in ((La.conv_states, Lb.conv_states), (La.recurrent_states, Lb.recurrent_states)):
                for i in da:
                    x, y = da[i], db.get(i)
                    if (x is None) != (y is None) or (x is not None and not torch.equal(x, y)):
                        return False
    return True


def cache_summary(cache):
    """[('lin', has_previous_state) | ('attn', n_entries)] per layer (checks and the position record)."""
    out = []
    for L in cache.layers:
        if is_linear_layer(L):
            out.append(("lin", bool(L.has_previous_state.get(0, False))))
        else:
            out.append(("attn", int(L.keys.shape[-2]) if getattr(L, "is_initialized", False) else 0))
    return out


def pair_plan(nh, nA, nB, aA, aB):
    """Two-fact collision (addendum 5): which cache entries each R2 condition takes from which harvest, and where
    the question starts. Harvests: 'A' = [head][A's turn], 'B' = [head][B's turn] (each its own session),
    'J' = [head][A's turn][B's turn] (one session). Parts are (source, indices into that source, RoPE shift).
      overlap  A and B at their stored positions (they share positions h .. h + max(nA, nB) - 1)
      shift    B's keys re-rotated by +nA so that B follows A (as in the joint session)
    Returns {condition: ([(source, indices, shift)], question start)}."""
    assert 0 <= aA and aA + 1 < nA and 0 <= aB and aB + 1 < nB
    head = list(range(nh))
    segA, segB = list(range(nh, nh + nA)), list(range(nh, nh + nB))
    minA, minB = [nh + aA, nh + aA + 1], [nh + aB, nh + aB + 1]
    jB, jminB = list(range(nh + nA, nh + nA + nB)), [nh + nA + aB, nh + nA + aB + 1]
    ov, sh = nh + max(nA, nB), nh + nA + nB
    return {"R2-joint": ([("J", head + segA + jB, 0)], sh),
            "R2-sep-overlap": ([("A", head + segA, 0), ("B", segB, 0)], ov),
            "R2-sep-shift": ([("A", head + segA, 0), ("B", segB, nA)], sh),
            "R2-min-joint": ([("J", head + minA + jminB, 0)], sh),
            "R2-min-sep-overlap": ([("A", head + minA, 0), ("B", minB, 0)], ov),
            "R2-min-sep-shift": ([("A", head + minA, 0), ("B", minB, nA)], sh)}


def hybrid_collide_names(base):
    """2B names: every R2 condition with the JOINT DeltaNet state ('-jstate'; R2-joint -> R2-joint-state), plus the
    two attention-only probes with the head-only DeltaNet state (-kv6). Returns {name: (base plan key, lin source)}."""
    out = {("R2-joint-state" if b == "R2-joint" else f"{b}-jstate"): (b, "J") for b in base}
    out.update({"R2-joint-kv6": ("R2-joint", "H"), "R2-sep-shift-kv6": ("R2-sep-shift", "H")})
    return out


# =============================================================== addendum 6 (A): binding IDs


_SUBJ_END = re.compile(r"([\w-]+)(?='s name is| is\b| was\b)")


def ea_positions(tok, ids, sentence, value):
    """Entity and attribute token positions of `sentence` (which must occur exactly once in the decoded ids):
      entity     the last token of the subject phrase (the word before the first "'s name is" / " is" / " was")
                 and the token after it (Feng & Steinhardt: the entity token and the next one)
      attribute  the value's first token and the token after it (the full stop, as in R-min)
      wrong      the sentence's first token and the token before the attribute (neither entity nor attribute;
                 the wrong-token control)
    Returns {"entity": [i, i + 1], "attr": [j, j + 1], "wrong": [w, j - 1]}."""
    spans, text = char_spans(tok, ids)
    s0 = text.find(sentence)
    assert s0 >= 0 and text.find(sentence, s0 + 1) < 0, "sentence not found exactly once"
    m = _SUBJ_END.search(sentence)
    assert m, f"no subject phrase in {sentence!r}"
    e = tokens_overlapping(spans, s0 + m.end(1) - 1, s0 + m.end(1))[-1]
    v = value.strip()
    vi = sentence.find(v, m.end(1))
    assert vi >= 0, f"value {v!r} not after the subject in {sentence!r}"
    a = tokens_overlapping(spans, s0 + vi, s0 + vi + 1)[0]
    w = tokens_overlapping(spans, s0, s0 + 1)[0]
    out = {"entity": [e, e + 1], "attr": [a, a + 1], "wrong": [w, a - 1]}
    assert not set(out["wrong"]) & (set(out["entity"]) | set(out["attr"])), out
    return out


def id_templates(pool, n, seed):
    """n two-pair 'name' contexts from the fact pool's person/pet slots: [(slot1, v1, slot2, v2)], two distinct
    slots and two distinct values per context (random, seeded)."""
    import random
    rng = random.Random(seed)
    slots = [s for s in pool["slots"] if s.get("kind") in ("person", "pet")]
    out = []
    while len(out) < n:
        s1, s2 = rng.sample(slots, 2)
        v1 = rng.choice(pool["names"][s1["names"]])
        v2 = rng.choice([x for x in pool["names"][s2["names"]] if x != v1])
        out.append((s1["slot"], v1, s2["slot"], v2))
    return out


def order_pc(X, order, k=4):
    """Dai's ordering-ID direction: of the top-k principal components of X [n, d] (centred), the one most
    correlated (|r|) with the order label [n]. Returns (unit direction [d], |r|)."""
    Xc = X.float() - X.float().mean(0)
    U, S, V = torch.linalg.svd(Xc, full_matrices=False)
    best, bv = -1.0, None
    o = order.float() - order.float().mean()
    for i in range(min(k, V.shape[0])):
        p = Xc @ V[i]
        r = (p * o).sum() / (p.norm() * o.norm()).clamp_min(1e-12)
        if abs(r.item()) > best:
            best, bv = abs(r.item()), V[i]
    return bv / bv.norm(), best


def gate_open(text, aliases):
    """G-match: the subject alias occurs (whole words, case-insensitive) in the user text."""
    return bool(find_all_aliases(text, aliases))


# =============================================================== generation analysis


def majority(labels, tie=None):
    """Majority of the non-None booleans; a tie (or no vote) gives `tie` (the original judge's label)."""
    v = [x for x in labels if x is not None]
    if not v:
        return tie
    t = sum(bool(x) for x in v)
    if 2 * t == len(v):
        return tie
    return 2 * t > len(v)


def balanced_yn(pairs):
    """[(consistent 'Yes'|'No', parsed 'yes'|'no'|None)] -> balanced accuracy (accY + accN) / 2 (unparsed =
    wrong; nan if a side is missing)."""
    acc = {}
    for side in ("Yes", "No"):
        g = [p == side.lower() for c, p in pairs if c == side]
        acc[side] = sum(g) / len(g) if g else float("nan")
    return (acc["Yes"] + acc["No"]) / 2


PUSH_CEILING = 0.15   # core_v1's push ceiling on clean use (B1's reference on Qwen2.5, which has no push@1 run)


def gen_rules(s):
    """The pre-registered B1-B6 and section-9 rules for one model from CI dicts ({mean, lo, hi}):
      subj           subj1 clean use on the use prompts
      subj_m_wrong   paired subj1 - wrongpos clean use
      bal_gain_subj  subj1 balanced yes/no accuracy gain vs nomem; bal_subj its accuracy
      wrong_gain     wrongpos clean-use gain vs nomem
      leak_gain      cross-binding leak, subj1 - nomem (the partner's target in the partner's answers)
      unrel_same     {condition: share of unrelated answers token-identical to nomem}
      reinst_m_subj  paired R-full - subj1 clean use
      analytic_gain  the pre-registered analytic read-out said GAIN (bool)
      subj_all       optional: clean-use CIs of every generated subj cell (the 'orphaned' rule: none beats the push)"""
    if any(math.isnan(s[k]["mean"]) for k in ("subj", "subj_m_wrong", "reinst_m_subj")):
        return {"B": {}, "labels": ["undetermined (no clean-use labels)"]}
    B = {}
    B["B1"] = s["subj"]["lo"] > PUSH_CEILING and s["subj_m_wrong"]["lo"] > 0
    B["B2"] = s["bal_gain_subj"]["lo"] > 0 and s["bal_subj"]["mean"] > 0.5
    B["B3"] = not (s["wrong_gain"]["lo"] > 0)
    B["B4"] = not (s["leak_gain"]["lo"] > 0)
    B["B5_qwen25"] = s["reinst_m_subj"]["mean"] >= 0 and not (s["reinst_m_subj"]["hi"] < 0)
    B["B6"] = all(v >= 0.95 for v in s["unrel_same"].values())
    labels = []
    if B["B1"] and B["B2"] and B["B3"] and B["B6"]:
        labels.append("holds (binding works)")
    if B["B1"] and not B["B3"]:
        labels.append("push in disguise")
    if not any(x["lo"] > PUSH_CEILING for x in s.get("subj_all", [s["subj"]])) and s["analytic_gain"]:
        labels.append("orphaned")
    if s["reinst_m_subj"]["lo"] > 0:
        labels.append("reinstatement wins")
    return {"B": B, "labels": labels or ["none of the section-9 rules fires"]}


def no_gain(stats):
    """PREREG: 'no gain' = the best cell's 95% CI of the mean dlog P(target) includes 0 or lies below it."""
    return stats is None or math.isnan(stats["mean"]) or stats["lo"] <= 0
