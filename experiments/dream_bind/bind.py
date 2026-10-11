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
