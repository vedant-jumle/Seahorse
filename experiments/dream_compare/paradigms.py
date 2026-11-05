#!/usr/bin/env python
"""dream_compare Stage B: the memory paradigms (PREREG.md "Stage B"). Library; stage_b.py runs it.

  render      every turn -> one chat message (the benchmark's frame, its session date, the 16,000-character cap)
  Memory      a context's sessions as session_pieces (head + one template piece per turn; long sessions in segments
              of whole turns <= SEG_MAX_TOKENS), the global token table (head once, then every turn's piece)
  Store       session-level prefill (SDPA), once per context: K, V of every memory token on the CPU at its
              session position; compose() restores turns at new positions (rope_shift per layer type); raw_keys()
              rotates keys back to position 0 (MemArt's scoring)
  retrieval   BM25 (retrieval.py) top-m turns in time order; PersonaMem: top-m sessions (their visible turns)
  conditions  P0 nomem, P1 full text, P2 text RAG, P3 KV RAG (the same turns, stored cache), P4 sparse (k per turn,
              S-loo / S-dream), P5 / P5s MemArt-style blocks (memart.py) at P3's / P4's budget
  Ragged      batched forward / generation of rows with DIFFERENT prefixes and prompts: left-padded cache slots and
              prompts, explicit masks from RoPE positions (sliding windows by position), per-row sampling streams;
              == the adapter's gen_batch per row (tested)
  selectors   dream_select's S-dream and S-loo on one stored turn (head + the turn's piece, harvested alone),
              device-safe re-implementations of its sel_dream / sel_loo (equal on one device: tested)
  storage     at-rest bytes per conversation (text bytes; K, V numbers x 4; Gemma: also V-only on K = V layers)
"""

import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("dream_compare_models", HERE / "models.py")
MA = _load("dream_compare_memart", HERE / "memart.py")
J = _load("dream_compare_judges", HERE / "judges.py")
R = _load("dream_compare_retrieval", HERE / "retrieval.py")
PC = _load("dream_compare_prep_common", HERE / "benchmarks" / "prep_common.py")


def DSR():
    """dream_select's run.py (selectors, dreamer)."""
    return M.DS()


def SEL():
    return M.DS().S


# ----------------------------------------------------------------------------------------------- constants (PREREG)

TURN_MAX_CHARS = 16000          # a longer turn is cut (all paradigms): 3 BEAM turns, 1 LongMemEval-S turn
SEG_MAX_TOKENS = 16384          # a longer session is encoded in segments of whole turns (BEAM)
K_SLICE = 8                     # P4: kept content positions per turn
M_TURNS = (3, 10)               # retrieved turns (P2-P4)
M_SESSIONS = (1, 3)             # PersonaMem: retrieved sessions
LOO_MAX_CONTENT = 200           # dream_select's cap: S-loo beyond it takes S-dream's picks (counted)
LOO_BATCH = 16
GEN_MAX_NEW = {"longmemeval": 128, "locomo": 48, "beam": 256, "personamem": 0}
SUBSET_N, SUBSET_SEED, N_SAMPLES = 100, 7, 4
LAT_N, LAT_DECODE = 20, 16
P1_CHUNK = 4096
NOT_MENTIONED = "Not mentioned in the conversation"
REFUSAL_EXTRA = re.compile(r"\bnot mentioned\b|\bno information\b|\bnot (?:specified|stated|provided)\b", re.I)

CONDS_GEN = ["P0", "P1", "P2-3", "P2-10", "P3-3", "P3-10", "P4loo-3", "P4loo-10", "P4dream-3", "P4dream-10",
             "P5-3", "P5-10", "P5s-3", "P5s-10"]
CONDS_PM = ["P0", "P1", "P2-1", "P2-3", "P3-1", "P3-3", "P4loo-1", "P4loo-3", "P4dream-1", "P4dream-3",
            "P5-1", "P5-3", "P5s-1", "P5s-3"]


def conds_for(bench):
    return CONDS_PM if bench == "personamem" else CONDS_GEN


def m_values(bench):
    return M_SESSIONS if bench == "personamem" else M_TURNS


def parse_cond(c):
    """'P4loo-10' -> ('P4', 'loo', 10); 'P0' -> ('P0', None, None)."""
    mm = re.fullmatch(r"(P\d)(s|loo|dream)?(?:-(\d+))?", c)
    assert mm, c
    return mm.group(1), mm.group(2), (int(mm.group(3)) if mm.group(3) else None)


# ----------------------------------------------------------------------------------------------- rendering (PREREG)


def cap_text(text):
    t = (text or "").strip()
    return (t[:TURN_MAX_CHARS].rstrip() + " [...]") if len(t) > TURN_MAX_CHARS else t


def render_turn(bench, turn):
    """(role, content) of a turn's chat message. LoCoMo (the neutral frame): every turn is a user message holding
    "[date] Speaker: text" (the transcript is a log the user provides; neither speaker is the assistant). PersonaMem:
    system (persona) turns become user messages. Content = "[session date] text" when the turn has a date."""
    text = cap_text(turn.text)
    if turn.date:
        text = f"[{turn.date}] {text}"
    if bench == "locomo":
        role = "user"
    elif turn.role == "system":
        role = "user"
    else:
        role = turn.role
    return role, text


def question_text(bench, item, context=None):
    """The question turn's text (identical for every paradigm)."""
    q = item.question.strip()
    if bench == "longmemeval":
        return f"Today is {item.question_date}. {q}" if item.question_date else q
    if bench == "locomo":
        a, b = context.meta.get("speaker_a"), context.meta.get("speaker_b")
        return (f"I'm reading a log of conversations between {a} and {b}. {q} Answer with a short phrase from the "
                f"log. If the log does not say, answer \"{NOT_MENTIONED}\".")
    return q


def is_refusal(text):
    AG = M.DS().AG
    t = text or ""
    return bool(AG.REFUSAL_RE.search(t) or REFUSAL_EXTRA.search(t))


# ----------------------------------------------------------------------------------------------- memory layout


@dataclass
class TurnMem:
    turn_id: str
    role: str
    content: str
    piece: list
    sess: int
    seg: int
    off: int            # offset of the piece in its segment, after the head
    g0: int = 0         # global index of the piece's first token
    n_content_tok: int = 0

    @property
    def n(self):
        return len(self.piece)


@dataclass
class Memory:
    H: list
    turns: list
    segments: list               # [[turn indices]]
    session_ids: list
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.by_id = {t.turn_id: i for i, t in enumerate(self.turns)}
        g = len(self.H)
        for t in self.turns:
            t.g0 = g
            g += t.n
        self.N = g

    @property
    def nh(self):
        return len(self.H)

    def seg_ranges(self):
        """[(g0, n)] global token range of each segment's turn tokens (MemArt's blocks never cross a segment)."""
        out = []
        for s in self.segments:
            a = self.turns[s[0]]
            out.append((a.g0, sum(self.turns[i].n for i in s)))
        return out

    def text_bytes(self, upto=None):
        ts = self.turns if upto is None else self.turns[:upto]
        return sum(len(t.content.encode("utf-8")) for t in ts)


def build_memory(C, bench_name, context):
    """The Memory of a context (all its turns; items see a prefix: visible_turns)."""
    tok = C.tok
    turns, segments, sids = [], [], []
    H0 = None
    for si, s in enumerate(context.sessions):
        msgs, ids = [], []
        for t in s.turns:
            role, content = render_turn(bench_name, t)
            msgs.append({"role": role, "content": content})
            ids.append(t.turn_id)
        if not msgs:
            continue
        H, T, _ = M.session_pieces(tok, msgs)
        H0 = H if H0 is None else H0
        assert H == H0, "the template head differs between sessions"
        seg, used = [], 0
        for tid, m, p in zip(ids, msgs, T):
            if seg and used + len(p) > SEG_MAX_TOKENS - len(H):
                segments.append(seg)
                seg, used = [], 0
            turns.append(TurnMem(tid, m["role"], m["content"], list(p), si, len(segments), used))
            seg.append(len(turns) - 1)
            used += len(p)
        if seg:
            segments.append(seg)
        sids.append(s.session_id)
    for t in turns:
        t.n_content_tok = len(tok(t.content, add_special_tokens=False).input_ids)
    return Memory(H0, turns, segments, sids, {"context_id": context.context_id})


def question_ids(C, qtext):
    """The question turn's tokens (user header + question + generation prompt), the template increment after a
    prior turn; asserted: head + it == the plain prompt."""
    H, _, Q = M.session_pieces(C.tok, [{"role": "user", "content": "x"}], qtext)
    full = M.DB().chat_ids(C.tok, qtext).tolist()
    assert full == H + Q, "head + question turn != the plain prompt"
    return Q


# ----------------------------------------------------------------------------------------------- the store


def layer_device(C, l):
    return next(C.blocks[l].self_attn.parameters()).device


def rope_rotate(C, K, delta, layer):
    """Keys K [B, h, S, d] of `layer` each rotated by its own delta (delta [S] long, any sign) with the layer's rotary
    embedding (partial RoPE: the zero-frequency dims stay). rope_rotate(K, full(d)) == models.rope_shift(K, d)."""
    delta = torch.as_tensor(delta, dtype=torch.long).to(K.device)
    cos, sin = M._rotary_call(C, K, delta[None], layer)
    return M._apply_rope(C, layer, K, cos.to(K.device), sin.to(K.device))


class Store:
    """K, V of every memory token on the CPU (fp32, post-RoPE at its session position), from one session-level
    prefill per segment (SDPA). K[l] [n_kv_l, N, d_l]; pos [N]."""

    def __init__(self, C, mem, raw_gpu_gb=4.0):
        self.mem = mem
        self.N = mem.N
        self.K, self.V = [], []
        for L in C.layout:
            self.K.append(torch.empty(L.n_kv, mem.N, L.head_dim))
            self.V.append(torch.empty(L.n_kv, mem.N, L.head_dim))
        self.pos = torch.empty(mem.N, dtype=torch.long)
        t0 = time.time()
        nh = mem.nh
        self.encoded_tokens = 0
        for si, seg in enumerate(mem.segments):
            ids = list(mem.H) + [x for i in seg for x in mem.turns[i].piece]
            with torch.no_grad():
                P, _, _ = M.harvest_kv(C, ids)
            self.encoded_tokens += len(ids)
            g0 = mem.turns[seg[0]].g0
            n = len(ids) - nh
            for l, (K, V) in enumerate(P.kv):
                if si == 0:
                    self.K[l][:, :nh] = K[0, :, :nh].cpu()
                    self.V[l][:, :nh] = V[0, :, :nh].cpu()
                self.K[l][:, g0:g0 + n] = K[0, :, nh:].cpu()
                self.V[l][:, g0:g0 + n] = V[0, :, nh:].cpu()
            if si == 0:
                self.pos[:nh] = torch.arange(nh)
            self.pos[g0:g0 + n] = torch.arange(nh, nh + n)
            del P
        self.encode_seconds = time.time() - t0
        self.blocks = MA.make_blocks(mem.seg_ranges())
        self.t2b = MA.token_to_block(self.blocks, mem.N)
        self._raw = {}
        per_layer = [K.numel() * 4 for K in self.K]
        self._raw_ok = sum(per_layer) <= raw_gpu_gb * 2 ** 30

    def raw_keys(self, C, l, device):
        """Layer l's keys rotated back to position 0 (MemArt scores without position), on `device`; kept on the GPU
        for the context when all layers fit raw_gpu_gb."""
        if l in self._raw:
            return self._raw[l]
        K = self.K[l].to(device)[None]
        raw = rope_rotate(C, K, -self.pos, l)[0]
        if self._raw_ok:
            self._raw[l] = raw
        return raw

    def nbytes(self):
        return sum(K.numel() + V.numel() for K, V in zip(self.K, self.V)) * 4

    def compose(self, C, parts, start):
        """A models.Prefix from stored parts [(global index list, RoPE shift)] in order: per layer the parts' entries
        are moved to the layer's device and their keys rotated by the shift (models.rope_shift per layer type)."""
        kv = []
        idxs = [torch.as_tensor(list(ix), dtype=torch.long) for ix, _ in parts]
        for l in range(len(self.K)):
            dev = layer_device(C, l)
            Ks, Vs = [], []
            for ix, (_, shift) in zip(idxs, parts):
                K = self.K[l][:, ix].to(dev)[None]
                V = self.V[l][:, ix].to(dev)[None]
                Ks.append(M.rope_shift(C, K, int(shift), l) if shift else K)
                Vs.append(V)
            kv.append((torch.cat(Ks, 2).contiguous(), torch.cat(Vs, 2).contiguous()))
        pos = torch.cat([self.pos[ix] + int(s) for ix, (_, s) in zip(idxs, parts)])
        assert pos.numel() < 2 or bool((pos[1:] > pos[:-1]).all()), "composed positions are not increasing"
        assert pos.numel() == 0 or int(pos.max()) < start
        return M.Prefix(kv, pos, int(start))


def head_part(mem):
    return (list(range(mem.nh)), 0)


def layout_text(mem, turn_idx):
    """P2's text layout of retrieved turns (time order): ids = head + their pieces; [start position of each piece];
    the question's start position."""
    ids, starts = list(mem.H), []
    for i in turn_idx:
        starts.append(len(ids))
        ids += mem.turns[i].piece
    return ids, starts, len(ids)


def p3_prefix(C, store, turn_idx):
    """P3: the retrieved turns' stored entries (whole pieces) restored at their P2 text positions."""
    mem = store.mem
    _, starts, qstart = layout_text(mem, turn_idx)
    parts = [head_part(mem)]
    for i, s in zip(turn_idx, starts):
        t = mem.turns[i]
        parts.append((list(range(t.g0, t.g0 + t.n)), s - int(store.pos[t.g0])))
    return store.compose(C, parts, qstart)


def p4_keep(mem, turn_idx, picks):
    """Slots of P3's prefix that P4 keeps: the head + each turn's picks (indices into its piece)."""
    keep, off = list(range(mem.nh)), mem.nh
    for i in turn_idx:
        keep += [off + int(p) for p in sorted(picks[i])]
        off += mem.turns[i].n
    return keep


# ----------------------------------------------------------------------------------------------- retrieval


def visible_turns(mem, bench, item):
    """Indices of the memory turns the item may see (PersonaMem: the first context_turns turns)."""
    n = len(mem.turns) if item.context_turns is None else item.context_turns
    return list(range(n))


def retrieve(bench, item, mem, m):
    """Turn indices (time order) retrieved for an item: BM25 top-m turns (retrieval.retrieve on the visible turns);
    PersonaMem: BM25 top-m sessions (document = the session's visible turns' text), all their visible turns."""
    if bench.name != "personamem":
        ids = R.retrieve(bench, item, m)
        return sorted(mem.by_id[t] for t in ids)
    vis = visible_turns(mem, bench, item)
    sess = {}
    for i in vis:
        sess.setdefault(mem.turns[i].sess, []).append(i)
    keys = sorted(sess)
    key = (id(bench), item.context_id, item.context_turns, "sessions")
    if key not in R._INDEX:
        R._INDEX[key] = R.BM25([" ".join(bench.contexts[item.context_id].turns()[i].text for i in sess[k]) for k in keys])
    top = R._INDEX[key].top(item.question, m)
    return sorted(i for j in top for i in sess[keys[j]])


def recall_of(bench, item, mem, turn_idx):
    return R.item_recall(bench, item, [mem.turns[i].turn_id for i in turn_idx])


# ----------------------------------------------------------------------------------------------- P1 (full text)


def prefill_text(C, ids, chunk=P1_CHUNK):
    """The text's K, V as a Prefix, from a chunked prefill on a plain cache (position masks on sliding models)."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    cache = M.make_cache([], 1)
    dev = M.input_device(C)
    n = len(ids)
    with torch.no_grad():
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            qp = torch.arange(a, b)
            kw = {}
            if C.sliding:
                kw["attention_mask"] = M.pos_masks(C, torch.arange(0, b), qp)
            o = C.body(input_ids=ids[a:b][None].to(dev), past_key_values=cache, position_ids=qp[None].to(dev),
                       use_cache=True, **kw)
            cache = o.past_key_values
            del o
    kv = [(K, V) for K, V in M.kv_list(cache)]
    return M.Prefix(kv, torch.arange(n), n)


def prefix_head(P, n):
    """The first n entries of a Prefix as views (no copy); the next turn starts at n."""
    if n == P.n:
        return M.Prefix(P.kv, P.pos, n, dict(P.meta))
    return M.Prefix([(K[:, :, :n], V[:, :, :n]) for K, V in P.kv], P.pos[:n], n, dict(P.meta))


def bytes_per_position(C):
    return M.kv_numbers_per_position(C.layout) * 4


# ----------------------------------------------------------------------------------------------- ragged batches


def _pad_id(C):
    p = C.tok.pad_token_id
    return int(p if p is not None else C.eos_list[0])


class Ragged:
    """A batch of rows {prefix: Prefix | None, ids: [prompt tokens], greedy, seed, p5: dict | None, q_len}. Layout per
    row: [pad slots][prefix entries] [pad tokens][prompt]: prefixes right-aligned in the first S slots, prompts
    right-aligned in the next T columns, so every row's last prompt token is the last column. Pad slots and pad
    tokens are masked as keys (a pad query sees only itself, so everything stays finite); masks come from RoPE
    positions (sliding layers: 0 <= q - k < window)."""

    def __init__(self, C, rows):
        self.C, self.rows = C, rows
        self.B = len(rows)
        self.S = max((r["prefix"].n if r.get("prefix") is not None else 0) for r in rows)
        self.T = max(len(r["ids"]) for r in rows)
        self.hook = None

    def _masks(self, qpos, qvalid, kpos, kvalid, diag_from=None):
        C = self.C
        dev = M.input_device(C)
        qpos, kpos = qpos.to(dev), kpos.to(dev)
        d = qpos[:, :, None] - kpos[:, None, :]
        allow = kvalid.to(dev)[:, None, :] & (d >= 0)
        if diag_from is not None:                       # pad queries see only themselves
            Tq = qpos.shape[1]
            pads = ~qvalid.to(dev)
            ar = torch.arange(Tq, device=dev)
            eye = torch.zeros_like(allow)
            eye[:, ar, diag_from + ar] = True
            allow = torch.where(pads[:, :, None], eye, allow)
        mn = torch.finfo(torch.float32).min
        f = lambda a: torch.where(a, 0.0, mn).to(torch.float32)[:, None]
        if not C.sliding:
            return f(allow)
        win = allow & (d < C.window)
        if diag_from is not None:
            win = torch.where((~qvalid.to(dev))[:, :, None], eye, win)
        return {M.FULL: f(allow), M.SLIDE: f(win)}

    def prefill(self):
        C, B, S, T = self.C, self.B, self.S, self.T
        from transformers import DynamicCache
        cache = DynamicCache()
        for l, L in enumerate(C.layout):          # layer by layer: at most one layer's batch buffer is duplicated
            dev = layer_device(C, l)
            p0 = self.rows[0].get("prefix")
            if B == 1 and p0 is not None and p0.n == S:
                K, V = p0.kv[l][0].to(dev), p0.kv[l][1].to(dev)       # one row, no padding: no extra buffer
            else:
                K = torch.zeros(B, L.n_kv, S, L.head_dim, device=dev)
                V = torch.zeros(B, L.n_kv, S, L.head_dim, device=dev)
                for b, r in enumerate(self.rows):
                    p = r.get("prefix")
                    if p is not None and p.n:
                        K[b, :, S - p.n:] = p.kv[l][0][0].to(dev)
                        V[b, :, S - p.n:] = p.kv[l][1][0].to(dev)
            cache.update(K, V, l)
            del K, V
        kpos = torch.zeros(B, S, dtype=torch.long)
        kvalid = torch.zeros(B, S, dtype=torch.bool)
        x = torch.full((B, T), _pad_id(C), dtype=torch.long)
        qpos = torch.zeros(B, T, dtype=torch.long)
        qvalid = torch.zeros(B, T, dtype=torch.bool)
        p5rows = {}
        for b, r in enumerate(self.rows):
            p = r.get("prefix")
            n = p.n if p is not None else 0
            if n:
                kpos[b, S - n:] = p.pos
                kvalid[b, S - n:] = True
            start = p.start if p is not None else 0
            ids = list(r["ids"])
            Tb = len(ids)
            x[b, T - Tb:] = torch.tensor(ids)
            qpos[b, T - Tb:] = torch.arange(start, start + Tb)
            qpos[b, :T - Tb] = start
            qvalid[b, T - Tb:] = True
            if r.get("p5"):
                ql = r.get("q_len", Tb)
                p5rows[b] = dict(r["p5"], slot0=S - n + r["p5"]["nh"], q_cols=(T - Tb, T - Tb + ql))
        masks = self._masks(qpos, qvalid, torch.cat([kpos, qpos], 1), torch.cat([kvalid, qvalid], 1), diag_from=S)
        if p5rows:
            self.hook = MA.MemArtHook(C, p5rows, M.kv_list, rope_rotate, S)
            self.hook.__enter__()
        dev = M.input_device(C)
        o = C.body(input_ids=x.to(dev), past_key_values=cache, position_ids=qpos.to(dev), attention_mask=masks,
                   use_cache=True)
        del masks
        self.cache = o.past_key_values
        self.kpos = torch.cat([kpos, qpos], 1)
        self.kvalid = torch.cat([kvalid, qvalid], 1)
        self.next_pos = qpos[:, -1] + 1
        return o.last_hidden_state

    def step(self, tok):
        C = self.C
        dev = M.input_device(C)
        qpos = self.next_pos[:, None]
        self.kpos = torch.cat([self.kpos, qpos], 1)
        self.kvalid = torch.cat([self.kvalid, torch.ones(self.B, 1, dtype=torch.bool)], 1)
        masks = self._masks(qpos, torch.ones_like(qpos, dtype=torch.bool), self.kpos, self.kvalid)
        o = C.body(input_ids=tok[:, None].to(dev), past_key_values=self.cache, position_ids=qpos.to(dev),
                   attention_mask=masks, use_cache=True)
        self.cache = o.past_key_values
        self.next_pos = self.next_pos + 1
        return o.last_hidden_state[:, -1]

    def close(self):
        if self.hook is not None:
            self.hook.__exit__()
        self.cache = None


def ragged_gen(C, rows, max_new, timing=None):
    """Generation of ragged rows (greedy, or sampled with C.sampler from a per-row stream seeded by row["seed"]: one
    uniform per step, so a row's draws never depend on the batch). -> [{"ids", "n_tok", "capped"}] per row; with
    timing (a dict) the first-token and decode times are recorded (synchronised)."""
    rb = Ragged(C, rows)
    B = rb.B
    dev = M.input_device(C)
    gens = [None if r.get("greedy", True) else torch.Generator().manual_seed(int(r["seed"])) for r in rows]
    greedy = torch.tensor([g is None for g in gens], device=dev)
    eos_t = torch.tensor(C.eos_list, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    seen = torch.zeros(B, C.vocab, dtype=torch.bool, device=dev) if C.sampler.presence else None
    ar = torch.arange(B, device=dev)
    steps = []
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            h = rb.prefill()[:, -1]
            for t in range(max_new):
                lg = C.lm_head(h).float().to(dev)
                if not bool(torch.isfinite(lg[~done]).all()):
                    raise FloatingPointError(f"non-finite logits at step {t}")
                nxt = lg.argmax(-1)
                if not bool(greedy.all()):
                    u = torch.zeros(B, 1, dtype=torch.float64)
                    for b, g in enumerate(gens):
                        if g is not None:
                            u[b, 0] = torch.rand(1, generator=g, dtype=torch.float64)
                    nxt = torch.where(greedy, nxt, C.sampler(lg, u.to(dev), seen))
                active = ~done
                nxt = torch.where(done, eos_t[0], nxt)
                steps.append(nxt)
                if seen is not None:
                    seen[ar, nxt] |= active
                done = done | (active & torch.isin(nxt, eos_t))
                if t == 0 and timing is not None:
                    sync()
                    timing["first_token_s"] = time.perf_counter() - t0
                    t1 = time.perf_counter()
                if bool(done.all()) or t == max_new - 1:
                    break
                h = rb.step(nxt)
        if timing is not None:
            sync()
            timing["decode_s"] = time.perf_counter() - t1
            timing["decode_steps"] = len(steps) - 1
            if rb.hook is not None:
                timing["p5_select_s"] = rb.hook.seconds
    finally:
        picked = rb.hook.picked if rb.hook is not None else None
        rb.close()
    toks = torch.stack(steps, 1).cpu().tolist() if steps else [[] for _ in rows]
    eset = set(C.eos_list)
    out = []
    for b, row in enumerate(toks):
        k = next((i for i, tk in enumerate(row) if tk in eset), None)
        ids_ = row if k is None else row[:k]
        out.append({"ids": ids_, "n_tok": len(ids_), "capped": k is None,
                    "p5_blocks": ({l: v for l, v in picked[b].items()} if picked and b in picked else None)})
    return out


def ragged_score(C, rows):
    """Teacher-forced log P of each row's last row["n_score"] prompt tokens (the reply) given everything before:
    -> [(sum log-prob, n tokens)]."""
    rb = Ragged(C, rows)
    out = []
    try:
        with torch.no_grad():
            h = rb.prefill()
            T = rb.T
            for b, r in enumerate(rows):
                L = int(r["n_score"])
                cols = torch.arange(T - L - 1, T - 1, device=h.device)
                lp = torch.log_softmax(M.logits(C, h[b, cols]), -1)
                tgt = torch.tensor(r["ids"][-L:], device=lp.device)
                out.append((float(lp.gather(-1, tgt[:, None])[:, 0].sum()), L))
    finally:
        rb.close()
    return out


# ----------------------------------------------------------------------------------------------- P4 selectors


class eager_attention:
    """Eager attention for the attention-score passes (S-dream): transformers 5 switches the model's attention
    implementation for the duration; transformers 4.46's SDPA classes fall back to eager when attentions are
    requested, so nothing is switched there."""

    def __init__(self, C):
        self.C = C
        self.prev = None

    def __enter__(self):
        m = self.C.model
        impl = getattr(m.config, "_attn_implementation", None)
        if impl != "eager" and hasattr(m, "set_attn_implementation"):
            self.prev = impl
            m.set_attn_implementation("eager")
        return self

    def __exit__(self, *exc):
        if self.prev is not None:
            self.C.model.set_attn_implementation(self.prev)
        return False


def multi_device(C):
    """True when the model's layers sit on more than one device (dream_bind's gen_batch keeps the logits on lm_head's
    device there: the adapter's masked loop is used instead)."""
    if not hasattr(C, "_b_multi"):
        C._b_multi = len({str(layer_device(C, l)) for l in range(len(C.blocks))}) > 1
    return C._b_multi


def gen1(C, ids, prefix=None, max_new=16):
    """One greedy row through the adapter's gen_batch (its masked loop on multi-device models)."""
    return M.gen_batch(C, torch.as_tensor(ids, dtype=torch.long), [(True, None)], prefix=prefix, max_new=max_new,
                       force_masks=multi_device(C))[0]


def head_mean_b(att):
    """dream_select's head_mean (eager attentions -> [layers, Tq, Tk] numpy, mean over heads), multi-device safe."""
    return torch.stack([a[0].float().mean(0).cpu() for a in att]).numpy()


def dream_b(C, store):
    """dream_select's dreamer (dream), through gen1: the same prompts, retry, parse and fallback."""
    D, S = DSR(), SEL()
    out = {"attempts": []}
    for p in (D.DREAM_PROMPT, D.DREAM_PROMPT_STRICT):
        ids = torch.tensor(D.DB.render_ids(C.tok, D.dream_messages(p, store.turn))[1], dtype=torch.long)
        g = gen1(C, ids, max_new=D.DREAM_MAX_NEW)
        text = C.tok.decode(g["ids"], skip_special_tokens=True)
        qs, strict, lines = S.parse_dreams(text, D.N_DREAM)
        out["attempts"].append({"raw": text, "questions": qs, "strict_ok": strict, "lines": lines})
        if len(qs) == D.N_DREAM:
            break
    best = max(out["attempts"], key=lambda a: len(a["questions"]))
    qs = best["questions"]
    out["fallback"] = len(qs) < D.N_DREAM
    if not qs:
        qs = best["lines"][:D.N_DREAM]
    if not qs:                                  # nothing at all (random dry-run models): one generic probe
        qs = ["What did I say?"]
        out["empty"] = True
    out["questions"] = qs
    out["retried"] = len(out["attempts"]) > 1
    return out


def turn_store(C, H, piece, content):
    """dream_select's store of one turn, harvested ALONE: head + the turn's piece (its role header included); content
    = the piece's tokens that spell its rendered content."""
    S = SEL()
    with torch.no_grad():
        P, _, _ = M.harvest_kv(C, list(H) + list(piece))
    fallback = False
    try:
        content_pos = S.content_positions(C.tok, piece, content)
    except AssertionError:
        fallback = True
        content_pos = list(range(min(3, len(piece) - 1), max(min(3, len(piece) - 1) + 1, len(piece) - 2)))
    return SimpleNamespace(turn=content, H=list(H), Seg=list(piece), nh=len(H), ns=len(piece), content=content_pos,
                           kv=P.kv, prefix=P, content_fallback=fallback)


def sel_dream_b(C, st, dreams):
    """dream_select's S-dream (sel_dream, primary score) on a turn store, through the adapter's forward (device-safe):
    per dreamed question, the head- and layer-mean attention from its content tokens and the last prompt token to
    each stored content position; summed over the questions."""
    D, S = DSR(), SEL()
    cols = [st.nh + c for c in st.content]
    prim = []
    with torch.no_grad(), eager_attention(C):
        for q in dreams["questions"]:
            ids = D.DB.qt_ids(C, q)
            _, att = M.forward(C, ids, st.prefix, attn=True)
            A = head_mean_b(att)
            try:
                qc = S.content_positions(C.tok, ids, q)
            except AssertionError:      # a degenerate dreamed line (random dry-run models): the last token only
                qc = []
            prim.append(S.query_mean(A.mean(0), qc + [len(ids) - 1], cols))
            del att
    return S.dream_combine(prim)


def _drop_cache(C, P, drops):
    """A DynamicCache of len(drops) rows: the whole stored turn minus one slot per row (per-layer devices)."""
    from transformers import DynamicCache
    T = P.n
    keep = torch.tensor([[i for i in range(T) if i != j] for j in drops])
    cache = DynamicCache()
    for l, (K, V) in enumerate(P.kv):
        idx = keep.to(K.device)
        cache.update(K[0][:, idx].permute(1, 0, 2, 3).contiguous(), V[0][:, idx].permute(1, 0, 2, 3).contiguous(), l)
    return cache, keep


def sel_loo_b(C, st, dreams):
    """dream_select's S-loo (sel_loo) on a turn store, device-safe: per dreamed question, the model's own greedy short
    answer with the whole turn; each content position's score = the drop in log P(own answer | question) when that
    position alone is removed; summed over the questions. Sliding models get position masks per row."""
    import numpy as np
    D = DSR()
    pf = st.prefix
    start = pf.start
    total = np.zeros(len(st.content))
    owns = []
    dev = M.input_device(C)
    with torch.no_grad():
        for q in dreams["questions"]:
            qi = D.DB.qt_ids(C, q)
            g = gen1(C, qi, prefix=pf, max_new=D.OWN_MAX_NEW)
            own = torch.tensor(g["ids"], dtype=torch.long)
            owns.append({"q": q, "answer": C.tok.decode(g["ids"], skip_special_tokens=True), "n_tok": len(g["ids"])})
            if len(own) == 0:
                continue
            ids = torch.cat([qi, own])
            Lq = len(qi)
            h, _ = M.forward(C, ids, pf)
            base = float(D._seq_lp(C, h[None], Lq, own)[0])
            drops = []
            for b in range(0, len(st.content), LOO_BATCH):
                js = [st.nh + c for c in st.content[b:b + LOO_BATCH]]
                cache, keep = _drop_cache(C, pf, js)
                qp = torch.arange(start, start + len(ids))
                kw = {}
                if C.sliding:
                    rb = Ragged.__new__(Ragged)
                    rb.C = C
                    kpos = torch.cat([pf.pos[keep], qp[None].expand(len(js), -1)], 1)
                    kw["attention_mask"] = rb._masks(qp[None].expand(len(js), -1), torch.ones(len(js), len(ids), dtype=torch.bool),
                                                     kpos, torch.ones_like(kpos, dtype=torch.bool))
                o = C.body(input_ids=ids[None].expand(len(js), -1).to(dev), past_key_values=cache,
                           position_ids=qp[None].expand(len(js), -1).to(dev), use_cache=True, **kw)
                drops += (base - D._seq_lp(C, o.last_hidden_state, Lq, own)).tolist()
                del o, cache
            total += np.array(drops)
    return total.tolist(), owns


def select_turn(C, H, turn, k=K_SLICE):
    """P4's picks for one memory turn (query-free: the turn alone): S-dream and S-loo top-k content positions
    (indices into the piece), dream_select's dreamer shared. Turns with <= k content tokens keep all of them; S-loo on
    more than LOO_MAX_CONTENT content tokens falls back to S-dream's picks."""
    D, S = DSR(), SEL()
    t0 = time.time()
    st = turn_store(C, H, turn.piece, turn.content)
    rec = {"n_content": len(st.content), "content_fallback": st.content_fallback}
    if len(st.content) <= k:
        rec.update(dream=list(st.content), loo=list(st.content), loo_fallback=False, dreams=None, seconds=time.time() - t0)
        return rec
    with torch.no_grad():
        dr = dream_b(C, st)
    sd = sel_dream_b(C, st, dr)
    rec["dream"] = S.topk(sd, st.content, k)
    rec["dreams"] = dr["questions"]
    if len(st.content) <= LOO_MAX_CONTENT:
        sl, owns = sel_loo_b(C, st, dr)
        rec["loo"] = S.topk(sl, st.content, k)
        rec["loo_fallback"] = False
        rec["own_answers"] = [o["answer"] for o in owns]
    else:
        rec["loo"] = list(rec["dream"])
        rec["loo_fallback"] = True
    rec["seconds"] = time.time() - t0
    del st
    return rec


def _parse_dream(D, S, text):
    qs, strict, lines = S.parse_dreams(text, D.N_DREAM)
    return {"raw": text, "questions": qs, "strict_ok": strict, "lines": lines}


def _finish_dream(D, attempts):
    out = {"attempts": attempts}
    best = max(attempts, key=lambda a: len(a["questions"]))
    qs = best["questions"]
    out["fallback"] = len(qs) < D.N_DREAM
    if not qs:
        qs = best["lines"][:D.N_DREAM]
    if not qs:
        qs = ["What did I say?"]
        out["empty"] = True
    out["questions"] = qs
    out["retried"] = len(attempts) > 1
    return out


def select_turns(C, H, turns, loo=(), k=K_SLICE, chunk=24, loo_rows=48, log=None):
    """P4's picks for many turns at once (query-free), batched with the ragged machinery; per turn the same result as
    select_turn up to batching numerics: dreams (dream_select's prompts, retry with the strict prompt) for every turn
    with > k content tokens -> S-dream picks; for the turns whose index is in `loo` (and <= LOO_MAX_CONTENT content
    tokens) S-loo: own answers (greedy, <= 24 tokens) and the leave-one-out drops, each drop a ragged row (the stored
    turn minus one slot, positions kept). -> {index: record}."""
    import numpy as np
    D, S = DSR(), SEL()
    out = {}
    idx = list(range(len(turns)))
    for c0 in range(0, len(idx), chunk):
        part = idx[c0:c0 + chunk]
        t0 = time.time()
        st = {i: turn_store(C, H, turns[i].piece, turns[i].content) for i in part}
        todo = [i for i in part if len(st[i].content) > k]
        for i in part:
            if i not in todo:
                c = list(st[i].content)
                out[i] = dict(n_content=len(c), content_fallback=st[i].content_fallback, dream=c, loo=c,
                              loo_fallback=False, dreams=None)
        # dreams
        att = {i: [] for i in todo}
        for prompt in (D.DREAM_PROMPT, D.DREAM_PROMPT_STRICT):
            need = [i for i in todo if not att[i] or len(att[i][-1]["questions"]) < D.N_DREAM]
            if prompt is D.DREAM_PROMPT_STRICT:
                need = [i for i in need if att[i]]
            rows = [dict(prefix=None, ids=D.DB.render_ids(C.tok, D.dream_messages(prompt, st[i].turn))[1]) for i in need]
            for b in range(0, len(rows), B_DREAM):
                gg = ragged_gen(C, rows[b:b + B_DREAM], D.DREAM_MAX_NEW)
                for i, g in zip(need[b:b + B_DREAM], gg):
                    att[i].append(_parse_dream(D, S, C.tok.decode(g["ids"], skip_special_tokens=True)))
        dreams = {i: _finish_dream(D, att[i]) for i in todo}
        for i in todo:
            sd = sel_dream_b(C, st[i], dreams[i])
            c = st[i].content
            out[i] = dict(n_content=len(c), content_fallback=st[i].content_fallback, dream=S.topk(sd, c, k),
                          dreams=dreams[i]["questions"], loo=None, loo_fallback=None)
        # S-loo
        lo = [i for i in todo if i in loo]
        for i in lo:
            if len(st[i].content) > LOO_MAX_CONTENT:
                out[i].update(loo=list(out[i]["dream"]), loo_fallback=True)
        lo = [i for i in lo if len(st[i].content) <= LOO_MAX_CONTENT]
        pairs = [(i, q) for i in lo for q in dreams[i]["questions"]]
        qids = {(i, q): D.DB.qt_ids(C, q).tolist() for i, q in pairs}
        owns = {}
        for b in range(0, len(pairs), B_DREAM):
            pp = pairs[b:b + B_DREAM]
            gg = ragged_gen(C, [dict(prefix=st[i].prefix, ids=qids[(i, q)]) for i, q in pp], D.OWN_MAX_NEW)
            for p, g in zip(pp, gg):
                owns[p] = g["ids"]
        specs = []                                  # (turn, q, drop slot or None)
        for i, q in pairs:
            if not owns[(i, q)]:
                continue
            specs.append((i, q, None))
            specs += [(i, q, st[i].nh + c) for c in st[i].content]
        lp = {}
        for b in range(0, len(specs), loo_rows):
            sp = specs[b:b + loo_rows]
            rows = []
            for i, q, j in sp:
                pf = st[i].prefix
                if j is not None:
                    pf = M.index_prefix(pf, [s for s in range(pf.n) if s != j])
                own = owns[(i, q)]
                rows.append(dict(prefix=pf, ids=qids[(i, q)] + own, n_score=len(own)))
            for s_, (v, _) in zip(sp, ragged_score(C, rows)):
                lp[s_] = v
            del rows
        for i in lo:
            tot = np.zeros(len(st[i].content))
            for q in dreams[i]["questions"]:
                if (i, q, None) not in lp:
                    continue
                base = lp[(i, q, None)]
                tot += np.array([base - lp[(i, q, st[i].nh + c)] for c in st[i].content])
            out[i].update(loo=S.topk(tot.tolist(), st[i].content, k), loo_fallback=False,
                          own_answers=[C.tok.decode(owns[(i, q)], skip_special_tokens=True) for q in dreams[i]["questions"]])
        dt = time.time() - t0
        for i in part:
            out[i]["seconds_chunk"] = dt
            out[i]["chunk_size"] = len(part)
        del st
        if log:
            log(f"  picks: {c0 + len(part)}/{len(idx)} turns ({len(lo)} with S-loo) in {dt:.0f}s")
    return out


B_DREAM = 16


def turn_key(t):
    return hashlib.sha1(f"{t.role}\x00{t.content}".encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------- storage accounting


def storage(C, mem, upto=None, k=K_SLICE):
    """At-rest bytes of one conversation (its first `upto` turns) per paradigm. Text = UTF-8 bytes of the rendered
    turns (what P1 / P2 store, and the BM25 index P3 / P4 need). KV = (head + turn tokens) x K,V numbers per
    position x 4 bytes; P4 = (head + min(k, content tokens) per turn) x (numbers x 4 + a 4-byte position); P5 = all
    tokens' K,V (no text) + MemArt's AABB index (2 x head_dim per KV head per layer per block, not used by our exact
    scoring). Gemma: *_vonly store V alone on its K = V layers."""
    ts = mem.turns if upto is None else mem.turns[:upto]
    text = sum(len(t.content.encode("utf-8")) for t in ts)
    n_tok = mem.nh + sum(t.n for t in ts)
    n_p4 = mem.nh + sum(min(k, t.n_content_tok) for t in ts)
    out = {}
    for tag, dedup in (("", False), ("_vonly", True)):
        num = M.kv_numbers_per_position(C.layout, dedup_k_eq_v=dedup)
        kv = n_tok * num * 4
        p4 = n_p4 * (num * 4 + 4)
        per_seg = {}
        for t in ts:
            per_seg[t.seg] = per_seg.get(t.seg, 0) + t.n
        n_blocks = sum(math.ceil(n / MA.BLOCK) for n in per_seg.values())
        aabb = n_blocks * sum(L.n_kv * L.head_dim * 2 for L in C.layout) * 4
        out.update({f"P0{tag}": 0, f"P1{tag}": text, f"P2{tag}": text, f"P3{tag}": text + kv, f"P4{tag}": text + p4,
                    f"P5{tag}": kv + aabb, f"kv_all{tag}": kv, f"kv_p4{tag}": p4})
    out.update(text_bytes=text, n_tokens=n_tok, n_p4_entries=n_p4, n_turns=len(ts))
    return out


if __name__ == "__main__":
    sys.exit("paradigms.py is a library (run stage_b.py)")
