"""dream_compare Stage A (PREREG.md, "Stage A: selectors"): the published KV-cache selectors as pure numpy scorers and
selectors (no torch, no model). The torch side (stage_a.py) reduces each layer's attention to the per-(KV head, stored
slot) statistics below while the pass runs; everything that turns those into scores and picks lives here, unit-tested on
toy tensors in test_stage_a.py.

Conventions
  w          one layer's softmax attention [Hq, Tq, Tk] (rows = queries, columns = keys; Tk >= the stored slots)
  n_kv       the layer's KV heads; query head h belongs to KV head h // G, G = Hq // n_kv (repeat_interleave, as
             transformers' repeat_kv and headmask.HeadMask)
  scores     {layer: [n_kv, n_stored]} per (layer, KV head, stored slot); stored slot = index into the harvest sequence
             [head][turn]; the head (slots < n_head) is the sink, always kept, never counted
  content    the selectable content positions as indices into the TURN (dream_select's convention); stored slot =
             n_head + content position
  per-head picks  {layer: [[content positions of KV head 0], [.. head 1], ...]}

Selectors (PREREG): KVzip (Kim et al. 2025; official attention/score.py::_get_score, reimplemented), SnapKV (Li et al.
2024; KVPress GQA mean, max-pool kernel 7), H2O (Zhang et al. 2023; recent / heavy split), VATP (Guo et al. 2024; x the
value's l1 norm), Expected Attention (KVPress expected_attention_press.py), StreamingLLM (Xiao et al. 2024), S-dream-VATP
(exploratory). None takes the benchmark question or answer: SnapKV-oracle is SnapKV's scorer fed the real question by
stage_a.sel_snapkv_oracle (the one flagged function).
"""

import math

import numpy as np

BUDGETS = ("k2", "k4", "k8", "k10")
GEN_BUDGETS = ("k4", "k8")
MAIN_K = "k8"

OURS = ("S-dream", "S-attn", "S-recon", "S-loo", "random", "gold@k")
BASELINES = ("KVzip-g", "KVzip-ph", "KVzip-sh", "SnapKV-dream-ph", "SnapKV-dream-sh", "SnapKV-oracle-ph", "H2O-ph",
             "H2O-sh", "VATP-ph", "EA-ph", "StreamingLLM", "S-dream-VATP")
SELECTIONS = OURS + BASELINES                     # 18 per budget
PER_HEAD = frozenset({"KVzip-g", "KVzip-ph", "SnapKV-dream-ph", "SnapKV-oracle-ph", "H2O-ph", "VATP-ph", "EA-ph"})
ORACLE = frozenset({"SnapKV-oracle-ph"})          # sees the real benchmark question (a ceiling that cheats)
LABEL = frozenset({"gold@k", "gold"})             # use the label
NON_ORACLE_BASELINES = tuple(c for c in BASELINES if c not in ORACLE)          # H4's candidates (11)
H2_PAIRS = (("KVzip-ph", "KVzip-sh"), ("SnapKV-dream-ph", "SnapKV-dream-sh"), ("H2O-ph", "H2O-sh"))
H3_METHODS = ("KVzip-ph", "SnapKV-dream-ph", "H2O-ph", "VATP-ph")
KVZIP_FORMS = ("KVzip-g", "KVzip-ph", "KVzip-sh")

KVZIP_PROMPT = "Repeat the previous context exactly."
SNAP_WINDOW = 32
SNAP_KERNEL = 7
EA_N_SINK = 4
EA_N_FUTURE = 512


def budget_k(name, n_content):
    """k for a budget name (dream_select's rule): k2 / k4 / k8, or k10 = max(2, round(0.10 * n_content)); capped."""
    k = max(2, int(round(0.10 * n_content))) if name == "k10" else int(name[1:])
    return min(k, n_content)


def entries_per_position(n_kv_per_layer):
    """E = stored K,V entries of one kept position: the sum over layers of the KV heads."""
    return int(sum(int(x) for x in n_kv_per_layer))


def budget_entries(k, n_kv_per_layer):
    """The matched budget in stored entries: k positions in every (layer, KV head) = k x E."""
    return int(k) * entries_per_position(n_kv_per_layer)


# ----------------------------------------------------------------------------------------------- reductions


def group_reduce(x, n_kv, op="mean"):
    """[Hq, ...] -> [n_kv, ...]: mean / max over the query heads of each GQA group (query head h -> KV head h // G)."""
    x = np.asarray(x, dtype=np.float64)
    hq = x.shape[0]
    assert hq % n_kv == 0, (hq, n_kv)
    g = x.reshape(n_kv, hq // n_kv, *x.shape[1:])
    return g.mean(1) if op == "mean" else g.max(1)


def maxpool1d(x, kernel=SNAP_KERNEL):
    """F.max_pool1d(x, kernel, stride 1, padding kernel // 2) along the last axis (padding = -inf)."""
    x = np.asarray(x, dtype=np.float64)
    p = kernel // 2
    pad = np.full(x.shape[:-1] + (p,), -np.inf)
    xp = np.concatenate([pad, x, pad], -1)
    n = x.shape[-1]
    return np.max(np.stack([xp[..., i:i + n] for i in range(kernel)], 0), 0)


# ----------------------------------------------------------------------------------------------- scorers (one layer)


def kvzip_layer(w, n_kv, n_stored):
    """KVzip's _get_score: for each stored entry the max softmax attention over the query heads of its GQA group and over
    every input row (repeat prompt + reconstruction). w = the input rows' attention [Hq, Tq, Tk]. -> [n_kv, n_stored]."""
    w = np.asarray(w, dtype=np.float64)[:, :, :n_stored]
    return group_reduce(w.max(1), n_kv, "max")


def snapkv_layer(w, n_kv, n_stored, window=SNAP_WINDOW, kernel=SNAP_KERNEL):
    """SnapKV's vote: the sum over the observation window (the last <= `window` rows of w = the question turn's rows)
    of the attention to each stored entry, mean over the GQA group, max-pooled over the stored axis. -> [n_kv, n_stored]."""
    w = np.asarray(w, dtype=np.float64)
    v = w[:, -window:, :n_stored].sum(1)
    return maxpool1d(group_reduce(v, n_kv, "mean"), kernel)


def h2o_layer(w, n_kv):
    """H2O's accumulated attention: the column sum over all rows of the harvest pass (w = [Hq, T, T]), mean over the
    GQA group. -> [n_kv, T]."""
    w = np.asarray(w, dtype=np.float64)
    return group_reduce(w.sum(1), n_kv, "mean")


def value_norms(V, p=1):
    """Per (KV head, slot) value-vector norm: V [n_kv, T, d] -> [n_kv, T] (l1 for VATP, l2 for Expected Attention)."""
    V = np.asarray(V, dtype=np.float64)
    return np.abs(V).sum(-1) if p == 1 else np.sqrt((V ** 2).sum(-1))


def vatp_layer(h2o, vnorm1):
    """VATP: the H2O score x the value's l1 norm (paper Eq. 4)."""
    return np.asarray(h2o, dtype=np.float64) * np.asarray(vnorm1, dtype=np.float64)


def rows_mean_layer(w, rows, n_kv, n_stored):
    """S-dream-VATP's attention part: the mean over the given query rows of the attention to each stored entry, mean
    over the GQA group. -> [n_kv, n_stored]."""
    w = np.asarray(w, dtype=np.float64)
    return group_reduce(w[:, list(rows), :n_stored].mean(1), n_kv, "mean")


def rope_matrix(cos, sin, rot_half):
    """The linear map of one rotary step as a matrix acting on ROW vectors (x' = x @ A): A = diag(cos) + Rh diag(sin),
    where rot_half(x) = x @ Rh. cos, sin: [d]; rot_half: a function on [n, d] arrays (the model's convention)."""
    cos, sin = np.asarray(cos, dtype=np.float64), np.asarray(sin, dtype=np.float64)
    d = cos.shape[0]
    Rh = rot_half(np.eye(d))
    return np.diag(cos) + Rh * sin[None, :]


def rotate_half(x):
    """transformers' rotate_half on the last axis: [x1, x2] -> [-x2, x1]."""
    x = np.asarray(x, dtype=np.float64)
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


def avg_rope_matrix(cos_rows, sin_rows, rot_half=rotate_half):
    """Expected Attention's RoPE matrix averaged over future positions: RoPE is linear in (cos, sin), so the mean of the
    per-position matrices is the matrix of the mean cos / sin. cos_rows, sin_rows: [P, d]."""
    return rope_matrix(np.asarray(cos_rows).mean(0), np.asarray(sin_rows).mean(0), rot_half)


def query_stats(q):
    """Per query head mean and (population) covariance of pre-RoPE queries q [Hq, n, d] -> mu [Hq, d], cov [Hq, d, d]."""
    q = np.asarray(q, dtype=np.float64)
    mu = q.mean(1)
    c = q - mu[:, None, :]
    return mu, np.einsum("hni,hnj->hij", c, c) / q.shape[1]


def ea_layer(mu, cov, A, K, V, scale, cols, use_cov=True):
    """Expected Attention (KVPress): with the averaged rotation A (row-vector convention: q' = q @ A), mu' = mu A and
    Sigma' = A^T Sigma A; z[h, j] = s mu'_h . k_j + (s^2 / 2) k_j^T Sigma'_h k_j for the keys of h's KV head; softmax
    over the columns `cols` (the turn's slots; others get 0); mean over the GQA group; x the value's l2 norm.
    mu [Hq, d], cov [Hq, d, d], A [d, d], K / V [n_kv, T, d] (K rotated at the stored positions). -> [n_kv, T]."""
    mu, cov, A = (np.asarray(x, dtype=np.float64) for x in (mu, cov, A))
    K, V = np.asarray(K, dtype=np.float64), np.asarray(V, dtype=np.float64)
    hq, n_kv = mu.shape[0], K.shape[0]
    G = hq // n_kv
    mu_r = mu @ A
    cov_r = np.matmul(np.matmul(A.T[None], cov), A[None])                    # A^T Sigma A per head
    Kq = np.repeat(K, G, axis=0)                                              # [Hq, T, d]
    z = scale * np.einsum("hd,htd->ht", mu_r, Kq)
    if use_cov:
        z = z + 0.5 * scale ** 2 * (np.matmul(Kq, cov_r) * Kq).sum(-1)
    cols = np.asarray(list(cols), dtype=int)
    p = np.zeros_like(z)
    zc = z[:, cols]
    zc = np.exp(zc - zc.max(1, keepdims=True))
    p[:, cols] = zc / zc.sum(1, keepdims=True)
    return group_reduce(p, n_kv, "mean") * value_norms(V, 2)


# ----------------------------------------------------------------------------------------------- selectors


def _clean(v):
    v = np.asarray(v, dtype=np.float64).copy()
    v[~np.isfinite(v)] = -np.inf
    return v


def _order(v):
    """Indices by score, highest first; ties to the earlier index; NaN / inf-less values last (dream_select's topk)."""
    return np.argsort(-_clean(v), kind="stable")


def topk_content(score_slots, content, n_head, k, exclude=()):
    """The k content positions with the highest score; score_slots = one score per stored slot ([n_stored]). Ties go
    to the earlier position. exclude = content positions not eligible. Sorted."""
    ex = set(exclude)
    cand = [c for c in content if c not in ex]
    k = min(k, len(cand))
    s = np.array([score_slots[n_head + c] for c in cand], dtype=np.float64)
    return sorted(int(cand[i]) for i in _order(s)[:k])


def per_head_topk(scores, content, n_head, k):
    """Per-head uniform: k content positions for every (layer, KV head). scores {layer: [n_kv, n_stored]}."""
    return {int(l): [topk_content(row, content, n_head, k) for row in np.asarray(S)] for l, S in scores.items()}


def global_topk(scores, content, n_head, n_entries):
    """KVzip's non-uniform head budget: the top n_entries (layer, KV head, content position) entries over the whole
    cache (KVzip's _threshold), ties to (earlier layer, head, position). Returns per-head picks (some heads may keep
    none)."""
    cols = np.asarray([n_head + int(c) for c in content], dtype=int)
    blocks, owner = [], []
    for l in sorted(scores):
        S = np.asarray(scores[l], dtype=np.float64)[:, cols]               # [n_kv, n_content]
        blocks.append(S.reshape(-1))
        owner += [(int(l), h) for h in range(S.shape[0])]
    v = np.concatenate(blocks)
    nc = len(cols)
    keep = _order(v)[:min(n_entries, len(v))]
    out = {int(l): [[] for _ in range(np.asarray(S).shape[0])] for l, S in scores.items()}
    for i in keep:
        l, h = owner[i // nc]
        out[l][h].append(int(content[i % nc]))
    return {l: [sorted(x) for x in hs] for l, hs in out.items()}


def shared_score(scores):
    """The shared form of a per-head score: the mean over all (layer, KV head) rows. -> [n_stored]."""
    rows = [np.asarray(S, dtype=np.float64) for _, S in sorted(scores.items())]
    return np.concatenate(rows, 0).mean(0)


def shared_topk(scores, content, n_head, k):
    return topk_content(shared_score(scores), content, n_head, k)


def recent_heavy(score_slots, content, n_head, k):
    """H2O's split (also VATP): the last k // 2 content positions (recent) + the top k - k // 2 of the others (heavy)."""
    k = min(k, len(content))
    recent = list(content[len(content) - k // 2:]) if k // 2 else []
    heavy = topk_content(score_slots, content, n_head, k - len(recent), exclude=recent)
    return sorted(set(recent) | set(heavy))


def per_head_recent_heavy(scores, content, n_head, k):
    return {int(l): [recent_heavy(row, content, n_head, k) for row in np.asarray(S)] for l, S in scores.items()}


def shared_recent_heavy(scores, content, n_head, k):
    return recent_heavy(shared_score(scores), content, n_head, k)


def streaming(content, k):
    """StreamingLLM: the first k // 2 and the last k - k // 2 content positions (the head is the sink)."""
    k = min(k, len(content))
    first = list(content[:k // 2])
    last = list(content[len(content) - (k - k // 2):]) if k - k // 2 else []
    return sorted(set(first) | set(last))


def picks_entries(picks):
    """Stored entries of per-head picks (the head not counted)."""
    return int(sum(len(x) for hs in picks.values() for x in hs))


def check_picks(picks, content, n_entries, n_kv_per_layer=None):
    """Per-head picks: exactly n_entries entries, every one a content position, sorted and unique per head; every layer
    has its KV heads (fails the run)."""
    cs = set(int(c) for c in content)
    tot = 0
    for l, hs in picks.items():
        if n_kv_per_layer is not None:
            assert len(hs) == n_kv_per_layer[int(l)], f"layer {l}: {len(hs)} heads, want {n_kv_per_layer[int(l)]}"
        for x in hs:
            assert list(x) == sorted(set(x)), f"layer {l}: picks not sorted / unique"
            bad = [c for c in x if c not in cs]
            assert not bad, f"layer {l}: picks that are not content positions: {bad}"
            tot += len(x)
    assert tot == n_entries, f"per-head picks hold {tot} entries, want {n_entries}"


def same_in_every_head(picks):
    """The shared positions if every (layer, KV head) keeps the same picks, else None."""
    rows = [tuple(x) for hs in picks.values() for x in hs]
    return list(rows[0]) if rows and all(r == rows[0] for r in rows) else None


def span_share(picks, span):
    """Share of (layer, KV head) rows whose picks hit the gold span (information only)."""
    if not span:
        return None
    rows = [x for hs in picks.values() for x in hs]
    return float(np.mean([bool(set(x) & set(span)) for x in rows])) if rows else None


def gap(cond, nomem, full, min_den):
    """(cond - nomem) / (full - nomem), or None when full - nomem < min_den or a value is missing / not finite."""
    if any(x is None or not math.isfinite(x) for x in (cond, nomem, full)):
        return None
    den = full - nomem
    if den < min_den:
        return None
    return (cond - nomem) / den
