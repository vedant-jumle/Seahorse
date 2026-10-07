"""Hierarchical bootstrap confidence intervals (numpy only).

Data are nested {item: {prompt: value}} where value is one number per (item, prompt): usually the mean
over that prompt's answers, or a PAIRED difference between two conditions on the same prompt (same
answers' random streams). The statistic is the macro mean: the mean over items of the mean over that
item's prompts. A bootstrap replicate resamples items with replacement and, inside every drawn item, its
prompts with replacement. Pairing: build the nested differences first (paired_diff), so both conditions
share every resample.
"""

import math

import numpy as np


def _clean(nested):
    out = {}
    for it, ps in nested.items():
        vals = [float(v) for v in ps.values() if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if vals:
            out[it] = np.asarray(vals, dtype=float)
    return out


def macro_mean(nested):
    d = _clean(nested)
    return float(np.mean([v.mean() for v in d.values()])) if d else float("nan")


def boot_ci(nested, n_boot=2000, seed=0, level=0.95):
    """(point, lo, hi, n_items): the macro mean and its percentile CI under the two-level bootstrap."""
    d = _clean(nested)
    if not d:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    items = list(d.values())
    n = len(items)
    point = float(np.mean([v.mean() for v in items]))
    per_item = np.empty((n, n_boot))
    for i, v in enumerate(items):  # prompts within each item
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        per_item[i] = v[idx].mean(1)
    pick = rng.integers(0, n, size=(n_boot, n))  # items
    stats = per_item[pick, np.arange(n_boot)[:, None]].mean(1)
    a = (1 - level) / 2
    lo, hi = np.quantile(stats, [a, 1 - a])
    return point, float(lo), float(hi), n


def paired_diff(a, b):
    """{item: {prompt: a - b}} over the (item, prompt) cells present (and finite) in both."""
    out = {}
    for it in a:
        if it not in b:
            continue
        cell = {}
        for p, va in a[it].items():
            vb = b[it].get(p)
            if va is None or vb is None or (isinstance(va, float) and math.isnan(va)) or \
                    (isinstance(vb, float) and math.isnan(vb)):
                continue
            cell[p] = va - vb
        if cell:
            out[it] = cell
    return out


def wrong_way(nested, sign=1.0, eps=0.0):
    """Share of items (by their mean) and of (item, prompt) cells that moved the WRONG way: value * sign
    < -eps (sign = +1 when a positive change is the expected direction)."""
    d = _clean(nested)
    if not d:
        return float("nan"), float("nan")
    items = [float(v.mean()) * sign < -eps for v in d.values()]
    cells = [float(x) * sign < -eps for v in d.values() for x in v]
    return sum(items) / len(items), sum(cells) / len(cells)
