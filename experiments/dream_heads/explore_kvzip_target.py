"""Exploratory follow-up on dream_heads: what does KVzip-g's uneven budget target?

Per item: KVzip-g k8 budget per (layer, KV group) vs R-full's per-group attention mass to memory at the last prompt
token, and vs per-group attribution onto d_comply (R-full - nomem). Then, across the k8 selections within an item:
does mass-weighted coverage (share of R-full's last-token memory attention preserved, heads weighted by their R-full
mass) track judged correctness / answer score / refusal better than plain mean coverage?
"""
import sys, glob, gzip, json
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/home/vedantjumle/projects/memories/Seahorse/experiments/dream_heads")
sys.path.insert(0, "/home/vedantjumle/projects/memories/Seahorse")
import analyze as AN  # noqa: E402

ROOT = "/home/vedantjumle/projects/memories/Seahorse/results/dream_heads_20261111"
SA = "/home/vedantjumle/projects/memories/Seahorse/results/dream_compare_a_20261105_1822"
SELS = ["KVzip-g", "KVzip-ph", "KVzip-sh", "S-loo", "S-dream", "random", "gold@k"]
rng = np.random.default_rng(7)


def boot(x, n=2000):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    bs = [np.mean(rng.choice(x, len(x))) for _ in range(n)]
    return float(np.mean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)


def fmt(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}] (n {t[3]})"


for m in ["qwen25_15b", "qwen25_7b", "llama32_3b"]:
    heads = json.load(open(f"{ROOT}/{m}/heads.json"))
    L, nq, nkv, g = heads["n_layers"], heads["n_q"], heads["n_kv"], heads["group"]
    recs = [json.load(gzip.open(f)) for f in sorted(glob.glob(f"{ROOT}/{m}/items/*.json.gz"))]
    J = AN.judged(f"{SA}/{m}", {r["info"]["question_id"]: r for r in recs})

    def jv(q, c):
        try:
            return J[c.replace("|", "@")][q]
        except Exception:
            return None

    r_mass, r_attr, r_late = [], [], []
    rows = {k: [] for k in ["wcov_j", "mcov_j", "wcov_a", "mcov_a", "wcov_r", "mcov_r", "d_j", "d_a"]}
    wcov_by_sel = {s: [] for s in SELS}
    for r in recs:
        q, info, C = r["info"]["question_id"], r["info"], r["conds"]
        ls = info["lstar"]
        sel = C["KVzip-g|k8"]["sel"]
        bud = np.array([[len(sel[str(l)][h]) for h in range(nkv)] for l in range(L)], float)  # [L, nkv]
        mass_full = np.array(C["R-full"]["mass"]).reshape(L, nkv, g).sum(-1)               # [L, nkv]
        dattr = (np.array(C["R-full"]["attr_lstar"]) - np.array(C["nomem"]["attr_lstar"])).reshape(L, nkv, g).sum(-1)
        r_mass.append(spearmanr(bud.ravel(), mass_full.ravel()).statistic)
        m2 = np.arange(L) <= ls
        r_attr.append(spearmanr(bud[m2].ravel(), dattr[m2].ravel()).statistic)
        r_late.append(spearmanr(bud.sum(1), np.arange(L)).statistic)
        # mass-weighted coverage per selection
        wq = np.array(C["R-full"]["mass"])                                                  # [L, nq]
        vals = []
        for s in SELS:
            k = f"{s}|k8"
            if k not in C:
                continue
            cov = np.array(C[k]["coverage"])
            ok_ = np.isfinite(cov); wc = float((cov * wq)[ok_].sum() / wq[ok_].sum())
            mc = float(np.nanmean(cov))
            wcov_by_sel[s].append(wc)
            vals.append((wc, mc, jv(q, k), C[k]["answer_score"], C[k]["refusal"]))
        if len(vals) >= 4:
            V = np.array([[v if v is not None else np.nan for v in row] for row in vals], float)
            ok = np.isfinite(V[:, 2])
            if ok.sum() >= 4 and np.ptp(V[ok, 2]) > 0:
                a = spearmanr(V[ok, 0], V[ok, 2]).statistic; b = spearmanr(V[ok, 1], V[ok, 2]).statistic
                rows["wcov_j"].append(a); rows["mcov_j"].append(b); rows["d_j"].append(a - b)
            a = spearmanr(V[:, 0], V[:, 3]).statistic; b = spearmanr(V[:, 1], V[:, 3]).statistic
            rows["wcov_a"].append(a); rows["mcov_a"].append(b); rows["d_a"].append(a - b)
            rows["wcov_r"].append(spearmanr(V[:, 0], V[:, 4]).statistic); rows["mcov_r"].append(spearmanr(V[:, 1], V[:, 4]).statistic)

    print(f"\n=== {m}  ({len(recs)} items)")
    print("KVzip-g k8 budget per (layer, KV group), within-item Spearman with:")
    print("  R-full last-token memory attention mass  ", fmt(boot(r_mass)))
    print("  R-full - nomem attribution on d_comply (l<=L*)", fmt(boot(r_attr)))
    print("  layer index (budget per layer)           ", fmt(boot(r_late)))
    print("Mass-weighted coverage of R-full's last-token memory attention, mean over items:")
    print("  " + ", ".join(f"{s} {np.mean(v):.2f}" for s, v in wcov_by_sel.items() if v))
    print("Within-item Spearman across k8 selections (mass-weighted cov | mean cov | difference):")
    print("  vs judged      ", fmt(boot(rows["wcov_j"])), "|", fmt(boot(rows["mcov_j"])), "|", fmt(boot(rows["d_j"])))
    print("  vs answer score", fmt(boot(rows["wcov_a"])), "|", fmt(boot(rows["mcov_a"])), "|", fmt(boot(rows["d_a"])))
    print("  vs refusal     ", fmt(boot(rows["wcov_r"])), "|", fmt(boot(rows["mcov_r"])))
