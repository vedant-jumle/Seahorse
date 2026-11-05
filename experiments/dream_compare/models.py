#!/usr/bin/env python
"""dream_compare Stage 0: the model adapter (PREREG.md, "Stage 0: infrastructure").

One spec per model instead of dream_select's hard-coded MODELS table and N_BLOCKS = 28: layers, layer types, KV heads
and head_dim per layer, the rotary embedding per layer type (partial rotation included), sliding windows, K = V layers,
the final logit soft-cap, chat-template quirks, dtype, device placement and the maximum context. The three models of
dream_select (qwen25_15b, qwen25_7b, llama32_3b) load through dream_select's / dream_bind's own code with exactly its
settings; Gemma 4 (and the optional Granite) load here, text decoder only, across GPUs with a device_map.

Cache utilities (wrappers that dispatch on the adapter; dream_bind's code is imported, never modified):
  kv_list / make_cache      per-layer shapes (Gemma 4: 8 x 256 on sliding layers, 2 x 512 on global layers)
  harvest_kv                a pass on a PLAIN DynamicCache: transformers 5's default cache for Gemma evicts sliding-layer
                            entries beyond the window (it keeps window - 1); a restored memory must keep every entry
  Prefix                    stored entries + their RoPE positions + where the next turn starts
  index_prefix / compose_prefix / rope_shift   per layer type (each type's own rotary embedding, partial rotary incl.)
  pos_masks                 explicit masks from RoPE POSITIONS (not cache slots): a restored entry in a sliding layer is
                            visible iff 0 <= q_pos - kv_pos < window. transformers derives the sliding mask from cache
                            slots, which is wrong once a sparse prefix makes slot distance < position distance (checked
                            in the tests: Gemma-4 tiny, max hidden diff 3.7 with slot masks vs 3e-6 with position masks)
  forward / gen_batch       teacher-forced pass / generation after a Prefix (position masks on sliding models; on the
                            all-full-attention models gen_batch is dream_bind's own gen_batch, unchanged)
  logits                    lm_head + Gemma's final logit soft-cap (30): C.lm_head is wrapped, so dream_bind's and
                            dream_select's code that calls C.lm_head sees the model's real logits

  python -I experiments/dream_compare/models.py --show        # the spec table
"""

import importlib.util
import inspect
import json
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
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DS = None


def DS():
    """dream_select's run.py (import only; it loads dream_bind's run.py as DS().DB)."""
    global _DS
    if _DS is None:
        _DS = _load("dream_compare_ds_run", REPO / "experiments" / "dream_select" / "run.py")
    return _DS


def DB():
    return DS().DB


def bind():
    return DS().bind


FULL, SLIDE = "full_attention", "sliding_attention"

# ----------------------------------------------------------------------------------------------- the specs
# rope: per layer type {theta, type, partial (fraction of head_dim rotated), scaling (rope_scaling extras)}.
# kv: per layer type (KV heads, head_dim). Facts checked against each model's config.json (test_dream_compare.py,
# from the HF cache where present; Gemma 4 read 2026-11-05 from huggingface.co/google/gemma-4-26B-A4B-it).

_G4_TYPES = tuple(FULL if (i + 1) % 6 == 0 else SLIDE for i in range(30))   # full at 5, 11, 17, 23, 29

SPECS = {
    "qwen25_15b": dict(
        label="Qwen2.5-1.5B", hf="Qwen/Qwen2.5-1.5B-Instruct", arch="qwen2", ds_key="qwen25_15b",
        n_layers=28, n_q_heads=12, layer_types=(FULL,) * 28, kv={FULL: (2, 128)},
        rope={FULL: dict(theta=1e6, type="default", partial=1.0)}, sliding_window=None, k_eq_v=(),
        softcap=None, chat=dict(date_string=None, thinking=None), dtype="fp32", n_gpus=1,
        hardware="gpu-a100-small (10 GB MIG) or gpu-v100", max_context=32768, tied=True),
    "qwen25_7b": dict(
        label="Qwen2.5-7B", hf="Qwen/Qwen2.5-7B-Instruct", arch="qwen2", ds_key="qwen25_7b",
        n_layers=28, n_q_heads=28, layer_types=(FULL,) * 28, kv={FULL: (4, 128)},
        rope={FULL: dict(theta=1e6, type="default", partial=1.0)}, sliding_window=None, k_eq_v=(),
        softcap=None, chat=dict(date_string=None, thinking=None), dtype="fp32", n_gpus=1,
        hardware="gpu-a100 (1 x 80 GB)", max_context=32768, tied=False),
    "llama32_3b": dict(
        label="Llama-3.2-3B", hf="meta-llama/Llama-3.2-3B-Instruct", arch="llama", ds_key="llama32_3b",
        n_layers=28, n_q_heads=24, layer_types=(FULL,) * 28, kv={FULL: (8, 128)},
        rope={FULL: dict(theta=5e5, type="llama3", partial=1.0,
                         scaling=dict(factor=32.0, high_freq_factor=4.0, low_freq_factor=1.0,
                                      original_max_position_embeddings=8192))},
        sliding_window=None, k_eq_v=(), softcap=None, chat=dict(date_string="26 Jul 2024", thinking=None),
        dtype="fp32", n_gpus=1, hardware="gpu-v100 (32 GB)", max_context=131072, tied=True),
    "gemma4_26b_a4b": dict(
        label="Gemma-4-26B-A4B", hf="google/gemma-4-26B-A4B-it", arch="gemma4", ds_key=None,
        n_layers=30, n_q_heads=16, layer_types=_G4_TYPES, kv={SLIDE: (8, 256), FULL: (2, 512)},
        rope={SLIDE: dict(theta=1e4, type="default", partial=1.0),
              FULL: dict(theta=1e6, type="proportional", partial=0.25)},
        sliding_window=1024, k_eq_v=(FULL,), softcap=30.0, chat=dict(date_string=None, thinking="gemma4"),
        dtype="fp32", n_gpus=2, hardware="gpu-a100 (2 x 80 GB; fp32 text decoder ~101 GB)", max_context=262144,
        tied=True, text_only=True, moe=dict(experts=128, top_k=8)),
    "granite42_30b": dict(
        label="Granite-4.2-30B", hf="ibm-granite/granite-4.2-30b", arch="granite", ds_key=None, optional=True,
        n_layers=64, n_q_heads=32, layer_types=(FULL,) * 64, kv={FULL: (8, 128)},
        rope={FULL: dict(theta=5e7, type="default", partial=1.0)}, sliding_window=None, k_eq_v=(),
        softcap=None, chat=dict(date_string=None, thinking="granite"), dtype="fp32", n_gpus=2,
        hardware="gpu-a100 (2 x 80 GB; fp32 ~117 GB)", max_context=131072, tied=False),
}
STAGE_MODELS = ("qwen25_15b", "qwen25_7b", "llama32_3b", "gemma4_26b_a4b")


@dataclass(frozen=True)
class LayerSpec:
    """One decoder layer's attention: type, KV heads, head_dim, rotary embedding, sliding window, K = V."""
    kind: str
    n_kv: int
    head_dim: int
    rope_theta: float
    rope_type: str
    partial: float
    window: int | None
    k_eq_v: bool

    @property
    def rot_dim(self):
        """Rotated dims per head (proportional / partial RoPE rotates partial * head_dim of them)."""
        return int(self.partial * self.head_dim // 2) * 2


def layout(spec):
    """[LayerSpec] per layer of a spec dict."""
    out = []
    for t in spec["layer_types"]:
        n_kv, hd = spec["kv"][t]
        r = spec["rope"][t]
        out.append(LayerSpec(t, n_kv, hd, float(r["theta"]), r["type"], float(r.get("partial", 1.0)),
                             spec["sliding_window"] if t == SLIDE else None, t in spec["k_eq_v"]))
    return out


def _cfg_get(c, k, d=None):
    if isinstance(c, dict):
        return c.get(k, d)
    try:   # transformers 5's heterogeneous configs (Gemma 4) raise on per-layer attributes read globally
        return getattr(c, k, d)
    except Exception:
        return d


def _per_layer(cfg, i):
    """(head_dim, KV heads) of layer i from a transformers-5 config object's per_layer_config, or None."""
    if isinstance(cfg, dict):
        return None
    try:
        p = cfg.per_layer_config[i]
        return p.head_dim, p.num_key_value_heads
    except Exception:
        return None


def layout_from_config(cfg):
    """[LayerSpec] per layer from an HF (text) config: a config.json dict or a config object. Qwen2 / Llama / Granite
    (one layer type) and Gemma 4 (sliding + full; the dict's global_head_dim / num_global_key_value_heads, or the
    object's per_layer_config; attention_k_eq_v; per-type rope_parameters)."""
    g = lambda k, d=None: _cfg_get(cfg, k, d)
    tc = g("text_config")
    if tc is not None:
        return layout_from_config(tc)
    n = g("num_hidden_layers")
    nq = g("num_attention_heads")
    hd = g("head_dim") or g("hidden_size") // nq
    n_kv = g("num_key_value_heads") or nq
    types = list(g("layer_types") or [FULL] * n)
    if _per_layer(cfg, 0) is not None:      # config object with per-layer dims (Gemma 4 in transformers 5)
        out = []
        rp = g("rope_parameters")
        for i, t in enumerate(types):
            d_i, kv_i = _per_layer(cfg, i)
            r = rp[t] if (isinstance(rp, dict) and t in rp) else rp
            out.append(LayerSpec(t, kv_i, d_i, float(r.get("rope_theta")), r.get("rope_type", "default"),
                                 float(r.get("partial_rotary_factor", 1.0) or 1.0),
                                 g("sliding_window") if t == SLIDE else None,
                                 bool(g("attention_k_eq_v", False)) and t == FULL))
        return out
    rp = g("rope_parameters")
    per_type = isinstance(rp, dict) and set(rp) <= {FULL, SLIDE} and bool(rp)
    rs = g("rope_scaling")
    out = []
    for t in types:
        if per_type:
            r = rp[t]
        elif isinstance(rp, dict) and rp:
            r = rp
        else:
            r = dict(rope_theta=g("rope_theta"), rope_type=(rs or {}).get("rope_type", "default"),
                     partial_rotary_factor=g("partial_rotary_factor", 1.0))
        kind = t if t in (FULL, SLIDE) else FULL
        is_global = kind == FULL and g("global_head_dim") is not None
        k_eq_v = bool(g("attention_k_eq_v", False)) and kind == FULL
        out.append(LayerSpec(kind, (g("num_global_key_value_heads") if (is_global and k_eq_v) else None) or n_kv,
                             g("global_head_dim") if is_global else hd, float(r.get("rope_theta")),
                             r.get("rope_type", "default"), float(r.get("partial_rotary_factor", 1.0) or 1.0),
                             g("sliding_window") if kind == SLIDE else None, k_eq_v))
    return out


def check_spec(key, cfg):
    """Mismatches between SPECS[key] and a real config.json (dict): [] when they agree."""
    s = SPECS[key]
    bad = []
    a, b = layout(s), layout_from_config(cfg)
    if len(a) != len(b):
        return [f"layers: spec {len(a)} vs config {len(b)}"]
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            bad.append(f"layer {i}: spec {x} vs config {y}")
    c = cfg.get("text_config", cfg)
    if c.get("num_attention_heads") != s["n_q_heads"]:
        bad.append(f"query heads: spec {s['n_q_heads']} vs config {c.get('num_attention_heads')}")
    if c.get("max_position_embeddings") != s["max_context"]:
        bad.append(f"max context: spec {s['max_context']} vs config {c.get('max_position_embeddings')}")
    if c.get("final_logit_softcapping") != s["softcap"]:
        bad.append(f"softcap: spec {s['softcap']} vs config {c.get('final_logit_softcapping')}")
    rs = c.get("rope_scaling") or {}
    want = s["rope"][s["layer_types"][0]].get("scaling")
    if want and any(rs.get(k) != v for k, v in want.items()):
        bad.append(f"rope scaling: spec {want} vs config {rs}")
    tied = cfg.get("tie_word_embeddings", c.get("tie_word_embeddings"))
    if tied is not None and bool(tied) != s["tied"]:
        bad.append(f"tied embeddings: spec {s['tied']} vs config {tied}")
    return bad


def kv_entries_per_position(spec_or_layout):
    """Stored K,V entries per kept position, summed over layers and KV heads (Stage A's budget unit: k positions cost
    k x this). Gemma 4: 25 x 8 + 5 x 2 = 210."""
    L = spec_or_layout if isinstance(spec_or_layout, list) else layout(spec_or_layout)
    return sum(l.n_kv for l in L)


def kv_numbers_per_position(spec_or_layout, dedup_k_eq_v=False):
    """Stored numbers per position (K and V, every layer and KV head). dedup_k_eq_v: on K = V layers (Gemma's global
    layers) the cache's V is RMSNorm(k_proj(h)) without a scale and K = RoPE(k_norm.weight * V), so storing V alone
    suffices (information only; the run stores both)."""
    L = spec_or_layout if isinstance(spec_or_layout, list) else layout(spec_or_layout)
    return sum(l.n_kv * l.head_dim * (1 if (dedup_k_eq_v and l.k_eq_v) else 2) for l in L)


# ----------------------------------------------------------------------------------------------- loading


class CappedHead(torch.nn.Module):
    """lm_head followed by the final logit soft-cap (cap * tanh(logits / cap)); a drop-in for C.lm_head."""

    def __init__(self, head, cap):
        super().__init__()
        self.head, self.cap = head, float(cap)
        self.in_features, self.out_features = head.in_features, head.out_features
        self.weight = head.weight

    def forward(self, h):
        x = self.head(h)
        return torch.tanh(x / self.cap) * self.cap


THINKING_TAILS = {"gemma4": "<|turn>model\n<|channel>thought\n<channel|>",     # the empty thought channel
                  "granite": "<|im_start|>assistant\n<think></think>"}            # Granite 4.2 thinks by default


def thinking_off(tok, kind):
    """Force enable_thinking=False in every chat-template call and assert the template's thinking-off tail (Gemma 4,
    Granite 4.2; Qwen3.x uses dream_bind's thinking_off)."""
    orig = tok.apply_chat_template

    def act(conversation, *a, **kw):
        kw.setdefault("enable_thinking", False)
        return orig(conversation, *a, **kw)
    tok.apply_chat_template = act
    s = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    assert s.endswith(THINKING_TAILS[kind]), f"{kind}: thinking not off: {s[-60:]!r}"
    assert "<|think|>" not in s, s
    return tok


def gemma_thinking_off(tok):
    return thinking_off(tok, "gemma4")


TEXT_PREFIXES = ("model.language_model.",)
SKIP_PREFIXES = ("model.vision_tower.", "model.embed_vision.", "model.audio_tower.", "model.embed_audio.")


def max_memory_map(n_gpus, gb_per_gpu):
    return {i: f"{gb_per_gpu}GiB" for i in range(n_gpus)}


def load_device_map(spec, device, attn, gb_per_gpu=66, local_files_only=None):
    """A model of SPECS loaded with transformers' from_pretrained, fp32, across the visible GPUs (device_map "auto",
    max_memory gb_per_gpu each) or on one device. Text-only checkpoints (Gemma 4 ships as *ForConditionalGeneration):
    the text config + key_mapping model.language_model.* -> model.*; the vision / audio towers are not loaded (asserted:
    nothing missing, only those prefixes unexpected)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    import transformers
    major = int(transformers.__version__.split(".")[0])
    kw = {}
    if local_files_only is not None:
        kw["local_files_only"] = local_files_only
    cfg = AutoConfig.from_pretrained(spec["hf"], **kw)
    dt = torch.float32
    lkw = dict(attn_implementation=attn, output_loading_info=True, **kw)
    lkw["dtype" if major >= 5 else "torch_dtype"] = dt
    n_gpu = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    if n_gpu > 1:
        lkw.update(device_map="auto", max_memory=max_memory_map(n_gpu, gb_per_gpu))
    else:
        lkw.update(device_map={"": device})
    if spec.get("text_only"):
        tc = cfg.text_config
        lkw.update(config=tc, key_mapping={r"^model\.language_model\.": "model."})
    model, info = AutoModelForCausalLM.from_pretrained(spec["hf"], **lkw)
    missing = [k for k in info.get("missing_keys", []) if k != "lm_head.weight" or not spec["tied"]]
    assert not missing, f"{spec['hf']}: parameters not in the checkpoint: {sorted(missing)[:6]}"
    unexpected = sorted(info.get("unexpected_keys", []))
    bad = [k for k in unexpected if not any(k.startswith(p) or k.startswith(p.replace("model.", "", 1))
                                             for p in SKIP_PREFIXES)]
    assert not bad, f"{spec['hf']}: unexpected checkpoint keys outside the vision / audio towers: {bad[:6]}"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    devmap = getattr(model, "hf_device_map", None)
    return model, {"n_unexpected_skipped": len(unexpected),
                   "skipped_prefixes": sorted({".".join(k.split(".")[:2]) for k in unexpected}),
                   "device_map_devices": sorted({str(v) for v in (devmap or {}).values()}),
                   "n_gpus_visible": n_gpu, "max_memory_gb_per_gpu": gb_per_gpu if n_gpu > 1 else None}


def input_device(C):
    return C.body.embed_tokens.weight.device


def setup(key, device="cuda", attn="eager", random_model=None, paris=True, gb_per_gpu=66, tiny=False):
    """The context C (dream_bind's layout: body, lm_head, blocks, tok, head, eos, sampler, ...) of a SPECS model, plus
    C.key, C.spec, C.layout, C.label and C.sliding. dream_select's three models: dream_select's own model block
    (run.model_config), dream_bind's loader (Llama's pinned date) and gen_context, i.e. exactly dream_select's settings
    (attn="eager" is theirs; "sdpa" is offered for Stage B's long passes). Others: load_device_map."""
    import yaml
    spec = SPECS[key]
    D = DS()
    cfg = yaml.safe_load(open(D.DB_CFG))
    t0 = time.time()
    if spec["ds_key"]:
        mc, label = D.model_config(spec["ds_key"], cfg)
        mc = dict(mc, eager=(attn == "eager"))
        dA = SimpleNamespace(random_model=random_model, dtype="fp32", device=device, tiny=tiny, gen_set=None,
                             model=spec["ds_key"])
        model, tok, linfo = D.DB.load(dA, cfg, mc)
    else:
        from transformers import AutoTokenizer
        label = spec["label"]
        mc = {"model": spec["hf"], "n_blocks": spec["n_layers"], "thinking_template": False,
              "text_only": bool(spec.get("text_only")), "eager": attn == "eager", "date_string": None}
        dA = SimpleNamespace(random_model=random_model, dtype="fp32", device=device, tiny=tiny, gen_set=None, model=key)
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        if random_model:
            model = random_model_for(key, tok)
            linfo = {"random_model": random_model}
        else:
            model, linfo = load_device_map(spec, device, attn, gb_per_gpu)
        if spec["chat"]["thinking"]:
            thinking_off(tok, spec["chat"]["thinking"])
        ls = getattr(model.config, "logits_scaling", 1.0)
        assert ls in (None, 1, 1.0), f"logits_scaling {ls}: C.lm_head would need the division (Granite)"
        impl = getattr(model.config, "_attn_implementation", None)
        assert impl == attn, f"attention {attn} required, got {impl}"
        linfo.update(attn_implementation=impl, load_seconds=round(time.time() - t0, 1))
    C = D.DB.context(dA, cfg, mc, model, tok)
    finish_context(C, key, cfg)
    C.load_info = linfo
    C.multi_device = len(linfo.get("device_map_devices") or []) > 1
    coh = None
    if paris and not random_model:
        coh = D.DB.coherence_check(C)
        assert coh["ok"], f"Paris check failed: {coh}"
    C.paris = coh
    C.label = label
    reset_peaks()
    return C


def finish_context(C, key, cfg):
    """Adapter fields on a dream_bind context (also used by the tests' tiny models): the spec / layout, the soft-capped
    head, the input device, dream_bind's generation settings and dream_select's first-token reference ids."""
    spec = SPECS[key]
    C.key, C.spec = key, spec
    C.layout = layout_from_config(C.model.config)
    C.sliding = any(l.kind == SLIDE for l in C.layout)
    C.window = next((l.window for l in C.layout if l.kind == SLIDE), None)
    cap = getattr(C.model.config, "final_logit_softcapping", None)
    if cap and not isinstance(C.lm_head, CappedHead):
        C.lm_head = CappedHead(C.lm_head, cap)
    C.softcap = cap
    C.dev = str(input_device(C))
    DB().gen_context(C, cfg)
    C.ref_ids = {w: int(DB().text_ids(C.tok, w)[0]) for w in ("I", "Sorry", "As")}
    return C


def random_model_for(key, tok, n_layers=None, seed=0, device="cpu", window=8):
    """A tiny random model of the spec's architecture with the real tokenizer's vocabulary (tests, CPU dry runs).
    Gemma 4: sliding + full layers with the real per-type structure (head_dim 16 / 32, KV 2 / 1, K = V on full layers,
    proportional partial RoPE 0.25 on full layers, soft-cap 30), MoE on, a small window to exercise the sliding mask."""
    spec = SPECS[key]
    torch.manual_seed(seed)
    if spec["arch"] == "qwen2":
        return DB().random_qwen2(tok, n_layers or 3, seed, device)
    if spec["arch"] == "llama":
        return DB().random_llama(tok, n_layers or 3, seed, device)
    if spec["arch"] == "gemma4":
        return tiny_gemma4(len(tok), n_layers=n_layers or 6, window=window, seed=seed).to(device)
    raise ValueError(f"no random model for {spec['arch']}")


def _tiny_plain(arch, vocab, n_layers, attn, seed):
    """dream_bind's tiny Qwen2 / Llama (hidden 64, 4 query / 2 KV heads) with a chosen attention implementation."""
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    Cfg, Cls = (Qwen2Config, Qwen2ForCausalLM) if arch == "qwen2" else (LlamaConfig, LlamaForCausalLM)
    torch.manual_seed(seed)
    c = Cfg(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=n_layers, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=4096, attn_implementation=attn)
    return Cls(c).float().eval()


def tiny_context(key, tok, n_layers=None, window=8, attn="eager", seed=0):
    """A dream_bind-style context of a tiny random model of SPECS[key]'s architecture with the real tokenizer (CPU):
    the tests' and the --random-model dry runs' model."""
    import yaml
    from seahorse.residual import decoder_layers, text_model
    spec = SPECS[key]
    cfg = yaml.safe_load(open(DS().DB_CFG))
    if spec["arch"] in ("qwen2", "llama"):
        model = (random_model_for(key, tok, n_layers, seed) if attn == "eager"
                 else _tiny_plain(spec["arch"], len(tok), n_layers or 3, attn, seed))
    else:
        model = tiny_gemma4(len(tok), n_layers or 6, window, seed, attn)
    if spec["chat"]["thinking"] and not getattr(tok, "_dc_thinking_off", False):
        thinking_off(tok, spec["chat"]["thinking"])
        tok._dc_thinking_off = True
    if spec["chat"]["date_string"] and not getattr(tok, "_dc_date", False):
        DB().date_pinned(tok, spec["chat"]["date_string"])
        tok._dc_date = True
    mc = {"model": spec["hf"], "n_blocks": len(decoder_layers(model)), "thinking_template": False, "text_only": False,
          "eager": attn == "eager", "date_string": spec["chat"]["date_string"]}
    C = SimpleNamespace(model=model, body=text_model(model), lm_head=model.lm_head, blocks=decoder_layers(model),
                        n=len(decoder_layers(model)), d=model.lm_head.in_features, dev="cpu", tok=tok, cfg=cfg, mc=mc,
                        tol=0.01, A=SimpleNamespace(tiny=True, random_model=spec["arch"], gen_set=None, model=key))
    finish_context(C, key, cfg)
    C.label, C.paris, C.load_info = spec["label"] + " (tiny random)", None, {"random_model": spec["arch"], "attn": attn}
    return C


def tiny_gemma4_config(vocab, n_layers=6, window=8, attn="eager"):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    types = [FULL if (i + 1) % 3 == 0 else SLIDE for i in range(n_layers)]
    types[-1] = FULL
    c = Gemma4TextConfig(
        vocab_size=vocab, vocab_size_per_layer_input=vocab, hidden_size=64, intermediate_size=64, moe_intermediate_size=32,
        num_experts=4, top_k_experts=2, enable_moe_block=True, num_attention_heads=4, num_key_value_heads=2,
        num_global_key_value_heads=1, head_dim=16, global_head_dim=32, attention_k_eq_v=True, num_hidden_layers=n_layers,
        layer_types=types, sliding_window=window, final_logit_softcapping=30.0, hidden_size_per_layer_input=0,
        max_position_embeddings=4096, tie_word_embeddings=True, use_bidirectional_attention="vision",
        rope_parameters={FULL: {"partial_rotary_factor": 0.25, "rope_theta": 1e6, "rope_type": "proportional"},
                         SLIDE: {"rope_theta": 1e4, "rope_type": "default"}},
        pad_token_id=0, bos_token_id=2, eos_token_id=1)
    c._attn_implementation = attn
    return c


def tiny_gemma4(vocab, n_layers=6, window=8, seed=0, attn="eager"):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(tiny_gemma4_config(vocab, n_layers, window, attn), attn_implementation=attn)
    return m.float().eval()


def reset_peaks():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def peak_gb():
    """{device: (allocated, reserved) GB} over every visible GPU (a 2-GPU model reports both)."""
    if not torch.cuda.is_available():
        return {}
    return {f"cuda:{i}": (round(torch.cuda.max_memory_allocated(i) / 2 ** 30, 2),
                          round(torch.cuda.max_memory_reserved(i) / 2 ** 30, 2)) for i in range(torch.cuda.device_count())}


# ----------------------------------------------------------------------------------------------- cache utilities


@dataclass
class Prefix:
    """A restored memory: kv [(K, V)] per layer ([1, n_kv_l, S, head_dim_l], K rotated at `pos`), pos [S] (the RoPE
    position of each entry, the same in every layer), start (the position the next turn starts at)."""
    kv: list
    pos: torch.Tensor
    start: int
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return int(self.pos.numel())

    def as_db(self):
        """dream_bind's (kv list, start) prefix."""
        return self.kv, self.start


def kv_list(pkv):
    """[(K, V)] per layer of a cache (dream_bind's kv_list; per-layer shapes as stored)."""
    return bind().kv_list(pkv)


def make_cache(kv, B=1):
    """A PLAIN DynamicCache holding kv for B rows (dream_bind's make_cache): full layers everywhere, so no entry of a
    sliding layer is ever evicted; masks come from pos_masks on sliding models."""
    return bind().make_cache(kv, B)


def _dev_idx(idx, t):
    return torch.as_tensor(idx, dtype=torch.long).to(t.device)


def index_kv(kv, idx):
    """kv entries `idx` (cache-slot indices) in every layer; multi-device safe (the index goes to each layer's device)."""
    return [(K[:, :, _dev_idx(idx, K)], V[:, :, _dev_idx(idx, V)]) for K, V in kv]


def index_prefix(P, idx, start=None):
    """The entries `idx` of a Prefix (at their stored positions); the next turn starts at P.start unless given."""
    ii = torch.as_tensor(idx, dtype=torch.long)
    return Prefix(index_kv(P.kv, ii), P.pos[ii].clone(), P.start if start is None else start, dict(P.meta))


def _rotary_call(C, x, pos, layer):
    """cos, sin of the layer's own rotary embedding at positions pos ([1, S]): Gemma 4's rotary takes the layer type;
    Qwen3.5's text mrope takes 3 position rows; the rest take (x, position_ids)."""
    rot = C.body.rotary_emb
    params = inspect.signature(rot.forward).parameters
    if "layer_type" in params:
        return rot(x, pos, C.model.config.layer_types[layer])
    if DB().is_hybrid(C):
        pos = pos[None].expand(3, *pos.shape)
    return rot(x, pos)


def _apply_rope(C, layer, K, cos, sin):
    blk = C.blocks[layer]
    mod = sys.modules[type(blk.self_attn).__module__]
    f = mod.apply_rotary_pos_emb
    names = list(inspect.signature(f).parameters)
    if names[:2] == ["q", "k"]:
        return f(K, K, cos, sin)[1]
    return f(K, cos.to(K.device), sin.to(K.device))


def rope_shift(C, K, delta, layer=None):
    """Stored keys K [B, n_kv, S, head_dim] of `layer` moved by `delta` positions with that layer's own rotary
    embedding (rotations compose additively; partial / proportional RoPE rotates only its rotated dims: the rest have
    frequency 0, cos 1, sin 0). layer None = dream_bind's rope_shift (the first attention layer's rotary: right for
    models with one rotary embedding)."""
    if layer is None:
        return DB().rope_shift(C, K, delta)
    S = K.shape[2]
    pos = torch.full((1, S), int(delta), dtype=torch.long, device=K.device)    # on K's device: the rotary moves
    cos, sin = _rotary_call(C, K, pos, layer)                                   # inv_freq to x's device (multi-GPU)
    return _apply_rope(C, layer, K, cos.to(K.device), sin.to(K.device))


def compose_prefix(C, parts, srcs, start):
    """A Prefix from several stored memories, in order: parts = [(source name, slot indices, RoPE shift)], srcs =
    {name: Prefix}. Per layer the parts' entries are concatenated, each part's keys rotated by its shift with that
    layer's rotary embedding (rope_shift per layer type). The next turn starts at `start`."""
    names = list(srcs)
    n_layers = len(srcs[names[0]].kv)
    out, pos = [], []
    for name, idx, shift in parts:
        pos.append(srcs[name].pos[torch.as_tensor(idx, dtype=torch.long)] + int(shift))
    for l in range(n_layers):
        Ks, Vs = [], []
        for name, idx, shift in parts:
            K, V = srcs[name].kv[l]
            K, V = K[:, :, _dev_idx(idx, K)], V[:, :, _dev_idx(idx, V)]
            Ks.append(rope_shift(C, K, shift, l) if shift else K)
            Vs.append(V)
        out.append((torch.cat(Ks, 2).contiguous(), torch.cat(Vs, 2).contiguous()))
    P = torch.cat(pos) if pos else torch.zeros(0, dtype=torch.long)
    assert bool((P[1:] > P[:-1]).all()) if P.numel() > 1 else True, "composed positions are not increasing"
    assert P.numel() == 0 or int(P.max()) < start, "a restored entry sits at or after the next turn's start"
    return Prefix(out, P, int(start))


def pos_masks(C, kv_pos, q_pos, B=1, dtype=torch.float32, device=None):
    """Additive masks [B, 1, Tq, Tk] from RoPE positions: full layers see kv_pos <= q_pos; sliding layers also need
    q_pos - kv_pos < window (transformers' sliding_window_overlay, by position instead of cache slot). Models with
    layer types (Gemma 4) get {layer type: mask}; others one 4D mask."""
    device = device or input_device(C)
    d = q_pos.to(device)[:, None] - kv_pos.to(device)[None, :]
    full = d >= 0
    mn = torch.finfo(dtype).min
    f = lambda m: torch.where(m, 0.0, mn).to(dtype)[None, None].expand(B, 1, -1, -1)
    if not C.sliding:
        return f(full)
    return {FULL: f(full), SLIDE: f(full & (d < C.window))}


def needs_pos_masks(C, prefix=None, q_end=None):
    """True on sliding models (always: masks from positions are exact; the slot masks are right only while every RoPE
    distance stays inside the window)."""
    return bool(C.sliding)


def harvest_kv(C, ids, prefix=None, attn=False):
    """One pass over ids (list / 1D tensor) after an optional Prefix, on a plain DynamicCache (nothing evicted):
    (Prefix of everything in the cache afterwards, hidden [T, d], attentions or None). Positions continue at
    prefix.start (0 without a prefix)."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    start = prefix.start if prefix else 0
    q_pos = torch.arange(start, start + len(ids))
    cache = make_cache(prefix.kv if prefix else [], 1)
    kw = {}
    if needs_pos_masks(C):
        kv_pos = torch.cat([prefix.pos, q_pos]) if prefix else q_pos
        kw["attention_mask"] = pos_masks(C, kv_pos, q_pos)
    dev = input_device(C)
    o = C.body(input_ids=ids[None].to(dev), past_key_values=cache, position_ids=q_pos[None].to(dev), use_cache=True,
               output_attentions=attn, **kw)
    kv = [(K.clone(), V.clone()) for K, V in kv_list(o.past_key_values)]
    pos = torch.cat([prefix.pos, q_pos]) if prefix else q_pos
    return Prefix(kv, pos, start + len(ids)), o.last_hidden_state[0], (o.attentions if attn else None)


def forward(C, ids, prefix=None, attn=False):
    """Teacher-forced pass of ids after an optional Prefix (not modified): (hidden [T, d], attentions or None)."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    start = prefix.start if prefix else 0
    q_pos = torch.arange(start, start + len(ids))
    cache = make_cache(prefix.kv if prefix else [], 1)
    kw = {}
    if needs_pos_masks(C):
        kv_pos = torch.cat([prefix.pos, q_pos]) if prefix else q_pos
        kw["attention_mask"] = pos_masks(C, kv_pos, q_pos)
    dev = input_device(C)
    o = C.body(input_ids=ids[None].to(dev), past_key_values=cache, position_ids=q_pos[None].to(dev), use_cache=True,
               output_attentions=attn, **kw)
    return o.last_hidden_state[0], (o.attentions if attn else None)


def logits(C, h):
    """fp32 logits (soft-capped where the model caps) of hidden states h; non-finite logits fail."""
    lg = C.lm_head(h).float()
    if not bool(torch.isfinite(lg).all()):
        raise FloatingPointError("non-finite logits")
    return lg


def answer_score(C, q_ids, a_ids, prefix=None):
    """log P(answer | prompt) teacher-forced (sum, mean per token), the answer starting the reply."""
    q_ids, a_ids = torch.as_tensor(q_ids, dtype=torch.long), torch.as_tensor(a_ids, dtype=torch.long)
    h, _ = forward(C, torch.cat([q_ids, a_ids]), prefix)
    rows = torch.arange(len(q_ids) - 1, len(q_ids) + len(a_ids) - 1, device=h.device)
    lp = torch.log_softmax(logits(C, h[rows]), -1)
    s = float(lp.gather(-1, a_ids.to(lp.device)[:, None])[:, 0].sum())
    return s, s / max(len(a_ids), 1)


def gen_batch(C, ids, rows, *, prefix=None, max_new, streams=None, first_ids=None, force_masks=False):
    """Generation after an optional Prefix. All-full-attention models on one device: dream_bind's gen_batch with
    (kv, start) unchanged. Sliding models, models spread over several GPUs (C.multi_device: dream_bind's loop keeps the
    logits on lm_head's device, which is not the input device there) or force_masks: the same loop (sampler,
    random-stream layout, eos handling) with position masks at every step and the logits moved to the input device."""
    if not (needs_pos_masks(C) or force_masks or getattr(C, "multi_device", False)):
        return DB().gen_batch(C, ids, rows, prefix=prefix.as_db() if prefix else None, max_new=max_new,
                              streams=streams, first_ids=first_ids)
    return _gen_masked(C, ids, rows, prefix=prefix, max_new=max_new, streams=streams, first_ids=first_ids)


def _gen_masked(C, ids, rows, *, prefix, max_new, streams, first_ids):
    """dream_bind.run.gen_batch's loop (no add / hybrid branches) with explicit position masks."""
    dev = input_device(C)
    ids = torch.as_tensor(ids, dtype=torch.long)
    B, T = len(rows), len(ids)
    start = prefix.start if prefix else 0
    cache = make_cache(prefix.kv if prefix else [], B)
    kv_pos = prefix.pos.clone() if prefix else torch.zeros(0, dtype=torch.long)
    x = ids[None].expand(B, -1).to(dev).contiguous()
    streams = streams or {}
    gens = {}
    for s, (seed, n_draw) in streams.items():
        g = torch.Generator(device=dev)
        g.manual_seed(int(seed))
        gens[s] = (g, n_draw)
    greedy = torch.tensor([bool(r[0]) for r in rows], device=dev)
    by_stream = {}
    for b, (gr, st) in enumerate(rows):
        if not gr:
            assert st is not None and st[0] in gens and 1 <= st[1] < gens[st[0]][1], f"row {b}: bad stream {st}"
            by_stream.setdefault(st[0], ([], []))
            by_stream[st[0]][0].append(b)
            by_stream[st[0]][1].append(st[1])
    by_stream = {s: (torch.tensor(bs, device=dev), torch.tensor(js, device=dev)) for s, (bs, js) in by_stream.items()}
    eos_t = torch.tensor(C.eos_list, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    seen = torch.zeros(B, C.vocab, dtype=torch.bool, device=dev) if C.sampler.presence else None
    ar = torch.arange(B, device=dev)
    steps, first = [], None
    for t in range(max_new):
        qp = torch.arange(start, start + T) if t == 0 else torch.tensor([start + T + t - 1])
        kv_pos = torch.cat([kv_pos, qp])
        M = pos_masks(C, kv_pos, qp, B)
        o = C.body(input_ids=x, past_key_values=cache, position_ids=qp[None].expand(B, -1).to(dev), use_cache=True,
                   attention_mask=M)
        cache = o.past_key_values
        lg = C.lm_head(o.last_hidden_state[:, -1]).float().to(dev)
        if not bool(torch.isfinite(lg[~done]).all()):
            raise FloatingPointError(f"non-finite logits at step {t}")
        if t == 0 and first_ids is not None:
            first = torch.log_softmax(lg, -1)[:, first_ids].cpu()
        u = torch.zeros(B, 1, dtype=torch.float64, device=dev)
        draws = {s: torch.rand(n, 1, generator=g, device=dev, dtype=torch.float64) for s, (g, n) in gens.items()}
        for s, (bs, js) in by_stream.items():
            u[bs] = draws[s][js]
        nxt = lg.argmax(-1)
        if not bool(greedy.all()):
            nxt = torch.where(greedy, nxt, C.sampler(lg, u, seen))
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
    eset = set(C.eos_list)
    out = []
    for b, row in enumerate(toks):
        k = next((i for i, tk in enumerate(row) if tk in eset), None)
        ids_ = row if k is None else row[:k]
        out.append({"ids": ids_, "n_tok": len(ids_), "capped": k is None,
                    "first": None if first is None else first[b].tolist()})
    return out


# ----------------------------------------------------------------------------------------------- sessions as turns


def session_pieces(tok, msgs, question=None):
    """Token pieces of [head][turn 1]...[turn n][question turn + generation prompt] for a chat of {role, content}
    messages (user / assistant; the first may not be a system message: it would replace the template's head). The head
    is dream_bind's (everything before the first user turn); each turn's tokens are the increment of the rendered
    prefix. Asserted: the pieces tokenise exactly as the whole rendered conversation. Returns (H, [turn ids], Q)."""
    DBm = DB()
    assert msgs and msgs[0]["role"] != "system", "a leading system message would replace the template head"
    head, _ = DBm.user_turn_start(tok)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids if s else []
    full_msgs = list(msgs) + ([{"role": "user", "content": question}] if question is not None else [])
    s_all = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=question is not None)
    assert s_all.startswith(head), (head[-40:], s_all[:80])
    cuts, prev = [], head
    for k in range(1, len(msgs) + 1):
        s_k = tok.apply_chat_template(list(msgs[:k]), tokenize=False, add_generation_prompt=False)
        if s_k.startswith(head) and s_all.startswith(s_k):
            cuts.append(s_k)
        else:   # a template that closes the last turn differently when it is last: cut at the next turn's header
            raise AssertionError(f"rendered prefix of {k} turns is not a prefix of the conversation: {s_k[-60:]!r}")
    pieces = []
    for s_k in cuts:
        pieces.append(s_k[len(prev):])
        prev = s_k
    qt = s_all[len(prev):]
    H = enc(head)
    T = [enc(p) for p in pieces]
    Q = enc(qt)
    assert H + [t for x in T for t in x] + Q == tok(s_all, add_special_tokens=False).input_ids, \
        "session pieces do not tokenise like the whole conversation"
    return H, T, Q


def show():
    for k, s in SPECS.items():
        L = layout(s)
        types = {}
        for l in L:
            types.setdefault(l.kind, []).append(l)
        desc = "; ".join(f"{t}: {len(v)} layers, {v[0].n_kv} KV x {v[0].head_dim}, rope {v[0].rope_type} "
                         f"theta {v[0].rope_theta:g} partial {v[0].partial}"
                         + (f", window {v[0].window}" if v[0].window else "") + (", K = V" if v[0].k_eq_v else "")
                         for t, v in types.items())
        print(f"{k:16s} {s['hf']:36s} {desc}; entries/pos {kv_entries_per_position(L)}; "
              f"{s['n_gpus']} GPU; ctx {s['max_context']}")


if __name__ == "__main__":
    if "--show" in sys.argv:
        show()
