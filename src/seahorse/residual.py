"""Model loading and residual-stream access via PyTorch forward hooks.

The hook point is the output of decoder block l (model.model.layers[l] for causal LMs; the
text decoder's layers for vision-language checkpoints such as Qwen3.5,
model.model.language_model.layers), i.e. the residual stream after that block's token mixer
and MLP. The same point is used for capture (writing) and injection (reading).
"""

from contextlib import contextmanager

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_TF_MAJOR = int(transformers.__version__.split(".")[0])
# parents of the decoder-layer ModuleList, tried in order (the first is every causal LM so far)
_TEXT_PATHS = ("model", "model.language_model", "language_model", "model.text_model")


def load_model(name, device="cuda", dtype=torch.float32):
    """Frozen model + tokenizer. Vision-language checkpoints (architecture *ForConditionalGeneration,
    e.g. Qwen3.5) are loaded with their own class and used through the text path only."""
    tok = AutoTokenizer.from_pretrained(name)
    archs = getattr(AutoConfig.from_pretrained(name), "architectures", None) or []
    cls = AutoModelForCausalLM
    if any(a.endswith("ForConditionalGeneration") for a in archs):
        from transformers import AutoModelForImageTextToText
        cls = AutoModelForImageTextToText
    kw = {"dtype": dtype} if _TF_MAJOR >= 5 else {"torch_dtype": dtype}
    model = cls.from_pretrained(name, **kw).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _get(obj, path):
    for a in path.split("."):
        obj = getattr(obj, a, None)
        if obj is None:
            return None
    return obj


def text_model(model):
    """The module that owns the decoder layers (embeddings -> layers -> final norm)."""
    for path in _TEXT_PATHS:
        m = _get(model, path)
        if m is not None and isinstance(getattr(m, "layers", None), torch.nn.ModuleList):
            return m
    raise AttributeError(f"no decoder layers found on {type(model).__name__}")


def decoder_layers(model):
    return text_model(model).layers


def _hidden(out):
    # transformers 4.x decoder layers return a tuple; newer versions a tensor
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


@contextmanager
def capture(model, layers):
    """Record the residual after each block in `layers` during a batch-1 forward.

    Yields a dict l -> [seq, d] tensor, filled once the forward pass has run.
    """
    store = {}
    blocks = decoder_layers(model)
    handles = []
    for l in layers:
        def hook(module, args, out, l=l):
            h = _hidden(out)
            assert h.shape[0] == 1, "capture expects batch size 1"
            store[l] = h[0].detach().clone()
        handles.append(blocks[l].register_forward_hook(hook))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def inject(model, layer, memory, alpha):
    """Apply h <- memory.read(h, alpha) at every position after block `layer`."""
    def hook(module, args, out):
        return _replace(out, memory.read(_hidden(out), alpha))

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
