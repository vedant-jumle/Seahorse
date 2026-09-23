"""Model loading and residual-stream access via PyTorch forward hooks.

The hook point is the output of decoder block l (model.model.layers[l]), i.e. the
residual stream after that block's attention and MLP. The same point is used for
capture (writing) and injection (reading).
"""

from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name, device="cuda", dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def decoder_layers(model):
    return model.model.layers


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
