import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from seahorse.memory import FastWeightMemory
from seahorse.residual import capture, decoder_layers, inject

LAYER = 1


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    ids = torch.randint(0, 128, (1, 10))
    return model, ids


def random_memory(d, scale=1.0):
    mem = FastWeightMemory(d, torch.zeros(d))
    mem.M = scale * torch.randn(d, d)
    return mem


def test_capture_matches_hidden_states(tiny):
    model, ids = tiny
    with torch.no_grad():
        with capture(model, [LAYER]) as store:
            out = model(ids, output_hidden_states=True)
    assert torch.allclose(store[LAYER], out.hidden_states[LAYER + 1][0], atol=1e-6)


def test_inject_alpha_zero_and_empty_memory_are_identity(tiny):
    model, ids = tiny
    d = model.config.hidden_size
    with torch.no_grad():
        ref = model(ids).logits
        with inject(model, LAYER, random_memory(d), alpha=0.0):
            a0 = model(ids).logits
        with inject(model, LAYER, FastWeightMemory(d, torch.zeros(d)), alpha=1.0):
            empty = model(ids).logits
    assert torch.equal(ref, a0)
    assert torch.equal(ref, empty)


def test_inject_changes_logits_and_is_removed(tiny):
    model, ids = tiny
    d = model.config.hidden_size
    with torch.no_grad():
        ref = model(ids).logits
        with inject(model, LAYER, random_memory(d), alpha=1.0):
            steered = model(ids).logits
        after = model(ids).logits
    assert not torch.allclose(ref, steered)
    assert torch.equal(ref, after)
    assert len(decoder_layers(model)[LAYER]._forward_hooks) == 0


def test_inject_only_affects_layers_at_and_after(tiny):
    model, ids = tiny
    d = model.config.hidden_size
    with torch.no_grad():
        with capture(model, [0, 2]) as ref:
            model(ids)
        with inject(model, LAYER, random_memory(d), alpha=1.0):
            with capture(model, [0, 2]) as steered:
                model(ids)
    assert torch.equal(ref[0], steered[0])
    assert not torch.allclose(ref[2], steered[2])
