import torch

from seahorse.memory import FastWeightMemory

D = 16


def make(mu=None):
    return FastWeightMemory(D, torch.zeros(D) if mu is None else mu)


def test_exact_recall_of_written_key():
    torch.manual_seed(0)
    mem = make(mu=torch.randn(D))
    h, delta = torch.randn(1, D), torch.randn(1, D)
    mem.write(delta, h)
    assert torch.allclose(mem.predict(mem.key(h)), delta, atol=1e-5)


def test_read_adds_recalled_delta_scaled_by_alpha():
    torch.manual_seed(1)
    mem = make()
    h, delta = torch.randn(1, D), torch.randn(1, D)
    mem.write(delta, h)
    assert torch.equal(mem.read(h, 0.0), h)
    assert torch.allclose(mem.read(h, 2.0), h + 2.0 * delta, atol=1e-5)


def test_orthogonal_keys_do_not_interfere():
    mem = make()
    h = torch.eye(D)[:2]
    delta = torch.randn(2, D)
    mem.write(delta, h)
    assert torch.allclose(mem.predict(mem.key(h)), delta, atol=1e-5)


def test_repeat_write_has_zero_error():
    torch.manual_seed(2)
    mem = make()
    h, delta = torch.randn(1, D), torch.randn(1, D)
    first, _ = mem.write(delta, h)
    second, _ = mem.write(delta, h)
    assert first[0] > 0.1
    assert second[0] < 1e-5


def test_contradiction_overwrites():
    torch.manual_seed(3)
    mem = make()
    h = torch.randn(1, D)
    old, new = torch.randn(1, D), torch.randn(1, D)
    mem.write(old, h)
    mem.write(new, h)
    assert torch.allclose(mem.predict(mem.key(h)), new, atol=1e-5)


def test_zero_gate_writes_nothing():
    torch.manual_seed(4)
    mem = make()
    mem.write(torch.randn(3, D), torch.randn(3, D), gate=torch.zeros(3))
    assert torch.count_nonzero(mem.M) == 0
