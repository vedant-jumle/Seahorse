"""Fast-weight memory acting on the residual stream at one layer.

read:  h <- h + alpha * M k(h)
write: e = delta - M k ;  M <- M + eta * g * e k^T        (delta rule)
key:   k(h) = normalize(h - mu)

With unit keys and eta=1, a write makes M reproduce `delta` exactly for that key,
and repeating the same write produces zero error (habituation).
"""

import torch
import torch.nn.functional as F


class FastWeightMemory:
    def __init__(self, d, mu, device=None, dtype=torch.float32, eps=1e-6):
        self.mu = mu.to(device=device, dtype=dtype)
        self.M = torch.zeros(d, d, device=device, dtype=dtype)
        self.eps = eps

    def key(self, h):
        """[..., d] residual -> [..., d] unit key."""
        return F.normalize(h - self.mu, dim=-1, eps=self.eps)

    def predict(self, k):
        """[..., d] keys -> [..., d] recalled deltas (M k)."""
        return k @ self.M.T

    def read(self, h, alpha):
        return h + alpha * self.predict(self.key(h))

    def write(self, delta, h_key, gate=None, eta=1.0):
        """Sequential delta-rule write, one token at a time in order.

        delta, h_key: [T, d]; gate: [T] in [0, 1] or None (= 1).
        Returns (error_norms, delta_norms), each [T]: how much of each delta the
        memory could not already predict.
        """
        keys = self.key(h_key)
        err_norms = torch.empty(keys.shape[0])
        delta_norms = delta.norm(dim=-1).cpu()
        for t in range(keys.shape[0]):
            k = keys[t]
            e = delta[t] - self.M @ k
            g = 1.0 if gate is None else float(gate[t])
            self.M += (eta * g) * torch.outer(e, k)
            err_norms[t] = e.norm()
        return err_norms, delta_norms
