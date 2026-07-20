"""
havq/model/vq.py
================
LAPA-style latent-action VQ over frozen V-JEPA clip embeddings, quantized with
NSVQ (Noise Substitution in Vector Quantization; Vali & Bäckström 2022) -- the
same quantizer LAPA uses. NOT classical VQ-VAE: there is no straight-through
estimator, no commitment loss, no EMA. The only training loss is reconstruction.

Per training pair (v_t, v_{t+H}), where v_* are L2-normalized clip embeddings:

    d          = f_enc([v_t ; v_{t+H}])           # MLP -> latent_dim
    z, idx     = quantize(d)                       # nearest codebook entry
    v_hat      = f_dec([v_t ; z])                  # MLP -> reconstructed v_{t+H}
    loss       = MSE(v_hat, v_{t+H})

Because the decoder already receives v_t, the low-capacity code z can only earn
its keep by carrying the *transition* v_t -> v_{t+H} -- i.e. the latent action.

NSVQ quantization (training)
----------------------------
Pick the hard nearest codebook entry q = argmin_k ||d - c_k||, then substitute
the (non-differentiable) quantization step with additive noise of *matched norm*:

    z = d + ||d - q|| * (n / ||n||),   n ~ N(0, I)

so E[||z - d||^2] = ||d - q||^2. This is fully differentiable: gradients reach
the encoder through d, and the codebook through the ||d - q|| term. At inference
(`encode_indices`) we drop the noise and use the hard assignment idx directly.

Dead-codebook replacement
-------------------------
Codes unused since the last reset are periodically overwritten with recent
encoder outputs (+ small jitter). The training loop drives the schedule via
config (`nsvq.replace_dead_codes_every`, `nsvq.replacement_warmup_batches`).

Copied and adapted from archive/v1-derisk/havq/vq.py; this repo stays
code-independent (no cross-module imports of the archived version).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def l2_normalize(V: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each row of a (N, D) array. Used identically at train and
    tokenize time so the codebook sees the same input distribution."""
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + eps)


def perplexity(idx: torch.Tensor, codebook_size: int) -> float:
    """Codebook perplexity exp(-sum p log p) over a batch of hard assignments.
    1.0 = collapse to a single code; codebook_size = perfectly uniform usage."""
    counts = torch.bincount(idx, minlength=codebook_size).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    dims = [in_dim] + list(hidden)
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers += [nn.Linear(dims[-1], out_dim)]
    return nn.Sequential(*layers)


class NSVQ(nn.Module):
    """Encoder + NSVQ codebook + decoder over (v_t, v_{t+H}) embedding pairs.

    Parameters
    ----------
    in_dim        : V-JEPA embedding dim D (encoder sees 2*D, decoder sees D+latent_dim)
    latent_dim    : f_enc output dim = codebook vector dim
    codebook_size : number of discrete codes |K|
    enc_hidden    : f_enc hidden widths
    dec_hidden    : f_dec hidden widths
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        codebook_size: int,
        enc_hidden: list[int],
        dec_hidden: list[int],
    ):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

        self.f_enc = _mlp(2 * in_dim, enc_hidden, latent_dim)
        self.f_dec = _mlp(in_dim + latent_dim, dec_hidden, in_dim)
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.normal_(self.codebook.weight, std=latent_dim ** -0.5)

        # codes used since the last replace_dead_codes() reset
        self.register_buffer("code_usage", torch.zeros(codebook_size, dtype=torch.long))
        # most recent batch of encoder outputs, used as the replacement pool
        self._replacement_pool: torch.Tensor | None = None

    def encode(self, vt: torch.Tensor, vtH: torch.Tensor) -> torch.Tensor:
        return self.f_enc(torch.cat([vt, vtH], dim=-1))

    def decode(self, vt: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.f_dec(torch.cat([vt, z], dim=-1))

    def quantize(self, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """d (B, latent_dim) -> (z, idx). Training: NSVQ noise substitution.
        Eval: hard nearest code."""
        with torch.no_grad():
            idx = torch.cdist(d, self.codebook.weight).argmin(dim=1)  # (B,)
        q = self.codebook(idx)  # (B, latent_dim), differentiable wrt codebook

        if self.training:
            self.code_usage.index_add_(0, idx, torch.ones_like(idx))
            self._replacement_pool = d.detach()
            n = torch.randn_like(d)
            n = n / (n.norm(dim=1, keepdim=True) + 1e-8)
            err = (d - q).norm(dim=1, keepdim=True)
            z = d + err * n
        else:
            z = q
        return z, idx

    def forward(self, vt: torch.Tensor, vtH: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.encode(vt, vtH)
        z, idx = self.quantize(d)
        vhat = self.decode(vt, z)
        return vhat, idx

    @torch.no_grad()
    def encode_indices(self, vt: torch.Tensor, vtH: torch.Tensor) -> torch.Tensor:
        """Inference tokenization: hard code index per pair, no noise."""
        d = self.encode(vt, vtH)
        return torch.cdist(d, self.codebook.weight).argmin(dim=1)

    @torch.no_grad()
    def replace_dead_codes(self) -> int:
        """Overwrite codes unused since the last reset with recent encoder
        outputs (+ jitter), then reset usage. Returns the number replaced."""
        dead = (self.code_usage == 0).nonzero(as_tuple=True)[0]
        n_dead = int(dead.numel())
        pool = self._replacement_pool
        if n_dead > 0 and pool is not None and pool.shape[0] > 0:
            sel = torch.randint(0, pool.shape[0], (n_dead,), device=pool.device)
            new = pool[sel] + 1e-2 * torch.randn(n_dead, self.latent_dim, device=pool.device)
            self.codebook.weight.data[dead] = new.to(self.codebook.weight.dtype)
        self.code_usage.zero_()
        return n_dead


if __name__ == "__main__":
    # shape / differentiability smoke test on random data
    torch.manual_seed(0)
    D, B = 1408, 32
    m = NSVQ(in_dim=D, latent_dim=64, codebook_size=256, enc_hidden=[512, 256], dec_hidden=[512, 512])
    vt, vtH = torch.randn(B, D), torch.randn(B, D)
    vhat, idx = m(vt, vtH)
    loss = torch.nn.functional.mse_loss(vhat, vtH)
    loss.backward()
    assert vhat.shape == (B, D) and idx.shape == (B,)
    assert m.codebook.weight.grad is not None and m.f_enc[0].weight.grad is not None
    print(f"ok: vhat {tuple(vhat.shape)}, idx {tuple(idx.shape)}, loss {loss.item():.4f}, "
          f"perplexity {perplexity(idx, 256):.1f}, replaced {m.replace_dead_codes()} dead codes")
