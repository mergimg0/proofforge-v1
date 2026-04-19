"""Muon optimizer: HeavyBall momentum + Newton-Schulz orthogonal projection.

Adapted from PufferLib 4.0 muon.py (lines 1-135). Removes PufferLib-specific
imports. Pure PyTorch implementation.

LoRA-specific considerations:
  LoRA matrices are shaped (r, hidden_dim) or (hidden_dim, r) where r=16.
  Newton-Schulz iteration on these small matrices is computationally cheap
  (5 iterations of small matrix multiplies). The orthogonal projection
  prevents rank degeneracy — multiple ranks learning the same direction —
  which wastes capacity in low-rank adapters.
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

__all__ = ["Muon"]

# Newton-Schulz iteration coefficients for 5 steps.
# Adapted from PufferLib muon.py:14-20.
NS_COEFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


def zeropower_via_newtonschulz5(G: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute the orthogonal projection of G via Newton-Schulz iteration.

    Projects the gradient onto the nearest orthogonal matrix. This prevents
    rank degeneracy in LoRA adapters by ensuring gradient directions are
    maximally spread across the available ranks.

    Adapted from PufferLib muon.py:22-41.
    """
    G = G.clone()
    x = G
    if G.size(-2) > G.size(-1):
        x = x.mT

    x = x / torch.clamp(G.norm(dim=(-2, -1)), min=eps)

    for a, b, c in NS_COEFS:
        s = x @ x.mT
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    if G.size(-2) > G.size(-1):
        x = x.mT

    return x.to(G.dtype)


class Muon(Optimizer):
    """Muon optimizer: HeavyBall + Newton-Schulz orthogonal projection.

    For 2D+ parameters (e.g., LoRA weight matrices), gradients are projected
    onto the nearest orthogonal matrix before the update step. 1D parameters
    (biases, norms) receive standard momentum SGD.

    Adapted from PufferLib muon.py:43-135.

    Args:
        params: Parameters to optimize.
        lr: Learning rate (default: 0.0025).
        weight_decay: L2 weight decay (default: 0.0).
        momentum: HeavyBall momentum coefficient (default: 0.95).
        eps: Numerical stability epsilon (default: 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 0.0025,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:  # type: ignore[union-attr]
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if momentum < 0.0:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "eps": eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Adapted from PufferLib muon.py:84-135.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )

                buf = state["momentum_buffer"]
                buf.mul_(momentum)
                buf.add_(grad)
                grad = grad + buf * momentum

                if grad.ndim >= 2:
                    grad = grad.view(grad.shape[0], -1)
                    grad = zeropower_via_newtonschulz5(grad)
                    grad *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5

                p.mul_(1 - lr * weight_decay)
                p.sub_(lr * grad.view(p.shape))

        return loss
