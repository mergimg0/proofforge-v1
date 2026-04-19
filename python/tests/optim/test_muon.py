"""Tests for proofforge.optim.muon.

Tests Muon optimizer step, orthogonal projection, and momentum buffer.
Requires torch but NOT gpytorch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from proofforge.optim.muon import Muon, zeropower_via_newtonschulz5


class TestNewtonSchulz:
    def test_preserves_gradient_scale(self):
        """Orthogonal projection should not collapse gradient magnitude."""
        G = torch.randn(16, 128)
        G_proj = zeropower_via_newtonschulz5(G)
        # Frobenius norm of projected gradient should be close to
        # sqrt(min(m, n)) for a well-conditioned orthogonal matrix
        expected_norm = G.size(-2) ** 0.5  # 16^0.5 = 4.0
        actual_norm = G_proj.norm().item()
        assert actual_norm > 0.5, "Projection collapsed gradient to near-zero"
        # The norm should be roughly consistent, not wildly different
        assert actual_norm < expected_norm * 5, "Projection inflated gradient abnormally"

    def test_output_shape(self):
        """Output shape must match input shape."""
        for shape in [(4, 32), (32, 4), (16, 16), (8, 128)]:
            G = torch.randn(*shape)
            G_proj = zeropower_via_newtonschulz5(G)
            assert G_proj.shape == G.shape

    def test_idempotent_approx(self):
        """Applying projection twice should give similar result (near-orthogonal)."""
        G = torch.randn(16, 64)
        proj1 = zeropower_via_newtonschulz5(G)
        proj2 = zeropower_via_newtonschulz5(proj1)
        # Already-orthogonal input should be nearly unchanged
        diff = (proj1 - proj2).norm() / (proj1.norm() + 1e-8)
        assert diff < 0.15, f"Projection not stable: relative diff = {diff:.4f}"


class TestMuonOptimizer:
    def test_step_reduces_loss(self):
        """A few Muon steps should reduce a simple quadratic loss."""
        torch.manual_seed(42)
        param = torch.nn.Parameter(torch.randn(8, 32))
        target = torch.randn(8, 32)
        optimizer = Muon([param], lr=0.01, momentum=0.95)

        initial_loss = ((param - target) ** 2).sum().item()
        for _ in range(10):
            optimizer.zero_grad()
            loss = ((param - target) ** 2).sum()
            loss.backward()
            optimizer.step()
        final_loss = ((param - target) ** 2).sum().item()

        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_momentum_buffer_initialization(self):
        """Momentum buffers should be created on first step."""
        param = torch.nn.Parameter(torch.randn(4, 8))
        optimizer = Muon([param], lr=0.01, momentum=0.95)

        assert len(optimizer.state) == 0

        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()

        assert param in optimizer.state
        assert "momentum_buffer" in optimizer.state[param]
        buf = optimizer.state[param]["momentum_buffer"]
        assert buf.shape == param.shape

    def test_1d_params_no_projection(self):
        """1D parameters (biases) should get standard momentum, not NS projection."""
        bias = torch.nn.Parameter(torch.randn(32))
        optimizer = Muon([bias], lr=0.01, momentum=0.95)

        initial = bias.clone().detach()
        loss = (bias ** 2).sum()
        loss.backward()
        optimizer.step()

        # Bias should have moved (gradient was non-zero)
        assert not torch.allclose(bias, initial)

    def test_weight_decay(self):
        """Weight decay should shrink parameter norms."""
        param = torch.nn.Parameter(torch.ones(4, 8) * 10.0)
        optimizer = Muon([param], lr=0.01, weight_decay=0.1, momentum=0.0)

        initial_norm = param.norm().item()
        # Zero gradients, only weight decay should act
        loss = torch.tensor(0.0, requires_grad=True)
        loss.backward()
        optimizer.step()

        assert param.norm().item() < initial_norm

    def test_invalid_lr(self):
        param = torch.nn.Parameter(torch.randn(4))
        with pytest.raises(ValueError, match="Learning rate"):
            Muon([param], lr=-0.01)

    def test_invalid_momentum(self):
        param = torch.nn.Parameter(torch.randn(4))
        with pytest.raises(ValueError, match="momentum"):
            Muon([param], lr=0.01, momentum=-1.0)
