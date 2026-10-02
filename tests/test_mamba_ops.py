import pytest
import torch
from detector.core.models.encoders.mamba_ops import SelectiveSSM, pure_pytorch_selective_scan

def test_selective_ssm_shape_and_grad():
    torch.manual_seed(42)
    B, L, D = 4, 16, 16
    x = torch.randn(B, L, D, requires_grad=True)
    ssm = SelectiveSSM(d_model=D, d_state=16)
    out = ssm(x)
    assert out.shape == (B, L, D)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_pure_pytorch_scan_numerical_finite():
    torch.manual_seed(42)
    B, L, D, N = 2, 8, 8, 16
    u = torch.randn(B, L, D)
    delta = torch.rand(B, L, D) * 0.1
    A = -torch.rand(D, N)
    B_tensor = torch.randn(B, L, N)
    C_tensor = torch.randn(B, L, N)
    D_tensor = torch.randn(D)
    
    y = pure_pytorch_selective_scan(u, delta, A, B_tensor, C_tensor, D_tensor)
    assert y.shape == (B, L, D)
    assert torch.isfinite(y).all()
