import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    selective_scan_fn = None
    HAS_MAMBA_CUDA = False


def pure_pytorch_selective_scan(u, delta, A, B, C, D=None):
    """
    Memory-efficient Pure PyTorch sequential fallback for selective scan.
    Computes discretization per-step to reduce peak VRAM from O(B*L*D*N) to O(B*D*N).

    Args:
        u: (B, L, D) input sequence
        delta: (B, L, D) step sizes
        A: (D, N) state matrix
        B: (B, L, N) input matrix
        C: (B, L, N) output matrix
        D: (D,) optional skip connection parameter
    Returns:
        y: (B, L, D) output sequence
    """
    batch_size, seq_len, d_model = u.shape
    d_state = A.shape[1]

    # Compute discretization in FP32 for numerical stability
    u_f32 = u.float()
    delta_f32 = delta.float()
    A_f32 = A.float()
    B_f32 = B.float()
    C_f32 = C.float()

    h = torch.zeros(batch_size, d_model, d_state, device=u.device, dtype=torch.float32)
    ys = []

    for t in range(seq_len):
        # deltaA_t: (B, D, N)
        deltaA_t = torch.exp(delta_f32[:, t].unsqueeze(-1) * A_f32.unsqueeze(0))
        # deltaB_u_t: (B, D, N)
        deltaB_u_t = (delta_f32[:, t] * u_f32[:, t]).unsqueeze(-1) * B_f32[:, t].unsqueeze(1)
        h = deltaA_t * h + deltaB_u_t
        # y_t: (B, D)
        y_t = (h * C_f32[:, t].unsqueeze(1)).sum(-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)
    if D is not None:
        y = y + u_f32 * D.float().unsqueeze(0).unsqueeze(0)

    return y.to(dtype=u.dtype)


class SelectiveSSM(nn.Module):
    """
    Dual-Engine Selective State Space Model block.
    Uses fused CUDA selective_scan_fn if installed and on CUDA; otherwise falls back to pure PyTorch.
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: str | int = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # Projections for selective parameters
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Initialize S4 parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        # dt_proj initialization
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize dt bias to ~0.001 to 0.1
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        batch_size, seq_len, d_model = x.shape
        A = -torch.exp(self.A_log.float())  # (D, N)

        # x_proj outputs dt_rank + B + C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt_part = x_dbl[:, :, : self.dt_rank]
        B_part = x_dbl[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C_part = x_dbl[:, :, self.dt_rank + self.d_state :]

        dt = F.softplus(self.dt_proj(dt_part))  # (B, L, D)

        if HAS_MAMBA_CUDA and x.is_cuda and selective_scan_fn is not None:
            # mamba_ssm expects (B, D, L)
            u_t = x.transpose(1, 2).contiguous()
            delta_t = dt.transpose(1, 2).contiguous()
            B_t = B_part.transpose(1, 2).contiguous()
            C_t = C_part.transpose(1, 2).contiguous()
            out = selective_scan_fn(
                u_t,
                delta_t,
                A,
                B_t,
                C_t,
                self.D.float(),
                z=None,
                delta_bias=None,
                delta_softplus=False,
            )
            return out.transpose(1, 2).contiguous().to(dtype=x.dtype)
        else:
            return pure_pytorch_selective_scan(x, dt, A, B_part, C_part, self.D)
