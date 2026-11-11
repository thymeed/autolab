"""Small selective diagonal SSM language model (Mamba-1 style, sequential scan).

Parameters are well under 20M at the defaults. The scan is a plain Python loop so a
hook can act on the recurrent state h (B, d_inner, d_state) after EVERY update, which
is exactly where QUESTION.md applies quantization. Slow for L=1024 but simple and
exact; a chunked parallel scan for training is a later optimisation.
A "channel" = one index along d_inner (it owns d_state state entries).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tasks import VOCAB


class SSMBlock(nn.Module):
    def __init__(self, d_model, expand=2, d_state=16, d_conv=4):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(1, d_model // 16)
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                              groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        # dt bias so that softplus(bias) is log-uniform in [1e-3, 1e-1]
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x, hook=None):
        B, L, _ = x.shape
        xi, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        xi = F.silu(self.conv(xi.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.x_proj(xi),
                                 [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                 # (B, L, d_inner)
        A = -torch.exp(self.A_log)                        # (d_inner, d_state)
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            a = torch.exp(dt[:, t, :, None] * A)          # (B, d_inner, d_state)
            b = dt[:, t, :, None] * Bm[:, t, None, :] * xi[:, t, :, None]
            h = a * h + b
            if hook is not None:
                h = hook(h)
            ys.append((h * Cm[:, t, None, :]).sum(-1))    # (B, d_inner)
        y = torch.stack(ys, dim=1) + xi * self.D
        return x + self.out_proj(y * F.silu(z))


class SSMLM(nn.Module):
    def __init__(self, d_model=192, n_layers=4, expand=2, d_state=16):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        self.blocks = nn.ModuleList(SSMBlock(d_model, expand, d_state) for _ in range(n_layers))
        self.norm_f = nn.LayerNorm(d_model)
        self.d_inner = expand * d_model

    def forward(self, tokens, hooks=None):
        x = self.embed(tokens)
        for i, blk in enumerate(self.blocks):
            x = blk(x, None if hooks is None else hooks[i])
        return self.norm_f(x) @ self.embed.weight.T       # tied output head

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
