# -*- coding: utf-8 -*-
from typing import Optional
import math
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):  # (..., D)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class FastSelectiveSSMBlock(nn.Module):
    """
    改动点（相对你目前的版本）：
      ✅ 双向扫描（forward + backward 再平均）
      ✅ 通道混合残差（补“空间/通道”交互）
      ✅ 残差门改为 GLU 风格（更稳更 expressive）
      ✅ A_log 多尺度初始化
      ✅ 保留 log-domain 前缀积与数值护栏（稳定）
    I/O: (B, L, d_model)
    """
    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_state: int = 64,
        dt_rank: int = 16,
        conv_kernel: int = 3,
        conv_dilation: int = 1,
        causal_conv: bool = True,
        use_rmsnorm: bool = True,
        dropout: float = 0.1,        # 小幅增大默认 dropout
        # 数值护栏
        beta_cap: float = 8.0,       # 更宽松的 Δ 上界
        log_clip: float = 20.0,
        eps: float = 1e-6,
        # B/C 缩放（1.0 接近你原版）
        b_scale: float = 1.0,
        c_scale: float = 1.0,
        # 双向启用开关
        bidirectional: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_state = n_state
        self.dt_rank = dt_rank
        self.beta_cap = beta_cap
        self.log_clip = log_clip
        self.eps = eps
        self.b_scale = b_scale
        self.c_scale = c_scale
        self.bidirectional = bidirectional

        # 归一化
        self.in_norm = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

        # 1) 线性映射到 2*d_inner，拆成主干与门控残差输入
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=True)

        # 2) 因果 depthwise Conv1d（局部高频）
        self.dw_conv = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=conv_kernel, groups=d_inner,
            dilation=conv_dilation,
            padding=(conv_kernel - 1) * conv_dilation if causal_conv else (conv_kernel // 2) * conv_dilation,
            bias=True
        )

        # 3) 生成 delta/B/C
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * n_state, bias=True)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)  # -> delta (B,L,d_inner)

        # 4) 连续参数 A 的对角化；D 跳连
        self.A_log = nn.Parameter(torch.empty(d_inner, n_state))
        self.D = nn.Parameter(torch.zeros(d_inner))

        # 5) 残差门（GLU）的参数：把 res 投到 2*d_inner，做 v * silu(g)
        self.res_glu = nn.Linear(d_inner, 2 * d_inner, bias=True)

        # 6) 通道混合残差（补“空间/通道”交互）
        self.chan_mix = nn.Sequential(
            RMSNorm(d_inner) if use_rmsnorm else nn.LayerNorm(d_inner),
            nn.Linear(d_inner, d_inner, bias=True),
            nn.Dropout(dropout),
        )

        # 7) 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight); nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.dw_conv.weight, gain=0.5); nn.init.zeros_(self.dw_conv.bias)
        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.5); nn.init.zeros_(self.x_proj.bias)
        nn.init.xavier_uniform_(self.dt_proj.weight, gain=0.5); nn.init.zeros_(self.dt_proj.bias)
        nn.init.xavier_uniform_(self.res_glu.weight, gain=0.5); nn.init.zeros_(self.res_glu.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.8); nn.init.zeros_(self.out_proj.bias)
        with torch.no_grad():
            # A_log 多尺度初始化：0.3~1.0，对应 A=-exp(A_log) ∈ (-e^1, -e^0.3)≈(-2.7, -1.35)
            t = torch.linspace(0, 1, steps=self.n_state)
            init = 0.3 + 0.7 * t
            self.A_log.copy_(init.log().unsqueeze(0).repeat(self.d_inner, 1))
            self.D.zero_()

    @torch.no_grad()
    def _causal_crop(self, x_conv: torch.Tensor, L: int) -> torch.Tensor:
        return x_conv[..., :L]

    # -------- SSM 核心：一次扫描（正向或反向用同一实现）---------
    def _scan_once(self, x_conv: torch.Tensor) -> torch.Tensor:
        """
        x_conv: (B, L, d_inner)
        return: (B, L, d_inner)
        """
        B, L, Din = x_conv.shape
        # 取特征 → (Δ, B_raw, C_raw)
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2n)
        delta_feat, B_raw, C_raw = torch.split(
            x_dbl, [self.dt_rank, self.n_state, self.n_state], dim=-1
        )

        # Δ：FP32、softplus 上界；(B,L,d_inner)
        delta32 = F.softplus(self.dt_proj(delta_feat.to(torch.float32)))
        if self.beta_cap is not None:
            delta32 = torch.clamp(delta32, max=self.beta_cap)
        delta = delta32  # FP32 保持更稳

        # B/C：tanh 缩放（避免初期过大）
        Bn = torch.tanh(B_raw).to(torch.float32) * self.b_scale   # (B,L,n)
        Cn = torch.tanh(C_raw).to(torch.float32) * self.c_scale   # (B,L,n)

        # A = -exp(A_log) < 0
        A = -torch.exp(self.A_log.to(torch.float32))              # (d_inner,n)

        # alpha = exp(delta * A)
        deltaA32 = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,d,n)

        # 输入项：delta * u * B
        u = x_conv.to(torch.float32)                               # (B,L,d)
        deltaB_u32 = (delta * u).unsqueeze(-1) * Bn.unsqueeze(2)  # (B,L,d,n)

        # log-domain 前缀积/和
        eps = self.eps
        log_clip = self.log_clip

        log_alpha = torch.log(torch.clamp(deltaA32, min=eps))     # (B,L,d,n)
        log_p_incl = torch.cumsum(log_alpha, dim=1)
        log_p_incl = torch.clamp(log_p_incl, min=-log_clip, max=log_clip)
        p_incl = torch.exp(log_p_incl)

        zero = torch.zeros(B, 1, Din, self.n_state, dtype=torch.float32, device=x_conv.device)
        log_p_excl = torch.cat([zero, log_p_incl[:, :-1, :, :]], dim=1)
        log_p_excl = torch.clamp(log_p_excl, min=-log_clip, max=log_clip)
        p_excl = torch.exp(log_p_excl) + eps

        z = deltaB_u32 / p_excl
        csum = torch.cumsum(z, dim=1)
        s_after32 = p_incl * csum

        # y = Σ_n s * C + D ⊙ u
        y32 = torch.einsum('bldn,bln->bld', s_after32, Cn)
        y32 = y32 + u * self.D.to(torch.float32).view(1, 1, -1)

        return y32.to(x_conv.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model) -> (B, L, d_model)
        """
        B, L, Dm = x.shape
        assert Dm == self.d_model

        # 归一化 + 线性
        h = self.in_norm(x)
        x_and_res = self.in_proj(h)                        # (B, L, 2*d_inner)
        x_proj_part, res = torch.split(x_and_res, self.d_inner, dim=-1)

        # 因果 DWConv
        x_conv = x_proj_part.transpose(1, 2)              # (B, d_inner, L)
        x_conv = self.dw_conv(x_conv)                     # (B, d_inner, L+pad)
        x_conv = self._causal_crop(x_conv, L)             # (B, d_inner, L)
        x_conv = x_conv.transpose(1, 2)                   # (B, L, d_inner)

        # SSM：双向扫描（可关）
        y_f = self._scan_once(x_conv)                     # (B, L, d_inner)
        if self.bidirectional:
            x_rev = torch.flip(x_conv, dims=[1])
            y_b = torch.flip(self._scan_once(x_rev), dims=[1])
            y = 0.5 * (y_f + y_b)
        else:
            y = y_f

        y = self.drop(y)

        # 残差门：GLU 风格（更稳、比纯乘法更容易训练）
        v, g = self.res_glu(res).chunk(2, dim=-1)         # (B,L,d_inner) x2
        y = y + v * F.silu(g)

        # 通道混合残差（补跨通道交互）
        y = y + self.chan_mix(y)

        # 输出
        out = self.out_proj(y)
        return out
