# augment.py
import math
import torch
import torch.nn.functional as F

def _to_btc(x, B_ref=None):
    """
    统一输出为 (B, T, C)
    - 3D: 直接返回
    - 2D:
        * 若第0维等于参考B，视为 (B, T) -> (B, T, 1)
        * 否则视为 (T, C) -> (1, T, C)
    - 1D: 视为 (T,) -> (1, T, 1) 或若提供B_ref可 reshape 为 (B_ref, -1, 1)
    返回: (x_btc, squeezed) 其中 squeezed 表示是否人为加了 batch 维=1
    """
    if x.dim() == 3:
        return x, False
    if x.dim() == 2:
        if B_ref is not None and x.size(0) == B_ref:
            return x.unsqueeze(-1), False  # (B,T)->(B,T,1)
        else:
            return x.unsqueeze(0), True     # (T,C)->(1,T,C)
    if x.dim() == 1:
        if B_ref is not None:
            T = x.numel() // B_ref
            return x.view(B_ref, T, 1), False
        else:
            return x.unsqueeze(0).unsqueeze(-1), True
    raise ValueError(f"Unsupported shape: {tuple(x.shape)}")

def _undo_btc(x, squeezed):
    return x.squeeze(0) if squeezed else x


# ------- 1) 抖动 + 缩放（数值域） -------
@torch.no_grad()
def aug_jitter_scale_pair(x, y, p=0.7, sigma=0.01, scale_low=0.9, scale_high=1.1, clamp=None):
    """
    成对抖动+缩放（安全版）：对 x 和 y 分别施加 *相同* 的缩放系数（按样本维度对齐）和各自的高斯噪声。
    不做 x||y 拼接，因此不依赖二者形状完全一致。
    兼容 2D/3D 等形状；缩放按“首维=批维”优先，否则用全局一个标量。
    """
    if torch.rand(1, device=x.device).item() > p:
        return x, y

    def _batch_size(t):
        return t.size(0) if t.dim() >= 2 else 1

    Bx, By = _batch_size(x), _batch_size(y)
    # 1) 同一随机缩放（尽量按 batch 维度对齐）
    if Bx == By and Bx > 1:
        scale = torch.empty(Bx, device=x.device).uniform_(scale_low, scale_high)  # (B,)
        x = x * scale.view(Bx, *([1] * (x.dim() - 1)))
        y = y * scale.view(By, *([1] * (y.dim() - 1)))
    else:
        s = float(torch.empty(1, device=x.device).uniform_(scale_low, scale_high))
        x = x * s
        y = y * s

    # 2) 各自加噪（幅度一致）
    if sigma and sigma > 0:
        x = x + torch.randn_like(x) * sigma
        y = y + torch.randn_like(y) * sigma

    # 3) 可选截断
    if clamp is not None:
        x = torch.clamp(x, clamp[0], clamp[1])
        y = torch.clamp(y, clamp[0], clamp[1])

    return x, y

# ------- 2) 时间扭曲（插值重采样） -------
@torch.no_grad()
def aug_time_warp_pair(x, y, p=0.5, strength=0.2):
    """
    对 (x||y) 的时间轴做单调的轻微拉伸/压缩，再线性插值回原长度。
    strength: 0~0.5，越大扭曲越强
    """
    if torch.rand(1).item() > p:
        return x, y

    xb, sx = _to_btc(x, B_ref=x.size(0) if x.dim()>=2 else None)
    yb, sy = _to_btc(y, B_ref=xb.size(0))
    z = torch.cat([xb, yb], dim=1)
    B, T, C = z.shape

    base = torch.linspace(0, 1, T, device=z.device).unsqueeze(0)  # (1,T)
    # 生成平滑的、严格递增的时间映射
    wiggle = (torch.randn(B, 4, device=z.device) * strength).cumsum(-1)
    ctrl = torch.cat([torch.zeros(B,1,device=z.device), wiggle, torch.zeros(B,1,device=z.device)], dim=1)
    # 上采样到 T 点（简易样条）
    t_new = F.interpolate(ctrl.unsqueeze(1), size=T, mode='linear', align_corners=True).squeeze(1)
    t_new = base + t_new
    t_new = (t_new - t_new[:, :1]) / (t_new[:, -1:] - t_new[:, :1] + 1e-8)  # 归一化到[0,1]

    # 线性插值：先到 (B,C,T) 再用 grid_sample
    z_chw = z.permute(0, 2, 1)  # (B,C,T)
    grid = t_new.unsqueeze(1) * 2 - 1  # [-1,1]
    zi = F.grid_sample(z_chw.unsqueeze(-1), grid.unsqueeze(-1), mode='bilinear', align_corners=True).squeeze(-1)
    zi = zi.permute(0, 2, 1)  # (B,T,C)

    xi, yi = z[:, :xb.size(1)], z[:, xb.size(1):]
    return _undo_btc(xi, sx), _undo_btc(yi, sy)

# ------- 3) 频域相位扰动（主频轻微相移） -------
@torch.no_grad()
def aug_fft_phase_pair(x, y, p=0.5, max_phase_shift=0.25*math.pi, topk=2):
    """
    对 (x||y) 在时间维做 rFFT，挑 top-k 主频做小角度相位扰动，再 irFFT。
    保留幅度，轻改相位 -> 改周期相位不改强度（常对季节性友好）
    """
    if torch.rand(1).item() > p:
        return x, y

    xb, sx = _to_btc(x, B_ref=x.size(0) if x.dim()>=2 else None)
    yb, sy = _to_btc(y, B_ref=xb.size(0))
    z = torch.cat([xb, yb], dim=1)
    B, T, C = z.shape

    # rFFT: (B, F, C)，F = T//2 + 1
    Z = torch.fft.rfft(z, dim=1)  # complex64/128
    mag = Z.abs()
    # 找每个通道的主频索引（排除直流分量 0）
    mag[:, 0] = -1
    idx = torch.topk(mag, k=min(topk, mag.size(1)-1), dim=1).indices  # (B, k, C)

    # 构造相移矩阵
    phase = torch.zeros_like(Z, dtype=Z.dtype)
    for b in range(B):
        # 给每个通道的 top-k 频率一个独立相移
        delta = (torch.rand(C, device=z.device) * 2 - 1) * max_phase_shift  # (C,)
        for k in range(idx.size(1)):
            f_idx = idx[b, k]  # (C,)
            phase[b, f_idx, torch.arange(C, device=z.device)] = torch.exp(1j * delta)

    Z_new = torch.where(phase==0, Z, Z * phase)
    z_new = torch.fft.irfft(Z_new, n=T, dim=1).real

    xi, yi = z[:, :xb.size(1)], z[:, xb.size(1):]
    return _undo_btc(xi, sx), _undo_btc(yi, sy)

@torch.no_grad()
def batch_mixup(x, y, alpha=0.4):
    """
    x: (B,T_in,C)  y: (B,T_out,C)
    返回混合后的 (x_mix, y_mix)
    """
    xb, sx = _to_btc(x, B_ref=x.size(0) if x.dim()>=2 else None)
    yb, sy = _to_btc(y, B_ref=xb.size(0))
    z = torch.cat([xb, yb], dim=1)
    B = xb.size(0)
    perm = torch.randperm(B, device=xb.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(xb.device)
    lam_x = lam.view(B,1,1)
    lam_y = lam.view(B,1,1)
    x_mix = xb * lam_x + xb[perm] * (1 - lam_x)
    y_mix = yb * lam_y + yb[perm] * (1 - lam_y)
    return _undo_btc(x_mix, sx), _undo_btc(y_mix, sy)

import torch
import torch.nn.functional as F
import random

@torch.no_grad()
def aug_pair_flip(x: torch.Tensor, y: torch.Tensor, p: float = 0.5):
    """
    成对时间翻转：同时将 x 和 y 在时间维翻转。
    兼容 (B,T,C) / (B,T) / (T,C) / (T,) 等，返回与输入同shape。
    """
    if torch.rand(1, device=x.device).item() > p:
        return x, y

    xb, sx = _to_btc(x)
    yb, sy = _to_btc(y, B_ref=xb.size(0))
    xb = xb.flip(1)
    yb = yb.flip(1)
    return _undo_btc(xb, sx), _undo_btc(yb, sy)


@torch.no_grad()
def aug_pair_random_crop(x: torch.Tensor,
                              y: torch.Tensor,
                              p: float = 0.5,
                              min_ratio: float = 0.5,
                              mode: str = "nearest",
                              max_retry: int = 5):
    """
    成对随机裁剪 + 重采样（不拼接版）：
    - 在“合并时间轴”上采样一个随机窗口 [start, start+L)，
      然后对 x(长度Tin，位于[0,Tin)) 与 y(长度Tout，位于[Tin,Tin+Tout)) 分别取交集段，
      再各自重采样回 Tin / Tout。
    - x 与 y 的通道数可以不同（不会拼接）。
    - 若窗口与 x 或 y 的交集长度为 0，则重试；超过 max_retry 直接返回原样（本样本跳过增强）。
    """
    if torch.rand(1, device=x.device).item() > p:
        return x, y

    xb, sx = _to_btc(x)                # (B, Tin, Cx)
    yb, sy = _to_btc(y, B_ref=xb.size(0))  # (B, Tout, Cy)
    B, Tin, Cx = xb.shape
    Tout, Cy = yb.size(1), yb.size(2)
    Tsum = Tin + Tout

    L_min = max(1, int(min_ratio * Tsum))
    if L_min > Tsum:
        L_min = Tsum

    out_x = torch.empty_like(xb)   # (B, Tin, Cx)
    out_y = torch.empty_like(yb)   # (B, Tout, Cy)

    for b in range(B):
        ok = False
        for _ in range(max_retry):
            L = random.randint(L_min, Tsum)
            start = 0 if L == Tsum else random.randint(0, Tsum - L)

            # 与 x 的交集：x 在 [0, Tin)
            xs = max(0, start)
            xe = min(Tin, start + L)
            Lx = xe - xs

            # 与 y 的交集：y 在 [Tin, Tin+Tout)
            ys = max(0, start - Tin)
            ye = min(Tout, start + L - Tin)
            Ly = ye - ys

            if Lx > 0 and Ly > 0:
                ok = True
                break

        if not ok:
            # 放弃本样本的此增强，保持原样
            out_x[b] = xb[b]
            out_y[b] = yb[b]
            continue

        # ---- 对 x 的交集段重采样回 Tin ----
        seg_x = xb[b, xs:xe, :]                                # (Lx, Cx)
        seg_x = seg_x.permute(1, 0).unsqueeze(0)               # (1, Cx, Lx)
        seg_xr = F.interpolate(seg_x, size=Tin, mode=mode, align_corners=False if mode!="nearest" else None)
        seg_xr = seg_xr.squeeze(0).permute(1, 0)               # (Tin, Cx)
        out_x[b] = seg_xr

        # ---- 对 y 的交集段重采样回 Tout ----
        seg_y = yb[b, ys:ye, :]                                # (Ly, Cy)
        seg_y = seg_y.permute(1, 0).unsqueeze(0)               # (1, Cy, Ly)
        seg_yr = F.interpolate(seg_y, size=Tout, mode=mode, align_corners=False if mode!="nearest" else None)
        seg_yr = seg_yr.squeeze(0).permute(1, 0)               # (Tout, Cy)
        out_y[b] = seg_yr

    return _undo_btc(out_x, sx), _undo_btc(out_y, sy)