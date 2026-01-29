from typing import Optional
from pathlib import Path

from torch import nn
from spikingjelly.activation_based import surrogate, neuron, functional
from ..base import NETWORKS
from ...module.spike_encoding import SpikeEncoder
from ...module.spike_attention import Block
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union
import math
from .utils_quant import QuantizeLinear, QuantizeEmbedding, act_quant_fn, AlphaInit                                          

detach_reset = True

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, constant=100): #, Clp_max=1):
        ctx.constant = constant
        # ctx.Clp_max = Clp_max
        return torch.div(torch.floor(torch.mul(tensor, constant)), constant)

    @staticmethod
    def backward(ctx, grad_output):
        # Clp_max = ctx.Clp_max
        return torch.nn.functional.hardtanh(grad_output), None 
        # return Clp_max * F.hardtanh(grad_output/Clp_max), None

Quantization_ = Quantization.apply

factor = math.log2(1 / math.log(2))

class ApproxSoftplus(nn.Module):
    def __init__(self, k=1.44):
        super(ApproxSoftplus, self).__init__()
        self.k = k  # 常数 k，用于近似 e^x 为 2^(k * x)

    def forward(self, x):
        result = torch.where(x < factor, torch.pow(2, x), x+1/math.log(2)-factor)
        return result
    
approx_softplus = ApproxSoftplus()

XC = -1.791995  # 分段点
C2 = -0.228245 # 线性部分的常数

class ApproxSiLU(nn.Module):
    """
    基于分段指数和线性的 SiLU 近似函数。
    注意：此函数在分段点 XC 处不光滑（导数不连续），且函数值不连续。
    请谨慎用于需要稳定梯度的训练。
    """
    def __init__(self):
        super(ApproxSiLU, self).__init__()
        self.xc = XC
        self.c2 = C2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 创建一个布尔张量，用于判断 x 是否小于分段点
        mask = x < self.xc
        
        # 1. 计算第一段（x < XC）: -2^x
        # 使用 torch.pow(2, x) 来计算 2^x，并取负
        result_low = -torch.pow(2.0, x)
        
        # 2. 计算第二段（x >= XC）: 2^(-x - 1) + x - 0.228245
        # 使用 torch.pow(2, -x - 1) 来计算指数部分
        result_high = torch.pow(2.0, -x - 1.0) + x + self.c2
        
        # 3. 根据 mask 合并结果
        # torch.where(condition, value_if_true, value_if_false)
        result = torch.where(mask, result_low, result_high)
        
        return result
    
approx_silu = ApproxSiLU()

@dataclass
class TSModelArgs:
    d_model: int         # 相当于输入特征数量，即 feature_number
    n_layer: int
    window_length: int   # 输入序列长度
    output_size: int     # 预测的时间步数（输出长度）
    d_state: int = 32
    expand: int = 0.2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 3
    conv_bias: bool = True
    bias: bool = False
    clamp: Optional[int] = None
    quantize: Optional[int] = None 
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
@NETWORKS.register_module("iSpikformer")
class iSpikformer(nn.Module):
    #def __init__(self, dim, d_model, T, dropout=0.0, **ssm_kwargs):
    def __init__(
        self,
        dim: int,
        max_length: int = 100,
        input_size: Optional[int] = None,
        d_ff: Optional[int] = None,
        num_steps: int = 4,
        encoder_type: Optional[str] = "conv",
        clamp: Optional[List[int]] = None,
        quantize: Optional[List[int]] = None, 
    ):
        """
        :param dim: 输入特征维度
        :param d_model: 模型内部/预测输出维度
        :param T: 脉冲仿真步数
        :param dropout: dropout 概率
        :param ssm_kwargs: 传递给 SS4D 层的额外参数
        """
        super().__init__()
        args = TSModelArgs(
            d_model=137,           # 输入特征数量 (feature_number)
            n_layer=1,             # 模型层数
            window_length=168,     # 输入序列长度 (window_length)
            output_size=24,        # 预测的时间步数 (output_size)
            d_state=32,            # 状态维度 (d_state)
            expand=0.2,              # 扩展因子 (expand)
            dt_rank='auto',        # 使用 'auto' 来自动计算 dt_rank
            d_conv=3,              # 卷积层参数 (d_conv)
            conv_bias=True,        # 卷积层是否使用偏置 (conv_bias)
            bias=False,             # 是否在其他层使用偏置 (bias)
            clamp = clamp,
            quantize = quantize
        )
        self.T = num_steps
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 3 // 2),
            ),
            nn.BatchNorm2d(4),
        )

        self.sequence_projector = nn.AdaptiveAvgPool1d(256)
        self.fc_out = nn.Linear(256, 24)

        
        # 输入已经是连续数值，可选地加入线性投影，这里直接用Identity
        self.input_proj = nn.Identity()
        
        # 叠加多个残差块（每个块内包含状态空间计算）
        # self.layers = nn.ModuleList([ResidualBlockTS(args) for _ in range(3)])
        self.layers = nn.ModuleList([
            ResidualBlockTS(args) #, clamp=self.clamp, quantize=self.quantize) 
            for _ in range(3)
        ])

        
        # Forecast Head: 将时间维度从 window_length 映射到 output_size
        # 注意：此处要求 window_length 固定，因为线性层的输入尺寸为 window_length
        self.forecast_head = nn.Linear(args.window_length, 24)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        # 对时间维度进行预测：先把 x 转置为 (B, d_model, L)
        x = x.transpose(1, 2)
        # 对每个特征通道独立，将 L 映射到 output_size
        output = self.forecast_head(x)  # (B, d_model, output_size)
        return output, output
    

class ResidualBlockTS(nn.Module):
    def __init__(self, args: TSModelArgs, clamp=None, quantize=None):
        """
        残差块：先归一化，再经过 MambaBlockTS，然后加上输入（残差连接）
        """
        super().__init__()
        self.args = args
        self.clamp = clamp
        self.quantize = quantize
        self.mixer = MambaBlockTS(args, self.clamp, self.quantize)
        
        # 使用 BatchNorm1d 替换 RMSNorm
        # self.norm = nn.BatchNorm1d(args.d_model)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        output = self.mixer(self.norm(x)) + x
        return output

class MambaBlockTS(nn.Module):
    def __init__(self, args: TSModelArgs, clamp=None, quantize=None):
        """
        Mamba 核心模块：
          - 先对输入进行线性映射（in_proj），拆分得到两部分
          - 使用 1D 卷积捕捉局部时序特征
          - 使用状态空间模型（SSM）进行全局时序建模
        """
        super().__init__()
        self.args = args
        self.clamp = clamp or [1]*7
        self.quantize = quantize or [8]*7
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        
        # 将 x 映射为状态空间模型的参数：输出 dt, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        # 将 dt 从 dt_rank 映射到 d_inner
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        self.res_proj = nn.Linear(args.d_inner, args.d_inner, bias=True)
        self.clip_delta = AlphaInit(torch.tensor(1.0))
        self.clip_delta1 = AlphaInit(torch.tensor(1.0))
        self.clip_c = AlphaInit(torch.tensor(1.0))
        self.clip_s = AlphaInit(torch.tensor(1.0))
        self.clip_x = AlphaInit(torch.tensor(1.0))
        self.clip_y = AlphaInit(torch.tensor(1.0))
        self.clip_y1 = AlphaInit(torch.tensor(1.0))
        self.clip_pts = AlphaInit(torch.tensor(1.0))
        
        # 初始化 A_log（取值基于 d_inner 和 d_state）及 D 参数
        A = repeat(torch.arange(1, args.d_state + 1, dtype=torch.float32), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        # self.res_proj = nn.Linear(args.d_inner, args.d_inner, bias=True)
    

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (B, L, d_model)
        Returns:
            Tensor, shape (B, L, d_model)
        """
        (b, l, d) = x.shape
        
        # in_proj 将 x 映射为 2*d_inner, 并拆分成两部分
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        # zlyan: add quantization
        (x_proj_part, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x_proj_part = act_quant_fn(x_proj_part, self.clip_c, 1, quant_method='elastic',
                                       symmetric=False, layerwise=True)
        # print(res)
        # res = act_quant_fn(res, self.clip_y1, 5, quant_method='elastic',
        #                                symmetric=True, layerwise=True)
        d_inner_s = self.args.d_inner

        # 将 x_proj_part 调整维度以便进行 1D 卷积： (B, d_inner, L)
        x_conv = rearrange(x_proj_part, 'b l d_inner_s -> b d_inner_s l')
        x_conv = self.conv1d(x_conv)[:, :, :l]  # 保持时间维度为 l
        x_conv = rearrange(x_conv, 'b d_inner_s l -> b l d_inner_s')
        
        # x_conv = F.silu(x_conv)
        x_conv = act_quant_fn(x_conv, self.clip_s, 1, quant_method='elastic',
                                       symmetric=False, layerwise=True)
        y = self.ssm(x_conv)
        
        # print(y)
        # print(torch.max(y), torch.min(y))
        # 使用 SiLU 激活残差部分
        y = act_quant_fn(y, self.clip_y, 1, quant_method='elastic',
                                       symmetric=True, layerwise=True)
        # res = self.res_proj(res)
        # res = F.silu(res)
        res = act_quant_fn(res, self.clip_y1, 5, quant_method='elastic',
                                       symmetric=False, layerwise=True)
        res = approx_silu(res)
        y = y * res
        # z = torch.cat([y, res], dim=-1)
        # y = torch.clamp(y, min=-3, max=3)
        # y = Quantization_(y, 32)
    
        # y = act_quant_fn(y, self.clip_y1, 6,
        #          quant_method='elastic',
        #          symmetric=False,
        #          layerwise=False)
        # print(self.out_proj.weight)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        """
        状态空间模型（SSM）：
         - 利用 x_proj 得到 dt, B, C 参数（其中 dt 表示步长）
         - 对 A（由 A_log 计算）进行离散化
         - 通过循环进行状态更新，完成状态空间模型的离散扫描
         
        Args:
            x: Tensor, shape (B, L, d_inner)
        Returns:
            Tensor, shape (B, L, d_inner)
        """
        (d_inner, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, n)
        D = self.D.float()
        
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = act_quant_fn(delta, self.clip_delta, 1, quant_method='elastic',
                                       symmetric=False, layerwise=True)
        delta = self.dt_proj(delta)
        # delta = act_quant_fn(delta, self.clip_pts, 5, quant_method='elastic',
        #                                symmetric=False, layerwise=True)
        # delta = approx_softplus(delta)
        delta = F.softplus(delta)
        # delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        # print(delta,torch.max(delta))
        delta = act_quant_fn(delta, self.clip_delta1, 1, quant_method='elastic',
                                       symmetric=False, layerwise=True)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """
        执行状态空间离散扫描：
          x(t+1) = A*x(t) + B*u(t)
          y(t)   = C*x(t) + D*u(t)
        这里采用顺序方式处理时间步（可优化为并行化实现）。
        
        Args:
            u: Tensor, (B, L, d_inner)
            delta: Tensor, (B, L, d_inner)
            A: Tensor, (d_inner, n)
            B: Tensor, (B, L, n)
            C: Tensor, (B, L, n)
            D: Tensor, (d_inner,)
        Returns:
            Tensor, (B, L, d_inner)
        """
        (b, l, d_inner) = u.shape
        n = A.shape[1]
        
        # 计算离散化参数
        # deltaA = torch.exp(einsum(delta, A, 'b l d_inner, d_inner n -> b l d_inner n'))
        deltaA = torch.round(einsum(delta, A, 'b l d_inner, d_inner n -> b l d_inner n'))
        deltaA = torch.pow(2, deltaA)
        deltaB_u = einsum(delta, B, u, 'b l d_inner, b l n, b l d_inner -> b l d_inner n')
        
        # 按时间步逐步扫描更新状态
        x = torch.zeros((b, d_inner, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            x = act_quant_fn(x, self.clip_x, 1, quant_method='elastic',
                                       symmetric=False, layerwise=True)
            y = einsum(x, C[:, i, :], 'b d_inner n, b n -> b d_inner')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + u * D
        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight