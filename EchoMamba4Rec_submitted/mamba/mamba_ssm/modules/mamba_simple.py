"""
Mamba状态空间模型核心实现

该文件实现了Mamba架构的核心组件 - 状态空间模型(SSM)。
Mamba是一种新型的序列建模架构，结合了循环神经网络的表达能力和Transformer的并行性。

核心特性：
1. 选择性状态空间模型 - 根据输入动态调整状态转移
2. 高效的并行训练和推理
3. 线性时间复杂度的序列建模
4. 强大的长序列建模能力

作者: Tri Dao, Albert Gu
版权: (c) 2023
在M2Rec中的作用: 作为序列编码器，用于建模用户的行为序列
"""

# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

# ======================== 可选依赖导入 ========================
# 尝试导入因果卷积函数，用于高效的1D卷积计算
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

# 尝试导入Triton优化的状态更新函数
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

# 尝试导入层归一化的优化实现
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    """
    Mamba状态空间模型
    
    这是Mamba架构的核心实现，通过选择性状态空间模型实现高效的序列建模。
    模型结合了RNN的表达能力和Transformer的并行训练优势。
    
    模型架构：
    Input -> Linear投影 -> 1D卷积 -> 状态空间模型 -> 输出投影 -> Output
    
    主要创新：
    1. 选择性机制：根据输入内容动态调整状态转移参数
    2. 硬件友好：设计了高效的CUDA kernel实现
    3. 线性复杂度：相对于序列长度的线性时间复杂度
    
    在M2Rec中的应用：
    - 编码用户的行为序列
    - 捕获长期依赖关系
    - 提供高效的序列表示
    """
    
    def __init__(
        self,
        d_model,                 # 模型维度，输入特征的维度
        d_state=16,              # SSM状态维度，控制状态空间的大小
        d_conv=4,                # 卷积核大小，用于局部特征提取
        expand=2,                # 扩展因子，内部维度相对于d_model的倍数
        dt_rank="auto",          # 时间步长参数的秩，"auto"表示自动设置
        dt_min=0.001,            # 时间步长的最小值
        dt_max=0.1,              # 时间步长的最大值
        dt_init="random",        # 时间步长的初始化方式："random"或"constant"
        dt_scale=1.0,            # 时间步长的缩放因子
        dt_init_floor=1e-4,      # 时间步长初始化的下限
        conv_bias=True,          # 卷积层是否使用偏置
        bias=False,              # 线性层是否使用偏置
        use_fast_path=False,     # 是否使用融合内核的快速路径
        layer_idx=None,          # 层索引，用于缓存管理
        device=None,             # 设备类型
        dtype=None,              # 数据类型
    ):
        """
        初始化Mamba状态空间模型
        
        该初始化函数构建了Mamba的所有核心组件，包括：
        1. 输入投影层
        2. 1D卷积层  
        3. 状态空间参数投影层
        4. 状态矩阵初始化
        5. 输出投影层
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # ======================== 基本参数设置 ========================
        self.d_model = d_model                    # 模型维度
        self.d_state = d_state                    # 状态空间维度
        self.d_conv = d_conv                      # 卷积核大小
        self.expand = expand                      # 扩展因子
        self.d_inner = int(self.expand * self.d_model)  # 内部维度 = expand * d_model
        
        # 自动计算dt_rank：通常设为d_model/16，确保参数效率
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path        # 是否使用优化路径
        self.layer_idx = layer_idx                # 层索引

        # ======================== 输入投影层 ========================
        # 将输入映射到内部维度的2倍（用于x和z两个分支）
        # shape: [d_model] -> [d_inner * 2]
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # ======================== 1D卷积层 ========================
        # 深度可分离卷积，用于局部特征提取和因果建模
        # shape: [d_inner, seq_len] -> [d_inner, seq_len]
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,     # 输入通道数
            out_channels=self.d_inner,    # 输出通道数（保持不变）
            bias=conv_bias,               # 是否使用偏置
            kernel_size=d_conv,           # 卷积核大小
            groups=self.d_inner,          # 分组卷积，每个通道独立卷积
            padding=d_conv - 1,           # 因果填充，确保不看到未来信息
            **factory_kwargs,
        )

        # ======================== 激活函数 ========================
        self.activation = "silu"                  # 激活函数类型
        self.act = nn.SiLU()                      # SiLU激活函数实例

        # ======================== 状态空间参数投影层 ========================
        # 从内部维度投影到时间步长参数(dt)和状态矩阵参数(B,C)
        # shape: [d_inner] -> [dt_rank + d_state * 2]
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        
        # 时间步长参数的进一步投影
        # shape: [dt_rank] -> [d_inner]
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)


        # ======================== 时间步长参数初始化 ========================
        # 特殊的dt投影初始化，保持初始化时的方差
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        
        if dt_init == "constant":
            # 常数初始化：所有权重设为相同值
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            # 随机初始化：在[-dt_init_std, dt_init_std]范围内均匀分布
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # ======================== 时间步长偏置初始化 ========================
        # 初始化dt偏置，使得F.softplus(dt_bias)在[dt_min, dt_max]范围内
        # 这确保了时间步长在合理的范围内，既不会太大也不会太小
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        # softplus的逆函数：https://github.com/pytorch/pytorch/issues/72759
        # 这样设置偏置可以确保经过softplus后得到期望的dt值
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # 标记该偏置不需要重新初始化
        self.dt_proj.bias._no_reinit = True

        # ======================== 状态转移矩阵A初始化 ========================
        # 使用S4D实数初始化方法
        # A矩阵控制状态如何随时间演化，初始化为对角矩阵
        # shape: [d_inner, d_state]，每个内部维度都有一个独立的状态转移矩阵
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        
        # 使用对数存储A矩阵，保持数值稳定性
        A_log = torch.log(A)  # 保持在fp32精度
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True    # 不应用权重衰减

        # ======================== 跳跃连接参数D ========================
        # D参数提供了一个跳跃连接，直接从输入到输出
        # 这类似于残差连接，帮助梯度流动和训练稳定性
        # shape: [d_inner]
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # 保持在fp32
        self.D._no_weight_decay = True        # 不应用权重衰减
        
        # ======================== 输出投影层 ========================
        # 将内部维度映射回模型维度
        # shape: [d_inner] -> [d_model]
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # ======================== M2Rec特有的多尺度处理组件 ========================
        # 以下组件是M2Rec项目中添加的，用于多尺度特征处理
        # 注意：这些组件在原始Mamba中不存在，是针对推荐系统的特殊优化
        
        # 傅里叶变换相关的投影层
        self.fourier_proj_real = nn.Linear(200, 200)     # 实部投影
        self.fourier_proj_image = nn.Linear(200, 200)    # 虚部投影
        self.fourier_proj_A = nn.Linear(2, 1)            # A矩阵的傅里叶投影
        self.fourier_proj_B = nn.Linear(2, 1)            # B矩阵的傅里叶投影
        self.fourier_proj_C = nn.Linear(66, 200)         # C矩阵的傅里叶投影
        
        # 时序卷积网络，用于多尺度特征提取
        self.tcn = nn.Conv1d(128, 128, 4, stride=3)       # 主要的时序卷积
        self.tcn_final = nn.Conv1d(200, 200, 2, stride=2) # 最终的时序卷积

        # todo 加入分频器替代
        self.filter_layer = FrequencyLayer(hidden_dropout_prob=0.2, hidden_size=200)
    def low_pass_filter_m_dim(self, input_tensor, cutoff_ratio=0.1):
        """
        沿第二维度(m维)的低通滤波器
        
        该函数在频域中对输入张量进行低通滤波，保留低频分量，去除高频噪声。
        这是M2Rec中的多尺度处理技术之一，用于平滑时序特征。
        
        Args:
            input_tensor (torch.Tensor): 输入张量，shape: (n, m, h)
                - n: batch维度
                - m: 时序维度（将在此维度上进行滤波）
                - h: 特征维度
            cutoff_ratio (float): 截止频率比例，保留的低频成分比例
        
        Returns:
            torch.Tensor: 滤波后的张量，shape与输入相同 (n, m, h)
            
        算法原理：
        1. 对m维度进行傅里叶变换，转到频域
        2. 创建低通滤波器掩码，只保留低频分量
        3. 应用掩码过滤高频分量
        4. 逆傅里叶变换，回到时域
        """
        
        # ======================== 第1步：傅里叶变换 ========================
        # 沿着第二维度(m维)进行快速傅里叶变换
        # shape: (n, m, h) -> (n, m, h) [复数域]
        fft_result = torch.fft.fft(input_tensor, dim=1)

        # ======================== 第2步：获取张量形状 ========================
        n, m, h = input_tensor.shape

        # ======================== 第3步：创建低通滤波器掩码 ========================
        # 初始化全零掩码，与FFT结果形状相同
        mask = torch.zeros_like(fft_result)

        # 计算截止频率：保留cutoff_ratio比例的低频分量
        m_cutoff = int(cutoff_ratio * m)

        # ======================== 第4步：设置掩码 ========================
        # 保留低频分量：频谱的前m_cutoff个和后m_cutoff个频率分量
        # 这是因为FFT结果的低频分量分布在两端
        mask[:, :m_cutoff, :] = 1        # 正频率的低频分量
        mask[:, -m_cutoff:, :] = 1       # 负频率的低频分量

        # ======================== 第5步：应用滤波器 ========================
        # 将掩码应用到FFT结果上，过滤掉高频分量
        filtered_fft = fft_result * mask

        # ======================== 第6步：逆傅里叶变换 ========================
        # 从频域转回时域，取实部作为最终结果
        # shape: (n, m, h) [复数域] -> (n, m, h) [实数域]
        filtered_output = torch.fft.ifft(filtered_fft, dim=1).real

        return filtered_output

    def forward(self, hidden_states, inference_params=None):
        """
        Mamba模型的前向传播
        
        这是Mamba状态空间模型的核心计算过程，实现了选择性状态空间建模。
        
        算法流程：
        1. 输入投影：将输入分解为x和z两个分支
        2. 卷积处理：对x分支进行1D卷积，提取局部特征
        3. 状态空间建模：计算状态转移参数(dt, B, C)
        4. 选择性扫描：执行状态空间模型的核心计算
        5. 门控融合：使用z分支对输出进行门控
        6. 输出投影：将结果映射回输出维度
        
        Args:
            hidden_states (torch.Tensor): 输入隐藏状态
                shape: (B, L, D) 
                - B: batch_size
                - L: sequence_length  
                - D: model_dimension
            inference_params: 推理参数，用于增量推理和状态缓存
            
        Returns:
            torch.Tensor: 输出隐藏状态，shape: (B, L, D)
        """
        # print("调用Mamba模块")
        # ======================== 第1步：获取输入形状 ========================
        batch, seqlen, dim = hidden_states.shape

        # ======================== 第2步：处理推理缓存（可选）========================
        conv_state, ssm_state = None, None
        if inference_params is not None:
            # 从缓存中获取卷积状态和SSM状态，用于增量推理
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # 如果是增量推理（序列偏移 > 0），使用单步更新
                # 状态会被就地更新
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # ======================== 第3步：输入投影和维度重排 ========================
        # 同时执行矩阵乘法和转置：BLH -> HBL
        # 这种重排可以提高内存访问效率
        # shape: (B, L, D) -> (B, d_inner*2, L)
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        
        # 添加偏置（如果存在）
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # ======================== 第4步：计算状态转移矩阵A ========================
        # A矩阵控制状态如何随时间演化
        # 使用负指数确保稳定性（特征值为负实数）
        # shape: (d_inner, d_state)
        A = -torch.exp(self.A_log.float())
        
        # ======================== 第5步：选择执行路径 ========================
        # 根据是否使用快速路径决定计算方式
        
        # 快速路径：使用融合内核进行高效计算
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:
            # 注意：快速路径不支持输出中间状态
            # 使用mamba_inner_fn融合内核，一次性完成所有计算
            out = mamba_inner_fn(
                xz,                              # 输入投影结果
                self.conv1d.weight,              # 卷积权重
                self.conv1d.bias,                # 卷积偏置
                self.x_proj.weight,              # 状态投影权重
                self.dt_proj.weight,             # 时间步长投影权重
                self.out_proj.weight,            # 输出投影权重
                self.out_proj.bias,              # 输出投影偏置
                A,                               # 状态转移矩阵
                None,                            # 输入相关的B矩阵（动态计算）
                None,                            # 输入相关的C矩阵（动态计算）
                self.D.float(),                  # 跳跃连接参数
                delta_bias=self.dt_proj.bias.float(),  # 时间步长偏置
                delta_softplus=True,             # 对时间步长应用softplus
            )

        else:
            # ======================== 第6步：标准路径 - 分步计算 ========================
            # 将输入投影结果分解为x和z两个分支
            # shape: (B, d_inner*2, L) -> 2 × (B, d_inner, L)
            x, z = xz.chunk(2, dim=1)
            # ======================== 第7步：计算卷积 ========================
            # 更新卷积状态（用于增量推理）
            if conv_state is not None:
                # 注意：如果直接取x[:, :, -self.d_conv:]，当seqlen < self.d_conv时会出错
                # 使用F.pad可以在seqlen < self.d_conv时用零填充，否则截断
                # shape: (B, D, W) 其中W是卷积窗口大小
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
                
            # 执行因果卷积计算
            if causal_conv1d_fn is None:
                # 标准卷积实现
                # shape: (B, d_inner, L) -> (B, d_inner, L)
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                # 使用优化的因果卷积函数
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # ======================== 第8步：计算状态空间参数 ========================
            # 注意：这里我们关心数据布局，避免额外的转置操作
            # 我们希望dt的维度排列为：d是最慢变化的维度，L是最快变化的维度
            # 这是ssm_scan内核期望的数据布局
            
            # M2Rec中的多尺度处理（注释掉的代码）：
            # 以下是M2Rec项目中尝试的傅里叶变换方法，用于多尺度特征处理
            # 目前这些代码被注释掉，可能是实验性功能
            # x_fft = torch.fft.fft(x,dim=-1)
            # x = torch.cat([x_fft.real.unsqueeze(-1),x_fft.imag.unsqueeze(-1)],dim=-1)
            # x = self.fourier_proj_C(self.tcn(x))
            # x = self.fourier_proj_A(x).squeeze(-1)
            
            # 投影到状态空间参数
            # shape: (B, d_inner, L) -> (B*L, d_inner) -> (B*L, dt_rank + d_state*2)
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            
            # 分解为时间步长参数dt和状态矩阵参数B, C
            # dt: 时间步长参数, shape: (B*L, dt_rank)
            # B:  输入到状态的映射矩阵, shape: (B*L, d_state)  
            # C:  状态到输出的映射矩阵, shape: (B*L, d_state)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            # 进一步投影dt参数
            # shape: (dt_rank, B*L) -> (d_inner, B*L) -> (B, d_inner, L)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
           
            # ======================== M2Rec中的多尺度处理（实验性功能）========================
            # 以下注释掉的代码是M2Rec项目中尝试的傅里叶变换方法
            # 目的是在频域进行多尺度特征处理，但目前未启用
            # dt_fft = torch.fft.fft(dt,dim=-1)
            # dt_fft = torch.fft.fft2(dt, dim=(1,2), norm='ortho')
            # dt = torch.cat([dt_fft.real.unsqueeze(-1),dt_fft.imag.unsqueeze(-1)],dim=-1)
            # df_real = self.fourier_proj_real(dt_fft.real)
            # df_imag = self.fourier_proj_image(dt_fft.imag)
            # data_fft = torch.fft.fft(dt,dim=-1)
            # data_ifft = torch.complex(data_fft.real,data_fft.imag)
            # dt = torch.fft.ifft(data_ifft, dim=-1).to(torch.float32)
            
            # ======================== 第9步：应用低通滤波 ========================
            # M2Rec的核心创新：对时间步长参数应用低通滤波
            # 这有助于平滑时序特征，去除高频噪声
            # shape保持不变: (B, d_inner, L)
            # todo
            # dt = self.low_pass_filter_m_dim(dt, cutoff_ratio=0.1)
            dt = self.filter_layer(dt)
            # ======================== 第10步：重排状态矩阵B和C ========================
            # 将B矩阵重新排列为适合selective_scan的格式
            # shape: (B*L, d_state) -> (B, d_state, L)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            
            # M2Rec中对B矩阵的傅里叶处理（注释掉）
            # B_fft = torch.fft.fft(B,dim=-1)
            # B = torch.cat([B_fft.real.unsqueeze(-1),B_fft.imag.unsqueeze(-1)],dim=-1)
            # B = self.fourier_proj(B).squeeze(-1)

            # 将C矩阵重新排列为适合selective_scan的格式
            # shape: (B*L, d_state) -> (B, d_state, L)
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            
            # M2Rec中对C矩阵的傅里叶处理（注释掉）
            # C_fft = torch.fft.fft(C,dim=-1)
            # C = torch.cat([C_fft.real.unsqueeze(-1),C_fft.imag.unsqueeze(-1)],dim=-1)
            # C = self.fourier_proj(C).squeeze(-1)

            # 验证激活函数类型
            assert self.activation in ["silu", "swish"]

            # ======================== 第11步：选择性扫描 - 核心SSM计算 ========================
            # 这是Mamba模型的核心：选择性状态空间扫描
            # 实现状态空间模型的前向传播：x[t+1] = A*x[t] + B*u[t], y[t] = C*x[t] + D*u[t]
            y = selective_scan_fn(
                x,                                   # 输入序列 (B, d_inner, L)
                dt,                                  # 时间步长参数 (B, d_inner, L)
                A,                                   # 状态转移矩阵 (d_inner, d_state)
                B,                                   # 输入矩阵 (B, d_state, L)
                C,                                   # 输出矩阵 (B, d_state, L)
                self.D.float(),                      # 跳跃连接参数 (d_inner,)
                z=z,                                 # 门控信号 (B, d_inner, L)
                delta_bias=self.dt_proj.bias.float(), # 时间步长偏置
                delta_softplus=True,                 # 对时间步长应用softplus激活
                return_last_state=ssm_state is not None,  # 是否返回最后状态
            )
            
            # ======================== 第12步：处理状态更新 ========================
            # 如果需要保存状态（用于增量推理）
            if ssm_state is not None:
                y, last_state = y               # 分离输出和最后状态
                ssm_state.copy_(last_state)     # 更新SSM状态缓存
                
            # ======================== 第13步：重排输出维度 ========================
            # 将输出从 (B, d_inner, L) 重排为 (B, L, d_inner)
            y = rearrange(y, "b d l -> b l d")
            
            # ======================== 第14步：输出投影 ========================
            # 将内部维度映射回模型维度
            # shape: (B, L, d_inner) -> (B, L, d_model)
            out = self.out_proj(y)
            
        # 返回最终输出，shape: (B, L, D)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """
        单步前向传播（增量推理）
        
        该方法用于自回归生成过程，一次只处理一个token。
        与完整的forward方法不同，这里使用状态缓存来实现高效的增量计算。
        
        Args:
            hidden_states (torch.Tensor): 当前输入，shape: (B, 1, D)
            conv_state (torch.Tensor): 卷积层的状态缓存，shape: (B, d_inner, d_conv)
            ssm_state (torch.Tensor): SSM的状态缓存，shape: (B, d_inner, d_state)
            
        Returns:
            tuple: (输出, 更新的卷积状态, 更新的SSM状态)
                - 输出: shape (B, 1, D)
                - conv_state: shape (B, d_inner, d_conv)
                - ssm_state: shape (B, d_inner, d_state)
        """
        dtype = hidden_states.dtype
        # 确保每次只处理一个token
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        
        # ======================== 第1步：输入投影 ========================
        # 移除序列维度并进行投影
        # shape: (B, 1, D) -> (B, D) -> (B, 2*d_inner)
        xz = self.in_proj(hidden_states.squeeze(1))
        
        # 分解为x和z两个分支
        # shape: (B, 2*d_inner) -> 2 × (B, d_inner)
        x, z = xz.chunk(2, dim=-1)

        # ======================== 第2步：卷积状态更新 ========================
        if causal_conv1d_update is None:
            # 标准的卷积状态更新实现
            # 将状态向左移动一位，为新输入腾出位置
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            # 将新输入放在最后位置
            conv_state[:, :, -1] = x
            # 计算卷积输出：状态与卷积核的加权和
            # shape: (B, d_inner, d_conv) × (d_inner, d_conv) -> (B, d_inner)
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            # 添加偏置并应用激活函数
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            # 使用优化的卷积更新函数
            x = causal_conv1d_update(
                x,                                          # 当前输入
                conv_state,                                 # 卷积状态
                rearrange(self.conv1d.weight, "d 1 w -> d w"),  # 卷积权重
                self.conv1d.bias,                           # 卷积偏置
                self.activation,                            # 激活函数
            )

        # ======================== 第3步：计算状态空间参数 ========================
        # 投影到状态空间参数
        # shape: (B, d_inner) -> (B, dt_rank + 2*d_state)
        x_db = self.x_proj(x)
        
        # 分解参数
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # 投影时间步长参数（注意：这里不添加dt_bias）
        # shape: (B, dt_rank) -> (B, d_inner)
        dt = F.linear(dt, self.dt_proj.weight)
        
        # 计算状态转移矩阵
        # shape: (d_inner, d_state)
        A = -torch.exp(self.A_log.float())

        # ======================== 第4步：SSM状态更新 ========================
        if selective_state_update is None:
            # 标准的SSM状态更新实现
            
            # 离散化A和B矩阵
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            
            # 计算离散化的状态转移矩阵
            # dA = exp(dt * A), shape: (B, d_inner, d_state)
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            
            # 计算离散化的输入矩阵
            # dB = dt * B, shape: (B, d_inner, d_state)
            dB = torch.einsum("bd,bn->bdn", dt, B)
            
            # 更新SSM状态：s[t+1] = dA * s[t] + dB * x[t]
            # shape: (B, d_inner, d_state)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            
            # 计算输出：y = C * s + D * x
            # shape: (B, d_inner)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            
            # 应用门控机制
            y = y * self.act(z)
        else:
            # 使用优化的状态更新函数
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, 
                z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        # ======================== 第5步：输出投影 ========================
        # 将内部维度映射回模型维度
        # shape: (B, d_inner) -> (B, d_model)
        out = self.out_proj(y)
        
        # 恢复序列维度并返回所有状态
        # shape: (B, d_model) -> (B, 1, d_model)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        为推理过程分配状态缓存
        
        该方法预先分配用于增量推理的状态缓存，包括卷积状态和SSM状态。
        这避免了在推理过程中重复分配内存，提高了推理效率。
        
        Args:
            batch_size (int): 批次大小
            max_seqlen (int): 最大序列长度（在该实现中未使用）
            dtype: 数据类型，如果为None则使用权重的数据类型
            **kwargs: 其他参数
            
        Returns:
            tuple: (卷积状态缓存, SSM状态缓存)
                - conv_state: shape (batch_size, d_inner, d_conv)
                - ssm_state: shape (batch_size, d_inner, d_state)
        """
        # 获取模型所在的设备
        device = self.out_proj.weight.device
        
        # ======================== 分配卷积状态缓存 ========================
        # 使用卷积层的数据类型，或者使用指定的数据类型
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        
        # 分配卷积状态缓存：存储卷积的历史输入
        # shape: (batch_size, d_inner, d_conv)
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, 
            device=device, dtype=conv_dtype
        )
        
        # ======================== 分配SSM状态缓存 ========================
        # 使用时间投影层的数据类型，或者使用指定的数据类型
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        
        # 分配SSM状态缓存：存储状态空间模型的隐藏状态
        # shape: (batch_size, d_inner, d_state)
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, 
            device=device, dtype=ssm_dtype
        )
        
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        """
        从缓存中获取推理状态
        
        该方法管理多层Mamba模型的状态缓存，每层都有独立的状态。
        支持状态的创建、获取和重置。
        
        Args:
            inference_params: 推理参数对象，包含状态缓存字典
            batch_size (int): 批次大小
            initialize_states (bool): 是否重新初始化状态为零
            
        Returns:
            tuple: (卷积状态, SSM状态)
                - conv_state: shape (batch_size, d_inner, d_conv)
                - ssm_state: shape (batch_size, d_inner, d_state)
        """
        # 确保层索引已设置
        assert self.layer_idx is not None
        
        # ======================== 检查缓存中是否存在当前层的状态 ========================
        if self.layer_idx not in inference_params.key_value_memory_dict:
            # 如果缓存中没有当前层的状态，创建新的状态
            
            # ======================== 创建卷积状态 ========================
            # shape: (batch_size, d_inner, d_conv)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,     # d_inner
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            
            # ======================== 创建SSM状态 ========================
            # shape: (batch_size, d_inner, d_state)
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,     # d_inner
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            
            # 将新创建的状态存储到缓存中
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            # ======================== 从缓存中获取已存在的状态 ========================
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            
            # TODO: 如果生成过程中批次大小发生变化，而我们重用相同的状态怎么办？
            if initialize_states:
                # 如果需要重新初始化，将状态重置为零
                conv_state.zero_()
                ssm_state.zero_()
                
        return conv_state, ssm_state

class FrequencyLayer(nn.Module):
    def __init__(self, hidden_dropout_prob, hidden_size):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.c = 3 // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    # 拆分高低频信号
    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        # 转换为频率信号

        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        # 前c个是低频信号
        low_pass[:, self.c:, :] = 0
        # 重新转换回时域
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        # 得到高频时域
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        # Add & Norm
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
