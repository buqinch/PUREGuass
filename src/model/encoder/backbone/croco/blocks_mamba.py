import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 辅助函数
def init_lecun_normal(tensor, gain=1.0):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1.0 / fan_in)
    return nn.init.trunc_normal_(tensor, std=std)

class SSMParams(nn.Module):
    """状态空间模型参数"""
    def __init__(self, d_model, d_state, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.zeros(self.d_model, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_model))
        
        # B和C矩阵
        self.B = nn.Parameter(torch.randn(self.d_model, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_model, self.d_state))
        
        self.bias = bias
        if bias:
            self.time_bias = nn.Parameter(torch.zeros(self.d_model))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        # A初始化为负值以使系统稳定
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        
        # B和C使用缩放的初始化
        init_lecun_normal(self.B, gain=0.5)
        init_lecun_normal(self.C, gain=0.5)
        
        # D初始化为1附近的小随机数
        nn.init.uniform_(self.D, 0.9, 1.1)
        
        if self.bias:
            nn.init.zeros_(self.time_bias)
            
    def forward(self, batch_size, seq_len, delta):
        """生成SSM参数"""
        # 扩展为批次
        A = -torch.exp(self.A_log.float())  # 负指数确保稳定性
        B = self.B.float()
        C = self.C.float()
        D = self.D.float()
        
        # 计算状态转换矩阵
        deltaA = torch.exp(torch.einsum('bld,dm->bldm', delta, A))
        
        return A, B, C, D, deltaA

class SelectiveSSM(nn.Module):
    """选择性状态空间模型 - Mamba的核心组件"""
    def __init__(self, d_model, d_state, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dropout = nn.Dropout(dropout)
        
        # SSM参数
        self.ssm_params = SSMParams(d_model, d_state)
        
        # 门控和投影
        self.time_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x, delta=None):
        batch, seq_len, dim = x.shape
        
        # 计算时间增量/选择性
        if delta is None:
            delta = F.softplus(self.time_projection(x))
        
        # 获取SSM参数
        A, B, C, D, deltaA = self.ssm_params(batch, seq_len, delta)
        
        # 准备状态向量
        x_state = torch.zeros(batch, dim, self.d_state, device=x.device)
        
        # 输出列表
        outputs = []
        
        # 序列扫描
        for t in range(seq_len):
            # 更新状态向量
            x_t = x[:, t]
            delta_A_t = deltaA[:, t]
            
            # 状态更新: x_state = A*x_state + B*x_t
            x_state = torch.bmm(x_state, delta_A_t) + torch.einsum('bd,dh->bdh', x_t, B)
            
            # 输出计算: y_t = C*x_state + D*x_t
            y_t = torch.einsum('bdh,dh->bd', x_state, C) + D * x_t
            
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [batch, seq_len, dim]
        
        return self.dropout(y)

class MambaBlock(nn.Module):
    """Mamba块，可以替换Transformer块"""
    def __init__(self, dim, d_state=16, d_conv=4, expand_factor=2, dropout=0.0):
        super().__init__()
        
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        
        # 展开维度
        self.hidden_dim = int(expand_factor * dim)
        
        # 输入投影
        self.in_proj = nn.Linear(dim, 2 * self.hidden_dim)
        
        # 卷积层捕获局部上下文
        self.conv = nn.Conv1d(
            self.hidden_dim, 
            self.hidden_dim, 
            kernel_size=d_conv, 
            padding=d_conv-1, 
            groups=self.hidden_dim,
            bias=True
        )
        
        # 选择性SSM
        self.ssm = SelectiveSSM(self.hidden_dim, d_state, dropout)
        
        # 输出投影
        self.out_proj = nn.Linear(self.hidden_dim, dim)
        
        # 规范化
        self.norm = nn.LayerNorm(dim)
        
        # 步长调整
        self.conv_stride_adj = nn.Conv1d(
            self.hidden_dim, 
            self.hidden_dim, 
            kernel_size=1,
            bias=True
        )
        
    def forward(self, x, pos=None):
        """前向传播"""
        # 保存残差连接
        residual = x
        
        # 应用层规范化
        x = self.norm(x)
        
        # 投影并分离到门控和值
        x = self.in_proj(x)
        x, gate = x.chunk(2, dim=-1)
        
        # 卷积处理
        x_conv = x.permute(0, 2, 1)  # [B, C, L]
        x_conv = self.conv(x_conv)
        
        # 调整卷积输出
        x_conv = x_conv[:, :, :x.shape[1]]  # 截断到原始序列长度
        x_conv = self.conv_stride_adj(x_conv)
        x_conv = x_conv.permute(0, 2, 1)  # [B, L, C]
        
        # 应用SiLU激活和门控
        x = F.silu(x_conv) * gate
        
        # 计算SSM
        x = self.ssm(x)
        
        # 输出投影
        x = self.out_proj(x)
        
        # 残差连接
        return residual + x

class Mamba(nn.Module):
    """完整的Mamba模型"""
    def __init__(self, dim, depth, d_state=16, d_conv=4, expand_factor=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # Mamba块堆叠
        self.layers = nn.ModuleList([
            MambaBlock(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
    def forward(self, x, pos=None):
        """前向传播"""
        for layer in self.layers:
            x = layer(x, pos)
            
        return x
