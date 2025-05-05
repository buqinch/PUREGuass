from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm import Mamba as MambaSSM
    
from .croco.blocks import DecoderBlock
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, make_batch_symmetric
from .croco.patch_embed import get_patch_embed
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding


inf = float('inf')


# 添加Mamba参数的配置
croco_mamba_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 768,
        'enc_num_heads': 16,
        'dec_num_heads': 12,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
    },
}


@dataclass
class BackboneCrocoCfg:
    name: Literal["croco", "croco_multi", "vit_mamba"]
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]
    patch_embed_cls: str = 'PatchEmbedDust3R'
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = 'token'
    # Mamba特有配置
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand_factor: float = 2.0
    mamba_dropout: float = 0.0


# ... existing code ...

class SSMKernel(nn.Module):
    """状态空间模型的核心实现"""
    def __init__(self, dim, d_state=16, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        # 将A矩阵改为共享参数，不再为每个特征维度创建单独的A
        self.A_log = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(dim, d_state))
        self.C = nn.Parameter(torch.randn(dim, d_state))
        self.D = nn.Parameter(torch.zeros(dim))
        
        # 简化步长投影
        self.dt_proj = nn.Linear(dim, 1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.normal_(self.A_log, mean=0.0, std=0.01)
        nn.init.normal_(self.B, mean=0.0, std=0.01)
        nn.init.normal_(self.C, mean=0.0, std=0.01)
    
    def forward(self, u, gate):
        """修改forward方法避免维度不匹配"""
        batch, seq_len, dim = u.shape
        
        # 从gate计算单一时间步长
        delta = torch.sigmoid(self.dt_proj(gate))  # [B, L, 1]
        
        # 构建负定A矩阵
        A = -torch.exp(self.A_log)  # [d_state, d_state]
        
        # 使用批量处理方式
        outputs = []
        h = torch.zeros(batch, self.d_state, device=u.device)
        
        for t in range(seq_len):
            # 当前时间步
            u_t = u[:, t]  # [B, D]
            delta_t = delta[:, t, 0]  # [B]
            
            # 批量计算状态更新
            # 为每个样本计算不同的A矩阵
            scaled_A = A * delta_t.view(-1, 1, 1)  # [B, d_state, d_state]
            A_exp = torch.matrix_exp(scaled_A)  # [B, d_state, d_state]
            
            # 更新状态
            h = torch.bmm(h.unsqueeze(1), A_exp).squeeze(1) + u_t @ self.B
            
            # 计算输出
            y = h @ self.C.t() + self.D * u_t
            
            outputs.append(y)
            
        return torch.stack(outputs, dim=1)  # [B, L, D]


class MambaAdapter(nn.Module):
    """将mamba-ssm库的Mamba模型适配到CroCo架构"""
    def __init__(self, dim, depth, d_state=16, d_conv=4, expand_factor=2.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        
        # 使用官方Mamba实现，这里的d_inner是扩展维度
        d_inner = int(expand_factor * dim)
        
        # 创建标准Mamba模型
        self.mamba = MambaSSM(
            d_model=dim,           # 输入/输出维度
            d_state=d_state,       # 状态维度
            d_conv=d_conv,         # 卷积核大小
            expand=expand_factor,  # 扩展因子
            # dropout=dropout        # dropout率
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 如果需要多层Mamba，创建一个ModuleList
        if depth > 1:
            self.layers = nn.ModuleList([deepcopy(self.mamba) for _ in range(depth)])
            # 移除单层mamba
            delattr(self, 'mamba')
        else:
            self.layers = [self.mamba]
    
    def forward(self, x, pos=None):
        """
        接收与ViT块相同的参数，但忽略pos
        x: [B, L, D] - 序列输入
        """
        # 对每层依次应用
        for layer in self.layers:
            x = layer(x)
        return x

class MambaBlock(nn.Module):
    """Mamba块实现，可替代Transformer块"""
    def __init__(self, dim, d_state=16, d_conv=4, expand_factor=2.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        
        self.hidden_dim = 257
        
        # 层规范化
        self.norm = nn.LayerNorm(dim)
        
        # 投影层
        self.in_proj = nn.Linear(dim, 2 * self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, dim)
        
        # 卷积层用于局部上下文
        self.conv = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=self.hidden_dim,
            bias=True
        )
        
        # SSM核心
        self.ssm = SSMKernel(self.hidden_dim, d_state=d_state)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, pos=None):
        """前向传播，兼容ViT块接口"""
        # 保存残差
        residual = x
        
        # 规范化
        x = self.norm(x)
        
        # 投影并分离
        x = self.in_proj(x)
        x, gate = x.chunk(2, dim=-1)
        
        # 卷积处理（局部上下文）
        x_conv = x.permute(0, 2, 1)  # [B, D, L]
        x_conv = self.conv(x_conv)
        x_conv = x_conv[:, :, :x.shape[1]]  # 调整到原始序列长度
        x_conv = x_conv.permute(0, 2, 1)  # [B, L, D]
        
        # 激活
        x = F.silu(x_conv)
        
        # 计算步长/选择性
        delta = F.softplus(gate)
        
        # SSM处理
        x = self.ssm(x, delta)
        
        # 输出投影
        x = self.out_proj(x)
        x = self.dropout(x)
        
        # 残差连接
        return residual + x


class Mamba(nn.Module):
    """多层Mamba块"""
    def __init__(self, dim, depth, d_state=16, d_conv=4, expand_factor=2.0, dropout=0.0):
        super().__init__()
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
        for layer in self.layers:
            x = layer(x, pos)
        return x


class AsymmetricVitMambaCroCo(CroCoNet):
    """混合ViT-Mamba模型，使用1/4 ViT + 1/2 Mamba + 1/4 ViT结构"""
    
    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        # 初始化基本参数
        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
            
        self.patch_embed_cls = cfg.patch_embed_cls
        
        # Mamba特有参数
        self.mamba_d_state = cfg.mamba_d_state
        self.mamba_d_conv = cfg.mamba_d_conv
        self.mamba_expand_factor = cfg.mamba_expand_factor
        self.mamba_dropout = cfg.mamba_dropout
        
        # 准备初始化参数
        self.croco_args = fill_default_args(croco_mamba_params[cfg.model], CroCoNet.__init__)
        
        # 初始化基类
        super().__init__(**self.croco_args)
        
        # 修改编码器为混合结构
        self._replace_middle_blocks_with_mamba()
        
        # 创建第二解码器（用于非对称处理）
        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(self.dec_blocks)
            
        # 相机内参处理
        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token':
            self.intrinsic_encoder = nn.Linear(9, 1024)
    
    def _replace_middle_blocks_with_mamba(self):
        """将中间1/2的编码器块替换为Mamba块"""
        total_blocks = len(self.enc_blocks)
        vit_blocks_first = total_blocks // 4
        vit_blocks_last = total_blocks // 4
        mamba_blocks_count = total_blocks - vit_blocks_first - vit_blocks_last
        
        # 保留前1/4和后1/4的ViT块
        first_blocks = self.enc_blocks[:vit_blocks_first]
        last_blocks = self.enc_blocks[-vit_blocks_last:] if vit_blocks_last > 0 else []
        
        # 创建中间的Mamba块
        mamba_section = MambaAdapter(
            dim=self.enc_embed_dim,
            depth=mamba_blocks_count,
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            expand_factor=self.mamba_expand_factor,
            dropout=self.mamba_dropout
        )
        
        # 构建新的编码器块列表
        new_enc_blocks = nn.ModuleList([])
        new_enc_blocks.extend(first_blocks)
        new_enc_blocks.append(mamba_section)
        new_enc_blocks.extend(last_blocks)
        
        # 替换编码器块
        self.enc_blocks = new_enc_blocks
    
    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # 从编码器到解码器的投影
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # 解码器的transformer块
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # 最终规范化层
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # 为第二个解码器复制所有权重（如果不存在）
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # 用于下游模型
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder':  [self.mask_token, self.patch_embed, self.enc_blocks, self.enc_norm, self.decoder_embed, self.dec_blocks, self.dec_blocks2, self.dec_norm],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ 无预测头 """
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # 将图像嵌入为patches
        x, pos = self.patch_embed(image, true_shape=true_shape)

        if intrinsics_embed is not None:
            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == 'token':
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pose = pos[:, 0:1, :].clone()
                add_pose[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
                pos = torch.cat((pos, add_pose), dim=1)

        # 位置嵌入
        assert self.enc_pos_embed is None

        # 应用编码器块
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, intrinsics_embed1=None, intrinsics_embed2=None):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0),
                                             torch.cat((intrinsics_embed1, intrinsics_embed2), dim=0) if intrinsics_embed1 is not None else None)
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1, intrinsics_embed1)
            out2, pos2, _ = self._encode_image(img2, true_shape2, intrinsics_embed2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2, force_asym=False):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # 获取true_shape（如果可用），否则假设图像形状是真实形状
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        intrinsics_embed1 = view1.get('intrinsics_embed', None)
        intrinsics_embed2 = view2.get('intrinsics_embed', None)

        if force_asym or not is_symmetrized(view1, view2):
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2, intrinsics_embed1, intrinsics_embed2)
        else:
            # 计算半个前向传递
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, extra_embed1=None, extra_embed2=None):
        final_output = [(f1, f2)]  # 投影前

        if extra_embed1 is not None:
            f1 = torch.cat((f1, extra_embed1), dim=-1)
        if extra_embed2 is not None:
            f2 = torch.cat((f2, extra_embed2), dim=-1)

        # 投影到解码器维度
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1端
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2端
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # 存储结果
            final_output.append((f1, f2))

        # 规范化最后的输出
        del final_output[1]  # 与final_output[0]重复
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, context, symmetrize_batch=False, return_views=False):
        b, v, _, h, w = context["image"].shape
        device = context["image"].device

        view1, view2 = ({'img': context["image"][:, 0]},
                        {'img': context["image"][:, 1]})

        # 在编码器中嵌入相机内参
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            intrinsic_emb = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree)
            view1['img'] = torch.cat((view1['img'], intrinsic_emb[:, 0]), dim=1)
            view2['img'] = torch.cat((view2['img'], intrinsic_emb[:, 1]), dim=1)

        if self.intrinsics_embed_loc == 'encoder' and (self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
            view1['intrinsics_embed'] = intrinsic_embedding[:, 0].unsqueeze(1)
            view2['intrinsics_embed'] = intrinsic_embedding[:, 1].unsqueeze(1)

        if symmetrize_batch:
            instance_list_view1, instance_list_view2 = [0 for _ in range(b)], [1 for _ in range(b)]
            view1['instance'] = instance_list_view1
            view2['instance'] = instance_list_view2
            view1['idx'] = instance_list_view1
            view2['idx'] = instance_list_view2
            view1, view2 = make_batch_symmetric(view1, view2)

            # 编码两张图像 --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, force_asym=False)
        else:
            # 编码两张图像 --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, force_asym=True)

        if self.intrinsics_embed_loc == 'decoder':
            # 注意：降采样硬编码为16
            intrinsic_emb = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree, downsample=16, merge_hw=True)
            dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, intrinsic_emb[:, 0], intrinsic_emb[:, 1])
        else:
            dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
            dec1, dec2 = list(dec1), list(dec2)
            for i in range(len(dec1)):
                dec1[i] = dec1[i][:, :-1]
                dec2[i] = dec2[i][:, :-1]

        if return_views:
            return dec1, dec2, shape1, shape2, view1, view2
        return dec1, dec2, shape1, shape2

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024


# 骨干网络封装
class BackboneVitMamba(Backbone):
    """ViT-Mamba混合骨干网络"""
    
    def __init__(self, cfg, d_in=3) -> None:
        super().__init__()
        self.model = AsymmetricVitMambaCroCo(cfg, d_in)
        
    def forward(self, context, symmetrize_batch=False):
        return self.model(context, symmetrize_batch)
        
    @property
    def patch_size(self) -> int:
        return self.model.patch_size
        
    @property
    def d_out(self) -> int:
        return self.model.d_out
