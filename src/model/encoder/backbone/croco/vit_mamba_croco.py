from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, DecoderBlock
from .croco import CroCoNet
from .misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, make_batch_symmetric
from .blocks_mamba import Mamba, MambaBlock

@dataclass
class BackboneCrocoCfg:
    name: Literal["croco", "croco_multi", "vit_mamba"]
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]
    patch_embed_cls: str = 'PatchEmbedDust3R'
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = 'token'
    mamba_d_state: int = 16            # Mamba状态维度
    mamba_d_conv: int = 4              # Mamba卷积核大小
    mamba_expand_factor: float = 2.0   # Mamba展开系数
    mamba_dropout: float = 0.0         # Mamba dropout率


class ViTMambaCrocoNet(CroCoNet):
    """混合ViT和Mamba的模型，结构为1/4ViT + 1/2Mamba + 1/4ViT"""
    
    def __init__(self, **kwargs):
        # 从kwargs中提取Mamba特定参数
        self.mamba_d_state = kwargs.pop('mamba_d_state', 16)
        self.mamba_d_conv = kwargs.pop('mamba_d_conv', 4)
        self.mamba_expand_factor = kwargs.pop('mamba_expand_factor', 2.0)
        self.mamba_dropout = kwargs.pop('mamba_dropout', 0.0)
        
        super().__init__(**kwargs)
        
        # 替换中间部分的编码器块为Mamba块
        self._replace_mid_blocks_with_mamba()
        
    def _replace_mid_blocks_with_mamba(self):
        """将中间1/2的ViT块替换为Mamba块"""
        enc_depth = self.enc_depth
        vit_blocks_start = enc_depth // 4
        vit_blocks_end = enc_depth - vit_blocks_start
        
        # 创建新的编码器块列表
        new_enc_blocks = nn.ModuleList()
        
        # 前1/4保持ViT块
        for i in range(vit_blocks_start):
            new_enc_blocks.append(self.enc_blocks[i])
        
        # 中间1/2替换为Mamba块
        mamba_section = Mamba(
            dim=self.enc_embed_dim,
            depth=vit_blocks_end - vit_blocks_start,
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            expand_factor=self.mamba_expand_factor,
            dropout=self.mamba_dropout
        )
        
        # 包装Mamba部分
        class MambaSection(nn.Module):
            def __init__(self, mamba_model):
                super().__init__()
                self.mamba = mamba_model
                
            def forward(self, x, pos=None):
                return self.mamba(x, pos)
                
        new_enc_blocks.append(MambaSection(mamba_section))
        
        # 后1/4保持ViT块
        for i in range(vit_blocks_end, enc_depth):
            new_enc_blocks.append(self.enc_blocks[i])
        
        # 替换编码器块
        self.enc_blocks = new_enc_blocks


class AsymmetricViTMambaCroCo(ViTMambaCrocoNet):
    """支持非对称解码器的ViT-Mamba混合模型"""
    
    def __init__(self, cfg, d_in=None):
        # 提取Mamba配置
        mamba_args = {
            'mamba_d_state': getattr(cfg, 'mamba_d_state', 16),
            'mamba_d_conv': getattr(cfg, 'mamba_d_conv', 4),
            'mamba_expand_factor': getattr(cfg, 'mamba_expand_factor', 2.0),
            'mamba_dropout': getattr(cfg, 'mamba_dropout', 0.0),
        }
        
        # 基本CroCo参数
        croco_args = fill_default_args(cfg.model, CroCoNet.__init__)
        croco_args.update(mamba_args)

        # 初始化其他成员变量
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
        
        # 初始化基类
        super().__init__(**croco_args)
        
        # 创建第二个解码器（用于非对称处理）
        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(self.dec_blocks)
            
        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token':
            self.intrinsic_encoder = nn.Linear(9, 1024)
    
    def load_state_dict(self, ckpt, **kw):
        # 复制所有权重到第二个解码器（如果不存在）
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)
    
    def _decoder(self, f1, pos1, f2, pos2, extra_embed1=None, extra_embed2=None):
        """非对称解码过程"""
        final_output = [(f1, f2)]  # 在投影前
        
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
        del final_output[1]  # 移除与final_output[0]重复的部分
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)
    
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
        # 获取真实形状（如果可用），否则假设图像形状是真实形状
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
    
    def forward(self, context, symmetrize_batch=False, return_views=False):
        b, v, _, h, w = context["image"].shape
        device = context["image"].device

        view1, view2 = ({'img': context["image"][:, 0]},
                        {'img': context["image"][:, 1]})

        # 在编码器中嵌入相机内参
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            from ....geometry.camera_emb import get_intrinsic_embedding
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

            # 编码两个图像 --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, force_asym=False)
        else:
            # 编码两个图像 --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, force_asym=True)

        if self.intrinsics_embed_loc == 'decoder':
            # 注意：降采样硬编码为16
            from ....geometry.camera_emb import get_intrinsic_embedding
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
