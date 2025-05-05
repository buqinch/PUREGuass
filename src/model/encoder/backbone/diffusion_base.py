import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=time.device) / (half - 1)
        )
        args = time[:, None].float() * freqs[None, :]
        return torch.cat((args.sin(), args.cos()), dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64):
        super().__init__()
        context_dim = context_dim or query_dim
        inner = heads * dim_head
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.proj_out = nn.Conv1d(inner, query_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Lq, _ = x.shape
        _, Lc, _ = context.shape
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)        # [B, Lc, inner]
        v = self.to_v(context)

        q = q.view(B, Lq, h, -1).permute(0, 2, 1, 3)   # → [B,h,Lq,dh]
        k = k.view(B, Lc, h, -1).permute(0, 2, 1, 3)   # → [B,h,Lc,dh]
        v = v.view(B, Lc, h, -1).permute(0, 2, 1, 3)


        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        out = out.permute(0, 2, 1, 3).reshape(B, Lq, -1)
        out = out.permute(0, 2, 1)
        return self.proj_out(out).permute(0, 2, 1)


class ResBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: Optional[int] = None, guidance_dim: Optional[int] = None):
        super().__init__()
        self.has_time = time_emb_dim is not None
        if self.has_time:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out)
            )
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, dim_out),
            nn.SiLU(),
            nn.Conv1d(dim_out, dim_out, kernel_size=3, padding=1)
        )
        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        self.guidance_attn = CrossAttention(dim_out, guidance_dim) if guidance_dim is not None else None

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None, guidance: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.block1(x)
        if self.has_time and time_emb is not None:
            t = self.time_mlp(time_emb)
            h = h + t.unsqueeze(-1)
        h = self.block2(h)
        if self.guidance_attn is not None and guidance is not None:
            # [B, C, L] → [B, L, C]
            if guidance.dim() == 3:
                g = guidance.permute(0, 2, 1)
            # [B, C] → 扩成 [B, L, C]
            elif guidance.dim() == 2:
                g = guidance.unsqueeze(1).repeat(1, h.size(2), 1)
            else:
                raise ValueError(f"Unexpected guidance shape {guidance.shape}")
            # cross-attn 要求 h_attn:[B, L, D], g:[B, L, C]
            h_attn = h.permute(0, 2, 1)
            h2 = self.guidance_attn(h_attn, g)
            h = h + h2.permute(0, 2, 1)

        return h + self.res_conv(x)


class GGDM(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        guidance_dim: Optional[int] = None,
        hidden_dims: List[int] = [256, 512, 1024, 512, 256],  # 对称，符合日志输出
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.guidance_dim = guidance_dim or feature_dim

        time_dim = hidden_dims[0] * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dims[0]),
            nn.Linear(hidden_dims[0], time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # 输入投影：C_in → hidden_dims[0]
        self.proj_in = nn.Conv1d(feature_dim, hidden_dims[0], kernel_size=1)

        # down path: hidden_dims[i] → hidden_dims[i+1]
        dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        self.downs = nn.ModuleList()
        for din, dout in dims:
            self.downs.append(nn.ModuleList([
                ResBlock(din, dout, time_dim, self.guidance_dim),
                ResBlock(dout, dout, time_dim, self.guidance_dim),
                nn.Conv1d(dout, dout, kernel_size=4, stride=2, padding=1)
            ]))

        # 中间层
        mid_ch = hidden_dims[-1]
        self.mid1 = ResBlock(mid_ch, mid_ch, time_dim, self.guidance_dim)
        self.mid_attn = CrossAttention(mid_ch, self.guidance_dim)
        self.mid2 = ResBlock(mid_ch, mid_ch, time_dim, self.guidance_dim)

        # up path: 正确使用 reversed dims 中的 din、dout
        self.ups = nn.ModuleList()
        prev_ch = mid_ch
        for din, dout in reversed(dims):
            # 上采样到 skip channels = dout
            up_conv = nn.ConvTranspose1d(prev_ch, dout, kernel_size=4, stride=2, padding=1)
            # concat 后 channels = dout + dout，再映射回 din
            block1 = ResBlock(dout * 2, din, time_dim, self.guidance_dim)
            block2 = ResBlock(din, din, time_dim, self.guidance_dim)
            self.ups.append(nn.ModuleList([up_conv, block1, block2]))
            prev_ch = din

        # 输出投影：hidden_dims[0] → C_in
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv1d(hidden_dims[0], feature_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, guidance: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, L = x.shape
        t = self.time_mlp(time)

        # proj in
        h = self.proj_in(x)
        # logger.info(f"[proj_in] → {h.shape}")

        # 处理超长序列
        extra = None
        if L > 256:
            extra = h[:, :, 256:].detach().clone()
            h = h[:, :, :256]
            # logger.info(f"[split]  h={h.shape}, extra={extra.shape}")

        # down sampling
        skips = []
        for idx, (block1, block2, down) in enumerate(self.downs):
            h = block1(h, t, guidance)
            h = block2(h, t, guidance)
            skips.append(h)
            h = down(h)
            # logger.info(f"[down {idx}] → {h.shape}")

        # middle
        h = self.mid1(h, t, guidance)
        if guidance is not None:
            hm = h.permute(0, 2, 1)
            gm = guidance.permute(0, 2, 1) if guidance.dim() == 3 else guidance
            h = h + self.mid_attn(hm, gm).permute(0, 2, 1)
        h = self.mid2(h, t, guidance)
        # logger.info(f"[mid] → {h.shape}")

        # up sampling
        for idx, (up_conv, block1, block2) in enumerate(self.ups):
            skip = skips.pop()
            h = up_conv(h)
            h = torch.cat([h, skip], dim=1)
            h = block1(h, t, guidance)
            h = block2(h, t, guidance)
            # logger.info(f"[up {idx}] → {h.shape}")

        # restore extra
        if extra is not None:
            h = torch.cat([h, extra], dim=2)
            # logger.info(f"[restore extra] → {h.shape}")

        out = self.final_conv(h)
        # logger.info(f"[final_conv] → {out.shape}")
        return out


class FeatureGGDM:
    def __init__(
        self,
        model: GGDM,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_diffusion_steps: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.device = device
        self.num_diffusion_steps = num_diffusion_steps

        self.betas = torch.linspace(beta_start, beta_end, num_diffusion_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        sa = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        so = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sa * x + so * noise, noise

    @torch.no_grad()
    def denoise_step(self, x: torch.Tensor, t: torch.Tensor, guidance: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_noise = self.model(x, t, guidance)
        alpha = self.alphas[t].view(-1, 1, 1)
        alpha_cum = self.alphas_cumprod[t].view(-1, 1, 1)
        beta = self.betas[t].view(-1, 1, 1)

        if t[0] > 0:
            noise = torch.randn_like(x)
            var = ((1 - alpha_cum) / (1 - alpha) * beta).sqrt()
        else:
            noise, var = 0, 0

        mean = (1 / alpha.sqrt()) * (x - (beta / ((1 - alpha_cum).sqrt() * (1 - alpha))) * pred_noise)
        return mean + var * noise

    @torch.no_grad()
    def optimize_features(self, features: torch.Tensor, guidance: Optional[torch.Tensor] = None, num_steps: int = 50, guidance_scale: float = 1.0, noise_level: float = 0.2) -> torch.Tensor:
        assert 0 < num_steps <= self.num_diffusion_steps
        x = features.to(self.device)
        guidance = guidance.to(self.device) if guidance is not None else None
        step_size = self.num_diffusion_steps // num_steps
        start_step = int(self.num_diffusion_steps * noise_level)
        times = list(range(start_step, 0, -step_size))[::-1]
        for t in times:
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            if guidance is not None and guidance_scale > 1.0:
                uncond = self.model(x, t_batch, None)
                cond   = self.model(x, t_batch, guidance)
                model_out = uncond + guidance_scale * (cond - uncond)
                x = self.denoise_step(x, t_batch, model_out)
            else:
                x = self.denoise_step(x, t_batch, guidance)
        return x