import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Any
from collections import OrderedDict
from timm.models.layers import DropPath

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from .csm_triton import cross_scan_fn, cross_merge_fn
from .csms6s import selective_scan_fn


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = Linear2d(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear2d(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InitMamba:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        disable_z=True,
        disable_z_act=False,
        output_with_act=False,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.with_dconv = d_conv > 1
        self.disable_z = disable_z
        self.disable_z_act = disable_z_act
        self.out_norm = LayerNorm2d(self.d_inner)
        self.forward_core = partial(self.forward_corev2, no_einsum=True)

        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear2d(self.d_model, d_proj, bias=bias)
        self.act = act_layer()
        
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs)

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        self.out_act = nn.GELU() if output_with_act else nn.Identity()
        self.out_proj = Linear2d(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = InitMamba.init_dt_A_D(
            self.d_state, 
            self.dt_rank, 
            self.d_inner, 
            dt_scale, 
            dt_init, 
            dt_min, 
            dt_max, 
            dt_init_floor, 
            k_group=self.k_group,
            )

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        ssoflex=True, # True: input 16 or 32 output 32 False: output dtype as input
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        selective_scan_backend = None,
        scan_force_torch = False,
        **kwargs,
    ):
        assert selective_scan_backend in [None, "oflex", "core", "mamba", "torch"]
        delta_softplus = True
        out_norm = self.out_norm

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        x_proj_bias = getattr(self, "x_proj_bias", None)
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=0, force_torch=scan_force_torch)
        if no_einsum:   # True
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
        Ds = self.Ds.to(torch.float) # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus).view(B, K, -1, H, W)
        y = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=0, force_torch=scan_force_torch).view(B, -1, H, W)
        y = out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


def create_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
    stride = patch_size // 2
    kernel_size = stride + 1
    padding = 1
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.Identity(),
        (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
        nn.Identity(),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.Identity(),
        (norm_layer(embed_dim) if patch_norm else nn.Identity()),
    )


def create_downsample_layer(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
    return nn.Sequential(
        nn.Identity(),
        nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
        nn.Identity(),
        norm_layer(out_dim),
    )


def create_block(
    dim=96,                     
    drop_path=[0.1, 0.1],       
    norm_layer=nn.LayerNorm,    # LayerNorm2d
    downsample=nn.Identity(),   # downsample
    ssm_d_state=16,             # 1
    ssm_ratio=2.0,              # 1.0
    ssm_dt_rank="auto",         # 'auto'
    ssm_act_layer=nn.SiLU,      # nn.SiLU
    ssm_conv=3,                 # 3
    ssm_conv_bias=True,         # False
    ssm_drop_rate=0.0,          # 0.0
    mlp_ratio=4.0,              # 4.0
    mlp_act_layer=nn.GELU,      # nn.GELU
    mlp_drop_rate=0.0
    ):

    blocks = []
    for d in range(len(drop_path)):
        blocks.append(VSSBlock(
            hidden_dim=dim, 
            drop_path=drop_path[d],
            norm_layer=norm_layer,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
        ))
    
    return nn.Sequential(OrderedDict(
        blocks=nn.Sequential(*blocks,),
        downsample=downsample,
    ))


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,                    # self.dims(i_layer)
        drop_path: float = 0,                   # len() = 4
        norm_layer: nn.Module = nn.LayerNorm,   # LayerNorm2d                 
        ssm_d_state: int = 16,                  # 1
        ssm_ratio=2.0,                          # 1.0
        ssm_dt_rank: Any = "auto",              # 'auto'
        ssm_act_layer=nn.SiLU,                  # nn.SiLU
        ssm_conv: int = 3,                      # 3
        ssm_conv_bias=True,                     # False
        ssm_drop_rate: float = 0,               # 0.0
        mlp_ratio=4.0,                          # 4.0
        mlp_act_layer=nn.GELU,                  # nn.GELU
        mlp_drop_rate: float = 0.0,             # 0.0
        post_norm: bool = False,                # 
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate)

    def forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x