import math
import warnings
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Any
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath
from mmengine.registry import MODELS
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from torch.nn.init import xavier_uniform_, constant_
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from .decode_head import BaseDecodeHead
from ..utils.mamba2 import LayerNorm2d, InitMamba, cross_scan_fn, cross_merge_fn, selective_scan_fn
from ..utils.dcnv3.functions import DCNv3Function


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()
    
    def _init_weights(self):
        xavier_uniform_(self.fc1.weight.data)
        constant_(self.fc1.bias.data, 0.)
        xavier_uniform_(self.fc2.weight.data)
        constant_(self.fc2.bias.data, 0.)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DeformableConv(nn.Module):
    def __init__(
            self,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            ffn_ratio=2,
            ffn_drop_rate=0.0,
            center_feature_scale=False,
            ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        hidden_dim = in_channels * ffn_ratio
        if in_channels % group != 0:
            raise ValueError(
                f'in_channels must be divisible by group, but got {in_channels} and {group}')
        _d_per_group = in_channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not self._is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set in_channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = in_channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=in_channels
                )
        self.bn = nn.SyncBatchNorm(in_channels)
        self.act = nn.GELU()
        
        self.offset = nn.Linear(in_channels, group * kernel_size * kernel_size * 2)
        self.mask = nn.Linear(in_channels, group * kernel_size * kernel_size)
        self.input_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = Mlp(in_channels, hidden_dim, out_channels, act_layer=nn.GELU, drop=ffn_drop_rate)
        self._init_weights()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, in_channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _init_weights(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)


    @staticmethod
    def _is_power_of_2(n):
        if (not isinstance(n, int)) or (n < 0):
            raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")

        return (n & (n - 1) == 0) and n != 0


    def forward(self, input):
        """
        :param query                       (B, H, W, C)
        :return output                     (B, H, W, C)
        """
        B, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = rearrange(input, 'b h w c -> b c h w')
        x1 = rearrange(self.act(self.bn(self.dw_conv(x1))), 'b c h w -> b h w c')
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(B, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(B, H, W, -1).type(dtype)
        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256)

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # B, H, W, groups -> B, H, W, groups, 1 -> B, H, W, groups, _d_per_group -> B, H, W, in_channel
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.in_channel // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = rearrange(self.out_proj(x), 'b h w c -> b c h w')
        return x


def create_block(
        dim,
        drop_path,
        ssm_d_state,
        ssm_ratio,
        ssm_dt_rank,
        ssm_drop_rate,
        if_up_sample,
        ffn_drop_rate,
):
    blocks = []
    for i in range(len(drop_path)):
        blocks.append(HFBlock(
            dim=dim, 
            drop_path=drop_path[i],
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_drop_rate=ssm_drop_rate,
            if_up_sample=if_up_sample,
            ffn_drop_rate=ffn_drop_rate,
            ))
    return nn.ModuleList(blocks)


class SS2D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        dropout=0.0,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.out_norm = LayerNorm2d(self.d_inner)
        self._forward = partial(self.forward_core, no_einsum=True)
        self.act = nn.GELU()
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
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

    def forward_core(
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

    def forward(self, x: torch.Tensor):
        y = self._forward(x)
        out = self.dropout(self.act(y))
        return out


class HFBlock(nn.Module):
    """ Hierarchical-Features Fusion Block. """
    def __init__(
        self,
        dim=96,                   
        drop_path=0,                 
        norm_layer=LayerNorm2d,                   
        ssm_d_state=16,                 
        ssm_ratio=2.0,                        
        ssm_dt_rank="auto",                                  
        ssm_drop_rate=0,  
        if_up_sample=False,   
        ffn_drop_rate=0.0,          
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.if_up_sample = if_up_sample
        self.conv = nn.Conv2d(dim*2, dim*2, kernel_size=1, stride=1)
        self.norm1 = norm_layer(dim*2, eps=1e-5)
        if self.if_up_sample:
            self.upsampling = nn.PixelShuffle(2)
            self.ssm = SS2D(
                d_model=dim // 2, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                dropout=ssm_drop_rate,
                )
            self.channel_mix = Mlp(dim // 2, dim, dim // 2, drop=ffn_drop_rate)
            self.norm2 = norm_layer(dim // 2, eps=1e-5)
        else:
            self.ssm = SS2D(
                d_model=dim * 2, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                dropout=ssm_drop_rate,
                )
            self.channel_mix = Mlp(dim * 2, dim * 4, dim * 2, drop=ffn_drop_rate)
            self.norm2 = norm_layer(dim * 2, eps=1e-5)
        self.drop_path = DropPath(drop_path)


    def forward(self, fmap_dp, fmap_sh):
        """
        Reconstruct high-resolution images for segmentation by fusing 
        different hierarchical low-resolution feature maps
        
        args: 
            fmap_dp: feature map from deep stage
            fmap_sh: feature map from shallow stage
        """
        x = self.norm1(self.conv(torch.cat([fmap_dp, fmap_sh], dim=1)))
        if self.if_up_sample:
            x = self.upsampling(x)
        x = x + self.norm2(self.ssm(x))
        x = rearrange(x, 'b c h w -> b h w c')
        x = x + self.drop_path(self.channel_mix(x))
        return rearrange(x, 'b h w c -> b c h w')


@MODELS.register_module()
class SegMambaHeadV62(BaseDecodeHead):
    def __init__(
        self,  
        depths=[2, 2, 2, 2], 
        in_channels=[768, 384, 192, 96],
        kernel_sizes=[3, 5, 7, 9],
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_drop_rate=0.0, 
        ffn_drop_rate: float=0.0,
        ffn_ratio=2,
        channels=512,
        drop_path_rate=0.1, 
        pretrained=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(in_channels=in_channels,
                         channels=channels,
                         init_cfg=init_cfg, **kwargs)
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        if isinstance(in_channels, int):
            in_channels = [int(in_channels * 2 ** i_layer) for i_layer in range(len(depths))]
        self.in_channels = in_channels
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Fusion and Upsampling layers
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            if_up_sample = True if i < len(depths) - 1 else False
            self.stages.append(create_block(
                dim=self.in_channels[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_drop_rate=ssm_drop_rate,
                if_up_sample=if_up_sample,
                ffn_drop_rate=ffn_drop_rate,
                ))
        
        # Deformable Transform
        self.deformable_conv = nn.ModuleList([
            DeformableConv(
                in_channels=dim,
                out_channels=dim, 
                kernel_size=ks,
                dw_kernel_size=None,
                stride=1,
                pad=(ks - 1) // 2,
                dilation=1,
                group=4,
                offset_scale=1.0,
                center_feature_scale=False,
                ffn_drop_rate=ffn_drop_rate,
            ) for dim, ks in zip(in_channels, kernel_sizes)
        ])
        # FFN
        in_features = in_channels[-1] * 2
        self.ffn = Mlp(in_features=in_features, 
                       hidden_features=in_features * ffn_ratio, 
                       drop=ffn_drop_rate)

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, strict=False, logger=None)
        elif self.init_cfg is not None:
            super().init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def forward(self, inputs):
        """
        args:
            input (List): 4 stage output of Encoder
        """
        inputs = self._transform_inputs(inputs)
        x_trans = []
        for input, dconv in zip(inputs, self.deformable_conv[::-1]):
            input = rearrange(input, 'b c h w -> b h w c')
            x = dconv(input)
            x_trans.append(x)
        x = x_trans[-1]
        index = -1
        for stage in self.stages:
            for block in stage:
                x = block(x, x_trans[index])
            index -= 1
        x = rearrange(x, 'b c h w -> b h w c')
        out = self.ffn(x)
        out = rearrange(out, 'b h w c -> b c h w')
        out = self.cls_seg(out)
        return out

