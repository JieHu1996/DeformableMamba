import math
import warnings
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Any
from functools import partial
from timm.models.layers import DropPath
from mmengine.registry import MODELS
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from ..utils.mamba2 import Mlp, LayerNorm2d, Linear2d, InitMamba, cross_scan_fn, cross_merge_fn, selective_scan_fn
from .decode_head import BaseDecodeHead


def create_stage(
    dim=96, 
    kernel_size=3,                    
    drop_path=[0.1, 0.1],       
    norm_layer=LayerNorm2d,  
    ssm_d_state=16,            
    ssm_ratio=2.0,             
    ssm_dt_rank="auto",                
    ssm_drop_rate=0.0,                     
    chan_act=nn.GELU,                      
    chan_drop_rate: float = 0.0,
    if_up_sample=False,
):

    blocks = []
    for i in range(len(drop_path)):
        blocks.append(DMF(
            dim=dim, 
            kernel_size=kernel_size,
            drop_path=drop_path[i],
            norm_layer=norm_layer,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_drop_rate=ssm_drop_rate,       
            chan_act=chan_act,                   
            chan_drop_rate=chan_drop_rate, 
            if_up_sample=if_up_sample))
    
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


class DFT(nn.Module):
    """ Deformable Feature Transformation. """
    def __init__(self, in_chans=768, out_chans=768, kernel_size=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=out_chans, 
                              kernel_size=kernel_size, padding=self.padding)
        self.offset_conv = nn.Conv2d(in_channels=in_chans, out_channels=2*kernel_size*kernel_size, 
                                     kernel_size=kernel_size, padding=self.padding)
        self.mask_conv = nn.Conv2d(in_channels=in_chans, out_channels=kernel_size * kernel_size, 
                                        kernel_size=kernel_size, padding=self.padding)
        self.norm = norm_layer(out_chans)
        self.act = nn.GELU()
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)      


    def transform(self, x):
        max_offset = torch.tensor((min(x.shape[-2], x.shape[-1]) // 4)).to(x.device)
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        mask = 2. * torch.sigmoid(self.mask_conv(x))
        x = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.proj.weight,
            bias=self.proj.bias,
            padding=self.padding,
            mask=mask,
            stride=1,
            )
        return x

    def forward(self, x):
        x = self.transform(x)
        x = self.act(self.norm(x))
        return x


class DMF(nn.Module):
    """ DeformableMamba Features Fusion Block. """
    def __init__(
        self,
        dim: int = 0, 
        kernel_size: int = 3,                  
        drop_path: float = 0,                 
        norm_layer: nn.Module = LayerNorm2d,                   
        ssm_d_state: int = 16,                 
        ssm_ratio=2.0,                        
        ssm_dt_rank: Any = "auto",                                  
        ssm_drop_rate: float = 0,  
        if_up_sample=False,                         
        chan_act=nn.GELU,                      
        chan_drop_rate: float = 0.0,            
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.if_up_sample = if_up_sample
        # 2d state space model
        self.global_branch = SS2D(
            d_model=dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            dropout=ssm_drop_rate,
            )  
        self.local_branch = DFT(in_chans=dim, out_chans=dim, kernel_size=kernel_size, norm_layer=norm_layer)
        self.conv = nn.Conv2d(dim*2, dim*2, kernel_size=1, stride=1)
        # self.norm1 = nn.SyncBatchNorm(dim * 2)
        # self.norm2 = nn.SyncBatchNorm(dim * 2)
        self.norm1 = norm_layer(dim*2, eps=1e-5)
        self.norm2 = norm_layer(dim*2, eps=1e-5)
        self.drop_path = DropPath(drop_path)
        if self.if_up_sample:
            self.upsampling = nn.PixelShuffle(2)
            self.channel_mix = Mlp(
                    in_features=dim // 2, 
                    hidden_features=dim, 
                    out_features=dim // 2,
                    act_layer=chan_act, 
                    drop=chan_drop_rate
                    )
            self.norm3 = nn.SyncBatchNorm(dim // 2)

    def forward(self, x_deep, x_shallow):
        """
        Reconstruct high-resolution images for segmentation by fusing 
        different hierarchical low-resolution feature maps
        
        args: 
            x_deep: Deep feature map
            x_shallow: Shallow feature map
        """
        x_deep = self.global_branch(x_deep)
        x_shallow = self.local_branch(x_shallow)
        x = self.norm1(torch.cat([x_deep, x_shallow], dim=1))
        x = x + self.drop_path(self.norm2(self.conv(x)))
        if self.if_up_sample:
            x = self.upsampling(x)
            x = x + self.channel_mix(self.norm3(x))
        return x


@MODELS.register_module()
class DeformableMambaHead(BaseDecodeHead):
    def __init__(
        self,  
        depths=[2, 2, 2, 2], 
        kernel_sizes=[3, 3, 3, 3],
        in_channels=[768, 384, 192, 96],
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_drop_rate=0.0,  
        chan_act=nn.GELU,
        chan_drop_rate:float=0.0,
        ffn_drop_rate: float=0.0,
        ffn_ratio=2,
        ffn_act_layer=nn.GELU,
        channels=512,
        drop_path_rate=0.1, 
        norm_layer="ln2d",
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
        norm_layer = LayerNorm2d
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            if_up_sample = True if i < len(depths) - 1 else False
            self.stages.append(create_stage(
                dim=self.in_channels[i],
                kernel_size=kernel_sizes[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_drop_rate=ssm_drop_rate,       
                chan_act=chan_act,                   
                chan_drop_rate=chan_drop_rate, 
                if_up_sample=if_up_sample,
                ))
        # FFN
        in_features = in_channels[-1] * 2
        hidden_features = in_features * ffn_ratio 
        self.ffn = Mlp(in_features=in_features, 
                       hidden_features=hidden_features, 
                       act_layer=ffn_act_layer, 
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
        x = inputs[-1]
        index = -1
        for stage in self.stages:
            for block in stage:
                x = block(x, inputs[index])
            index -= 1
        out = self.ffn(x)
        out = self.cls_seg(out)
        return out


