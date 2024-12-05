import math
import warnings
import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from typing import Any
from timm.models.layers import DropPath
from mmcv.cnn import ConvModule
from mmengine.registry import MODELS
from mmengine.model.weight_init import (constant_init, kaiming_init, trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from ..utils.mamba2 import SS2D, Mlp, LayerNorm2d
from .decode_head import BaseDecodeHead


def create_stage(
    dim=96,                     
    drop_path=[0.1, 0.1],       
    norm_layer=LayerNorm2d,  
    ssm_d_state=16,            
    ssm_ratio=2.0,             
    ssm_dt_rank="auto",         
    ssm_act_layer=nn.SiLU,      
    ssm_conv=3,                
    ssm_conv_bias=True,        
    ssm_drop_rate=0.0,                              
    if_up_sample=False,
):

    blocks = []
    blocks.append(HFBlock(
        dim=dim, 
        drop_path=drop_path,
        norm_layer=norm_layer,
        ssm_d_state=ssm_d_state,
        ssm_ratio=ssm_ratio,
        ssm_dt_rank=ssm_dt_rank,
        ssm_act_layer=ssm_act_layer,
        ssm_conv=ssm_conv,
        ssm_conv_bias=ssm_conv_bias,
        ssm_drop_rate=ssm_drop_rate,                
        if_up_sample=if_up_sample))
    
    return nn.ModuleList(blocks)


class DDConv2d(nn.Module):
    """
    Depth-wise deformable 2d-Convlution.
    """
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=1, padding=1, dilation=1, norm_layer=LayerNorm2d):
        super().__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_chans)
        self.offset_conv = nn.Conv2d(in_chans, 2*kernel_size*kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.mask_conv = nn.Conv2d(in_chans, kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.norm = norm_layer(out_chans)
        self.act = nn.GELU()
        self._init_weights()
    
    def _init_weights(self):
        for n, m in self.named_modules():
            if n == 'pointwise':
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            else:
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0.)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)

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
            stride=self.stride,
            dilation=self.dilation,
            mask=mask,
            )
        return x

    def forward(self, x):
        x = self.pointwise(self.transform(x))
        x = self.act(self.norm(x))
        return x


class HFBlock(nn.Module):
    """ Hierarchical-Features Fusion Block. """
    def __init__(
        self,
        dim: int = 0,                   
        drop_path: float = 0,                 
        norm_layer: nn.Module = LayerNorm2d,                   
        ssm_d_state: int = 16,                 
        ssm_ratio=2.0,                        
        ssm_dt_rank: Any = "auto",              
        ssm_act_layer=nn.SiLU,                  
        ssm_conv: int = 3,                    
        ssm_conv_bias=True,                     
        ssm_drop_rate: float = 0,    
        if_up_sample=False,                                 
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.if_up_sample = if_up_sample
        # 2d state space model
        self.ssm = nn.ModuleList([SS2D(
            d_model=dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            ) for _ in range(len(drop_path))
        ])
        self.drop_path = [DropPath(dpr) for dpr in drop_path]
        self.conv = nn.Conv2d(dim*2, dim*2, kernel_size=1, stride=1)
        self.norm = norm_layer(dim*2, eps=1e-5)
        if self.if_up_sample:
            self.upsampling = nn.PixelShuffle(2)

    def forward(self, x_deep, x_shallow):
        """
        Reconstruct high-resolution images for segmentation by fusing 
        different hierarchical low-resolution feature maps
        
        args: 
            x_deep: Deep feature map
            x_shallow: Shallow feature map
        """
        x = x_deep
        for i in range(len(self.ssm)):
            x = x + self.drop_path[i](self.ssm[i](x))
        x = torch.cat([x, x_shallow], dim=1)
        x = self.norm(self.conv(x))
        if self.if_up_sample:
            x = self.upsampling(x)
        return x


@MODELS.register_module()
class SegMambaHeadV6(BaseDecodeHead):
    def __init__(
        self,  
        depths=[2, 2, 2, 2], 
        in_channels=[768, 384, 192, 96],
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ffn_drop_rate: float = 0.0,
        ffn_ratio=2,
        channels=512,
        ffn_act_layer=nn.GELU,
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
        ssm_act_layer = nn.SiLU
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            if_up_sample = True if i < len(depths) - 1 else False
            self.stages.append(create_stage(
                dim=self.in_channels[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,                  
                if_up_sample=if_up_sample,
                ))
        # FFN
        in_features = in_channels[-1] * 2
        hidden_features = in_features * ffn_ratio 
        self.ffn = Mlp(in_features=in_features, 
                       hidden_features=hidden_features, 
                       act_layer=ffn_act_layer, 
                       drop=ffn_drop_rate)
        self.ddc = nn.ModuleList([DDConv2d(in_chans=dim, out_chans=dim) for dim in in_channels[::-1]])

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
        inputs = [ddc(input) for ddc, input in zip(self.ddc, inputs)]
        x = inputs[-1]
        index = -1
        for stage in self.stages:
            for block in stage:
                x = block(x, inputs[index])
            index -= 1
        out = self.ffn(x)
        out = self.cls_seg(out)
        return out


