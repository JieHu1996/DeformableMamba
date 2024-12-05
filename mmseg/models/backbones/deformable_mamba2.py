import math
import warnings
import torch
import torch.nn as nn
import torchvision
from torch.nn.init import xavier_uniform_, constant_
from timm.models.layers import DropPath, trunc_normal_
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from mmengine.logging import print_log
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.model.weight_init import (constant_init, kaiming_init, trunc_normal_)
from ..utils.mamba2 import LayerNorm2d, create_patch_embed, create_downsample_layer, create_block

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class DWConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,
                                   padding=padding, groups=in_chans, bias=bias)
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)

        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


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
    

class DeformableConv2(nn.Module):
    """ Deformable Feature Transformation. """
    def __init__(self, 
                 in_chans=768, 
                 out_chans=768, 
                 kernel_size=3, 
                 norm_layer=nn.SyncBatchNorm,
                 drop_path=0.0
                 ):
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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() 

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
        x = x + self.drop_path(self.act(self.norm(self.transform(x))))
        return x


class LocalFocuser(nn.Module):
    def __init__(self, dim, kernel_size, drop_path, downsample, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim        
        blocks = [
            DeformableConv2(
                in_chans=dim, 
                out_chans=dim, 
                kernel_size=kernel_size,
                drop_path=drop_path[i],
                **kwargs) for i in range(len(drop_path))
                ]
        self.blocks = nn.Sequential(*blocks)
        self.downsample = downsample

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.downsample(x)
        return x


@MODELS.register_module()
class DeformableMAMBAv2(BaseModule):
    def __init__(
            self, 
            patch_size=4, 
            in_chans=3, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",        
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0, 
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            drop_path_rate=0.1, 
            patch_norm=True, 
            deform_depths=[1, 1, 1, 1],
            kernel_sizes=[3, 5, 7, 9],
            pretrained=None, 
            init_cfg=None,
            **kwargs):
        
        super().__init__(**kwargs)

        NORM = {'ln2d': LayerNorm2d, 
                'bn': nn.BatchNorm2d, 
                'syncbn': nn.SyncBatchNorm}   

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
                
        dpr1 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr2 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(deform_depths))]

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(len(depths))]
        self.num_features = dims[-1]
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        

        norm_layer = LayerNorm2d
        ssm_act_layer = nn.SiLU
        mlp_act_layer = nn.GELU

        self.patch_embed = create_patch_embed(
            in_chans=in_chans, 
            embed_dim=dims[0], 
            patch_size=patch_size, 
            patch_norm=patch_norm, 
            norm_layer=norm_layer, 
            )

        self.layers = nn.ModuleList()
        self.deform_layers = nn.ModuleList()
        downsample = []
        for i in range(len(depths)):
            downsample = [create_downsample_layer(
                self.dims[i], 
                self.dims[i + 1], 
                norm_layer=norm_layer,
            ) if (i < len(depths) - 1) else nn.Identity() for _ in range(2)]

            self.layers.append(
                create_block(
                    dim = self.dims[i],
                    drop_path = dpr1[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=norm_layer,
                    downsample=downsample[0],
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
        
            self.deform_layers.append(
                LocalFocuser(
                    dim=self.dims[i], 
                    kernel_size=kernel_sizes[i],
                    drop_path=dpr2[sum(deform_depths[:i]):sum(deform_depths[:i + 1])],
                    downsample=downsample[1],))
                
        self.fusion = nn.ModuleList([
            nn.Conv2d(self.dims[i], self.dims[i], kernel_size=1, bias=False) for i in range(len(self.dims))])

        self.final_norm = nn.ModuleList([
            LayerNorm2d(self.dims[i]) for i in range(len(self.dims))])        

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            load_state_dict(self, state_dict, strict=False, logger=None)

        # init global norm  
        if 'deform_layers' not in state_dict:
            print_log('Init Deformable Layers', logger='current')
            for n, m in self.named_modules():
                if any(k in n for k in ['deform_layers', 'final_norm', 'fusion']):
                    if isinstance(m, nn.Conv2d):
                        if 'offset' in n or 'mask' in n:
                            constant_init(m, val=1.0, bias=0.)
                        else:
                            kaiming_init(m, mode='fan_in', bias=0.)
                    elif isinstance(m, (LayerNorm2d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                        constant_init(m, val=1.0, bias=0.)

    # def feature_fusion(self, x_global, x_local):
    #     pass

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x_global = self.patch_embed(x)
        x_local = x_global
        outs = []
        
        for i, layer, deform_layer in zip(range(4), self.layers, self.deform_layers):
            out_global, x_global = layer_forward(layer, x_global)
            out_local, x_local = layer_forward(deform_layer, x_local)
            # out = self.feature_fuion(out_global, out_local)
            out = self.final_norm[i](self.fusion[i](out_global + out_local))
            outs.append(out.contiguous())

        return outs




