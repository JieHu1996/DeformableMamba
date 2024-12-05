import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.modules.batchnorm import _BatchNorm
import math
import warnings
from einops import rearrange
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_
from ..utils.mamba2 import LayerNorm2d, create_patch_embed, create_downsample_layer, create_block
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from mmengine.logging import print_log
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.model.weight_init import (constant_init, kaiming_init, trunc_normal_)

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


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


@MODELS.register_module()
class DeformableMAMBA(BaseModule):
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
            ssm_conv_bias=False,
            ssm_drop_rate=0.0, 
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            drop_path_rate=0.1, 
            patch_norm=True, 
            pretrained=None, 
            init_cfg=None,
            norm_layer='ln2d',
            **kwargs):
        
        # kwargs.update(norm_layer=norm_layer)
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
                
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depths decay rule

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(len(depths))]
        self.num_features = dims[-1]
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        

        norm_layer = LayerNorm2d
        ssm_act_layer = nn.SiLU
        mlp_act_layer = nn.GELU

        self.patch_embed = create_patch_embed(in_chans=in_chans, 
                                              embed_dim=dims[0], 
                                              patch_size=patch_size, 
                                              patch_norm=patch_norm, 
                                              norm_layer=norm_layer, 
                                            )

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            downsample = create_downsample_layer(
                self.dims[i], 
                self.dims[i + 1], 
                norm_layer=norm_layer,
            ) if (i < len(depths) - 1) else nn.Identity()

            self.layers.append(create_block(
                dim = self.dims[i],
                drop_path = dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
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
        
        self.final_norm = []
        for i in range(len(self.dims)):
            self.final_norm.append(NORM['ln2d'](self.dims[i]))
        self.final_norm = nn.ModuleList(self.final_norm)
        

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
        if 'final_norm' not in state_dict:
            print_log('Init global norm', logger='current')
            for n, m in self.named_modules():
                if 'global norm' in n:
                    constant_init(m, val=1.0, bias=0.)


    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        
        for i, layer in enumerate(self.layers):
            out, x = layer_forward(layer, x)
            out = self.final_norm[i](out)
            outs.append(out.contiguous())

        return outs



