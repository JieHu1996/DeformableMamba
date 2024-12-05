import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from ..utils.mamba2 import create_block, create_downsample_layer, LayerNorm2d
from .decode_head import BaseDecodeHead


class VSS(BaseModel):
    def __init__(self,
                 num_classes=171,
                 input_dim=768,
                 depth=2,
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
                 norm_layer="LN",):
        super().__init__()
        self.num_classes = num_classes
        self.dim = input_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        norm_layer = LayerNorm2d
        ssm_act_layer = nn.SiLU
        mlp_act_layer = nn.GELU
        self.layers = nn.ModuleList()
        downsample = create_downsample_layer(
                self.dim, 
                self.num_classes, 
                norm_layer=norm_layer,
            )
        self.layers.append(create_block(
            dim = self.dim,
            downsample=downsample,
            drop_path= dpr,
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
        self.layers.append(create_block(
            dim = self.num_classes,
            drop_path= dpr,
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

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

@MODELS.register_module()
class MambaHead(BaseDecodeHead):
    def __init__(self,                  
                 num_classes=171,
                 input_dim=768,
                 depth=2,
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
                 norm_layer="LN",
                 **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.vss_module = VSS(
                 num_classes=num_classes,
                 input_dim=input_dim,
                 depth=depth,
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
                 drop_path_rate=drop_path_rate, 
                 norm_layer=norm_layer,)
        
    def forward(self, x):
        output = self.vss_module(x[0])
        return output
