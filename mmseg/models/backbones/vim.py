# # Copyright (c) 2015-present, Facebook, Inc.
# # All rights reserved.
# import warnings
# from mmseg.registry import MODELS
# from einops import rearrange
# from mmengine.logging import print_log
# from mmengine.model import BaseModule
# from mmengine.model.weight_init import (constant_init, kaiming_init,
#                                         trunc_normal_)
# from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
# from torch.nn.modules.batchnorm import _BatchNorm
# from mmseg.models.utils import resize, PatchEmbed

# import torch
# import torch.nn as nn
# from functools import partial
# from torch import Tensor
# from typing import Optional
# from timm.models.layers import DropPath, to_2tuple
# import math
# from mamba_ssm.modules.mamba_simple import Mamba

# try:
#     from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# except ImportError:
#     RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# class PatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         # assert not torch.isnan(self.proj.weight).any(), "proj weight contains NaN values"
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x
    

# class Block(nn.Module):
#     def __init__(
#         self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
#     ):
#         """
#         Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

#         This Block has a slightly different structure compared to a regular
#         prenorm Transformer block.
#         The standard block is: LN -> MHA/MLP -> Add.
#         [Ref: https://arxiv.org/abs/2002.04745]
#         Here we have: Add -> LN -> Mixer, returning both
#         the hidden_states (output of the mixer) and the residual.
#         This is purely for performance reasons, as we can fuse add and LayerNorm.
#         The residual needs to be provided (except for the very first block).
#         """
#         super().__init__()
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         self.mixer = mixer_cls(dim)
#         self.norm = norm_cls(dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         if self.fused_add_norm:
#             assert RMSNorm is not None, "RMSNorm import fails"
#             assert isinstance(
#                 self.norm, (nn.LayerNorm, RMSNorm)
#             ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

#     def forward(
#         self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
#     ):
#         r"""Pass the input through the encoder layer.

#         Args:
#             hidden_states: the sequence to the encoder layer (required).
#             residual: hidden_states = Mixer(LN(residual))
#         """
#         # msg1 = 'before norm'
#         # msg2 = f'hidden_states: {hidden_states}'
#         # msg3 = f'residual: {residual}'
#         # self.logger2.debug(msg1)
#         # self.logger2.debug(msg2)
#         # self.logger2.debug(msg3)
        
#         if not self.fused_add_norm:
#             if residual is None:
#                 residual = hidden_states
#             else:
#                 residual = residual + self.drop_path(hidden_states)
# #            assert not torch.isnan(self.norm.weight).any(), "LayerNorm weight contains NaN values"
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
# #            assert not torch.isnan(hidden_states).any(), "hidden states contains NaN values"
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             if residual is None:
#                 hidden_states, residual = fused_add_norm_fn(
#                     hidden_states,
#                     self.norm.weight,
#                     self.norm.bias,
#                     residual=residual,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     eps=self.norm.eps,
#                 )
#             else:
#                 hidden_states, residual = fused_add_norm_fn(
#                     self.drop_path(hidden_states),
#                     self.norm.weight,
#                     self.norm.bias,
#                     residual=residual,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     eps=self.norm.eps,
#                 )   
#         # msg4 = 'after norm'
#         # msg5 = f'hidden_states: {hidden_states}'
#         # msg6 = f'residual: {residual}'
#         # self.logger2.debug(msg4)
#         # self.logger2.debug(msg5)
#         # self.logger2.debug(msg6)
 
#         hidden_states = self.mixer(hidden_states, inference_params=inference_params)
#         return hidden_states, residual

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# def create_block(
#     d_model,
#     ssm_cfg=None,
#     norm_epsilon=1e-5,
#     drop_path=0.,
#     rms_norm=False,
#     residual_in_fp32=False,
#     fused_add_norm=False,
#     layer_idx=None,
#     device=None,
#     dtype=None,
#     if_bimamba=False,
#     bimamba_type="none",
#     if_devide_out=False,
#     init_layer_scale=None,
# ):
#     if if_bimamba:
#         bimamba_type = "v1"
#     if ssm_cfg is None:
#         ssm_cfg = {}
#     factory_kwargs = {"device": device, "dtype": dtype}
#     mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
#     norm_cls = partial(
#         nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
#     )
#     block = Block(
#         d_model,
#         mixer_cls,
#         norm_cls=norm_cls,
#         drop_path=drop_path,
#         fused_add_norm=fused_add_norm,
#         residual_in_fp32=residual_in_fp32,
#     )
#     block.layer_idx = layer_idx
#     return block


# @MODELS.register_module()
# class VisionMamba(BaseModule):
#     def __init__(self, 
#                  img_size=224, 
#                  patch_size=16, 
#                  stride=16,
#                  depth=24, 
#                  embed_dim=192, 
#                  channels=3,
#                  out_indices=-1, 
#                  ssm_cfg=None, 
#                  drop_rate=0.,
#                  drop_path_rate=0.1,
#                  norm_epsilon: float = 1e-5, 
#                  rms_norm: bool = False, 
#                  fused_add_norm=False,
#                  residual_in_fp32=False,
#                  device=None,
#                  dtype=None,
#                  if_bidirectional=False,
#                  final_pool_type='none',
#                  if_abs_pos_embed=False,
#                  flip_img_sequences_ratio=-1.,
#                  if_bimamba=False,
#                  bimamba_type="none",
#                  interpolate_mode='bicubic',
#                  patch_norm=False,
#                  if_cls_token=False,
#                  if_devide_out=False,
#                  init_layer_scale=None,
#                  with_cls_token=False,
#                  pretrained=None,
#                  init_cfg=None,
#                  **kwargs):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         # add factory_kwargs into kwargs
#         kwargs.update(factory_kwargs) 
#         super().__init__(init_cfg=init_cfg)

#         assert not (init_cfg and pretrained), \
#             'init_cfg and pretrained cannot be set at the same time'
#         if isinstance(pretrained, str):
#             warnings.warn('DeprecationWarning: pretrained is deprecated, '
#                             'please use "init_cfg" instead')
#             self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
#         elif pretrained is not None:
#             raise TypeError('pretrained must be a str or None')
        
#         self.patch_norm = patch_norm
#         self.img_size = img_size
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         self.if_bidirectional = if_bidirectional
#         self.final_pool_type = final_pool_type
#         self.if_abs_pos_embed = if_abs_pos_embed
#         self.flip_img_sequences_ratio = flip_img_sequences_ratio
#         self.interpolate_mode = interpolate_mode
#         self.if_cls_token = if_cls_token
#         self.with_cls_token = with_cls_token
#         self.num_tokens = 1 if if_cls_token else 0
#         self.h = self.w = int(img_size // patch_size)
#         self.patch_size = patch_size
#         self.num_patches =  self.h * self.w

#         # pretrain parameters
#         self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)

#         if if_cls_token:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#             self.num_tokens = 1
            
#         if if_abs_pos_embed:
#             self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.embed_dim))
#             self.pos_drop = nn.Dropout(p=drop_rate)

#         if isinstance(out_indices, int):
#             if out_indices == -1:
#                 out_indices = depth - 1
#             self.out_indices = [out_indices]
#         elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
#             self.out_indices = out_indices
#         else:
#             raise TypeError('out_indices must be type of int, list or tuple')

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
#         inter_dpr = [0.0] + dpr
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
#         self.layers = nn.ModuleList(
#             [
#                 create_block(
#                     embed_dim,
#                     ssm_cfg=ssm_cfg,
#                     norm_epsilon=norm_epsilon,
#                     rms_norm=rms_norm,
#                     residual_in_fp32=residual_in_fp32,
#                     fused_add_norm=fused_add_norm,
#                     layer_idx=i,
#                     if_bimamba=if_bimamba,
#                     bimamba_type=bimamba_type,
#                     drop_path=inter_dpr[i],
#                     if_devide_out=if_devide_out,
#                     init_layer_scale=init_layer_scale,
#                     **factory_kwargs,
#                 )
#                 for i in range(depth)
#             ]
#         )
        
#         self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
#             embed_dim, eps=norm_epsilon, **factory_kwargs
#         )

#     def init_weights(self):
#         if (isinstance(self.init_cfg, dict)
#                 and self.init_cfg.get('type') == 'Pretrained'):
#             checkpoint = CheckpointLoader.load_checkpoint(
#                 self.init_cfg['checkpoint'], logger=None, map_location='cpu')

#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             else:
#                 state_dict = checkpoint

#             if 'pos_embed' in state_dict.keys():
#                 if self.pos_embed.shape != state_dict['pos_embed'].shape:
#                     print_log(msg=f'Resize the pos_embed shape from '
#                               f'{state_dict["pos_embed"].shape} to '
#                               f'{self.pos_embed.shape}')
#                     h, w = self.img_size, self.img_size
#                     pos_size = int(
#                         math.sqrt(state_dict['pos_embed'].shape[1] - 1))
#                     state_dict['pos_embed'] = self.resize_pos_embed(
#                         state_dict['pos_embed'],
#                         (h // self.patch_size, w // self.patch_size),
#                         (pos_size, pos_size), self.interpolate_mode)

#             load_state_dict(self, state_dict, strict=False, logger=None)
#             for n, m in self.named_modules():
#                 if isinstance(m, nn.Conv2d):
#                     kaiming_init(m, mode='fan_in', bias=0.)
#         elif self.init_cfg is not None:
#             super().init_weights()
#         else:
#             # We only implement the 'jax_impl' initialization implemented at
#             # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
#             trunc_normal_(self.pos_embed, std=.02)
#             trunc_normal_(self.cls_token, std=.02)
#             for n, m in self.named_modules():
#                 if isinstance(m, nn.Linear):
#                     trunc_normal_(m.weight, std=.02)
#                     if m.bias is not None:
#                         if 'ffn' in n:
#                             nn.init.normal_(m.bias, mean=0., std=1e-6)
#                         else:
#                             nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.Conv2d):
#                     kaiming_init(m, mode='fan_in', bias=0.)
#                 elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
#                     constant_init(m, val=1.0, bias=0.)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

#     @staticmethod
#     def resize_pos_embed(pos_embed, input_shape, pos_shape, mode):
#         """Resize pos_embed weights.

#         Resize pos_embed using bicubic interpolate method.
#         Args:
#             pos_embed (torch.Tensor): Position embedding weights.
#             input_shpae (tuple): Tuple for (downsampled input image height,
#                 downsampled input image width).
#             pos_shape (tuple): The resolution of downsampled origin training
#                 image.
#             mode (str): Algorithm used for upsampling:
#                 ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
#                 ``'trilinear'``. Default: ``'nearest'``
#         Return:
#             torch.Tensor: The resized pos_embed of shape [B, L_new, C]
#         """

#         assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
#         token_position = (pos_shape[0] * pos_shape[1]) // 2
#         new_position = (input_shape[0] * input_shape[1]) // 2
#         cls_token_weight = pos_embed[:, token_position, :]
#         pos_embed_weight = torch.cat((pos_embed[:, :token_position, :], pos_embed[:, token_position + 1:, :]), dim=1)
#         pos_embed_weight = pos_embed_weight.reshape(
#             1, pos_shape[0], pos_shape[1], pos_embed.shape[2]).permute(0, 3, 1, 2).contiguous()
#         pos_embed_weight = resize(
#             pos_embed_weight, size=input_shape, align_corners=False, mode=mode)
#         pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
#         cls_token_weight = cls_token_weight.unsqueeze(0)
#         new_position = pos_embed_weight.shape[1] // 2
#         pos_embed = torch.cat((pos_embed_weight[:, :new_position, :], cls_token_weight, 
#                                pos_embed_weight[:, new_position:, :]), dim=1)
#         return pos_embed

#     def forward(self, x, inference_params=None):
#         # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#         # with slight modifications to add the dist_token
#         # assert not torch.isnan(x).any(), "input contains NaN values"
#         x = self.patch_embed(x)
#         # assert not torch.isnan(x).any(), "patch embeded input contains NaN values"
#         B, M, _ = x.shape

#         if self.if_cls_token:
#             cls_token = self.cls_token.expand(B, -1, -1)
#             token_position = M // 2
#             # add cls token in the middle
#             x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)

#         if self.if_abs_pos_embed:
#             x = x + self.pos_embed
#             x = self.pos_drop(x)
        
#         if not self.with_cls_token:
#             x = torch.cat((x[:, :token_position, :], x[:, token_position + 1:, :]), dim=1)

#         # mamba impl
#         residual = None
#         hidden_states = x
#         outs = []
#         for i, layer in enumerate(self.layers):
#             hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
#             if i in self.out_indices:
#                 x_hs, x_res = hidden_states, residual
#                 if not self.fused_add_norm:
#                     if x_res is None:
#                         x_res = x_hs
#                     else:
#                         x_res = x_res + self.drop_path(x_hs)
#                     x_hs = self.norm_f(x_res.to(dtype=self.norm_f.weight.dtype))
#                 else:
#                     # Set prenorm=False here since we don't need the residual
#                     fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
#                     x_hs = fused_add_norm_fn(
#                         self.drop_path(x_hs),
#                         self.norm_f.weight,
#                         self.norm_f.bias,
#                         eps=self.norm_f.eps,
#                         residual=x_res,
#                         prenorm=False,
#                         residual_in_fp32=self.residual_in_fp32,
#                     )
#                     if self.with_cls_token:
#                         x_hs = torch.cat((x_hs[:, :token_position, :], x_hs[:, token_position + 1:, :]), dim=1)
#                     x = rearrange(x_hs, 'b (h w) d -> b d h w', h=self.h, w=self.w).contiguous()                               
#                     outs.append(x)
#         return tuple(outs)


# if __name__ == '__main__':
#     test_pos_embed = False
#     if test_pos_embed:
#         pos_embed = torch.randn(1, 4, 2)
#         cls_token = torch.ones(1, 1, 2) * 2
#         pos_embed = torch.cat((pos_embed[:, :2, :], cls_token, pos_embed[:, 2:, :]), dim=1)
#         token_position = pos_embed.shape[1] // 2
#         input_shape = (4, 4)
#         pos_shape = (2, 2)
#         mode = 'bicubic'
#         def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
#             assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
#             token_position = (pos_shape[0] * pos_shape[1]) // 2
#             new_position = (input_shape[0] * input_shape[1]) // 2
#             cls_token_weight = pos_embed[:, token_position, :]
#             pos_embed_weight = torch.cat((pos_embed[:, :token_position, :], pos_embed[:, token_position + 1:, :]), dim=1)
#             pos_embed_weight = pos_embed_weight.reshape(
#                 1, pos_shape[0], pos_shape[1], pos_embed.shape[2]).permute(0, 3, 1, 2).contiguous()
#             pos_embed_weight = resize(
#                 pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
#             pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
#             cls_token_weight = cls_token_weight.unsqueeze(0)
#             new_position = pos_embed_weight.shape[1] // 2
#             pos_embed = torch.cat((pos_embed_weight[:, :new_position, :], cls_token_weight, 
#                                 pos_embed_weight[:, new_position:, :]), dim=1)
#             return pos_embed
#         new_pos_embed = resize_pos_embed(pos_embed, input_shape, pos_shape, mode)
    
#     input = torch.randn(2, 3, 224, 224).cuda()
#     # out_indices = [7, 15, 23]
#     encoder = VisionMamba(img_size=224, 
#                         depth=24, 
#                         embed_dim=192, 
#                         # out_indices=out_indices, 
#                         rms_norm=True, 
#                         residual_in_fp32=True, 
#                         fused_add_norm=True, 
#                         final_pool_type='all', 
#                         if_abs_pos_embed=True,  
#                         bimamba_type="v2",
#                         interpolate_mode='bicubic', 
#                         if_cls_token=True, 
#                         with_cls_token=True,).cuda()
#     output = encoder(input)

#     print('end')
