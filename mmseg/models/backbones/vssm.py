import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

from ..utils.mamba2 import LayerNorm2d, create_patch_embed, create_downsample_layer, create_block

from mmseg.registry import MODELS


@MODELS.register_module()
class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
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
        norm_layer="LN", # "BN", "LN2D" 
        posembed=False,
        imgsize=224,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(len(depths))]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        norm_layer = LayerNorm2d
        ssm_act_layer = nn.SiLU
        mlp_act_layer = nn.GELU

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
        self.patch_embed = create_patch_embed(in_chans=in_chans, 
                                              embed_dim=dims[0], 
                                              patch_size=patch_size, 
                                              patch_norm=patch_norm, 
                                              norm_layer=norm_layer, 
                                            )
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            downsample = create_downsample_layer(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
            ) if (i_layer < len(depths) - 1) else nn.Identity()

            self.layers.append(create_block(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
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

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}


    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    # def flops(self, shape=(3, 224, 224), verbose=True):
    #     # shape = self.__input_shape__[1:]
    #     supported_ops={
    #         "aten::silu": None, # as relu is in _IGNORED_OPS
    #         "aten::neg": None, # as relu is in _IGNORED_OPS
    #         "aten::exp": None, # as relu is in _IGNORED_OPS
    #         "aten::flip": None, # as permute is in _IGNORED_OPS
    #         # "prim::PythonOp.CrossScan": None,
    #         # "prim::PythonOp.CrossMerge": None,
    #         "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
    #     }

    #     model = copy.deepcopy(self)
    #     model.cuda().eval()

    #     input = torch.randn((1, *shape), device=next(model.parameters()).device)
    #     params = parameter_count(model)[""]
    #     Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

    #     del model, input
    #     return sum(Gflops.values()) * 1e9
    #     return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

# compatible with openmmlab
@MODELS.register_module()
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint from {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        
        return outs


def vmamba_tiny_s1l8():
    return VSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v0", forward_type="v05_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d"), 
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )


if __name__ == "__main__":
    model_ref = vmamba_tiny_s1l8()

    model = VSSM(
        depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2, 
        patch_size=4, in_chans=3, num_classes=1000, 
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
        ssm_init="v2", forward_type="m0_noz", 
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2", 
        use_checkpoint=False, posembed=False, imgsize=224, 
    )
    print(parameter_count(model)[""])
    print(model.flops()) # wrong
    model.cuda().train()
    model_ref.cuda().train()

    def bench(model):
        import time
        inp = torch.randn((128, 3, 224, 224)).cuda()
        for _ in range(30):
            model(inp)
        torch.cuda.synchronize()
        tim = time.time()
        for _ in range(30):
            model(inp)
        torch.cuda.synchronize()
        tim1 = time.time() - tim

        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        tim = time.time()
        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        tim2 = time.time() - tim

        return tim1 / 30, tim2 / 30
    
    print(bench(model_ref))
    print(bench(model))

    breakpoint()


