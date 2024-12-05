_base_ = [
    '../_base_/datasets/s2d3d.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
# ckpt
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa

# model settings
crop_size = (640, 640)
norm_cfg = dict(type='LN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        pretrained=checkpoint,
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    decode_head=dict(
        type='MusterHead',
        embed_dims=768,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=(2, 2, 2, 2),
        num_heads=(32, 16, 8, 4),
        strides=(2, 2, 2, 4),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        act_cfg=dict(type='GELU'),
        # norm_cfg=dict(type='LN'),
        with_cp=False,
        init_cfg=None,
        in_channels=[768, 384, 192, 96],
        in_index=[0, 1, 2, 3],
        # pool_scales=(1, 2, 3, 6),
        channels=192,
        # dropout_ratio=0.1,
        num_classes=13,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(512, 512)))

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)