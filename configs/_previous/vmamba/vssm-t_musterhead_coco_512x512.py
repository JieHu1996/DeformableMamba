_base_ = [
    '../_base_/datasets/coco-stuff10k.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
# ckpt
checkpoint = '/home/ka/ka_iar/ka_ba9856/mmsegmentation/ckpts/vssm1_tiny_0230s_ckpt_epoch_264.pth'
# model settings
norm_cfg = dict(type='LN', requires_grad=True)
crop_size = (512, 512)
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
        type='Backbone_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained=checkpoint,
        dims=96,
        depths=(2, 2, 8, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=1.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        norm_layer="ln2d",),
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
        num_classes=171,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))

# # optimizer
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
#     paramwise_cfg=dict(custom_keys={
#         'pos_block': dict(decay_mult=0.),
#         'norm': dict(decay_mult=0.)
#     }))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=4000),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=4000,
#         end=200000,
#         by_epoch=False,
#     )
# ]

# # By default, models are trained on 8 GPUs with 2 images per GPU
# train_dataloader = dict(batch_size=2)
# val_dataloader = dict(batch_size=1)
# test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'pos_block': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.)
    }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=4000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=4000,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)