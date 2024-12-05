_base_ = [
    '../_base_/datasets/woodscape.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
# ckpt
checkpoint = '/home/ka/ka_iar/ka_ba9856/mmsegmentation/ckpts/vssm1_tiny_0230s_ckpt_epoch_264.pth'

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
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
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))

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

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)