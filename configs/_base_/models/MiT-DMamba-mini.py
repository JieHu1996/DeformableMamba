# ckpt
checkpoint = 'ckpts/mit_b2.pth'

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,)

model = dict(
    type='EncoderDecoder',
    pretrained=checkpoint,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MiT',
        patch_size=4, 
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True, 
        depths=[3, 4, 6, 3], 
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0, 
        drop_path_rate=0.1),
    decode_head=dict(
        type='DeformableMambaHead',
        num_classes=171, 
        depths=[1, 1, 1, 1], 
        in_channels=[512, 320, 128, 64],
        in_index=[0, 1, 2, 3],
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_drop_rate=0.1,
        ffn_ratio=2,
        channels=128,
        drop_path_rate=0.2, 
        norm_layer="ln2d",
        input_transform='multiple_select',
        init_cfg=None,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))