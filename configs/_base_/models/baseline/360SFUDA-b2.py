# ckpt
checkpoint = 'ckpts/mit_b2.pth'

# model settings
find_unused_parameters=True
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
        type='PanSFUDAHead',
        feature_strides=[4, 8, 16, 32],
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        embedding_dim=256, 
        align_corners=False,
        input_transform='multiple_select',
        num_classes=13,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))