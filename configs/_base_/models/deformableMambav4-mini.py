# ckpt
checkpoint = 'ckpts/vssm-tiny.pth'

# model settings
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
    )
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        type='DeformableMAMBAv4',
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
        type='DeformableMambaHead',
        num_classes=171, 
        depths=[1, 1, 1, 1], 
        in_channels=[768, 384, 192, 96],
        in_index=[0, 1, 2, 3],
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_drop_rate=0.1,
        ffn_ratio=2,
        channels=192,
        drop_path_rate=0.2, 
        norm_layer="ln2d",
        input_transform='multiple_select',
        init_cfg=None,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))