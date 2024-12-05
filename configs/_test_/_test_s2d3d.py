_base_ = [
    '../_base_/models/deformableMamba-mini.py',
    '../_base_/datasets/s2d3d.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# model settings
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=13, 
        depths=[1, 1, 1, 1], 
        kernel_sizes=[3, 3, 3, 3]),
    test_cfg=dict(mode='whole'))


# lr scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=4000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=4000,
        by_epoch=False,
    )
]


train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=130)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=130))

# data loader
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader