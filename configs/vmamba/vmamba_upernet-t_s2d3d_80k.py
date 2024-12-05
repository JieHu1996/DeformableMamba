_base_ = [
    '../_base_/models/vmamba-tiny.py',
    '../_base_/datasets/s2d3d.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# model settings
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=13),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(600, 600)))


# lr scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        by_epoch=False,
    )
]


# hooks
iters_per_epoch = 130
default_hooks = dict(
    logger=dict(type='LoggerHook', 
                interval=iters_per_epoch, 
                interval_exp_name=iters_per_epoch))

# data loader
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader