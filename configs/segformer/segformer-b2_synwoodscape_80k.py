_base_ = [
    '../_base_/models/baseline/segformer-b2.py',
    '../_base_/datasets/synwoodscape.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# model settings
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=24,),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(600, 600)))

# dataloader
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader