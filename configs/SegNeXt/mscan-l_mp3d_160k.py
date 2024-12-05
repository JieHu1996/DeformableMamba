_base_ = [
    '../_base_/models/baseline/mscan-l.py',
    '../_base_/datasets/mp3d.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# model setting
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=20,),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(600, 600)))

# dataloader
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader