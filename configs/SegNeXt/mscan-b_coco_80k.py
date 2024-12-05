_base_ = [
    '../_base_/models/baseline/mscan-b.py',
    '../_base_/datasets/coco-stuff10k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# model setting
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=171,),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))

# dataloader
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader