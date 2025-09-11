

_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/gtav.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
# model settings
model = dict(
    pretrained=None,
    type='EncoderDecoder',
    backbone=dict(
        type='lsnet_b',
        style='pytorch',
        pretrained= 'pretrained/lsnet_b.pth',
        frozen_stages=-1,
    ),
    neck=dict(
        type='LSNetFPN',
        in_channels=[128, 256, 384, 512],
        out_channels=256,
        num_outs=4,
        # num_extra_trans_convs=1,
        ),
    decode_head=dict(num_classes=150))


train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=60000,
    val_interval=1000,
    val_begin=0)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        save_best='mIoU'))

work_dir = 'workdir'
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size_divisor=32,
    test_cfg=dict(size_divisor=32))

crop_size = (1024, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        scale=(2048, 1024)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# gpu_multiples = 2  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# # optimizer
# optimizer = dict(type='AdamW', lr=0.0001 * gpu_multiples, weight_decay=0.0001)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=80000 // gpu_multiples)
# checkpoint_config = dict(by_epoch=False, interval=8000 // gpu_multiples)
# evaluation = dict(interval=8000 // gpu_multiples, metric='mIoU')