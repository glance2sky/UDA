

_base_ = [
    '../_base_/models/mib_b5.py', '../_base_/datasets/cityscapes512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
]


train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=60000,
    val_interval=500,
    val_begin=55000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        save_best='mIoU'))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=60000,
        by_epoch=False,
    )
]

work_dir = 'workdir'

data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size_divisor=32,
    test_cfg=dict(size_divisor=32))