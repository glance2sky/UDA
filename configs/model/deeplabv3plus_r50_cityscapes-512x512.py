

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]


# model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

# 下面的代码要到具体的model的配置文件中去更改
# model = uda_model

data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size_divisor=32,
    test_cfg=dict(size_divisor=32))

model = dict(
    data_preprocessor=data_preprocessor,
)



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
        max_keep_ckpts=1,
        save_best='mIoU'))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=7e-05, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

# param_scheduler = [
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=0,
#         end=60000,
#         by_epoch=False,
#     )
# ]

work_dir = 'workdir'


custom_hooks = [dict(type='DebugVisualizationHook')]

# visualizer = dict(
#     type='Visualizer',
#     vis_backends=[dict(type='TensorboardVisBackend')]
# )

# randomness = dict(seed=1579917503)

# load_from = 'workdir/uda_hierarchy3_rcs_crop_grad_accum_hloss/best_mIoU_iter_34000.pth'
# load_from = 'workdir/uda_hierarchy2_rcs_crop_grad_accum/best_mIoU_iter_9000.pth'
# load_from = 'workdir/best_69.33/best_mIoU_iter_51000.pth'