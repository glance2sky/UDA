

_base_ = [
    '../_base_/models/daformer_HHHead.py', '../_base_/datasets/uda_gta2cityscapes512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
]


model = dict(
    uda_model=dict(
        decode_head=dict(type='HHHead',
                         tree_params={'i2c':'data/cityscapes/cityscapes_i2c.txt',
                                      'json':'data/cityscapes/cityscapes_hierarchy3.json',},
                         c=0.5,
                         temp=0.5),
        # init_cfg=dict(type='Pretrained', checkpoint='workdir/DAformer_HHHead_zero_shot2/best_mIoU_iter_10000.pth')
    ),
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    # imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_lambda=0,
    # imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    debug_iter=250
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
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
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

data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size_divisor=32,
    test_cfg=dict(size_divisor=32))

custom_hooks = [dict(type='DebugVisualizationHook')]

visualizer = dict(
    type='HHLocalVisualizer',
)
randomness = dict(seed=0)

# load_from = 'workdir/uda_hierarchy3_rcs_crop_grad_accum_hloss/best_mIoU_iter_34000.pth'
# load_from = 'workdir/uda_hierarchy2_rcs_crop_grad_accum/best_mIoU_iter_9000.pth'
# load_from = 'workdir/uda_re_hierarchy3_rcs_crop_grad_accum_hloss/best_mIoU_iter_14000.pth'
# load_from = 'workdir/test_gradient_explore_logmin/best_mIoU_iter_43000.pth'

