dataset_type = 'UDADataset'

source_type = 'GTAVDataset'
source_root = 'data/gta'
target_type = 'CityscapesDataset'
target_root = 'data/cityscapes'


data_root = 'data/gta'
crop_size = (512, 512)
source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

target_train_pipeline = source_train_pipeline
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(
#         type='Resize',
#         scale=(2048, 512)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    # collate_fn=dict(type='default_collate'),
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        rare_class_sampling=dict(min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5),
        source_dataset=dict(
            type=source_type,
            data_root=source_root,
            # serialize_data=False,
            data_prefix=dict(
                img_path='images', seg_map_path='labels'),
            pipeline=source_train_pipeline),
        target_dataset=dict(
            type=target_type,
            data_root=target_root,
            data_prefix=dict(
                img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
            pipeline=target_train_pipeline)

        ))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        # type=source_type,
        data_root='data/cityscapes/',
        # indices=1,
        # data_root=source_root,
        # indices=300,
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        # data_prefix=dict(
        #     img_path='images', seg_map_path='labels'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

