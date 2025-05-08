from configs.dsdl.voc import train_dataloader
from torch.sparse import sampled_addmm
# custom_imports = dict(imports=['mmseg.datasets'], allow_failed_imports=False)
data_type = 'CityscapesDataset'
data_root = 'data/cityscapes'
crop_size = (480,264)

data_prefix = dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train')
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(720,396)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip',prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', scale=(720,396), keep_ratio=True),
    dict(type='PackSegInputs')
]
val_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler = dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        # _scope_='mmseg',
        type=data_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline,
        indices=100,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler = dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        # _scope_='mmseg',
        type=data_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=val_pipeline,
        indices=50
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler = dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        # _scope_='mmseg',
        type=data_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        pipeline=test_pipeline,
        indices=50
    )
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
val_evaluator = test_evaluator

train_cfg = dict(by_epoch=True, max_epochs=15, val_begin=2, val_interval=1)
val_cfg = dict(mode='whole')
test_cfg = dict()
