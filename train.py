import torch
from torch.optim import AdamW

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner
from numpy.ma.core import outer

from mmseg.registry import MODELS
from mmseg.registry import DATASETS
init_default_scope('mmseg')
#
# model_cfg = Config.fromfile('configs/_base_/models/mib_b5.py')
# dataset_cfg = Config.fromfile('configs/_base_/datasets/cityscapes.py')
#
# runner = Runner(
#     model=model_cfg['model'],
#     work_dir='./work_dir',
#     train_dataloader=dataset_cfg['train_dataloader'],
#     optim_wrapper=dict(
#         type=AmpOptimWrapper, optimizer=dict(type=AdamW, lr=2e-4)
#     ),
#     train_cfg=dataset_cfg['train_cfg'],
#     val_dataloader=dataset_cfg['val_dataloader'],
#     val_cfg=dict(),
#     val_evaluator=dataset_cfg['val_evaluator'],
#     default_scope='mmseg'
# )
# runner.train()
config = Config.fromfile('configs/model/san-vit-b16_coco-stuff164k-640x640.py')
runner = Runner.from_cfg(config)
runner.train()
