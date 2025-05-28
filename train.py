import torch
from torch.optim import AdamW

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import Runner
from numpy.ma.core import outer

from mmseg.registry import MODELS
from mmseg.registry import DATASETS

# config = Config.fromfile('configs/model/san-vit-b16_coco-stuff164k-640x640.py')
# runner = Runner.from_cfg(config)
# # runner.train()
# print(runner.test())

# config = Config.fromfile('configs/model/daformer_mitb5_b2_cityscapes-512x512.py')
# runner = Runner.from_cfg(config)
# runner.train()

# config = Config.fromfile('configs/model/daformer_mitb5_b2_gta-512x512.py')
# runner = Runner.from_cfg(config)
# runner.train()

# config = Config.fromfile('configs/model/daformer_mitb5_b2_synthia-512x512.py')
# runner = Runner.from_cfg(config)
# runner.test()

# config = Config.fromfile('configs/model/hyperformer_mitb5_b2_cityscapes-512x512.py')
#
# runner = Runner.from_cfg(config)
# runner.train()

config = Config.fromfile('configs/model/uda_daformer_HHHead_gta2cityscapes512.py')
runner = Runner.from_cfg(config)
runner.train()