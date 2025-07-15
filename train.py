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
# runner = Runner.from_cfg(config)
# runner.load_checkpoint('workdir/DAformer_HHHead_zero_shot2/best_mIoU_iter_10000.pth')
# runner.test()

# config = Config.fromfile('workdir/uda_hierarchy3_rcs_crop_grad_accum_hloss/vis_data/config.py')
# runner = Runner.from_cfg(config)
# # runner.load_checkpoint('workdir/DAformer_HHHead_zero_shot2/best_mIoU_iter_10000.pth', revise_keys=[(r'^', 'model.')])
# runner.train()

config = Config.fromfile('configs/model/uda_daformer_HHHead_gta2cityscapes512.py')
runner = Runner.from_cfg(config)
# runner.load_checkpoint('workdir/uda_hierarchy3_rcs_crop_grad_accum_hloss_poincare_hcl/best_mIoU_iter_42000.pth')
# runner.load_checkpoint('workdir/DAformer_HHHead_zero_shot2/best_mIoU_iter_10000.pth', revise_keys=[(r'^', 'model.')])
runner.train()