# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union, List

from scipy.special.cython_special import pseudo_huber

from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
from mmengine.optim import OptimWrapper
# from mmengine.runner import Runner
from mmengine.logging import MessageHub

import mmcv
import numpy as np
import torch
from torch import Tensor
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from configs.model.uda_daformer_HHHead_gta2cityscapes512 import train_cfg
# from mmcv.ops.point_sample import normalize
# from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import UDA, build_segmentor
from mmseg.models.UDA.uda_decorator import UDADecorator
# from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
#                                                 get_mean_std, strong_transform)
# from mmseg.models.utils.visualization import subplotimg
# from mmseg.utils.utils import downscale_label_ratio
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
# from mmseg.models.decode_heads import FloatingRegionScore, init_mask, select_pixels_to_label

import kornia
import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms


from mmseg.utils.uda_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)



def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm

def exponential_weight_schedule(t, T, w_min, w_max, alpha=5.0):
    return w_min + (w_max - w_min) * (1 - math.exp(-alpha * t / T))

@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self,
                 uda_model,
                 data_preprocessor: OptConfigType = None,
                 mix='class',
                 blur=True,
                 color_jitter_strength=0.2,
                 color_jitter_probability=0.2,
                 pseudo_threshold=0.968,
                 pseudo_weight_ignore_top=15,
                 pseudo_weight_ignore_bottom=120,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg = None
                 ):
        super(DACS, self).__init__(uda_model,
                                   data_preprocessor,
                                   init_cfg)

        self.train_cfg = {}
        self.message_hub = MessageHub.get_current_instance()

        # self.local_iter = 0
        # self.max_iters = cfg['max_iters'] # cfg配置文件里面暂时没有这个参数，可能是后面手动添加的。经过检查确实前期将runner的max_iters加入了
        self.alpha = 0.999
        self.pseudo_threshold = pseudo_threshold
        self.psweight_ignore_top = pseudo_weight_ignore_top
        self.psweight_ignore_bottom = pseudo_weight_ignore_bottom
        # self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        # self.fdist_classes = cfg['imnet_feature_dist_classes']
        # self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        # self.enable_fdist = self.fdist_lambda > 0
        self.mix = mix
        self.blur = blur
        self.color_jitter_s = color_jitter_strength
        self.color_jitter_p = color_jitter_probability
        # self.debug_img_interval = cfg['debug_img_interval']
        # self.print_grad_magnitude = cfg['print_grad_magnitude']
        # assert self.mix == 'class'

        # self.debug_fdist_mask = None
        # self.debug_gt_rescale = None

        # self.class_probs = {}
        ema_cfg = deepcopy(uda_model)
        self.model = build_segmentor(uda_model)
        self.ema_model = build_segmentor(ema_cfg)


        # if self.enable_fdist:
        #     self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        # else:
        #     self.imnet_model = None

        # self.transforms = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    def get_model(self):
        return self.model

    def get_ema_model(self):
        return self.ema_model


    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def source_loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        losses = dict()

        source_x = self.get_model().extract_feat(inputs)
        source_loss_decode = self.get_model().decode_head.loss(source_x, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(source_loss_decode, 'source_decode'))
        # loss_decode = self.get_model()._decode_head_forward_train(source_x, data_samples)
        # losses.update(loss_decode)
        # for m in self.get_ema_model().modules():
        #     if isinstance(m, _DropoutNd):
        #         m.training = False
        #     if isinstance(m, DropPath):
        #         m.training = False
        # target_data['inputs'] = torch.stack(target_data['inputs'])
        # probs, cprobs = self.get_ema_model().encode_decode(target_data['inputs'], target_data['data_samples'])
        # pseudo_label = self.get_ema_model().decode_head.decide(probs)
        # target_data.data_samples.seg_sem_map = pseudo_label
        # target_x = self.get_model().extract_feat(target_data)
        # target_loss_decode = self.get_model().decode_head.loss(target_x, target_data['data_samples'], self.train_cfg)





        # log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        # outputs = dict(
        #     log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return losses


    def target_loss(self, data: dict) -> dict:
        losses = dict()
        source_data = data[0]
        target_data = data[1]
        inputs = target_data['inputs']
        data_samples = target_data['data_samples']
        gt_semantic_seg = source_data['data_samples'][0].gt_sem_seg.data.unsqueeze(0)
        batch_size = inputs.shape[0]

        means, stds = get_mean_std(source_data['inputs'][0].device)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        with torch.no_grad():
            target_feat = self.get_ema_model().extract_feat(inputs)
            probs, _ = self.get_ema_model().decode_head.forward(target_feat, data_samples[0].img_shape)

            pseudo_prob, pseudo_label = torch.max(probs, dim=1)
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=inputs.device)

            if self.psweight_ignore_top > 0:
                # Don't trust pseudo-labels in regions with potential
                # rectification artifacts. This can lead to a pseudo-label
                # drift from sky towards building or traffic light.
                pseudo_weight[:, :self.psweight_ignore_top, :] = 0
            if self.psweight_ignore_bottom > 0:
                pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=inputs.device)

            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mix_masks = get_class_masks(gt_semantic_seg)

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((source_data['inputs'][i], inputs[i])),
                    target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)

        target_x = self.get_model().extract_feat(mixed_img)
        target_loss_decode = self.get_model().decode_head.loss(target_x, data_samples,
                                                               self.train_cfg, pseudo_weight)

        losses.update(add_prefix(target_loss_decode, 'mix'))
        return losses




    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.

        optim_wrapper.zero_grad()

        source_data = data['source']
        target_data = data['target']
        cur_iter = self.message_hub.get_info('iter')
        if cur_iter == 0:
            self._init_ema_weights()
        if cur_iter > 0:
            self._update_ema(cur_iter)

        source_data = self.data_preprocessor(source_data, True)
        source_losses = self._run_forward(source_data, mode='source_loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(source_losses)  # type: ignore
        optim_wrapper.backward(parsed_losses)

        target_data['inputs'] = torch.stack(target_data['inputs'])
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        target_data = self.data_preprocessor(target_data, True)
        target_losses = self.target_loss([source_data, target_data])
        parsed_losses, t_log_vars = self.parse_losses(target_losses)  # type: ignore
        optim_wrapper.backward(parsed_losses)
        log_vars.update(t_log_vars)

        optim_wrapper.step()
        log_vars.pop('loss', None)

        return log_vars



# 在源域图片上评估了双曲确定性，最终mIoU小下降了一点；
# 250302_1018_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_0a8e9