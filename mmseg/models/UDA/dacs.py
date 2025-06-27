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
from mmengine.structures import PixelData

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
from mmseg.visualization.hyper_hierarchy_visualizer import HHLocalVisualizer
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
# from mmseg.models.decode_heads import FloatingRegionScore, init_mask, select_pixels_to_label

import kornia
import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms


from mmseg.utils.uda_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform, downscale_label_ratio)



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
                 imnet_feature_dist_lambda=0,
                 imnet_feature_dist_classes=[],
                 imnet_feature_dist_scale_min_ratio=0,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg = None,
                 debug_iter = 200,
                 ):
        super(DACS, self).__init__(uda_model,
                                   data_preprocessor,init_cfg)

        self.train_cfg = {}
        self.message_hub = MessageHub.get_current_instance()
        self.message_hub.update_info_dict({'debug_iter':debug_iter})
        self.debug_iter = debug_iter

        # self.local_iter = 0
        # self.max_iters = cfg['max_iters'] # cfg配置文件里面暂时没有这个参数，可能是后面手动添加的。经过检查确实前期将runner的max_iters加入了
        self.alpha = 0.999
        self.pseudo_threshold = pseudo_threshold
        self.psweight_ignore_top = pseudo_weight_ignore_top
        self.psweight_ignore_bottom = pseudo_weight_ignore_bottom
        self.fdist_lambda = imnet_feature_dist_lambda
        self.fdist_classes = imnet_feature_dist_classes
        self.fdist_scale_min_ratio = imnet_feature_dist_scale_min_ratio
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = mix
        self.blur = blur
        self.color_jitter_s = color_jitter_strength
        self.color_jitter_p = color_jitter_probability
        # self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = False
        # assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None
        self.visualizer = HHLocalVisualizer.get_current_instance()

        # self.class_probs = {}
        ema_cfg = deepcopy(uda_model) # cfg配置文件里面暂时没有这个参数，可能是后面手动添加的。经过检查确实前期将配置文件中的model配置字典加入了。
        fea_cfg = deepcopy(uda_model)
        self.model = build_segmentor(uda_model)
        self.ema_model = build_segmentor(ema_cfg)


        if self.enable_fdist:
            self.imnet_model = build_segmentor(fea_cfg)
        else:
            self.imnet_model = None

        # self.transforms = transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
    def get_model(self):
        return self.model


    def get_ema_model(self):
        return self.ema_model

    def get_imnet_model(self):
        return self.imnet_model

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

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, inputs: Tensor, data_samples: SampleList, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(inputs)
            feat_imnet = [f.detach() for f in feat_imnet]
        gt = torch.cat([samples.gt_sem_seg.data.unsqueeze(0) for samples in data_samples], dim=0)
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self.parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

# TODO: 为了保持框架一致，train_step方法不应该被重构，loss应该被重构，模仿encoder_decoder
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
        # with open('debug/train_train_b.txt', 'a', encoding='utf-8') as file:
        #     print('*************source*************', file=file)
        source_loss_decode = self.get_model().decode_head.loss(source_x, data_samples,
                                            self.train_cfg)

        if self.enable_fdist:
            losses['features'] = source_x


        losses.update(add_prefix(source_loss_decode, 'source'))
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


    def target_loss(self, data: dict, cur_iter: int) -> dict:
        losses = dict()
        source_data = data[0]
        target_data = data[1]
        inputs = target_data['inputs']
        data_samples = target_data['data_samples']
        # gt_semantic_seg = source_data['data_samples'][0].gt_sem_seg.data.unsqueeze(0)
        gt_semantic_seg = torch.cat([gt_l.gt_sem_seg.data.unsqueeze(0) for gt_l in source_data['data_samples']])
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

        target_feat = self.get_ema_model().extract_feat(inputs)
        probs, _ = self.get_ema_model().decode_head.forward(target_feat, data_samples[0].pad_shape)

        probs = probs[:,:19,:,:]
        pseudo_prob, pseudo_label = torch.max(probs.detach(), dim=1)
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
        mix_masks = get_class_masks(gt_semantic_seg, cur_iter, )

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

        for i in range(len(data_samples)):
            data_samples[i].gt_sem_seg.data = mixed_lbl[i]


        target_x = self.get_model().extract_feat(mixed_img)
        target_loss_decode = self.get_model().decode_head.loss(target_x, data_samples,
                                                               self.train_cfg, pseudo_weight)


        losses.update(add_prefix(target_loss_decode, 'mix'))

        if self.message_hub.get_info('iter') % self.message_hub.get_info('debug_iter') == 0:
            s_t_img = torch.cat((source_data['inputs'][0], target_data['inputs'][0]), dim=1)
            s_t_img = self.visualizer.return_add_datasample(image=s_t_img,
                                                            draw_gt=False,
                                                            draw_pred=False)
            mixed_img_label = self.visualizer.return_add_datasample(image=mixed_img[0],
                                                                    data_sample=PixelData(data=mixed_lbl[0]),
                                                                    draw_gt=True,
                                                                    draw_pred=False,
                                                                    )
            draw_img = self.visualizer.return_add_datasample(image=mixed_img[0],
                                                                    draw_gt=False,
                                                                    draw_pred=False,
                                                                    )
            draw_ps_weight = (pseudo_weight[0].unsqueeze(0)).expand(3,-1,-1).cpu()*255
            draw_ps_weight = np.clip(draw_ps_weight.numpy(), 0, 255).astype(np.uint8)
            draw_ps_weight = np.ascontiguousarray(draw_ps_weight.transpose(1, 2, 0))

            draw_mix_mask = (mix_masks[0][0].cpu()).expand(3,-1,-1)*255
            draw_mix_mask = np.clip(draw_mix_mask.numpy(), 0, 255).astype(np.uint8)
            draw_mix_mask = np.ascontiguousarray(draw_mix_mask.transpose(1,2,0))
            cat_mix_img = np.concatenate((draw_img,mixed_img_label), axis=0)
            cat_mask_weight = np.concatenate((draw_mix_mask, draw_ps_weight), axis=0)
            debug_img = np.concatenate((s_t_img,cat_mix_img,cat_mask_weight), axis=1)
            self.visualizer.add_datasample(name='target_mix_{}'.format(self.message_hub.get_info('iter')),
                                           image=debug_img,
                                           draw_gt=False,
                                           draw_pred=False)
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


        source_data = data['source']
        target_data = data['target']
        cur_iter = self.message_hub.get_info('iter')
        if cur_iter == 0:
            self._init_ema_weights()
        if cur_iter > 0:
            self._update_ema(cur_iter)

        if self.message_hub.get_info('iter') % 100 == 0:
            for i in range(len(source_data['data_samples'])):
                source_data['data_samples'][i].set_data({'ori_img':source_data['inputs'][i]})
                target_data['data_samples'][i].set_data({'ori_img':target_data['inputs'][i]})

        source_data = self.data_preprocessor(source_data, True)

        source_losses = self._run_forward(source_data, mode='source_loss')  # type: ignore
        if self.enable_fdist:
            src_feat = source_losses.pop('features')
        source_parsed_losses, log_vars = self.parse_losses(source_losses)  # type: ignore
        # optim_wrapper.zero_grad()
        optim_wrapper.backward(source_parsed_losses, retain_graph=True)
        # optim_wrapper.step()
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            print(f'Source Seg. Grad.: {grad_mag}')

        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(**source_data, feat=src_feat)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'source'))




        target_data['inputs'] = torch.stack(target_data['inputs'])
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # with optim_wrapper.optim_context(self):
        target_data = self.data_preprocessor(target_data, True)
        target_losses = self.target_loss([source_data, target_data], cur_iter)
        target_parsed_losses, t_log_vars = self.parse_losses(target_losses)  # type: ignore
        log_vars.update(t_log_vars)
        optim_wrapper.backward(target_parsed_losses, retain_graph=True)

        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            print(f'Target Seg. Grad.: {grad_mag}')

        optim_wrapper.step()
        optim_wrapper.zero_grad()
        log_vars.pop('loss', None)

        return log_vars



# 在源域图片上评估了双曲确定性，最终mIoU小下降了一点；
# 250302_1018_gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_0a8e9