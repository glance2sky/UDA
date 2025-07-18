
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union, List

import torch
from torch import Tensor

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models import build_segmentor
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
from mmengine.optim import OptimWrapper

class UDADecorator(BaseSegmentor):

    def __init__(self, uda_model,
                 data_preprocessor: OptConfigType = None,
                 init_cfg = None):
        super(BaseSegmentor, self).__init__(
            data_preprocessor, init_cfg
        )

        self.test_cfg = uda_model['test_cfg']
        self.num_classes = uda_model['decode_head']['num_classes']

    def get_model(self):
        return self.model

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      return_feat=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=return_feat)
        return losses

    def loss(self):
        pass

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        self.visualizer.add_temp_embeding_for_poincare(label=data_samples[0].gt_sem_seg.data)
        return self.get_model().predict(inputs, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        pass

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'source_loss':
            return self.source_loss(inputs, data_samples)
        elif mode == 'target_loss':
            return self.target_loss(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
