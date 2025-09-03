# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

import json
import math
from ..utils.tree import Tree



class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super().__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@MODELS.register_module()
class ASPPHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super().__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output



from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
PROJ_EPS = 1e-3
class HyperMLR(nn.Module):
    """Multinomial logistic regression in hyperbolic space."""

    def __init__(self, out_channels, num_classes, c=0.5):
        """Initialize the model.

        Args:
            num_classes (int): Number of classes
            out_channels (int): Number of channels of the input features
            c (float, optional): Hyperbolic curvature. Defaults to 1.
        """
        super().__init__()
        self.c = c
        self.K = torch.tensor(c, dtype=float)
        self.num_classes = num_classes
        self.P_MLR = Parameter(torch.empty((num_classes, out_channels), dtype=torch.float32))
        self.A_MLR = Parameter(torch.empty((num_classes, out_channels), dtype=torch.float32))
        kaiming_uniform_(self.P_MLR, a=math.sqrt(5))
        kaiming_uniform_(self.A_MLR, a=math.sqrt(5))

    def _hyper_logits(self, inputs):
        """Compute the logits in hyperbolic space.

        Args:
            inputs (torch.Tensor): Tensor of shape (B, C, H, W)
        """
        # B = batch size
        # C = number of channels
        # H, W = height and width of the input
        # O = number of classes

        # P_MLR: (O,C)
        # A_MLR: (O,C)
        # output: (B,H,W,O)

        # normalize inputs and P_MLR
        xx = torch.norm(inputs, dim=1)**2  # (B,H,W)
        pp = torch.norm(-self.P_MLR, dim=1)**2  # (O,)
        P_kernel = -self.P_MLR[:, :, None, None]  # (O,C,1,1)

        # compute cross correlations
        px = torch.nn.functional.conv2d(input=inputs, weight=P_kernel, stride=1,
                                        padding='same', dilation=1, groups=1)  # (B,O,H,W)
        pp = pp.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # (1,O,1,1)

        # c^2 * | X|^2 * |-P|^2
        sqsq = self.K * xx.unsqueeze(1) * self.K * pp  # (B,O,H,W)

        # Rewrite mob add as alpha * p + beta * x
        # where alpha = A / D and beta = B / D
        A = 1 + 2 * self.K * px + self.K * xx.unsqueeze(1)  # (B,O,H,W)
        B = 1 - self.K * pp  # (1,O,1,1)
        D = 1 + 2 * self.K * px + sqsq  # (B,O,H,W)
        D = torch.max(D, torch.tensor(1e-12, device=inputs.device))
        alpha = A / D  # (B,O,H,W)
        beta = B / D  # (B,O,H,W)

        # Calculate mobius addition norm indepently from the mobius addition
        # (B,O,H,W)
        mobaddnorm = ((alpha ** 2 * pp) + (beta ** 2 * xx.unsqueeze(1)) + (2 * alpha * beta * px))
        # now in order to project the mobius addition onto the hyperbolic disc
        # we need to divide vectors whos l2norm : |x| (not |x|^2) are higher than max norm
        maxnorm = (1.0 - PROJ_EPS) / torch.sqrt(self.K)
        project_normalized = torch.where(  # (B,O,H,W)
            torch.sqrt(mobaddnorm) > maxnorm,  # condition
            maxnorm / torch.max(torch.sqrt(mobaddnorm), torch.tensor(1e-12, device=inputs.device)),  # if true
            torch.ones_like(mobaddnorm))  # if false
        mobaddnormprojected = torch.where(  # (B,O,H,W)
            torch.sqrt(mobaddnorm) < maxnorm,  # condition
            mobaddnorm,  # if true
            torch.ones_like(mobaddnorm) * maxnorm ** 2)  # if false

        A_norm = torch.norm(self.A_MLR, dim=1)  # (O,)
        normed_A = torch.nn.functional.normalize(self.A_MLR, dim=1)  # (O,C)

        # TODO：源码的A_kernel进行了一次转置。这里需不需要？不需要，kernel形状早已转置了
        A_kernel = normed_A[:, :, None, None]  # (O,C,1,1)
        xdota = beta * torch.nn.functional.conv2d(inputs, weight=A_kernel)  # (B,O,H,W)
        pdota = (alpha * torch.sum(-self.P_MLR * normed_A, dim=1)[None, :, None, None])  # (B,O,H,W)
        mobdota = xdota + pdota  # (B,O,H,W)
        mobdota *= project_normalized  # equiv to project mob add to max norm before dot
        lamb_px = 2.0 / torch.max(1 - self.K * mobaddnormprojected, torch.tensor(1e-12, device=inputs.device))
        sineterm = torch.sqrt(self.K) * mobdota * lamb_px
        lambda_term = 2.0 # / (1 - self.K * pp)  # (1,O,1,1)
        out = lambda_term / torch.sqrt(self.K) * A_norm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * \
            torch.asinh(sineterm)  # (B,O,H,W)
        return out

    def forward(self, x):
        logits = self._hyper_logits(x)
        return logits


from typing import List, Tuple
from mmseg.utils import ConfigType, SampleList, add_prefix
from torch import Tensor
import copy
@MODELS.register_module()
class ASPPHead_HH(ASPPHead):
    def __init__(self, tree_params=None, c=0.5,**kwargs):
        super(ASPPHead_HH, self).__init__(**kwargs)

        tree_params['i2c'] = self.txt2dict(tree_params['i2c'])
        with open(tree_params['json']) as f:
            tree_params['json'] = json.load(f)

        self.tree = Tree(**tree_params)
        self.embedding_layer = ConvModule(512, 256, kernel_size=(1, 1), norm_cfg=None, act_cfg=None)
        self.c = c
        # self.hyper_mlr = HyperMLR(512,self.tree.M, c=c)
        self.hyper_mlr = HyperMLR(256, self.tree.M, c=c)  # for test

    @staticmethod
    def txt2dict(fn):
        """ Reads txt file and converts to idx2concept dictionary"""
        with open(fn, 'r') as f:
            ls = f.readlines()
        d = {}
        i = 0
        for l in ls:
            _, c = l.split(':')
            d[i] = c.strip()
            i += 1
        return d


    @staticmethod
    def torch_project_hyp_vecs(x, c, dim=-1):
        """
        Project hyperbolic vectors to ensure they stay within the Poincaré ball.

        Args:
            x: Input tensor
            c: Curvature
            dim: Dimension to compute norm over

        Returns:
            Clipped tensor within the Poincaré ball
        """
        PROJ_EPS = 1e-5
        max_norm = (1.0 - PROJ_EPS) / math.sqrt(c)

        # Compute norms along specified dimension
        norms = torch.norm(x, p=2, dim=dim, keepdim=True)

        # Clip norms
        clipped = torch.clamp(norms, max=max_norm)

        # Project vectors
        return x * (clipped / (norms + PROJ_EPS))

    def torch_exp_map_zero(self, inputs, c, EPS=1e-7):
        """
        PyTorch implementation of exponential mapping from Euclidean to hyperbolic space (Poincaré ball model).

        Args:
            inputs: Input tensor of shape [n, d] (Euclidean coordinates)
            c: Curvature of the hyperbolic space (positive scalar)
            EPS: Small constant for numerical stability

        Returns:
            Projected points in the Poincaré ball
        """
        sqrt_c = torch.sqrt(torch.tensor(c, device=inputs.device))

        # Add epsilon to avoid division by zero
        inputs = inputs + EPS

        # Compute norm along the last dimension
        norm = torch.norm(inputs, p=2, dim=1, keepdim=True)

        # Compute scaling factor gamma
        gamma = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)

        # Scale the input vectors
        scaled_inputs = gamma * inputs

        # Project to Poincaré ball
        return self.torch_project_hyp_vecs(scaled_inputs, c, dim=1)

    def embedding_norm(self, x, min_scale=0.1, max_scale=0.9):
        radius = 1.0 / torch.sqrt(torch.tensor(self.c))
        target_min = min_scale * radius
        target_max = max_scale * radius
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp_min(1e-5)
        scale = target_min + (target_max - target_min) * (x_norm - x_norm.min()) / (x_norm.max() - x_norm.min())
        x_normalized = x * (scale / x_norm)
        return x_normalized


    def hrc_softmax(self, logits):
        logits = logits - torch.max(logits, dim=1, keepdim=True).values
        exp_logits = torch.exp(logits)
        with torch.amp.autocast(enabled=False,device_type='cuda'):
            Z = torch.einsum('bijk,li->bljk',exp_logits,self.tree.sibmat.cuda())
        cond_probs = exp_logits / torch.clamp(Z, min=1e-15)
        return cond_probs

    def get_joints(self, cond_probs):
        log_probs = torch.log(torch.max(cond_probs, torch.tensor(1e-4)))
        log_sum_p = torch.einsum('bijk,li->bljk',log_probs,self.tree.hmat.cuda())
        joints = torch.exp(log_sum_p)
        return joints

    def run(self, projected_embedding, input_size):
        logits = self.hyper_mlr(projected_embedding)
        logits = resize(
            input=logits,
            size=input_size,
            mode='bilinear',
            align_corners=self.align_corners)
        cond_probs = self.hrc_softmax(logits)
        joints = self.get_joints(cond_probs)


        return joints, cond_probs

    def cls_seg(self, feat, input_size):
        """new head"""
        if self.dropout is not None:
            feat = self.dropout(feat)
        embedding = self.embedding_layer(feat)
        embedding = self.embedding_norm(embedding)
        # projected_embedding = self.torch_exp_map_zero(embedding, c=0.5)
        projected_embedding = self.torch_exp_map_zero(embedding, c=self.c)
        probs, cprobs = self.run(projected_embedding, input_size)
        # predictions = self.decide(probs)
        return probs, cprobs

    def forward(self, inputs, img_size):
        output = self._forward_feature(inputs)
        output = self.cls_seg(output, img_size)
        return output

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType, seg_weight=None) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.
            seg_weight: None

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        probs, cprobs = self.forward(inputs, batch_data_samples[0].pad_shape)

        # losses = self.loss_by_feat(cprobs, batch_data_samples, seg_weight)
        losses = self.hierarchy_loss(probs, batch_data_samples, seg_weight)
        return losses

    def hierarchy_loss(self, seg_logits: Tensor,
                       batch_data_samples: SampleList, seg_weight=None) -> dict:

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()

        # h_labels_weight = self.cal_label_weight(temperature=0.5)

        h_labels = self.generate_hrc_labels(seg_label)
        new_batch_data = self.reflect_labels_logits(seg_logits, h_labels)
        for i, data in enumerate(new_batch_data):
            seg_logits = data[0]
            seg_label = data[1]
            label_flat = seg_label.view(seg_label.shape[0], -1)

            num_target = seg_logits.shape[1]
            valid_mask = (label_flat < num_target)
            valid_labels = label_flat[valid_mask]

            log_probs = torch.log(torch.clamp(seg_logits, min=1e-8))
            flat_cprobs = log_probs.permute(0, 2, 3, 1).contiguous().view((log_probs.shape[0], -1, log_probs.shape[1]))
            valid_probs = flat_cprobs[valid_mask]

            pos_logp = torch.gather(valid_probs, dim=1, index=valid_labels.unsqueeze(1))
            d_loss = -torch.mean(pos_logp)
            loss['{}_depth'.format(i + 1)] = d_loss
            if 'loss_ce' not in loss.keys():
                loss['loss_ce'] = d_loss
            else:
                loss['loss_ce'] = loss['loss_ce'] + d_loss

        return loss

    def generate_hrc_labels(self, label):
        labels = []
        B = label.shape[0]
        for b in range(B):
            b_label = []
            for i in self.tree.depth_idx.keys():
                temp_label = copy.deepcopy(label[b][0])
                for ic, ia in self.tree.abs2con[i].items():
                    temp_label[temp_label == ic] = ia

                b_label.append(temp_label.unsqueeze(0))
            labels.append(torch.cat(b_label, dim=0).unsqueeze(0))


        return torch.cat(labels, dim=0)

    def reflect_labels_logits(self, seg_logits: Tensor, labels: Tensor) -> list:

        h1_label_mask = (labels[0][0] == 23)
        h2_label_mask = (labels[0][1] == 16)
        h3_label_mask = (labels[0][2] == 16)

        B = labels.shape[0]
        depth = labels.shape[1]
        data = []
        batch_data = []
        for b in range(B):
            for d in range(depth):
                values, indices = torch.sort(torch.Tensor(list(set(self.tree.abs2con[d + 1].values()))))
                tem_seg = torch.cat(
                    [seg_logits[b][i].unsqueeze(0) for i in sorted(set(self.tree.abs2con[d + 1].values())) if i != 255],
                    dim=0)
                for i, v in zip(indices, values):
                    if i != 255:
                        labels[b][d][labels[b][d] == v] = i
                data.append({'input_feat': tem_seg, 'label': labels[b][d]})
        for d in range(depth):
            temp_input = []
            temp_label = []
            for b in range(B):
                temp_input.append(data[d + b * depth]['input_feat'].unsqueeze(0))
                temp_label.append(data[d + b * depth]['label'].unsqueeze(0))
            batch_data.append([torch.cat(temp_input, dim=0), torch.cat(temp_label, dim=0)])

        return batch_data