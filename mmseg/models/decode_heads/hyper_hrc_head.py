
from ..utils.tree import Tree
from mmseg.ops import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .sep_aspp_head import DepthwiseSeparableASPPModule
from .segformer_head import MLP


import torch
import copy
import math
import json
from typing import List, Tuple
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_

import geoopt.manifolds.stereographic.math as gmath

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.utils import ConfigType, SampleList, add_prefix
from mmseg.visualization.hyper_hierarchy_visualizer import HHLocalVisualizer
from mmseg.models.utils import PoincareManifold

from mmengine.logging import MessageHub

PROJ_EPS = 1e-3


def tensor_hook(grad):
    if grad.abs().max() > 1e6:
        print(f": max:{grad.abs().max().item():.3e}")
        import traceback
        traceback.print_stack(limit=5)
    print(grad.abs().max().item())

    return grad



class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))

    else:
        raise NotImplementedError(type)


class HyperMapper(object):
    """A class to map between euclidean and hyperbolic space and compute distances."""

    def __init__(self, c=1.) -> None:
        """Initialize the hyperbolic mapper.

        Args:
            c (float, optional): Hyperbolic curvature. Defaults to 1.0
        """
        self.c = c
        self.K = torch.tensor(-self.c, dtype=float)

    def expmap(self, x, dim=-1, min_scale=0.1, max_scale=0.9):
        """Exponential mapping from Euclidean to hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (..., d)
        """
        radius = 1.0 / torch.sqrt(torch.tensor(self.c))
        target_min = min_scale * radius
        target_max = max_scale * radius
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp_min(1e-5)
        scale = target_min + (target_max - target_min)*(x_norm - x_norm.min()) / (x_norm.max() - x_norm.min())
        x_normalized = x * (scale / x_norm)
        x_hyp = gmath.expmap0(x_normalized, k=self.K, dim=dim)

        x_hyp = gmath.project(x_hyp, k=self.K, dim=dim)
        return x_hyp

    def expmap2(self, inputs, dim=-1):
        PROJ_EPS = 1e-3
        EPS = 1e-6
        sqrt_c = torch.sqrt(torch.abs(self.K))
        inputs = inputs + EPS  # protect div b 0
        norm = torch.norm(inputs, dim=dim)
        gamma = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)  # sh ncls
        scaled_inputs = gamma.unsqueeze(dim) * inputs
        return gmath.project(scaled_inputs, k=self.K, dim=dim, eps=PROJ_EPS)

    def logmap(self, x):
        """Logarithmic mapping from hyperbolic to Euclidean space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (..., d)
        """
        return gmath.project(gmath.logmap0(x.double(), k=self.K), k=self.K)

    def poincare_distance(self, x, y):
        """Poincare distance between two points in hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)
            y (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        return gmath.dist(x, y, k=self.K)

    def poincare_distance_origin(self, x, dim=-1):
        """Poincare distance between two points in hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        return gmath.dist0(x, k=self.K, dim=dim)

    def cosine_distance(self, x, y):
        """Cosine distance between two points.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)
            y (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)


TEST_EPS = 1e-6
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
        self.P_MLR = Parameter(torch.empty((num_classes, out_channels), dtype=torch.float))
        self.A_MLR = Parameter(torch.empty((num_classes, out_channels), dtype=torch.float))
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

@HEADS.register_module()
class HHHead(BaseDecodeHead):
    EPS = torch.tensor(1e-12)
    def __init__(self, decoder_params=None, tree_params=None,c=1.0,temp=0.5, **kwargs):
        super(HHHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = decoder_params
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

        tree_params['i2c'] = txt2dict(tree_params['i2c'])
        with open(tree_params['json']) as f:
            tree_params['json'] = json.load(f)

        # option 1: custom definehyper MLR
        self.c = c
        # self.K = torch.tensor(-self.c, dtype=torch.float)
        self.K = c

        self.tree = Tree(temp=temp,**tree_params)
        self.embedding_layer = ConvModule(256,512, kernel_size=(1,1), norm_cfg=None, act_cfg=None)
        self.hyper_mlr = HyperMLR(512,self.tree.M, c=self.c)
        self.mapper = HyperMapper(c=self.c)
        # option 2: use other defined hyperMLR
        # self.manifold_dec = PoincareManifold(
        #     k=self.c,
        #     learn_k=False,
        #     embed_dim=1024,
        #     num_classes=self.tree.M,
        #     clip_r=1.0,
        #     enc_type='euclidean',
        #     manifold_type='poincare'
        # )


        # self.debug_img = True
        self.debug_img = False

        self.visualizer = HHLocalVisualizer.get_current_instance()
        self.message_hub = MessageHub.get_current_instance()

    def torch_project_hyp_vecs(self, x, dim=1):
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
        max_norm = (1.0 - PROJ_EPS) / math.sqrt(self.c)

        # Compute norms along specified dimension
        norms = torch.norm(x, p=2, dim=dim, keepdim=True)

        # Clip norms
        clipped = torch.clamp(norms, max=max_norm)

        # Project vectors
        return x * (clipped / (norms + PROJ_EPS))
        # x_hyp = gmath.project(x, k=self.K, dim=dim)
        # return x_hyp

    def torch_exp_map_zero(self, inputs, dim=1):
        """
        PyTorch implementation of exponential mapping from Euclidean to hyperbolic space (Poincaré ball model).

        Args:
            inputs: Input tensor of shape [n, d] (Euclidean coordinates)
            c: Curvature of the hyperbolic space (positive scalar)
            EPS: Small constant for numerical stability

        Returns:
            Projected points in the Poincaré ball
        """
        EPS = 1e-7
        sqrt_c = torch.sqrt(torch.tensor(self.c, device=inputs.device))

        # Add epsilon to avoid division by zero
        inputs = inputs + EPS

        # Compute norm along the last dimension
        norm = torch.norm(inputs, p=2, dim=1, keepdim=True)

        # Compute scaling factor gamma
        gamma = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)

        # Scale the input vectors
        scaled_inputs = gamma * inputs

        # Project to Poincaré ball
        return self.torch_project_hyp_vecs(scaled_inputs, dim=1)
        # x_hyp = gmath.expmap0(inputs, k=self.K, dim=dim)
        # x_hyp = gmath.project(x_hyp, k=self.K, dim=dim)
        # return x_hyp

    def hrc_softmax(self, logits):
        logits = logits - torch.max(logits, dim=1, keepdim=True).values
        exp_logits = torch.exp(logits)
        with torch.amp.autocast(enabled=False,device_type='cuda'):
            Z = torch.einsum('bijk,li->bljk',exp_logits,self.tree.sibmat.cuda())
        cond_probs = exp_logits / torch.clamp(Z, min=1e-15)
        return cond_probs

    def get_joints(self, cond_probs):
        # cond_probs.register_hook(tensor_hook)
        # TODO:
        log_probs = torch.log(torch.max(cond_probs, torch.tensor(1e-12)))
        # log_probs.register_hook(tensor_hook)
        # log_probs.register_hook(tensor_hook)
        log_sum_p = torch.einsum('bijk,li->bljk',log_probs,self.tree.hmat.cuda())
        # log_sum_p.register_hook(tensor_hook)
        joints = torch.exp(log_sum_p)
        # if joints.grad:
        #     print(joints.grad.abs().max().item())
        # # joints.register_hook(tensor_hook)
        return joints

    def run(self, projected_embedding, input_size):
        logits = self.hyper_mlr(projected_embedding)
        logits = resize(
            input=logits,
            size=input_size,
            mode='bilinear',
            align_corners=self.align_corners)
        cond_probs = self.hrc_softmax(logits)
        # cond_probs.register_hook(tensor_hook)
        joints = self.get_joints(cond_probs)
        # joints.register_hook(tensor_hook)


        return joints, cond_probs

    def decide(self, probs, unseen=None):
        """
        PyTorch implementation of class decision function.

        Args:
            probs: Probability tensor of shape [..., num_classes]
            unseen: List of unseen class indices to exclude (optional)

        Returns:
            Predicted class indices tensor
        """
        if unseen is None:
            unseen = []
        if len(probs.shape) == 3:
            probs = probs.unsqueeze(0)

        # Gather target class probabilities
        cls_probs = probs[:, :self.tree.K, :, :]  # Equivalent to tf.gather with np.arange

        # Apply hierarchy matrix filtering
        # hmat_subset = self.tree.hmat[:self.tree.K, :self.tree.K].to(probs.device)
        #TODO:这里在condition probs的基础上又计算了一次condition probs，是否有些不妥？是否有更高的方法区处理那些中间类
        # 而且如果目标类别全是叶子节点，那么目标类别之间也就不可能有层级关系。
        # cls_probs = torch.matmul(cls_probs, hmat_subset)

        # Handle unseen classes
        if len(unseen) > 0:
            # Convert unseen to tensor and move to same device
            unseen_tensor = torch.tensor(unseen, device=probs.device)

            # Gather probabilities for unseen classes only
            cls_gather = cls_probs[..., unseen_tensor]

            # Get predictions within unseen classes
            predict_ = torch.argmax(cls_gather, dim=-1)
            predictions = unseen_tensor[predict_]
        else:
            # Standard argmax over all classes
            predictions = torch.argmax(cls_probs, dim=1)

        return predictions

    def cls_seg(self, feat, input_size):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        # feat.register_hook(tensor_hook)
        embedding = self.embedding_layer(feat)
        # embedding = feat
        # embedding.register_hook(tensor_hook)

        # projected_embedding = self.mapper.expmap(embedding, dim=1)
        projected_embedding = self.torch_exp_map_zero(embedding)
        # projected_embedding.register_hook(tensor_hook)
        probs, cprobs = self.run(projected_embedding, input_size)
        # probs.register_hook(tensor_hook)
        # predictions = self.decide(probs)
        return probs, cprobs

    def r_cls_seg(self, feat, input_size):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        # feat.register_hook(tensor_hook)
        logits = self.manifold_dec(feat)
        embed = self.manifold_dec.embed(feat)
        logits = resize(
            input=logits,
            size=input_size,
            mode='bilinear',
            align_corners=self.align_corners)
        cond_probs = self.hrc_softmax(logits)
        # cond_probs.register_hook(tensor_hook)
        joints = self.get_joints(cond_probs)
        # joints.register_hook(tensor_hook)

        return joints, cond_probs



    def forward(self, inputs, img_size):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        probs, cprobs = self.cls_seg(x, img_size)
        # probs.register_hook(tensor_hook)

        # probs, cprobs = self.cls_seg(x, img_size)
        # probs.register_hook(tensor_hook)



        return probs, cprobs

# manifold vision
#     def forward(self, inputs, img_size):
#         x = inputs
#         n, _, h, w = x[-1].shape
#         # for f in x:
#         #     mmcv.print_log(f'{f.shape}', 'mmseg')
#
#         os_size = x[0].size()[2:]
#         _c = {}
#         for i in self.in_index:
#             # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
#             _c[i] = x[i]
#             if _c[i].dim() == 3:
#                 _c[i] = _c[i].permute(0, 2, 1).contiguous()\
#                     .reshape(n, -1, x[i].shape[2], x[i].shape[3])
#             # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
#             if _c[i].size()[2:] != os_size:
#                 # mmcv.print_log(f'resize {i}', 'mmseg')
#                 _c[i] = resize(
#                     _c[i],
#                     size=os_size,
#                     mode='bilinear',
#                     align_corners=self.align_corners)
#
#         x = torch.cat(list(_c.values()), dim=1)
#         probs, cprobs = self.r_cls_seg(x, img_size)
#         # probs.register_hook(tensor_hook)
#
#         # probs, cprobs = self.cls_seg(x, img_size)
#         # probs.register_hook(tensor_hook)
#
#
#         return probs, cprobs


    def feat_embed(self,inputs, img_size):
        x = inputs
        n, _, h, w = x[-1].shape

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = x[i]
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous() \
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = torch.cat(list(_c.values()), dim=1)
        logits = self.manifold_dec(x)
        embed = self.manifold_dec.embed(x)
        logits = resize(
            input=logits,
            size=img_size,
            mode='bilinear',
            align_corners=self.align_corners)
        cond_probs = self.hrc_softmax(logits)
        # cond_probs.register_hook(tensor_hook)
        joints = self.get_joints(cond_probs)
        # joints.register_hook(tensor_hook)

        return joints, cond_probs, embed

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
        # probs.register_hook(tensor_hook)
        # B = probs.shape[0]
        # for b in range(B):
        #     cls_uni = torch.unique(batch_data_samples[b].gt_sem_seg.data)
        #     if 16 in cls_uni:
        #         cls_mask = (batch_data_samples[b].gt_sem_seg.data == 16)
        #         num_cls = torch.sum(cls_mask)
        #         label_num_cls = torch.sum(probs[b][:19].argmax(0)==16)
        #         rig_num_cls = torch.sum(probs[b][:19].argmax(0)[cls_mask[0]] == 16)
        #         iou = rig_num_cls / (num_cls + label_num_cls - rig_num_cls)
        #         with open('debug/train_train_b.txt', 'a', encoding='utf-8') as file:
        #             print('train in label: {} --- train in pred: {} --- right pred: {} --- iou: {}'.format(num_cls, label_num_cls, rig_num_cls, iou), file=file)



        # debug_info = self.debug_class(16, probs, batch_data_samples)
        # losses = self.loss_by_feat(cprobs, batch_data_samples, seg_weight)
        losses = self.hierarchy_loss(probs, batch_data_samples, seg_weight)
        # losses.update(add_prefix(debug_info, 'debug'))
        return losses

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList, seg_weight=None) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.
            seg_weight: None

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        # criterion = torch.nn.NLLLoss(ignore_index=255)
        # loss['ce'] = criterion(torch.log(seg_logits + 1e-8), seg_label.squeeze())


        label_flat = seg_label.view(seg_label.shape[0], -1)
        num_target = self.tree.M
        valid_mask = (label_flat < num_target)
        valid_labels = label_flat[valid_mask]

        log_probs = torch.log(torch.clamp(seg_logits, min=1e-8))
        log_sum_p = torch.einsum('bijk,li->bljk', log_probs, self.tree.hmat.cuda())
        log_sum_p = log_sum_p[:,:19,:,:]
        flat_cprobs = log_sum_p.permute(0,2,3,1).contiguous().view((log_sum_p.shape[0],-1,log_sum_p.shape[1]))
        valid_cprobs = flat_cprobs[valid_mask]

        if seg_weight is not None:
            flat_weight = seg_weight.view(seg_weight.shape[0], -1)
            valid_weight = flat_weight[valid_mask]
            indices = valid_labels.view(-1, 1).expand(-1, valid_cprobs.size(1))
            pos_logp = torch.gather(valid_cprobs, dim=1, index=indices)[:, 0]
            loss['loss_ce'] = -torch.mean(pos_logp * valid_weight)
            return loss


        indices = valid_labels.view(-1, 1).expand(-1, valid_cprobs.size(1))
        pos_logp = torch.gather(valid_cprobs, dim=1, index=indices)[:, 0]
        loss['loss_ce'] = -torch.mean(pos_logp)




        return loss

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        probs, cprobs = self.forward(inputs, batch_img_metas[0]['img_shape'])
        # self.visualizer.add_temp_embeding_for_poincare(embed=embed)
        if probs.shape[1] != self.tree.K:
            probs = probs[:,:self.tree.K,:,:]

        # self.visualizer.visualize_hyperbolic(i2c=self.tree.i2c, manifold=self.manifold_dec.dec_manifold, poincare=True)

        return self.predict_by_feat(probs, batch_img_metas)

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
        # data[0]['input_feat'].register_hook(tensor_hook)
        # data[1]['input_feat'].register_hook(tensor_hook)
        # data[2]['input_feat'].register_hook(tensor_hook)
        for d in range(depth):
            temp_input = []
            temp_label = []
            for b in range(B):
                temp_input.append(data[d + b * depth]['input_feat'].unsqueeze(0))
                temp_label.append(data[d + b * depth]['label'].unsqueeze(0))
            batch_data.append([torch.cat(temp_input, dim=0), torch.cat(temp_label, dim=0)])



        return batch_data

    def hierarchy_loss(self, seg_logits: Tensor,
                       batch_data_samples: SampleList, seg_weight=None) -> dict:

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        # seg_logits.register_hook(tensor_hook)

        h_labels = self.generate_hrc_labels(seg_label)
        # h_labels_weight = self.cal_label_weight(temperature=0.5)
        new_batch_data = self.reflect_labels_logits(seg_logits, h_labels)
        # new_batch_data[0][0].register_hook(tensor_hook)
        # new_batch_data[1][0].register_hook(tensor_hook)
        # new_batch_data[2][0].register_hook(tensor_hook)
        # labels_weight = self.reflect_labels_weight(h_labels_weight)
        # labels_weight = self.reflect_siblings_labels_weight()
        # weight_mask = self.generate_weight_mask(labels_weight, new_batch_data)

        if self.debug_img:
            if self.message_hub.get_info('iter') % self.message_hub.get_info('debug_iter') == 0:
                if 'gta' in batch_data_samples[0].img_path:
                    debug_name = 'source_hierarchy_img_{}'.format(self.message_hub.get_info('iter'))
                else:
                    debug_name = 'mix_hierarchy_img_{}'.format(self.message_hub.get_info('iter'))

                self.visualizer.draw_hierarchy_map(name=debug_name,
                                                   image=batch_data_samples[0].ori_img,
                                                   batch_data=new_batch_data,
                                                   hierarchy_classes=self.tree.hie_classes_name,
                                                   draw_gt=True,
                                                   draw_pred=True)

        for i, data in enumerate(new_batch_data):
            seg_logits = data[0]
            # seg_logits.register_hook(tensor_hook)
            seg_label = data[1]
            label_flat = seg_label.view(seg_label.shape[0], -1)

            num_target = seg_logits.shape[1]
            valid_mask = (label_flat < num_target)
            valid_labels = label_flat[valid_mask]

            log_probs = torch.log(torch.clamp(seg_logits, min=1e-12))
            # log_probs = self.stable_log(seg_logits)
            # log_probs.register_hook(tensor_hook)
            flat_cprobs = log_probs.permute(0, 2, 3, 1).contiguous().view((log_probs.shape[0], -1, log_probs.shape[1]))
            valid_probs = flat_cprobs[valid_mask]
            # valid_probs.register_hook(tensor_hook)

            # if seg_weight is not None:
            #     seg_weight = torch.ones((seg_weight.shape), device=seg_weight.device)
            #     flat_weight = seg_weight.view(seg_weight.shape[0], -1)
            #     valid_weight = flat_weight[valid_mask]
            # else:
            #     flat_weight = weight_mask[i].view(weight_mask[i].shape[0], -1)
            #     valid_weight = flat_weight[valid_mask]

            pos_logp = torch.gather(valid_probs, dim=1, index=valid_labels.unsqueeze(1))
            # d_loss = -torch.mean(pos_logp * valid_weight.unsqueeze(1))
            d_loss = -torch.mean(pos_logp)
            loss['{}_depth'.format(i + 1)] = d_loss
            if 'loss_ce' not in loss.keys():
                loss['loss_ce'] = d_loss
            else:
                loss['loss_ce'] = loss['loss_ce'] + d_loss
        # loss['loss_ce'].register_hook(tensor_hook)

        return loss

    def debug_class(self, d_class, probs, batch_data_samples):

        ancesor = self.tree.hmat[d_class].view(1, -1, 1, 1).cuda()
        d_class_probs = ancesor * probs

        class_mask = torch.cat([data_sample.gt_sem_seg.data == d_class for data_sample in batch_data_samples],
                               dim=0).unsqueeze(1).float()
        mask_class_probs = class_mask * d_class_probs
        sum_class_probs = torch.sum(mask_class_probs, dim=(2, 3))
        mean_class_probs = sum_class_probs / torch.sum(class_mask, dim=(2, 3))

        debug_info = {}

        for i in range(ancesor.shape[1]):
            if ancesor.squeeze()[i] != 0:
                debug_info[self.tree.i2n[i]] = mean_class_probs[0][i]

        return debug_info

    def cal_label_weight(self, temperature):

        class_hie_weight = {}
        for depth in self.tree.hclass_num.keys():
            classes = torch.tensor(list(self.tree.hclass_num[depth].keys()))
            nums = torch.tensor(list(self.tree.hclass_num[depth].values()), dtype=torch.int64)
            class_hie_weight[depth] = {}
            freq = nums / torch.sum(nums)
            freq = 1 - freq
            freq = torch.softmax(freq / temperature, dim=-1)

            class_hie_weight[depth]['classes'] = classes
            class_hie_weight[depth]['weight'] = freq

        return class_hie_weight

    def reflect_labels_weight(self, labels_weight):

        h_cls2wgh = {}
        for depth in labels_weight.keys():
            classes = labels_weight[depth]['classes']
            weight = labels_weight[depth]['weight']
            h_cls2wgh[depth] = {int(c):w for c,w in zip(classes, weight)}


        re_h_cls2wgh = {}
        for depth in labels_weight.keys():
            re_h_cls2wgh[depth] = {}
            values, indices = torch.sort(torch.Tensor(list(set(self.tree.abs2con[depth].values()))))
            for v, i in zip(values, indices):
                re_h_cls2wgh[depth][int(i)] = h_cls2wgh[depth][int(v)]
        return re_h_cls2wgh

    def reflect_siblings_labels_weight(self):

        h_cls2wgh = self.tree.siblings_weight
        re_h_cls2wgh = {}
        for depth in h_cls2wgh.keys():
            re_h_cls2wgh[depth] = {}
            values, indices = torch.sort(torch.Tensor(list(set(self.tree.abs2con[depth].values()))))
            for v, i in zip(values, indices):
                re_h_cls2wgh[depth][int(i)] = h_cls2wgh[depth][int(v)]
        return re_h_cls2wgh

    def generate_weight_mask(self,labels_weight, new_batch_data):

        dep_weight = {}
        for depth in range(len(new_batch_data)):
            labels = new_batch_data[depth][1]
            weights = torch.ones((labels.shape), device=labels.device)
            for i, w in labels_weight[depth+1].items():
                weights[labels == i] = w*2*(depth+1)
            dep_weight[depth] = weights
        return dep_weight

    def stable_log(self,probs):
        probs = torch.clamp(probs, min=1e-4, max=1.0)
        log_probs = torch.where(probs < 1e-7,
                                -1e7 * (1-probs),
                                torch.log(probs))
        return log_probs







