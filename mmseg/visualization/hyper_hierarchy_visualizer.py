import json
from typing import Dict, List, Optional, Union
import os

import cv2
import umap
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import master_only
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer
import mmengine.fileio as fileio
import mmcv

from mmseg.registry import VISUALIZERS
from mmseg.structures import SegDataSample
from mmseg.utils import get_classes, get_palette
from mmseg.visualization.local_visualizer import SegLocalVisualizer
# from mmseg.models.utils import resize

from lib.geoopt.manifolds.lorentz.math import lorentz_to_poincare, poincare_to_lorentz


class GradientMonitor:
    def __init__(self, model_name):
        self.model_name = model_name

    def start_monitor(self, model):
        self.model = model
        self.gradients = {name: [] for name, _ in model.named_parameters() if _.requires_grad}
        self._register_hooks()

    def _register_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda grad, n=name: self._hook_fn(grad, n)
                )

    def _hook_fn(self, grad, name):
        if grad is not None:
            self.gradients[name].append(grad.detach().cpu().numpy())

    def get_stats(self):
        stats = {}
        for name, grads in self.gradients.items():
            if len(grads) > 0:
                flat_grads = np.concatenate([g.flatten() for g in grads])
                stats[name] = {
                    'mean': float(np.mean(flat_grads)),
                    'std': float(np.std(flat_grads)),
                    'max': float(np.max(np.abs(flat_grads))),
                    'hist': np.histogram(flat_grads, bins=50)
                }
        return stats

    def save_histogram_json(self, cur_step, save_dir, ):

        summary_json = {}
        for name,grads in self.gradients.items():
            # layer_json = {}
            summary_json[name+'_source'] = {}
            summary_json[name+'_mix'] = {}
            for step in range(len(grads)):
                if step % 2 == 0:
                    mode = '_source'
                else:
                    mode = '_mix'
                step_layer_json = {}
                bins, values = np.histogram(grads[step].flatten())
                step_layer_json['bins'] = bins.tolist()
                step_layer_json['values'] = values.tolist()
                summary_json[name+mode]['step_{}'.format(cur_step - len(grads)//2 + step//2 + 1)] = step_layer_json

        json_dir = os.path.join(save_dir, 'grad_json')
        os.makedirs(json_dir, exist_ok=True)
        with open(os.path.join(json_dir, 'step_{}.json'.format(cur_step)), 'w') as f:
            json.dump(summary_json, f, indent=4)

        self.gradients = {name: [] for name in self.gradients.keys()}





@VISUALIZERS.register_module()
class HHLocalVisualizer(SegLocalVisualizer):
    """
    Local Visualizer
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 classes: Optional[List] = None,
                 palette: Optional[List] = None,
                 dataset_name: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, classes, palette, dataset_name, alpha, **kwargs)

        self.grads_moni = GradientMonitor('hyper')
        self.temp_embedding = None
        self.temp_label = None


    def add_datasample(
            self,
            name: str,
            image: Union[np.ndarray, str],
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = True) -> None:
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if isinstance(image, str):
            img_bytes = fileio.get(image)
            image = mmcv.imfrombytes(img_bytes, flag='color', backend='cv2')
        if isinstance(image,torch.Tensor):
            if image.dtype == torch.uint8:
                image = image.permute(1,2,0).cpu().numpy()
                image = np.ascontiguousarray(image)
            elif image.dtype == torch.float32:
                image = image.cpu() * torch.tensor([58.395, 57.12, 57.375]).unsqueeze(1).unsqueeze(1)
                image = image + torch.tensor([123.675, 116.28, 103.53]).unsqueeze(1).unsqueeze(1)
                image = np.clip(image.numpy(), 0, 255).astype(np.uint8)
                image = np.ascontiguousarray(image.transpose(2,0,1))





        if draw_gt and data_sample is not None:
            if 'gt_sem_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_sem_seg(image, data_sample.gt_sem_seg,
                                                 classes, palette, with_labels)

            if 'gt_depth_map' in data_sample:
                gt_img_data = gt_img_data if gt_img_data is not None else image
                gt_img_data = self._draw_depth_map(gt_img_data,
                                                   data_sample.gt_depth_map)

        if draw_pred and data_sample is not None:

            if 'pred_sem_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing semantic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_sem_seg(image,
                                                   data_sample.pred_sem_seg,
                                                   classes, palette,
                                                   with_labels)

            if 'pred_depth_map' in data_sample:
                pred_img_data = pred_img_data if pred_img_data is not None \
                    else image
                pred_img_data = self._draw_depth_map(
                    pred_img_data, data_sample.pred_depth_map)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            drawn_img = image

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)

    def draw_hierarchy_map(self,
                           name: str,
                           image: Union[torch.Tensor, np.ndarray, str,list],
                           batch_data: list,
                           hierarchy_classes: list,
                           draw_gt: bool = True,
                           draw_pred: bool = True,
                           show: bool = False,
                           wait_time: float = 0,
                           # TODO: Supported in mmengine's Viusalizer.
                           out_file: Optional[str] = None,
                           step: int = 0,
                           with_labels: Optional[bool] = True
                           ):
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)


        if isinstance(image, str):
            img_bytes = fileio.get(image)
            image = mmcv.imfrombytes(img_bytes, flag='color', backend='cv2')
        if isinstance(image, torch.Tensor):
            image = image.permute(1,2,0).cpu().numpy()
            image = np.ascontiguousarray(image)



        # remake the gt_sem_seg and pred_sem_seg
        draw_img_list = []
        for depth in range(len(batch_data)):

            tem_seg_logit = batch_data[depth][0][0,...]
            tem_gt = batch_data[depth][1][0,...]
            _, tem_pred_seg = torch.max(tem_seg_logit, dim=-3)
            tem_pred_seg = PixelData(data=tem_pred_seg)
            tem_gt = PixelData(data=tem_gt)
            tem_classes = hierarchy_classes[depth]
            tem_palette = palette[:len(tem_classes)]

            gt_img_data = None
            pred_img_data = None

            if draw_gt and batch_data is not None:
                gt_img_data = self._draw_sem_seg(image, tem_gt,
                                                 tem_classes, tem_palette, with_labels)

            if draw_pred and batch_data is not None:
                pred_img_data = self._draw_sem_seg(image,
                                                   tem_pred_seg,
                                                   tem_classes, tem_palette,
                                                   with_labels)

            if gt_img_data is not None and pred_img_data is not None:
                drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=0)
            elif gt_img_data is not None:
                drawn_img = gt_img_data
            else:
                drawn_img = pred_img_data
            draw_img_list.append(drawn_img)

        drawn_img = np.concatenate(draw_img_list, axis=1)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(mmcv.rgb2bgr(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, step)

    def return_add_datasample(
            self,
            image: Union[np.ndarray, torch.Tensor, str],
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = True) -> np.ndarray:
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if isinstance(image, str):
            img_bytes = fileio.get(image)
            image = mmcv.imfrombytes(img_bytes, flag='color', backend='cv2')
        if isinstance(image, torch.Tensor):
            if image.dtype == torch.uint8:
                image = image.permute(1,2,0).cpu().numpy()
                image = np.ascontiguousarray(image)
            elif image.dtype == torch.float32:
                image = image.cpu() * torch.tensor([58.395, 57.12, 57.375]).unsqueeze(1).unsqueeze(1)
                image = image + torch.tensor([123.675, 116.28, 103.53]).unsqueeze(1).unsqueeze(1)
                image = np.clip(image.numpy(), 0, 255).astype(np.uint8)
                image = np.ascontiguousarray(image.transpose(1,2,0))

        if draw_gt and data_sample is not None:
            gt_img_data = self._draw_sem_seg(image, data_sample,
                                                 classes, palette, with_labels)

        if draw_pred and data_sample is not None:
            pred_img_data = self._draw_sem_seg(image,
                                               data_sample,
                                               classes, palette,
                                               with_labels)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            drawn_img = image

        return drawn_img

    def write_grad_json(self, cur_step):
        for name, backend in self._vis_backends.items():
            self.grads_moni.save_histogram_json(cur_step, backend._save_dir)

    def add_temp_embeding_for_poincare(self,embed=None, label=None):


        if embed is not None:
            self.temp_embedding = embed.cpu()
        else:
            self.temp_label = label.cpu()

        if self.temp_embedding is not None and self.temp_label is not None:
            h, w = self.temp_label.shape[-2], self.temp_label[-1]
            self.temp_embedding = F.interpolate(
                                    input=self.temp_embedding,
                                    size=self.temp_label.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False)
            self.temp_embedding = self.temp_embedding.flatten(2).permute(0,2,1)
            self.temp_label = self.temp_label.flatten(0)

            unique_cls = torch.unique(self.temp_label)
            embed_list = [torch.mean(self.temp_embedding[0][self.temp_label == cls], dim=0, keepdim=True) for cls in unique_cls if
                          cls != 255]
            self.temp_embedding = torch.cat(embed_list, dim=0)
            self.temp_label = unique_cls




    def visualize_hyperbolic(self, i2c=None, manifold=None, poincare=False):
        """ Plots hyperbolic data on Poincaré ball and tangent space

        Note: This function only supports curvature k=1.
        """

        data = self.temp_embedding
        labels = self.temp_label
        if i2c is not None:
            labels = [int(i) for i in labels if i != 255]
        fig = plt.figure(figsize=(14, 7))

        # 2D embeddings
        if (data.shape[-1] == 2 and poincare) or (data.shape[-1] == 3 and not poincare):
            if poincare:
                data_P = data.cpu()
            else:
                data_P = lorentz_to_poincare(data, k=manifold.k).cpu()
        # Dimensionality reduction to 2D
        else:
            if poincare:
                data = poincare_to_lorentz(data.to(manifold.c.device), manifold.c)
            reducer = umap.UMAP(output_metric='hyperboloid')
            data = reducer.fit_transform(data.cpu().numpy())
            data = manifold.add_time(torch.tensor(data).to(manifold.c.device))
            data_P = lorentz_to_poincare(data, k=manifold.c).cpu()

        ax = fig.add_subplot(1, 2, 1)
        plt.scatter(data_P[:, 0], data_P[:, 1], c=labels, s=20)
        # Draw Poincaré boundary
        boundary = plt.Circle((0, 0.75), 0.5, color='k', fill=False)
        ax.add_patch(boundary)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal', adjustable='box')

        plt.colorbar()
        plt.xlabel("$z_0$")
        plt.ylabel("$z_1$")
        ax.set_title("Poincaré Ball")

        # Plot hyperbolic embeddings in tangent space of the origin
        if poincare:
            z_all_T = (manifold.logmap0(data_P.to('cuda'))).detach().cpu()
        else:
            z_all_T = (manifold.logmap0(data)).detach().cpu()
            z_all_T = z_all_T[..., 1:]

        ax = fig.add_subplot(1, 2, 2)
        plt.scatter(z_all_T[:, 0], z_all_T[:, 1], c=labels, s=1)
        ax.set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.xlabel("$z_0$")
        plt.ylabel("$z_1$")
        ax.set_title("Tangent Space")

        return fig