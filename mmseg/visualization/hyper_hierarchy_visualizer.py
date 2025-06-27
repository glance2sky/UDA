from typing import Dict, List, Optional, Union

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer
import mmengine.fileio as fileio
import mmcv

from mmseg.registry import VISUALIZERS
from mmseg.structures import SegDataSample
from mmseg.utils import get_classes, get_palette
from mmseg.visualization.local_visualizer import SegLocalVisualizer

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

