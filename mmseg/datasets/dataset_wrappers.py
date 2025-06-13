# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
from typing import List, Optional, Sequence, Union
import os.path as osp

import json
import numpy as np
import torch

from mmengine import print_log
from mmengine.dataset import ConcatDataset, force_full_init, BaseDataset
from mmseg.registry import DATASETS, TRANSFORMS

from .cityscapes import CityscapesDataset

def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()

@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup.

    Args:
        dataset (ConcatDataset or dict): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self,
                 dataset: Union[ConcatDataset, dict],
                 pipeline: Sequence[dict],
                 skip_type_keys: Optional[List[str]] = None,
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)

        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, ConcatDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`ConcatDataset` instance, but got {type(dataset)}')

        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self._metainfo = self.dataset.metainfo
        self.num_samples = len(self.dataset)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indices'):
                indices = transform.get_indices(self.dataset)
                if not isinstance(indices, collections.abc.Sequence):
                    indices = [indices]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indices
                ]
                results['mix_results'] = mix_results

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys

@DATASETS.register_module()
class UDADataset:
    """A wrapper of UDA dataset.

    Args:
        source_dataset (Union[dict, BaseDataset]): The dataset to be mixed.
        target_dataset (Union[dict, BaseDataset]): The dataset to be mixed.

        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self,
                 # datasets: Sequence[Union[BaseDataset, dict]],
                 source_dataset: Union[dict, BaseDataset],
                 target_dataset: Union[dict, BaseDataset],
                 lazy_init: bool = False,
                 rare_class_sampling=None,
                 ignore_keys: Union[str, List[str], None] = None):

        self.datasets = []
        if isinstance(source_dataset, dict):
            self.datasets.append(DATASETS.build(source_dataset))
        elif isinstance(source_dataset, BaseDataset):
            self.datasets.append(source_dataset)
        else:
            raise TypeError('source datasets should be config or '
                            f'`BaseDataset` instance, but got {type(source_dataset)}')
        if isinstance(target_dataset, dict):
            self.datasets.append(DATASETS.build(target_dataset))
        elif isinstance(target_dataset, BaseDataset):
            self.datasets.append(target_dataset)
        else:
            raise TypeError('target datasets should be config or '
                            f'`BaseDataset` instance, but got {type(target_dataset)}')

        # for i, dataset in enumerate(datasets):
        #     if isinstance(dataset, dict):
        #         self.datasets.append(DATASETS.build(dataset))
        #     elif isinstance(dataset, BaseDataset):
        #         self.datasets.append(dataset)
        #     else:
        #         raise TypeError(
        #             'elements in datasets sequence should be config or '
        #             f'`BaseDataset` instance, but got {type(dataset)}')
        self.source_dataset = self.datasets[0]
        self.target_dataset = self.datasets[1]

        self.ignore_index = self.target_dataset.ignore_index
        self.CLASSES = self.target_dataset.METAINFO['classes']
        self.PALETTE = self.target_dataset.METAINFO['palette']
        assert self.target_dataset.ignore_index == self.source_dataset.ignore_index
        assert self.target_dataset.METAINFO['classes'] == self.source_dataset.METAINFO['classes']
        assert self.target_dataset.METAINFO['palette'] == self.source_dataset.METAINFO['palette']

        rcs_cfg = rare_class_sampling
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                'data/gta', self.rcs_class_temp)
            print_log(f'RCS Classes: {self.rcs_classes}')
            print_log(f'RCS ClassProb: {self.rcs_classprob}')

            with open(osp.join('data/gta', 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.source_dataset.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self.source_dataset, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

            self.samples_with_class_dist = {}
            for cls, num in samples_with_class_and_n.items():
                self.samples_with_class_dist[cls] = [n[1] for n in num if n[1] >= self.rcs_min_pixels]








        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        # c = 16
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = int(f1.split('_label')[0])-1
        s1 = self.source_dataset[i1]
        assert f1 == s1['data_samples'].seg_map_path.split('/')[-1]
        if self.rcs_min_crop_ratio > 0: # 实际上这段代码没有用
            for j in range(100):
                n_class = torch.sum(s1['data_samples'].gt_sem_seg.data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio: # 源域图像的该类别像素数量本来就大于min_pixels
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.

                s1 = self.source_dataset[i1]
        i2 = np.random.choice(range(len(self.target_dataset))) # 目标域不进行稀有类采样，随即采样
        s2 = self.target_dataset[i2]

        # 很奇怪，只有源域GTA的gt_semantic_seg被返回了，目标域cityscapes没有用到
        return {'source':s1, 'target':s2}

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the first dataset in ``self.datasets``.

        Returns:
            dict: Meta information of first dataset.
        """
        # Prevent `self._metainfo` from being modified by outside.
        return {}

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return
        for d in self.datasets:
            d.full_init()
        # Get the cumulative sizes of `self.datasets`. For example, the length
        # of `self.datasets` is [2, 3, 4], the cumulative sizes is [2, 5, 9]
        self._fully_initialized = True

    def __len__(self):
        return len(self.source_dataset) * len(self.target_dataset)

    def __getitem__(self,idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s1 = self.source_dataset[idx // len(self.target_dataset)]
            s2 = self.target_dataset[idx % len(self.target_dataset)]
        return {'source':s1, 'target':s2}