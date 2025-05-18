# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
from typing import List, Optional, Sequence, Union

import numpy as np

from mmengine.dataset import ConcatDataset, force_full_init, BaseDataset
from mmseg.registry import DATASETS, TRANSFORMS



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


        self.ignore_keys = []

        meta_keys: set = set()
        for dataset in self.datasets:
            meta_keys |= dataset.metainfo.keys()
        # Only use metainfo of first dataset.
        self._metainfo = self.datasets[0].metainfo
        for i, dataset in enumerate(self.datasets, 1):
            for key in meta_keys:
                if key in self.ignore_keys:
                    continue
                if key not in dataset.metainfo:
                    raise ValueError(
                        f'{key} does not in the meta information of '
                        f'the {i}-th dataset')
                first_type = type(self._metainfo[key])
                cur_type = type(dataset.metainfo[key])
                if first_type is not cur_type:  # type: ignore
                    raise TypeError(
                        f'The type {cur_type} of {key} in the {i}-th dataset '
                        'should be the same with the first dataset '
                        f'{first_type}')
                if (isinstance(self._metainfo[key], np.ndarray)
                        and not np.array_equal(self._metainfo[key],
                                               dataset.metainfo[key])
                        or (not isinstance(self._metainfo[key], np.ndarray)
                            and self._metainfo[key] != dataset.metainfo[key])):
                    raise ValueError(
                        f'The meta information of the {i}-th dataset does not '
                        'match meta information of the first dataset')

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the first dataset in ``self.datasets``.

        Returns:
            dict: Meta information of first dataset.
        """
        # Prevent `self._metainfo` from being modified by outside.
        return copy.deepcopy(self._metainfo)

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
        s1 = self.source_dataset[idx // len(self.target_dataset)]
        s2 = self.target_dataset[idx % len(self.target_dataset)]
        return {'source':s1, 'target':s2}