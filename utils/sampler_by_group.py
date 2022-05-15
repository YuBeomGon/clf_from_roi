# https://github.com/pytorch/vision/blob/main/references/detection/group_by_aspect_ratio.py
# https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py
'''
LBP papsmear data size is very varing, need to sorting, and grouped similar size, using batch_sampler
this is just option for sampler,
'''

import bisect
import copy
import math
from collections import defaultdict
from itertools import repeat, chain

import numpy as np
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torch.utils.data.sampler import BatchSampler, Sampler
from torch.utils.model_zoo import tqdm

def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Args:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, sampler, group_ids, batch_size, drop_last=False):
    # def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(f"sampler should be an instance of torch.utils.data.Sampler, but got sampler={sampler}")
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.groups = list(set(self.group_ids))
            
        # buffer the indices of each group until batch size is reached
        self.buffer_per_group = {k: [] for k in self.groups}
        self.drop_last = drop_last

    def __iter__(self):
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            group_buffer = self.buffer_per_group[group_id]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]  # yield a copy of the list
                del group_buffer[:]
                
    def __len__(self):
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")                

#    def __len__(self):
#        return len(self.sampler) // self.batch_size


def _compute_aspect_ratios_slow(dataset, indices=None):
    print(
        "Your dataset doesn't support the fast path for "
        "computing the aspect ratios, so will iterate over "
        "the full dataset and load every image instead. "
        "This might take some time..."
    )
    if indices is None:
        indices = range(len(dataset))

    class SubsetSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    sampler = SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=14,  # you might want to increase it for faster processing
        collate_fn=lambda x: x[0],
    )
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, (img, _) in enumerate(data_loader):
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(dataset, indices)

    if isinstance(dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)

    # slow path
    return _compute_aspect_ratios_slow(dataset, indices)


def _quantize(x, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def create_aspect_ratio_groups(dataset, k=0):
    aspect_ratios = compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    print(f"Using {fbins} as bins for aspect ratio quantization")
    print(f"Count of instances per bin: {counts}")
    return groups


def create_area_groups(dataset, k=0):
    # check clf_from_roi/prepare_data/notebooks/group_by_area.ipynb
    areas = dataset.df.area
    if k == 0 :
        bins = [150]
    elif k ==1 :
        bins = [120, 220]
    elif k ==2 :
        bins = [100, 180, 280]
    elif k ==3 :
        bins = [100, 140, 200, 300]
    groups = _quantize(areas, bins)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [400]
    print(f"Using {fbins} as bins for aspect ratio quantization")
    print(f"Count of instances per bin: {counts}")
    return groups    