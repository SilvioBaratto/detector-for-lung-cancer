"""
Visualization utilities for CT scans and nodules.
"""

import matplotlib
matplotlib.use('nbagg')

import numpy as np
import matplotlib.pyplot as plt
import torch

from config import get_config
from dataset import (
    ClassificationDataset,
    get_candidate_repository,
    get_ct_cache,
)
from model.model_segmentation import SegmentationMask, MaskTuple
from util.util import IrcTuple

clim = (-1000.0, 300)


def findPositiveSamples(start_ndx=0, limit=100):
    """Find positive (nodule) samples from the dataset."""
    config = get_config()
    repo = get_candidate_repository(config)
    candidates = repo.get_all()

    positiveSample_list = []
    for sample in candidates:
        if sample.is_nodule:
            print(len(positiveSample_list), sample)
            positiveSample_list.append(sample)

        if len(positiveSample_list) >= limit:
            break

    return positiveSample_list


def showCandidate(series_uid, batch_ndx=None, **kwargs):
    """Display a candidate nodule with multiple views."""
    config = get_config()
    ct_cache = get_ct_cache(config)

    ds = ClassificationDataset(
        config=config,
        series_uid=series_uid,
        sort_by='random',
        **kwargs
    )
    pos_list = [i for i, x in enumerate(ds.candidates) if x.is_nodule]

    if batch_ndx is None:
        if pos_list:
            batch_ndx = pos_list[0]
        else:
            print("Warning: no positive samples found; using first negative sample.")
            batch_ndx = 0

    ct = ct_cache.get_classification(series_uid)
    ct_t, pos_t, _, _, center_irc_t = ds[batch_ndx]
    center_irc = IrcTuple(*center_irc_t) if hasattr(center_irc_t, '__iter__') else center_irc_t
    ct_a = ct_t[0].numpy()

    fig = plt.figure(figsize=(30, 50))

    group_list = [
        [9, 11, 13],
        [15, 16, 17],
        [19, 21, 23],
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('index {}'.format(int(center_irc.index)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_array[int(center_irc.index)], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc.row)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_array[:, int(center_irc.row)], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc.col)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct.hu_array[:, :, int(center_irc.col)], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc.index)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[ct_a.shape[0] // 2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc.row)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:, ct_a.shape[1] // 2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc.col)), fontsize=30)
    for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
        label.set_fontsize(20)
    plt.imshow(ct_a[:, :, ct_a.shape[2] // 2], clim=clim, cmap='gray')
    plt.gca().invert_yaxis()

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index), fontsize=30)
            for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
                label.set_fontsize(20)
            plt.imshow(ct_a[index], clim=clim, cmap='gray')

    print(series_uid, batch_ndx, bool(pos_t[0]), pos_list)


def build2dLungMask(series_uid, center_ndx):
    """Build 2D lung mask for a specific slice."""
    config = get_config()
    ct_cache = get_ct_cache(config)

    mask_model = SegmentationMask().to('cuda')
    ct = ct_cache.get_segmentation(series_uid)

    ct_g = torch.from_numpy(
        ct.hu_array[center_ndx].astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to('cuda')
    pos_g = torch.from_numpy(
        ct.positive_mask[center_ndx].astype(np.float32)
    ).unsqueeze(0).unsqueeze(0).to('cuda')
    input_g = ct_g / 1000

    label_g, neg_g, pos_g, lung_mask, mask_dict = mask_model(input_g, pos_g)

    mask_tup = MaskTuple(**mask_dict)

    return mask_tup
