"""
Segmentation visualization script.

Generates visualization of segmentation masks.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import get_config
from dataset import get_candidate_repository, get_ct_cache
from model.model_segmentation import SegmentationMask, MaskTuple
from analysis.vis import build2dLungMask
from util.util import xyz2irc

# Get configuration and data access
config = get_config()
candidate_repo = get_candidate_repository(config)
ct_cache = get_ct_cache(config)

candidateInfo_list = candidate_repo.get_all()
series_list = sorted(set(c.series_uid for c in candidateInfo_list))


def transparent_cmap(cmap, N=255):
    """Copy colormap and set alpha values."""
    mycmap = copy.deepcopy(cmap)
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.75, N + 4)
    return mycmap


tgray = transparent_cmap(plt.get_cmap('gray'))
tpurp = transparent_cmap(plt.get_cmap('Purples'))
tblue = transparent_cmap(plt.get_cmap('Blues'))
tgreen = transparent_cmap(plt.get_cmap('Greens'))
torange = transparent_cmap(plt.get_cmap('Oranges'))
tred = transparent_cmap(plt.get_cmap('Reds'))

clim = (0, 1.3)
start_ndx = 3
mask_model = SegmentationMask().to('cuda')

ct_list = []
nit_ndx = start_ndx
for nit_ndx in range(start_ndx, start_ndx + 3):
    candidate = candidateInfo_list[nit_ndx]
    ct = ct_cache.get_segmentation(candidate.series_uid)
    center_irc = xyz2irc(
        candidate.center_xyz,
        ct.metadata.origin_xyz,
        ct.metadata.spacing_xyz,
        ct.metadata.direction,
    )

    ct_list.append((ct, center_irc))
start_ndx = nit_ndx + 1

fig = plt.figure(figsize=(60, 90))
subplot_ndx = 0
for ct_ndx, (ct, center_irc) in enumerate(ct_list):
    mask_tup = build2dLungMask(ct.series_uid, int(center_irc.index))

    for attr_ndx, attr_str in enumerate(mask_tup._fields):
        subplot_ndx = 1 + 3 * 2 * attr_ndx + 2 * ct_ndx
        subplot = fig.add_subplot(len(mask_tup), len(ct_list) * 2, subplot_ndx)
        subplot.set_title(attr_str)

        plt.imsave('test.png', ct.hu_array[int(center_irc.index)], cmap='RdGy')

fig = plt.figure(figsize=(40, 10))

subplot = fig.add_subplot(1, 4, 4)
subplot.set_title('mal mask', fontsize=30)
for label in (subplot.get_xticklabels() + subplot.get_yticklabels()):
    label.set_fontsize(20)

if ct_list:
    ct, center_irc = ct_list[-1]
    plt.imsave('output.png', ct.hu_array[int(center_irc.index)], cmap='gray', vmin=-1000, vmax=2000)
