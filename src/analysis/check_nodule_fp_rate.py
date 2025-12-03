"""
False positive rate analysis application.

Checks the false positive rate of the segmentation and classification pipeline.
"""

import argparse
import glob
import hashlib
import os
import sys

import numpy as np
import scipy.ndimage.measurements as measure

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

from config import get_config
from dataset import (
    CandidateInfo,
    SegmentationDataset,
    ClassificationDataset,
    get_candidate_repository,
    get_ct_cache,
)
from model.model_segmentation import get_segmentation_model
from model.model_classification import get_model
from util.util import enumerateWithEstimate, irc2xyz
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class FalsePosRateCheckApp:
    """Application for checking false positive rates."""

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--series-uid',
            help='Limit inference to this Series UID only.',
            default=None,
            type=str,
        )
        parser.add_argument('--include-train',
            help="Include data that was in the training set.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--segmentation-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default=None,
        )
        parser.add_argument('--classification-path',
            help="Path to the saved classification model",
            nargs='?',
            default=None,
        )
        parser.add_argument('--tb-prefix',
            default='p2ch13',
            help="Data prefix to use for Tensorboard run.",
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.config = get_config()

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if not self.cli_args.segmentation_path:
            self.cli_args.segmentation_path = self.initModelPath('seg')
        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.initModelPath('cls')

        self.seg_model, self.cls_model = self.initModels()

        # Initialize data access
        self.candidate_repo = get_candidate_repository(self.config)
        self.ct_cache = get_ct_cache(self.config)

    def initModelPath(self, type_str):
        """Find model path by pattern matching."""
        pretrained_path = os.path.join(
            'data', 'part2', 'models',
            f'{type_str}_*_*.*.state',
        )
        file_list = glob.glob(pretrained_path)
        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([pretrained_path, file_list])
            raise

    def initModels(self):
        """Initialize segmentation and classification models."""
        with open(self.cli_args.segmentation_path, 'rb') as f:
            log.debug(self.cli_args.segmentation_path)
            log.debug(hashlib.sha1(f.read()).hexdigest())

        seg_dict = torch.load(self.cli_args.segmentation_path)

        seg_model = get_segmentation_model(
            'ultralightunet',
            in_channels=7,
            n_classes=1,
        )
        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        with open(self.cli_args.classification_path, 'rb') as f:
            log.debug(self.cli_args.classification_path)
            log.debug(hashlib.sha1(f.read()).hexdigest())

        cls_dict = torch.load(self.cli_args.classification_path)

        cls_model = get_model('mifnet')
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)
            seg_model = seg_model.to(self.device)
            cls_model = cls_model.to(self.device)

        self.conv_list = nn.ModuleList([
            self._make_circle_conv(radius).to(self.device) for radius in range(1, 8)
        ])

        return seg_model, cls_model

    def initSegmentationDl(self, series_uid):
        """Initialize segmentation data loader."""
        seg_ds = SegmentationDataset(
            config=self.config,
            context_slices=3,
            series_uid=series_uid,
            full_ct=True,
        )
        return DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=1,
            pin_memory=self.use_cuda,
        )

    def initClassificationDl(self, candidate_list):
        """Initialize classification data loader."""
        cls_ds = _CandidateDataset(
            candidates=candidate_list,
            config=self.config,
            ct_cache=self.ct_cache,
        )
        return DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=1,
            pin_memory=self.use_cuda,
        )

    def main(self):
        """Main analysis loop."""
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        # Get validation set series
        val_ds = ClassificationDataset(
            config=self.config,
            val_stride=10,
            is_validation=True,
        )
        val_set = set(c.series_uid for c in val_ds.candidates)

        # Get all candidates
        all_candidates = self.candidate_repo.get_all()
        positive_set = set(c.series_uid for c in all_candidates if c.is_nodule)

        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(c.series_uid for c in all_candidates)

        train_list = sorted(series_set - val_set) if self.cli_args.include_train else []
        val_list = sorted(series_set & val_set)

        total_tp = total_tn = total_fp = total_fn = 0
        total_missed_pos = 0
        missed_pos_dist_list = []
        missed_pos_cit_list = []

        candidate_dict = self.candidate_repo.get_by_series()

        series_iter = enumerateWithEstimate(val_list + train_list, "Series")

        for _series_ndx, series_uid in series_iter:
            ct, _output_g, _mask_g, clean_g = self.segmentCt(series_uid)

            seg_candidate_list, _seg_centerIrc_list, _ = self.clusterSegmentationOutput(
                series_uid, ct, clean_g
            )
            if not seg_candidate_list:
                continue

            cls_dl = self.initClassificationDl(seg_candidate_list)
            results_list = []

            for batch_ndx, batch_tup in enumerate(cls_dl):
                input_t, label_t, index_t, series_list, center_t = batch_tup

                input_g = input_t.to(self.device)
                with torch.no_grad():
                    _logits_g, probability_g = self.cls_model(input_g)
                probability_t = probability_g.to('cpu')

                for i, _series_uid in enumerate(series_list):
                    assert series_uid == _series_uid
                    results_list.append((center_t[i], probability_t[i, 0].item()))

            # Match annotations with segmentation results
            tp = tn = fp = fn = 0
            missed_pos = 0

            candidate_list = candidate_dict.get(series_uid, [])
            candidate_list = [c for c in candidate_list if c.is_nodule]

            found_cit_list: list[CandidateInfo | None] = [None] * len(results_list)

            for candidate in candidate_list:
                min_dist = (999, None)

                for result_ndx, (result_center_irc_t, nodule_probability_t) in enumerate(results_list):
                    result_center_xyz = irc2xyz(
                        result_center_irc_t,
                        origin_xyz=ct.metadata.origin_xyz,
                        vxSize_xyz=ct.metadata.spacing_xyz,
                        direction_a=ct.metadata.direction,
                    )
                    delta_xyz_t = torch.tensor(result_center_xyz) - torch.tensor(candidate.center_xyz)
                    distance_t = (delta_xyz_t ** 2).sum().sqrt()

                    min_dist = min(min_dist, (distance_t, result_ndx))

                distance_cutoff = max(10, candidate.diameter_mm / 2)
                if min_dist[0] < distance_cutoff:
                    found_dist, result_ndx = min_dist
                    if result_ndx is None:
                        continue
                    nodule_probability_t = results_list[result_ndx][1]

                    assert candidate.is_nodule

                    if nodule_probability_t > 0.5:
                        tp += 1
                    else:
                        fn += 1

                    found_cit_list[result_ndx] = candidate
                else:
                    log.warning(f"!!! Missed positive {candidate}; {min_dist} min dist !!!")
                    missed_pos += 1
                    missed_pos_dist_list.append(float(min_dist[0]))
                    missed_pos_cit_list.append(candidate)

            log.info(f"{series_uid}: {missed_pos} missed pos, {fn} fn, {fp} fp, {tp} tp, {tn} tn")
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_missed_pos += missed_pos

        with open(self.cli_args.segmentation_path, 'rb') as f:
            log.info(self.cli_args.segmentation_path)
            log.info(hashlib.sha1(f.read()).hexdigest())
        with open(self.cli_args.classification_path, 'rb') as f:
            log.info(self.cli_args.classification_path)
            log.info(hashlib.sha1(f.read()).hexdigest())

        log.info(f"total: {total_missed_pos} missed pos, {total_fn} fn, {total_fp} fp, {total_tp} tp, {total_tn} tn")
        for cit, dist in zip(missed_pos_cit_list, missed_pos_dist_list):
            log.info(f"    Missed by {dist}: {cit}")

    def segmentCt(self, series_uid):
        """Run segmentation on a CT scan."""
        with torch.no_grad():
            ct = self.ct_cache.get_classification(series_uid)
            output_g = torch.zeros(ct.hu_array.shape, dtype=torch.float32, device=self.device)

            seg_dl = self.initSegmentationDl(series_uid)
            for batch_tup in seg_dl:
                input_t, label_t, series_list, slice_ndx_list = batch_tup

                input_g = input_t.to(self.device)
                prediction_g = self.seg_model(input_g)

                for i, slice_ndx in enumerate(slice_ndx_list):
                    output_g[slice_ndx] = prediction_g[i, 0]

            mask_g = output_g > 0.5
            clean_g = self.erode(mask_g.unsqueeze(0).unsqueeze(0), 1)[0][0]

        return ct, output_g, mask_g, clean_g

    def _make_circle_conv(self, radius):
        """Create circular convolution kernel."""
        diameter = 1 + radius * 2

        a = torch.linspace(-1, 1, steps=diameter) ** 2
        b = (a[None] + a[:, None]) ** 0.5

        circle_weights = (b <= 1.0).to(torch.float32)

        conv = nn.Conv3d(1, 1, kernel_size=(1, diameter, diameter), padding=(0, radius, radius), bias=False)
        conv.weight.data.fill_(1)
        conv.weight.data *= circle_weights / circle_weights.sum()

        return conv

    def erode(self, input_mask, radius, threshold=1):
        """Apply erosion to mask."""
        conv = self.conv_list[radius - 1]
        input_float = input_mask.to(torch.float32)
        result = conv(input_float)

        return result >= threshold

    def clusterSegmentationOutput(self, series_uid, ct, clean_g):
        """Cluster segmentation output into candidates."""
        clean_a = clean_g.cpu().numpy()
        candidateLabel_a, candidate_count = measure.label(clean_a)  # type: ignore[misc]
        centerIrc_list = measure.center_of_mass(  # type: ignore[misc]
            ct.hu_array.clip(-1000, 1000) + 1001,
            labels=candidateLabel_a,
            index=list(range(1, candidate_count + 1)),
        )

        candidate_list = []
        for i, center_irc in enumerate(centerIrc_list):
            assert np.isfinite(center_irc).all()
            center_xyz = irc2xyz(
                center_irc,
                origin_xyz=ct.metadata.origin_xyz,
                vxSize_xyz=ct.metadata.spacing_xyz,
                direction_a=ct.metadata.direction,
            )
            diameter_mm = 0.0

            candidate = CandidateInfo(
                is_nodule=False,
                has_annotation=False,
                is_malignant=False,
                diameter_mm=diameter_mm,
                series_uid=series_uid,
                center_xyz=tuple(center_xyz),
            )
            candidate_list.append(candidate)

        return candidate_list, centerIrc_list, candidateLabel_a


class _CandidateDataset(Dataset):
    """Minimal dataset for classification of detected candidates."""

    CHUNK_SIZE = (32, 48, 48)

    def __init__(self, candidates, config, ct_cache):
        super().__init__()
        self.candidates = candidates
        self.config = config
        self.ct_cache = ct_cache

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, idx):
        candidate = self.candidates[idx]
        ct = self.ct_cache.get_classification(candidate.series_uid)
        chunk, center_irc = ct.extract_chunk(candidate.center_xyz, self.CHUNK_SIZE)

        ct_tensor = torch.from_numpy(chunk).unsqueeze(0).float()

        label = torch.tensor([
            not candidate.is_nodule,
            candidate.is_nodule,
        ], dtype=torch.long)

        label_idx = 1 if candidate.is_nodule else 0

        return ct_tensor, label, label_idx, candidate.series_uid, center_irc


if __name__ == '__main__':
    FalsePosRateCheckApp().main()
