"""
End-to-end nodule analysis application.

Runs the complete detection pipeline:
1. Segmentation - U-Net identifies candidate regions
2. Classification - CNN classifies candidates as nodule vs non-nodule
3. Malignancy - (optional) Classifies nodules as benign vs malignant
"""

import argparse
import glob
import os
import sys

import numpy as np
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology

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
logging.getLogger("segmentation").setLevel(logging.WARNING)
logging.getLogger("classification").setLevel(logging.WARNING)


def print_confusion(label, confusions, do_mal):
    """Print confusion matrix."""
    row_labels = ['Non-Nodules', 'Benign', 'Malignant']

    if do_mal:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Benign', 'Pred. Malignant']
    else:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Nodule']
        confusions[:, -2] += confusions[:, -1]
        confusions = confusions[:, :-1]

    cell_width = 16
    f = '{:>' + str(cell_width) + '}'
    print(label)
    print(' | '.join([f.format(s) for s in col_labels]))
    for i, (l, r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        if i == 0:
            r[1] = ''
        print(' | '.join([f.format(item) for item in r]))


def match_and_score(detections, truth, threshold=0.5, threshold_mal=0.5):
    """
    Match detections to ground truth and compute confusion matrix.

    Returns 3x4 confusion matrix:
    - Rows: Truth: Non-Nodules, Benign, Malignant
    - Cols: Not Detected, Detected by Seg, Detected as Benign, Detected as Malignant
    """
    true_nodules = [c for c in truth if c.is_nodule]
    truth_diams = np.array([c.diameter_mm for c in true_nodules])
    truth_xyz = np.array([c.center_xyz for c in true_nodules])

    detected_xyz = np.array([n[2] for n in detections])
    # Detection classes:
    # 1 -> detected by seg but filtered by cls
    # 2 -> detected as benign nodule (or nodule if no malignancy model)
    # 3 -> detected as malignant nodule
    detected_classes = np.array([
        1 if d[0] < threshold else (2 if d[1] < threshold_mal else 3)
        for d in detections
    ])

    confusion = np.zeros((3, 4), dtype=np.int_)

    if len(detected_xyz) == 0:
        for tn in true_nodules:
            confusion[2 if tn.is_malignant else 1, 0] += 1
    elif len(truth_xyz) == 0:
        for dc in detected_classes:
            confusion[0, dc] += 1
    else:
        normalized_dists = (
            np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1)
            / truth_diams[:, None]
        )
        matches = normalized_dists < 0.7
        unmatched_detections = np.ones(len(detections), dtype=np.bool_)
        matched_true_nodules = np.zeros(len(true_nodules), dtype=np.int_)

        for i_tn, i_detection in zip(*matches.nonzero()):
            matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection])
            unmatched_detections[i_detection] = False

        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud:
                confusion[0, dc] += 1
        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion[2 if tn.is_malignant else 1, dc] += 1

    return confusion


class NoduleAnalysisApp:
    """End-to-end nodule analysis application."""

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for inference',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=48,
            type=int,
        )
        parser.add_argument('--run-validation',
            help='Run over validation rather than a single CT.',
            action='store_true',
            default=False,
        )
        parser.add_argument('--include-train',
            help="Include data that was in the training set.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--segmentation-path',
            help="Path to the saved segmentation model",
            nargs='?',
            default='../../models/segmentation/seg_2022-07-21_17.29.23_final_seg.best.state',
        )
        parser.add_argument('--cls-model',
            help="Model class name to use for the classifier.",
            action='store',
            default='mifnet',
        )
        parser.add_argument('--classification-path',
            help="Path to the saved classification model",
            nargs='?',
            default='../../models/classification/cls_2022-07-21_19.09.18_nodule-nonnodule.best.state',
        )
        parser.add_argument('--malignancy-model',
            help="Model class name to use for the malignancy classifier.",
            action='store',
            default='mifnet',
        )
        parser.add_argument('--malignancy-path',
            help="Path to the saved malignancy classification model",
            nargs='?',
            default='../../models/classification/cls_2022-07-22_10.25.28_malben-finetune-twolayer.best.state',
        )
        parser.add_argument('--tb-prefix',
            default='nodule_analysis',
            help="Data prefix to use for Tensorboard run.",
        )
        parser.add_argument('series_uid',
            nargs='?',
            default=None,
            help="Series UID to use.",
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.config = get_config()

        if not (bool(self.cli_args.series_uid) ^ self.cli_args.run_validation):
            raise ValueError("One and only one of series_uid and --run-validation should be given")

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Initialize model paths if not provided
        if not self.cli_args.segmentation_path:
            self.cli_args.segmentation_path = self.initModelPath('seg')
        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.initModelPath('cls')

        self.seg_model, self.cls_model, self.malignancy_model = self.initModels()

        # Initialize data access
        self.candidate_repo = get_candidate_repository(self.config)
        self.ct_cache = get_ct_cache(self.config)

    def initModelPath(self, type_str):
        """Find model path by pattern matching."""
        local_path = os.path.join(
            '..', '..', 'models',
            f'{type_str}_*_*.best.state',
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                '..', '..', 'models',
                f'{type_str}_*_*.*.state',
            )
            file_list = glob.glob(pretrained_path)

        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug([local_path, file_list])
            raise

    def initModels(self):
        """Initialize all models."""
        log.debug(self.cli_args.segmentation_path)
        seg_dict = torch.load(self.cli_args.segmentation_path)

        seg_model = get_segmentation_model(
            'ultralightunet',
            in_channels=7,
            n_classes=1,
        )
        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        log.debug(self.cli_args.classification_path)
        cls_dict = torch.load(self.cli_args.classification_path)

        cls_model = get_model(self.cli_args.cls_model)
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)
            seg_model.to(self.device)
            cls_model.to(self.device)

        malignancy_model = None
        if self.cli_args.malignancy_path:
            malignancy_model = get_model(self.cli_args.malignancy_model)
            malignancy_dict = torch.load(self.cli_args.malignancy_path)
            malignancy_model.load_state_dict(malignancy_dict['model_state'])
            malignancy_model.eval()
            if self.use_cuda:
                malignancy_model.to(self.device)

        return seg_model, cls_model, malignancy_model

    def initSegmentationDl(self, series_uid):
        """Initialize segmentation data loader for a series."""
        seg_ds = SegmentationDataset(
            config=self.config,
            context_slices=3,
            series_uid=series_uid,
            full_ct=True,
        )
        return DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

    def initClassificationDl(self, candidate_list):
        """Initialize classification data loader for candidates."""
        # Create a temporary dataset with just these candidates
        cls_ds = _CandidateDataset(
            candidates=candidate_list,
            config=self.config,
            ct_cache=self.ct_cache,
        )
        return DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
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

        # Get all series with nodules
        all_candidates = self.candidate_repo.get_all()
        positive_set = set(c.series_uid for c in all_candidates if c.is_nodule)

        # Determine which series to process
        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(c.series_uid for c in all_candidates)

        if self.cli_args.include_train:
            train_list = sorted(series_set - val_set)
        else:
            train_list = []
        val_list = sorted(series_set & val_set)

        # Build candidate dict by series
        candidate_dict = self.candidate_repo.get_by_series()

        # Process all series
        series_iter = enumerateWithEstimate(val_list + train_list, "Series")
        all_confusion = np.zeros((3, 4), dtype=np.int_)

        for _, series_uid in series_iter:
            ct = self.ct_cache.get_classification(series_uid)
            mask_a = self.segmentCt(ct, series_uid)

            candidate_list = self.groupSegmentationOutput(series_uid, ct, mask_a)
            classifications_list = self.classifyCandidates(ct, candidate_list)

            if not self.cli_args.run_validation:
                print(f"Found nodule candidates in {series_uid}:")
                for prob, prob_mal, center_xyz, center_irc in classifications_list:
                    if prob > 0.5:
                        s = f"nodule prob {prob:.3f}, "
                        if self.malignancy_model:
                            s += f"malignancy prob {prob_mal:.3f}, "
                        s += f"center xyz {center_xyz}"
                        print(s)

            if series_uid in candidate_dict:
                one_confusion = match_and_score(
                    classifications_list, candidate_dict[series_uid]
                )
                all_confusion += one_confusion
                print_confusion(
                    series_uid, one_confusion, self.malignancy_model is not None
                )

        print_confusion("Total", all_confusion, self.malignancy_model is not None)

    def classifyCandidates(self, ct, candidate_list):
        """Classify candidates as nodule/non-nodule and malignant/benign."""
        cls_dl = self.initClassificationDl(candidate_list)
        classifications_list = []

        for batch_ndx, batch_tup in enumerate(cls_dl):
            input_t, _, _, series_list, center_list = batch_tup

            input_g = input_t.to(self.device)
            with torch.no_grad():
                _, probability_nodule_g = self.cls_model(input_g)
                if self.malignancy_model is not None:
                    _, probability_mal_g = self.malignancy_model(input_g)
                else:
                    probability_mal_g = torch.zeros_like(probability_nodule_g)

            for center_irc, prob_nodule, prob_mal in zip(
                center_list,
                probability_nodule_g[:, 1].tolist(),
                probability_mal_g[:, 1].tolist()
            ):
                center_xyz = irc2xyz(
                    center_irc,
                    origin_xyz=ct.metadata.origin_xyz,
                    vxSize_xyz=ct.metadata.spacing_xyz,
                    direction_a=ct.metadata.direction,
                )
                cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)
                classifications_list.append(cls_tup)

        return classifications_list

    def segmentCt(self, ct, series_uid):
        """Run segmentation on a CT scan."""
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_array, dtype=np.float32)
            seg_dl = self.initSegmentationDl(series_uid)

            for input_t, _, _, slice_ndx_list in seg_dl:
                input_g = input_t.to(self.device)
                prediction_g = self.seg_model(input_g)

                for i, slice_ndx in enumerate(slice_ndx_list):
                    output_a[slice_ndx] = prediction_g[i].cpu().numpy()

            mask_a = output_a > 0.5
            mask_a = morphology.binary_erosion(mask_a, iterations=1)  # type: ignore[misc]

        return mask_a

    def groupSegmentationOutput(self, series_uid, ct, clean_a):
        """Group connected components in segmentation output."""
        candidateLabel_a, candidate_count = measurements.label(clean_a)  # type: ignore[misc]

        centerIrc_list = measurements.center_of_mass(  # type: ignore[misc]
            ct.hu_array.clip(-1000, 1000) + 1001,
            labels=candidateLabel_a,
            index=np.arange(1, candidate_count + 1),
        )

        candidate_list = []
        for i, center_irc in enumerate(centerIrc_list):
            center_xyz = irc2xyz(
                center_irc,
                origin_xyz=ct.metadata.origin_xyz,
                vxSize_xyz=ct.metadata.spacing_xyz,
                direction_a=ct.metadata.direction,
            )
            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])

            candidate = CandidateInfo(
                is_nodule=False,
                has_annotation=False,
                is_malignant=False,
                diameter_mm=0.0,
                series_uid=series_uid,
                center_xyz=tuple(center_xyz),
            )
            candidate_list.append(candidate)

        return candidate_list


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
    NoduleAnalysisApp().main()
