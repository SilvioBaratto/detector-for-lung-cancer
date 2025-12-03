"""
Segmentation model training application.

Trains U-Net model for nodule segmentation in CT slices.

Design Principles:
- Dependency Inversion: Dependencies injected via constructor
- Single Responsibility: Training logic only, logging/checkpointing delegated
"""

import argparse
import datetime
import sys
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import Config, get_config
from dataset import (
    SegmentationDataset,
    TrainingSegmentationDataset,
    get_ct_cache,
)
from model.model_segmentation import (
    get_segmentation_model,
    list_segmentation_models,
    SegmentationAugmentation,
)
from training.losses import DiceLoss, LossStrategy
from training.checkpointing import ModelCheckpointer, create_checkpointer
from training.logging import TrainingLogger, create_training_logger
from util.util import enumerateWithEstimate
from util.device import get_best_device, is_mps_device
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Metrics indices
METRICS_LOSS_NDX = 1
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9
METRICS_SIZE = 10


class SegmentationTrainingApp:
    """
    Training application for segmentation models.

    Supports dependency injection for testing and flexibility:
    - model: Segmentation model (default: UltraLightUNet via registry)
    - loss_fn: Loss function (default: DiceLoss)
    - logger: Training logger (default: TensorBoard)
    - checkpointer: Model checkpointer (default: ModelCheckpointer)
    """

    def __init__(
        self,
        sys_argv: Optional[list[str]] = None,
        # Dependency injection
        config: Optional[Config] = None,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[LossStrategy] = None,
        logger: Optional[TrainingLogger] = None,
        checkpointer: Optional[ModelCheckpointer] = None,
    ):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=16,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly scaling.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by adding noise.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--tb-prefix',
            default='segmentation',
            help="Data prefix to use for Tensorboard run.",
        )
        parser.add_argument('--model',
            default='ultralightunet',
            choices=list_segmentation_models(),
            help="Model architecture to use (default: ultralightunet for lightweight)",
        )
        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='none',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        # Use injected config or create default
        self.config = config or get_config()

        self.totalTrainingSamples_count = 0

        # Build augmentation settings
        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        # Device setup (MPS > CUDA > CPU)
        self.device = get_best_device()
        self.use_cuda = self.device.type in ('cuda', 'mps')

        # Dependency injection with defaults
        if model is not None:
            self.segmentation_model = model
            self.augmentation_model = SegmentationAugmentation(**self.augmentation_dict)
        else:
            self.segmentation_model, self.augmentation_model = self.initModel()

        self.optimizer = self.initOptimizer()
        self.dice_loss = loss_fn or DiceLoss()

        # Initialize logger (injected or created)
        self.logger = logger or create_training_logger(
            base_dir="../../runs",
            tb_prefix=self.cli_args.tb_prefix,
            time_str=self.time_str,
            comment=self.cli_args.comment,
            mode="seg",
        )

        # Initialize checkpointer (injected or created)
        self.checkpointer = checkpointer or create_checkpointer(
            base_dir="../../models",
            tb_prefix=self.cli_args.tb_prefix,
            model_type="seg",
            time_str=self.time_str,
            comment=self.cli_args.comment,
        )

    def initModel(self):
        """Initialize segmentation and augmentation models."""
        model_name = self.cli_args.model
        log.info(f"Initializing model: {model_name}")

        segmentation_model = get_segmentation_model(
            model_name,
            in_channels=7,
            n_classes=1,
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            if self.device.type == 'cuda':
                log.info(f"Using CUDA; {torch.cuda.device_count()} devices.")
                if torch.cuda.device_count() > 1:
                    segmentation_model = nn.DataParallel(segmentation_model)
                    augmentation_model = nn.DataParallel(augmentation_model)
            elif is_mps_device(self.device):
                log.info("Using MPS (Apple Silicon)")
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def initOptimizer(self):
        """Initialize the optimizer."""
        return Adam(self.segmentation_model.parameters())

    def initTrainDl(self):
        """Initialize training data loader."""
        train_ds = TrainingSegmentationDataset(
            config=self.config,
            val_stride=10,
            is_validation=False,
            context_slices=3,
        )

        batch_size = self.cli_args.batch_size
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            batch_size *= torch.cuda.device_count()

        return DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

    def initValDl(self):
        """Initialize validation data loader."""
        val_ds = SegmentationDataset(
            config=self.config,
            val_stride=10,
            is_validation=True,
            context_slices=3,
        )

        batch_size = self.cli_args.batch_size
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            batch_size *= torch.cuda.device_count()

        return DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

    def initTensorboardWriters(self):
        """Initialize TensorBoard writers (now delegated to logger)."""
        # Logger is initialized in __init__, this is kept for compatibility
        pass

    def main(self):
        """Main training loop."""
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        validation_cadence = 5

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info(
                f"Epoch {epoch_ndx} of {self.cli_args.epochs}, "
                f"{len(train_dl)}/{len(val_dl)} batches of size "
                f"{self.cli_args.batch_size}*{torch.cuda.device_count() if self.use_cuda else 1}"
            )

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                self.saveModel('seg', epoch_ndx, score == best_score)
                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        # Close the logger
        self.logger.close()

    def doTraining(self, epoch_ndx, train_dl):
        """Execute one training epoch."""
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffle()

        batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} Training",
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        """Execute validation."""
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            batch_iter = enumerateWithEstimate(
                val_dl,
                f"E{epoch_ndx} Validation",
                start_ndx=val_dl.num_workers,
            )

            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         classification_threshold=0.5):
        """Compute loss for a batch."""
        input_t, label_t, _series_list, _slice_ndx_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        # Augment during training if enabled
        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)

        # Dice loss with false negative weighting
        diceLoss_g = self.dice_loss(prediction_g, label_g, reduction="none")
        fnLoss_g = self.dice_loss(prediction_g * label_g, label_g, reduction="none")

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1] > classification_threshold).to(torch.float32)

            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])
            fp = (predictionBool_g * (~label_g)).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    def logImages(self, epoch_ndx, mode_str, dl):
        """Log sample images to TensorBoard."""
        self.segmentation_model.eval()
        ct_cache = get_ct_cache(self.config)

        # Take first 12 series (sorted for consistency)
        images = sorted(dl.dataset.series_list)[:12]

        for series_ndx, series_uid in enumerate(images):
            ct = ct_cache.get_segmentation(series_uid)

            for slice_ndx in range(6):
                # Select six equidistant slices
                ct_ndx = slice_ndx * (ct.num_slices - 1) // 5
                ct_t, label_t, _, _ = dl.dataset._get_slice(series_uid, ct_ndx)

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                # Normalize CT values for display
                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                ctSlice_a = ct_t[dl.dataset.context_slices].numpy()

                # Build visualization image
                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                image_a[:, :, 0] += prediction_a & (1 - label_a)  # False positive: red
                image_a[:, :, 0] += (1 - prediction_a) & label_a  # False negative: orange
                image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5
                image_a[:, :, 1] += prediction_a & label_a  # True positive: green
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                tag = f'{mode_str}/{series_ndx}_prediction_{slice_ndx}'
                step = self.totalTrainingSamples_count

                if mode_str == 'trn':
                    self.logger.log_training_image(tag, image_a, step, dataformats='HWC')
                else:
                    self.logger.log_validation_image(tag, image_a, step, dataformats='HWC')

                if epoch_ndx == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                    image_a[:, :, 1] += label_a  # Green for labels
                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    label_tag = f'{mode_str}/{series_ndx}_label_{slice_ndx}'
                    if mode_str == 'trn':
                        self.logger.log_training_image(label_tag, image_a, step, dataformats='HWC')
                    else:
                        self.logger.log_validation_image(label_tag, image_a, step, dataformats='HWC')

        self.logger.flush()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        """Log metrics to TensorBoard and console."""
        log.info(f"E{epoch_ndx} {type(self).__name__}")

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['seg_loss/all'] = float(metrics_a[METRICS_LOSS_NDX].mean())

        metrics_dict['seg_percent_all/tp'] = sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['seg_percent_all/fn'] = sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['seg_percent_all/fp'] = sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100

        precision = metrics_dict['seg_pr/precision'] = (
            sum_a[METRICS_TP_NDX] / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        )
        recall = metrics_dict['seg_pr/recall'] = (
            sum_a[METRICS_TP_NDX] / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)
        )
        metrics_dict['seg_pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(
            f"E{epoch_ndx} {mode_str:8} "
            f"{metrics_dict['seg_loss/all']:.4f} loss, "
            f"{metrics_dict['seg_pr/precision']:.4f} precision, "
            f"{metrics_dict['seg_pr/recall']:.4f} recall, "
            f"{metrics_dict['seg_pr/f1_score']:.4f} f1 score"
        )
        log.info(
            f"E{epoch_ndx} {mode_str + '_all':8} "
            f"{metrics_dict['seg_loss/all']:.4f} loss, "
            f"{metrics_dict['seg_percent_all/tp']:-5.1f}% tp, "
            f"{metrics_dict['seg_percent_all/fn']:-5.1f}% fn, "
            f"{metrics_dict['seg_percent_all/fp']:-9.1f}% fp"
        )

        # Use injected logger
        step = self.totalTrainingSamples_count
        if mode_str == 'trn':
            self.logger.log_training_metrics(metrics_dict, step)
        else:
            self.logger.log_validation_metrics(metrics_dict, step)

        self.logger.flush()
        return metrics_dict['seg_pr/recall']

    def saveModel(self, _type_str, epoch_ndx, isBest=False):
        """Save model checkpoint using injected checkpointer."""
        metrics = {
            'epoch': epoch_ndx,
            'total_samples': self.totalTrainingSamples_count,
        }

        self.checkpointer.save(
            model=self.segmentation_model,
            optimizer=self.optimizer,
            epoch=epoch_ndx,
            total_samples=self.totalTrainingSamples_count,
            metrics=metrics,
            is_best=isBest,
        )


if __name__ == '__main__':
    SegmentationTrainingApp().main()
