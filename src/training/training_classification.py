"""
Classification model training application.

Trains nodule classification (nodule vs non-nodule) or
malignancy classification (benign vs malignant) models.

Design Principles:
- Dependency Inversion: Dependencies injected via constructor
- Single Responsibility: Training logic only, logging/checkpointing delegated
"""

import argparse
import datetime
from typing import Optional

import numpy as np
from matplotlib import pyplot

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from config import Config, get_config
from dataset import (
    ClassificationDataset,
    MalignancyDataset,
    Augmentation3D,
)
from model.model_classification import get_model, list_models
from training.losses import CrossEntropyLoss, LossStrategy
from training.metrics import ClassificationMetricsCalculator
from training.checkpointing import ModelCheckpointer, create_checkpointer
from training.logging import TrainingLogger, create_training_logger
from util.util import enumerateWithEstimate
from util.device import get_best_device, get_num_workers, is_mps_device
from util.logconf import logging


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Metrics indices for backward compatibility
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_PRED_P_NDX = 2
METRICS_LOSS_NDX = 3
METRICS_SIZE = 4


class ClassificationTrainingApp:
    """
    Training application for classification models.

    Supports dependency injection for testing and flexibility:
    - model: Classification model (default: MIFNet via registry)
    - loss_fn: Loss function (default: CrossEntropyLoss)
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
            import sys
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=24,
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
        parser.add_argument('--malignant',
            help="Train the model to classify nodules as benign or malignant.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--finetune',
            help="Start finetuning from this model.",
            default='',
        )
        parser.add_argument('--finetune-depth',
            help="Number of blocks (counted from the head) to include in finetuning",
            type=int,
            default=1,
        )
        parser.add_argument('--tb-prefix',
            default='classification',
            help="Data prefix to use for Tensorboard run.",
        )
        parser.add_argument('--model',
            default='mifnet',
            choices=list_models(),
            help="Model architecture to use (default: mifnet for lightweight)",
        )
        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dlwpt',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        # Use injected config or create default
        self.config = config or get_config()

        self.totalTrainingSamples_count = 0

        # Augmentation settings
        self.augmentation = Augmentation3D(
            flip=True,
            offset=0.1,
            scale=0.2,
            rotate=True,
            noise=25.0,
        )

        # Device setup (MPS > CUDA > CPU)
        self.device = get_best_device()
        self.use_cuda = self.device.type in ('cuda', 'mps')

        # Dependency injection with defaults
        self.model = model or self.initModel()
        self.optimizer = self.initOptimizer()
        self.loss_fn = loss_fn or CrossEntropyLoss()
        self.metrics_calculator = ClassificationMetricsCalculator()

        # Initialize logger (injected or created)
        self.logger = logger or create_training_logger(
            base_dir="runs",
            tb_prefix=self.cli_args.tb_prefix,
            time_str=self.time_str,
            comment=self.cli_args.comment,
            mode="cls",
        )

        # Initialize checkpointer (injected or created)
        model_type = 'mal' if self.cli_args.malignant else 'cls'
        self.checkpointer = checkpointer or create_checkpointer(
            base_dir="../../models",
            tb_prefix=self.cli_args.tb_prefix,
            model_type=model_type,
            time_str=self.time_str,
            comment=self.cli_args.comment,
        )

    def initModel(self):
        """Initialize the classification model."""
        model_name = self.cli_args.model
        log.info(f"Initializing model: {model_name}")
        cls_model = get_model(model_name)

        if self.cli_args.finetune:
            d = torch.load(self.cli_args.finetune, map_location='cpu')
            model_blocks = [
                n for n, subm in cls_model.named_children()
                if len(list(subm.parameters())) > 0
            ]
            finetune_blocks = model_blocks[-self.cli_args.finetune_depth:]
            log.info(f"Finetuning from {self.cli_args.finetune}, blocks {' '.join(finetune_blocks)}")

            cls_model.load_state_dict(
                {
                    k: v for k, v in d['model_state'].items()
                    if k.split('.')[0] not in model_blocks[-1]
                },
                strict=False,
            )
            for n, p in cls_model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False)

        if self.use_cuda:
            if self.device.type == 'cuda':
                log.info(f"Using CUDA; {torch.cuda.device_count()} devices.")
                if torch.cuda.device_count() > 1:
                    cls_model = nn.DataParallel(cls_model)
            elif is_mps_device(self.device):
                log.info("Using MPS (Apple Silicon)")
            cls_model = cls_model.to(self.device)

        return cls_model

    def initOptimizer(self):
        """Initialize the optimizer."""
        lr = 0.003 if self.cli_args.finetune else 0.001
        return SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def initTrainDl(self):
        """Initialize training data loader."""
        DatasetClass = MalignancyDataset if self.cli_args.malignant else ClassificationDataset

        train_ds = DatasetClass(
            config=self.config,
            val_stride=10,
            is_validation=False,
            balance_ratio=1,
            augmentation=self.augmentation,
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
        DatasetClass = MalignancyDataset if self.cli_args.malignant else ClassificationDataset

        val_ds = DatasetClass(
            config=self.config,
            val_stride=10,
            is_validation=True,
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
        validation_cadence = 5 if not self.cli_args.finetune else 1

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

                model_type = 'mal' if self.cli_args.malignant else 'cls'
                self.saveModel(model_type, epoch_ndx, score == best_score)

        # Close the logger
        self.logger.close()

    def doTraining(self, epoch_ndx, train_dl):
        """Execute one training epoch."""
        self.model.train()
        train_dl.dataset.shuffle()

        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} Training",
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
                augment=True,
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        """Execute validation."""
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                f"E{epoch_ndx} Validation",
                start_ndx=val_dl.num_workers,
            )

            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                    augment=False,
                )

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, augment=True):
        """Compute loss for a batch."""
        input_t, label_t, index_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        index_g = index_t.to(self.device, non_blocking=True)

        if augment:
            input_g = self.augmentation(input_g)

        logits_g, probability_g = self.model(input_g)

        loss_g = self.loss_fn(
            logits_g, label_g[:, 1],
            reduction="none",
        )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = index_g
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = predLabel_g
        metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:, 1]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        return loss_g.mean()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        """Log metrics to TensorBoard and console."""
        log.info(f"E{epoch_ndx} {type(self).__name__}")

        pos = 'mal' if self.cli_args.malignant else 'pos'
        neg = 'ben' if self.cli_args.malignant else 'neg'

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] == 0
        negPred_mask = metrics_t[METRICS_PRED_NDX] == 0
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())
        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        truePos_count = pos_correct
        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = float(metrics_t[METRICS_LOSS_NDX].mean())
        metrics_dict['loss/neg'] = float(metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean())
        metrics_dict['loss/pos'] = float(metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean())

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = neg_correct / neg_count * 100
        metrics_dict['correct/pos'] = pos_correct / pos_count * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float64(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = \
            truePos_count / np.float64(truePos_count + falseNeg_count)
        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        # ROC curve and AUC
        threshold = torch.linspace(1, 0, steps=89)
        tpr = (metrics_t[None, METRICS_PRED_P_NDX, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        fpr = (metrics_t[None, METRICS_PRED_P_NDX, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count
        fp_diff = fpr[1:] - fpr[:-1]
        tp_avg = (tpr[1:] + tpr[:-1]) / 2
        auc = float((fp_diff * tp_avg).sum())
        metrics_dict['auc'] = auc

        log.info(
            f"E{epoch_ndx} {mode_str:8} {metrics_dict['loss/all']:.4f} loss, "
            f"{metrics_dict['correct/all']:-5.1f}% correct, "
            f"{metrics_dict['pr/precision']:.4f} precision, "
            f"{metrics_dict['pr/recall']:.4f} recall, "
            f"{metrics_dict['pr/f1_score']:.4f} f1 score, "
            f"{auc:.4f} auc"
        )

        # Use injected logger
        step = self.totalTrainingSamples_count

        # Replace pos/neg with appropriate labels
        renamed_metrics = {
            k.replace('pos', pos).replace('neg', neg): v
            for k, v in metrics_dict.items()
        }

        if mode_str == 'trn':
            self.logger.log_training_metrics(renamed_metrics, step)
            # Log ROC figure
            fig = pyplot.figure()
            pyplot.plot(fpr.numpy(), tpr.numpy())
            self.logger.log_training_figure('roc', fig, step)
            pyplot.close(fig)
            # Log histograms
            bins = np.linspace(0, 1)
            self.logger.log_training_histogram(
                f'label_{neg}', metrics_t[METRICS_PRED_P_NDX, negLabel_mask], step, bins
            )
            self.logger.log_training_histogram(
                f'label_{pos}', metrics_t[METRICS_PRED_P_NDX, posLabel_mask], step, bins
            )
        else:
            self.logger.log_validation_metrics(renamed_metrics, step)
            # Log ROC figure
            fig = pyplot.figure()
            pyplot.plot(fpr.numpy(), tpr.numpy())
            self.logger.log_validation_figure('roc', fig, step)
            pyplot.close(fig)
            # Log histograms
            bins = np.linspace(0, 1)
            self.logger.log_validation_histogram(
                f'label_{neg}', metrics_t[METRICS_PRED_P_NDX, negLabel_mask], step, bins
            )
            self.logger.log_validation_histogram(
                f'label_{pos}', metrics_t[METRICS_PRED_P_NDX, posLabel_mask], step, bins
            )

        return metrics_dict['auc'] if self.cli_args.malignant else metrics_dict['pr/f1_score']

    def saveModel(self, _type_str, epoch_ndx, isBest=False):
        """Save model checkpoint using injected checkpointer."""
        metrics = {
            'epoch': epoch_ndx,
            'total_samples': self.totalTrainingSamples_count,
        }

        self.checkpointer.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch_ndx,
            total_samples=self.totalTrainingSamples_count,
            metrics=metrics,
            is_best=isBest,
        )


if __name__ == '__main__':
    ClassificationTrainingApp().main()
