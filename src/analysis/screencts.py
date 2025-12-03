"""
CT screening application.

Screens CT scans and computes lung mask ratios.
"""

import argparse
import sys

from torch.utils.data import Dataset, DataLoader

from config import get_config
from dataset import get_candidate_repository, get_ct_cache
from util.util import enumerateWithEstimate, prhist
from util.logconf import logging
from .vis import build2dLungMask

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class LunaScreenCtDataset(Dataset):
    """Dataset for screening CT scans."""

    def __init__(self):
        config = get_config()
        self.ct_cache = get_ct_cache(config)
        repo = get_candidate_repository(config)
        candidates = repo.get_all()
        self.series_list = sorted(set(c.series_uid for c in candidates))

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, ndx):
        series_uid = self.series_list[ndx]
        ct = self.ct_cache.get_classification(series_uid)
        mid_ndx = ct.hu_array.shape[0] // 2

        mask_tup = build2dLungMask(series_uid, mid_ndx)

        return series_uid, float(mask_tup.dense_mask.sum() / (mask_tup.raw_dense_mask.sum() or 1))


class LunaScreenCtApp:
    """Application for screening CT scans."""

    def __init__(self, sys_argv=None):
        if sys_argv is None:
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

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        """Main screening loop."""
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        self.prep_dl = DataLoader(
            LunaScreenCtDataset(),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        series2ratio_dict = {}

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Screening CTs",
            start_ndx=self.prep_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            series_list, ratio_list = batch_tup
            for series_uid, ratio_float in zip(series_list, ratio_list):
                series2ratio_dict[series_uid] = ratio_float

        prhist(list(series2ratio_dict.values()))


if __name__ == '__main__':
    LunaScreenCtApp().main()
