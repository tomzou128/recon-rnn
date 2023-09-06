import argparse
from loguru import logger
from config import cfg, update_config
from tqdm import tqdm
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import transforms, find_dataset_def
from utils import tocuda
from tools.evaluation_utils import eval_depth_batched, ResultsAverager


def args():
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of VisFusion')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


args = args()
update_config(cfg, args)

transform = [
    transforms.ResizeDepth((256, 192), 'gt_depths'),
    transforms.ToTensor()
]
transforms = transforms.Compose(transform)
scene_path = os.path.join(cfg.TEST.TUPLE_PATH, cfg.MODE + '.txt')
Dataset = find_dataset_def(cfg.DATASET)
dataset = Dataset(cfg, transforms)
img_loader = DataLoader(dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS, drop_last=False)


def evaluate():
    all_frame_metrics = ResultsAverager('Evaluation', 'frame_metrics')
    img_loader.dataset.epoch = 0

    for batch_idx, sample in tqdm(enumerate(img_loader)):
        sample = tocuda(sample)
        metric_dict = eval_depth_batched(sample['pred_depths'], sample['gt_depths'], cfg.TEST.NEAR, cfg.TEST.FAR)

        n_sample = sample['gt_depths'].shape[0]
        for i in range(n_sample):
            sample_metrics = {}
            for key in list(metric_dict.keys()):
                sample_metrics[key] = metric_dict[key][i]
            all_frame_metrics.update_results(sample_metrics)

    all_frame_metrics.compute_final_average()
    all_frame_metrics.pretty_print_results(print_running_metrics=False)
    all_frame_metrics.print_sheets_friendly(
        include_metrics_names=True,
        print_running_metrics=False
    )

    out_path = os.path.join(cfg.OUTDIR, 'evaluate')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    all_frame_metrics.output_json(
        os.path.join(out_path, f'all_frame_avg_metrics_{cfg.MODE}.json')
    )


if __name__ == '__main__':
    evaluate()
