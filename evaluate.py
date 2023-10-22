import argparse
from config import cfg, update_config
from tqdm import tqdm
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
Dataset = find_dataset_def(cfg.DATA.NAME)
dataset = Dataset(cfg, transforms)
img_loader = DataLoader(dataset, cfg.DATA.BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.N_WORKERS, drop_last=False)


def evaluate():
    all_frame_metrics = ResultsAverager('frame_metrics')
    print(len(img_loader))
    for batch_idx, sample in tqdm(enumerate(img_loader)):
        sample = tocuda(sample)
        metric_dict = eval_depth_batched(sample['pred_depths'].flatten(0, 1), sample['gt_depths'].flatten(0, 1), cfg.TEST.MIN_DEPTH, cfg.TEST.MAX_DEPTH)

        all_frame_metrics.update_batch(metric_dict)

        if batch_idx % 100 == 0:
            all_frame_metrics.print_metrics()

    all_frame_metrics.print_metrics()

    # out_path = os.path.join(cfg.OUTDIR, 'evaluate')
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    # all_frame_metrics.output_json(
    #     os.path.join(out_path, f'all_frame_avg_metrics_{cfg.MODE}.json')
    # )


if __name__ == '__main__':
    evaluate()
