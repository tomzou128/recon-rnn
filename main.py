import argparse
import os
import time
import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loguru import logger

from utils import tensor2float, save_scalars, make_nograd_func, tocuda  # DictAverageMeter, SaveScene, make_nograd_func
from datasets import transforms, find_dataset_def
from models import DepthRNN
from config import cfg, update_config
from datasets.sampler import DistributedSampler
from ops.comm import *
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

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:12356',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    # parse arguments and check
    args = parser.parse_args()
    return args


args = args()
update_config(cfg, args)

cfg.defrost()
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
print('number of gpus: {}'.format(num_gpus))
cfg.DISTRIBUTED = num_gpus > 1

if cfg.DISTRIBUTED:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="gloo", init_method="env://")
    logger.info(f"Running on GPU:{torch.cuda.get_device_name(args.local_rank)}")
    synchronize()
cfg.LOCAL_RANK = args.local_rank
cfg.freeze()

torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# create logger
if is_main_process():
    if not os.path.isdir(cfg.LOGDIR + '/log'):
        os.makedirs(cfg.LOGDIR + '/log')

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_path = os.path.join(cfg.LOGDIR, 'log', f'{current_time_str}_{cfg.MODE}.log')
    print('creating log file', logfile_path)
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO")

    tb_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'log'))

# Augmentation
if cfg.MODE == 'train':
    n_views = cfg.TRAIN.N_VIEWS
    random_rotation = cfg.TRAIN.RANDOM_ROTATION_3D
    random_translation = cfg.TRAIN.RANDOM_TRANSLATION_3D
    paddingXY = cfg.TRAIN.PAD_XY_3D
    paddingZ = cfg.TRAIN.PAD_Z_3D
else:
    n_views = cfg.TEST.N_VIEWS
    random_rotation = False
    random_translation = False
    paddingXY = 0
    paddingZ = 0

transform = [
    transforms.ResizeDepth((256, 192), 'gt_depths'),
    transforms.ToTensor(),
]
transforms = transforms.Compose(transform)

# dataset, dataloader
scene_path = os.path.join(cfg.TEST.TUPLE_PATH, cfg.MODE + '.txt')
Dataset = find_dataset_def(cfg.DATASET)
if cfg.MODE == "train":
    train_dataset = Dataset(cfg, transforms)
    if cfg.DISTRIBUTED:
        train_sampler = DistributedSampler(train_dataset, shuffle=False)
        TrainImgLoader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, sampler=train_sampler,
                                                     num_workers=cfg.TRAIN.N_WORKERS, pin_memory=True, drop_last=True)
    else:
        TrainImgLoader = DataLoader(train_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TRAIN.N_WORKERS,
                                    drop_last=True)
elif cfg.MODE == "test":
    test_dataset = Dataset(cfg, transforms)
    if cfg.DISTRIBUTED:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, sampler=test_sampler,
                                                    num_workers=cfg.TEST.N_WORKERS, pin_memory=True, drop_last=False)
    else:
        TestImgLoader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS,
                                   drop_last=False)

# model, optimizer
model = DepthRNN(cfg)

if cfg.DISTRIBUTED:
    model.cuda()
    model = DistributedDataParallel(
        model, device_ids=[cfg.LOCAL_RANK], output_device=cfg.LOCAL_RANK,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False,
        find_unused_parameters=True
    )
else:
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, lr=cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WD)


# main function
def train():
    # load parameters
    start_epoch = 0
    if cfg.RESUME:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(saved_models) != 0:
            # use the latest checkpoint file
            loadckpt = os.path.join(cfg.LOGDIR, saved_models[-1])
            logger.info("resuming " + str(loadckpt))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
            state_dict = torch.load(loadckpt, map_location=map_location)
            model.load_state_dict(state_dict['model'], strict=False)
            optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            start_epoch = state_dict['epoch'] + 1
    elif cfg.LOADCKPT != '':
        # load checkpoint file specified by args.loadckpt
        logger.info("loading model {}".format(cfg.LOADCKPT))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
        state_dict = torch.load(cfg.LOADCKPT, map_location=map_location)
        model.load_state_dict(state_dict['model'], strict=False)
        optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        start_epoch = state_dict['epoch'] + 1

    logger.info("start at epoch {}".format(start_epoch))
    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    milestones = [int(epoch_idx) for epoch_idx in cfg.TRAIN.LREPOCHS.split(':')[0].split(',')]
    lr_gamma = 1 / float(cfg.TRAIN.LREPOCHS.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, cfg.TRAIN.EPOCHS):
        logger.info('Epoch {}:'.format(epoch_idx))
        TrainImgLoader.dataset.epoch = epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % cfg.SUMMARY_FREQ == 0
            start_time = time.time()
            try:
                loss, scalar_outputs = train_sample(sample)
                if is_main_process():
                    logger.info('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx,
                                                                                                     cfg.TRAIN.EPOCHS,
                                                                                                     batch_idx,
                                                                                                     len(TrainImgLoader),
                                                                                                     loss,
                                                                                                     time.time() - start_time))
                if do_summary and is_main_process():
                    save_scalars(tb_writer, 'train', scalar_outputs, global_step, 1)
                del scalar_outputs

            except Exception as e:
                torch.cuda.empty_cache()
                print(f'{str(e)}')
                continue

        lr_scheduler.step()

        # checkpoint
        if (epoch_idx + 1) % cfg.SAVE_FREQ == 0 and is_main_process():
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(cfg.LOGDIR, epoch_idx))


def validate():
    raise NotImplementedError()


def test(from_latest=False):
    ckpt_list = []

    if cfg.LOADCKPT != '':
        saved_models = [cfg.LOADCKPT]
    else:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if from_latest:
            saved_models = saved_models[-1:]

    for ckpt in saved_models:
        if ckpt not in ckpt_list:
            # use the latest checkpoint file
            loadckpt = os.path.join(cfg.LOGDIR, ckpt)
            logger.info("resuming " + str(loadckpt))
            state_dict = torch.load(loadckpt)
            model.load_state_dict(state_dict['model'], strict=False)
            epoch_idx = state_dict['epoch']

            # just paste not debug yet
            test_dataset = Dataset(cfg, transforms)
            if cfg.DISTRIBUTED:
                test_sampler = DistributedSampler(test_dataset, shuffle=False)
                TestImgLoader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                                                            sampler=test_sampler,
                                                            num_workers=cfg.TEST.N_WORKERS, pin_memory=True,
                                                            drop_last=False)
            else:
                TestImgLoader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False,
                                           num_workers=cfg.TEST.N_WORKERS,
                                           drop_last=False)

            all_frame_metrics = ResultsAverager('Evaluation', 'frame_metrics')
            batch_len = len(TestImgLoader)
            for batch_idx, sample in enumerate(TestImgLoader):
                sample = tocuda(sample)

                start_time = time.time()
                loss, scalar_outputs, outputs = test_sample(sample)
                logger.info('Epoch {}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, batch_idx,
                                                                                            len(TestImgLoader), loss,
                                                                                            time.time() - start_time))

                metric_dict = eval_depth_batched(outputs[1], sample['gt_depths'][:, 1:], cfg.TEST.NEAR, cfg.TEST.FAR)

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
                os.path.join(out_path, f'all_frame_avg_metrics_{cfg.MODE}_online.json')
            )

            # ckpt_list.append(ckpt)


def train_sample(sample):
    model.train()

    optimizer.zero_grad()
    outputs, loss_dict = model(sample, is_training=True)
    loss = loss_dict['total_loss']
    if not type(loss) == float:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return tensor2float(loss), tensor2float(loss_dict)
    else:
        return 0, tensor2float(loss_dict)


@make_nograd_func
def test_sample(sample, is_training=False):
    model.eval()
    outputs, loss_dict = model(sample, is_training)
    loss = loss_dict['total_loss']

    return tensor2float(loss), tensor2float(loss_dict), outputs


if __name__ == '__main__':
    if cfg.MODE == "train":
        train()
    else:
        test()
