import argparse
import os
import time
import datetime
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loguru import logger

from utils import tensor2float, save_scalars, make_nograd_func, tocuda  # DictAverageMeter, SaveScene, make_nograd_func
from datasets import transforms, find_dataset_def
from models import DepthRNN
from config import cfg, update_config
from tools.evaluation_utils import eval_depth_batched, ResultsAverager

# Wrapper around Python's native multiprocessing
import torch.multiprocessing as mp
# Takes input data and distribute across GPUs
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# Initialize and destroy our distributed process group
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Typically each GPU runs one process, and set up group so that processes can communicate with each other
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
    logger.info(f'init_process_group, rank: {rank}, world_size:{world_size}')
    torch.cuda.set_device(rank)


def load_data(cfg):
    Dataset = find_dataset_def(cfg.DATA.NAME)
    transform = [
        transforms.ResizeDepth((256, 192), 'gt_depths'),
        transforms.ToTensor(),
    ]
    trans = transforms.Compose(transform)
    dataset = Dataset(cfg, trans)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.DATA.BATCH_SIZE,
                            sampler=DistributedSampler(dataset, shuffle=True) if cfg.DISTRIBUTED else None,
                            num_workers=cfg.DATA.N_WORKERS, pin_memory=True, drop_last=False)
    return dataset, dataloader


def load_new_model(cfg):
    model = DepthRNN(cfg).cuda()
    if cfg.DISTRIBUTED:
        model = DDP(model)
    if cfg.MODE == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WD)
        return model, optimizer
    else:
        return model


def log_setup(cfg):
    if not os.path.isdir(cfg.LOGDIR + '/log'):
        os.makedirs(cfg.LOGDIR + '/log')

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_path = os.path.join(cfg.LOGDIR, 'log', f'{current_time_str}_{cfg.MODE}.log')
    print('creating log file', logfile_path)
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO")

    tb_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'log'))
    return tb_writer


def train(rank: int, ngpus_per_node: int, cfg):
    ddp_setup(rank, ngpus_per_node)

    dataset, dataloader = load_data(cfg)
    model, optimizer = load_new_model(cfg)

    if rank == 0:
        tb_writer = log_setup(cfg)

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
    else:
        logger.info('Start from new model')

    logger.info("start at epoch {}".format(start_epoch))
    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    milestones = [int(epoch_idx) for epoch_idx in cfg.TRAIN.LREPOCHS.split(':')[0].split(',')]
    lr_gamma = 1 / float(cfg.TRAIN.LREPOCHS.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, cfg.TRAIN.EPOCHS):
        logger.info('Epoch {}:'.format(epoch_idx))
        dataloader.sampler.set_epoch(epoch_idx)

        # training
        for batch_idx, sample in enumerate(dataloader):
            sample = tocuda(sample)
            global_step = len(dataloader) * epoch_idx + batch_idx
            do_summary = global_step % cfg.SUMMARY_FREQ == 0
            start_time = time.time()
            try:
                loss, scalar_outputs = train_sample(model, optimizer, sample)
                if rank == 0:
                    logger.info('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx,
                                                                                                     cfg.TRAIN.EPOCHS,
                                                                                                     batch_idx,
                                                                                                     len(dataloader),
                                                                                                     loss,
                                                                                                     time.time() - start_time))
                if do_summary and rank == 0:
                    save_scalars(tb_writer, 'train', scalar_outputs, global_step, 1)
                del scalar_outputs

            except Exception as e:
                torch.cuda.empty_cache()
                print(f'{str(e)}')
                continue

        lr_scheduler.step()

        # checkpoint
        if (epoch_idx + 1) % cfg.SAVE_FREQ == 0 and rank == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(cfg.LOGDIR, epoch_idx))

    destroy_process_group()


def validate():
    raise NotImplementedError()


def test():
    dataset, dataloader = load_data(cfg)
    model = load_new_model(cfg)

    # Load model
    state_dict = torch.load(os.path.join(cfg.LOGDIR, cfg.LOADCKPT))
    model.load_state_dict(state_dict['model'], strict=False)
    epoch_idx = state_dict['epoch']

    all_frame_metrics = ResultsAverager('all_frame_metrics')
    if cfg.TEST.PRINT_GT:
        gt_frame_metrics = ResultsAverager('all_GT_frame_metrics')

    step_frame_metrics = []
    for i in range(1, 8):
        step_frame_metrics.append(ResultsAverager(f'step_{i}_frame_metrics'))

    for batch_idx, sample in enumerate(dataloader):
        sample = tocuda(sample)

        start_time = time.time()
        loss, scalar_outputs, outputs = test_sample(model, sample)
        logger.info('Epoch {}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, batch_idx,
                                                                                    len(dataloader), loss,
                                                                                    time.time() - start_time))
        full_depths = torch.stack(outputs['full_preds'], dim=2).squeeze(1)
        gt_depths = sample['gt_depths'][:, 1:]
        B, T = full_depths.shape[:2]

        metric_dict = eval_depth_batched(full_depths.flatten(0, 1), gt_depths.flatten(0, 1), cfg.TEST.MIN_DEPTH,
                                         cfg.TEST.MAX_DEPTH)
        all_frame_metrics.update_batch(metric_dict)

        if cfg.TEST.PRINT_GT:
            pred_depths = sample['pred_depths'][:, 1:]
            gt_metric_dict = eval_depth_batched(pred_depths.flatten(0, 1), gt_depths.flatten(0, 1),
                                                cfg.TEST.MIN_DEPTH,
                                                cfg.TEST.MAX_DEPTH)
            gt_frame_metrics.update_batch(gt_metric_dict)

        for key, values in metric_dict.items():
            metric_dict[key] = values.unflatten(0, (B, T))
        for i, metric in enumerate(step_frame_metrics):
            step_metric_dict = {}
            for key, values in metric_dict.items():
                step_metric_dict[key] = values[:, i]
            metric.update_batch(step_metric_dict)

        if batch_idx % 20 == 0:
            all_frame_metrics.print_metrics()

    all_frame_metrics.print_metrics()

    if cfg.TEST.PRINT_GT:
        gt_frame_metrics.print_metrics()

    for metric in step_frame_metrics:
        metric.print_metrics()

    # out_path = os.path.join(cfg.OUTDIR, 'evaluate')
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    # all_frame_metrics.output_json(
    #     os.path.join(out_path, f'all_frame_avg_metrics_{cfg.MODE}_online.json')
    # )

    # ckpt_list.append(ckpt)


def train_sample(model, optimizer, sample):
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
def test_sample(model, sample):
    model.eval()
    outputs, loss_dict = model(sample, is_training=True)
    loss = loss_dict['total_loss']

    return tensor2float(loss), tensor2float(loss_dict), outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ReconRNN')
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
                        help='gpu ids for multiprocessing training',
                        default='0',
                        type=str)
    args = parser.parse_args()
    update_config(cfg, args)
    # gpu_ids = args.gpu.split(',')
    # world_size = len(gpu_ids)
    cfg.defrost()
    ngpus_per_node = torch.cuda.device_count()
    cfg.DISTRIBUTED = ngpus_per_node > 1 and cfg.MODE == 'train'
    cfg.freeze()

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    if cfg.MODE == "train":
        mp.spawn(train, args=(ngpus_per_node, cfg), nprocs=ngpus_per_node)
    else:
        test()
