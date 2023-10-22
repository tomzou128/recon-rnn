from yacs.config import CfgNode as CN

_C = CN()

_C.MODE = 'train'
_C.LOADCKPT = ''
_C.LOGDIR = './checkpoints'
_C.OUTDIR = './out'
_C.RESUME = False
_C.SCENE = None
_C.SUMMARY_FREQ = 20
_C.SAVE_FREQ = 1
_C.SEED = 1
_C.LOCAL_RANK = 0
_C.DISTRIBUTED = False

# dataset related
_C.DATA = CN()
_C.DATA.NAME = 'scannet_depth_rnn'
_C.DATA.PATH = ''
_C.DATA.TUPLE_PATH = ''
_C.DATA.TUPLE_FILE = ''
_C.DATA.BATCH_SIZE = 1
_C.DATA.N_WORKERS = 8
_C.DATA.N_VIEWS = 5
_C.DATA.LOAD_IMG = False

# train
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 40
_C.TRAIN.LR = 0.001
_C.TRAIN.LREPOCHS = '12,24,36:2'
_C.TRAIN.WD = 0.0
_C.TRAIN.FINETUNE_LAYER = None
_C.TRAIN.MIN_DEPTH = 0.25
_C.TRAIN.MAX_DEPTH = 5.0


# test
_C.TEST = CN()
_C.TEST.LOAD_IMG = False
_C.TEST.PRINT_GT = False
_C.TEST.MIN_DEPTH = 0.25
_C.TEST.MAX_DEPTH = 5.0

# model
_C.MODEL = CN()
_C.MODEL.N_LAYER = 3

_C.MODEL.BACKBONE2D = CN()
_C.MODEL.BACKBONE2D.ARC = 'fpn-mnas'

_C.MODEL.SPARSEREG = CN()
_C.MODEL.SPARSEREG.DROPOUT = False

# _C.MODEL.TOP_K_OCC = [9, 9, 9]
# _C.MODEL.N_VIEWS = 9
# _C.MODEL.PASS_LAYERS = 2
# _C.MODEL.SINGLE_LAYER_MESH = False

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


def check_config(cfg):
    pass
