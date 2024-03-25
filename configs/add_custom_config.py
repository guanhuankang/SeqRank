from detectron2.config import CfgNode as CN

def add_custom_config(cfg, num_gpus=1):
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.PE = "APE"

    cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.NAME = "ResNeXt50"
    cfg.MODEL.BACKBONE.NUM_FEATURES = (256,512,1024,2048)
    cfg.MODEL.BACKBONE.FEATURE_KEYS = ["res2", "res3", "res4", "res5"]

    cfg.MODEL.NECK = CN()
    cfg.MODEL.NECK.NAME = "FPN"

    cfg.MODEL.SIS_HEAD = CN()
    cfg.MODEL.SIS_HEAD.NAME = "FoveaModule"
    cfg.MODEL.SIS_HEAD.NUM_BLOCKS = 6
    cfg.MODEL.SIS_HEAD.KEY_FEATURES = ["res5", "res4", "res3"]
    cfg.MODEL.SIS_HEAD.MASK_KEY = "res2"

    cfg.MODEL.GAZE_SHIFT_HEAD = CN()
    cfg.MODEL.GAZE_SHIFT_HEAD.NAME = "SequentialRankingModule"
    cfg.MODEL.GAZE_SHIFT_HEAD.NUM_BLOCKS = 2
    cfg.MODEL.GAZE_SHIFT_HEAD.KEY = "res5"
    
    cfg.MODEL.MODULES = CN()
    cfg.MODEL.MODULES.MULTIQ = CN()
    cfg.MODEL.MODULES.MULTIQ.GRID_SIZES = [(1,1), (2,2), (3,3)]

    cfg.MODEL.COMMON = CN()
    cfg.MODEL.COMMON.EMBED_DIM = 256
    cfg.MODEL.COMMON.NUM_HEADS = 8
    cfg.MODEL.COMMON.HIDDEN_DIM = 2048
    cfg.MODEL.COMMON.DROPOUT_ATTN = 0.0
    cfg.MODEL.COMMON.DROPOUT_FFN = 0.0
    cfg.MODEL.COMMON.NUM_QUERIES = 100

    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.LOSS = CN()
    cfg.LOSS.CLS_COST = 10.0
    cfg.LOSS.MASK_CE_COST = 5.0
    cfg.LOSS.MASK_DICE_COST = 5.0
    cfg.LOSS.BBOX_L1_COST = 5.0
    cfg.LOSS.BBOX_GIOU_COST = 2.0
    cfg.LOSS.OBJ_POS = 1.0
    cfg.LOSS.OBJ_NEG = 0.1
    cfg.LOSS.SAL_COST = 5.0
    cfg.LOSS.SAL_POS = 1.0
    cfg.LOSS.SAL_NEG = 1.0
    cfg.LOSS.SAL_TERMINATE = True
    cfg.LOSS.AUX = "disable"
    cfg.LOSS.AUX_WEIGHT = 0.4
    cfg.LOSS.NUM_POINTS = 12544

    cfg.DATASETS.ROOT = "assets"

    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.IMS_PER_GPU = 32
    cfg.SOLVER.NUM_GPUS = num_gpus

    cfg.TEST.AUG = CN()
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.UPPER_BOUND = False
    cfg.TEST.EVAL_SAVE = False
    cfg.TEST.METRICS_OF_INTEREST = ["mae"]
    cfg.TEST.THRESHOLD = 0.5

    cfg.INPUT.FT_SIZE_TRAIN = 800
    cfg.INPUT.FT_SIZE_TEST = 800
