# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
import numpy as np
import cv2

from detectron2.config import get_cfg
# from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from .predictor import VisualizationDemo
from configs.add_custom_config import add_custom_config
from seqrank import *

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_custom_config(cfg, num_gpus=1)
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get_visualization(masks, height=480, width=640):
    # [[61, 87, 234], [99, 192, 251], [188, 176, 100], [153, 102, 68], [119, 85, 8]]
    color_maps = [[61, 87, 234], [99, 192, 251], [188, 176, 100], [153, 102, 68]] + [(np.array([119, 85, 8]).astype(float) * x).astype(np.uint8) for x in np.linspace(1., 0.2, 100)]  ## Tian
    # color_maps = [(222,38,12), (234,87,61), (235,152,61), (184,210,79), (100,176,188), (5, 62, 86)]  ## Ours
    # color_maps = [cm.jet(x)[0:3] for x in np.linspace(1.0, 0.0, num_ranks)]
    
    H, W = (height, width) if len(masks)<=0 else masks[0].shape
    n = len(masks)
    vis = np.zeros((H,W,3), dtype=np.uint8)
    for i in range(n):
        j = n - i -1
        m = masks[j]
        xs, ys = np.where(m > 0.5)
        vis[xs, ys] = color_maps[j]
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

def get_demo():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    
    return demo
