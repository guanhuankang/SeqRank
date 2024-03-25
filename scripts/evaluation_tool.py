import os, sys

import torch

sys.path.append("..")

from PIL import Image
import numpy as np
import tqdm, json, copy

from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch
)

from detectron2.config import get_cfg
from detectron2.utils import comm, logger
from detectron2.data import DatasetCatalog
from configs import add_custom_config
from dataset import register_sor_dataset, sor_dataset_mapper_test
from pycocotools.mask import decode
from evaluation import SOREvaluator

def setup(args):
    cfg = get_cfg()
    add_custom_config(cfg, num_gpus=args.num_gpus)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    # logger.setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="toy")
    register_sor_dataset(cfg)
    return cfg

def loadJson(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def main(args):
    cfg = setup(args)
    dataset_name = cfg.DATASETS.TEST[0]
    groundtruth = DatasetCatalog.get(dataset_name)

    sor_evaluator = SOREvaluator(cfg, dataset_name)
    name = "unknown"
    path = cfg.MODEL.WEIGHTS

    pred_instances = torch.load(path)
    for gt_zip in tqdm.tqdm(groundtruth):
        gt = sor_dataset_mapper_test(gt_zip, cfg)
        instances = copy.deepcopy(pred_instances.get(gt["image_name"], {"masks": []}))
        instances["masks"] = [ decode(x).astype(float) for x in instances["masks"]]
        sor_evaluator.process([gt], [instances])

    results = sor_evaluator.evaluate()
    print(results)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, f"dataset_{dataset_name}_method_{name}.csv"), "w") as f:
        f.write(str(results))

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )