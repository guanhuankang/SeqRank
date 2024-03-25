import numpy as np
from PIL import Image
import os, scipy, torch
from pycocotools.mask import encode

import itertools
import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator
from .metrics import Metrics

class UpperBoundMatcher:
    def __init__(self):
        pass

    def cost(self, a, b):
        """ -IOU as cost """
        its = (a * b).sum()
        uni = (a + b).sum() - its
        return -its / (uni + 1e-6)

    def match(self, preds, gts):
        """

        Args:
            preds: list of n numpy.array or tensor.Tensor (same shape)
            gts: list of m numpy.array or tensor.Tensor with same shape (same shape)

        Returns:
            k: the top k indices are the best match, the remainders are appended at the end randomly
            indices: a list of int indicate the optimal order of preds.
                Get optimal preds with preds[indices]
        """
        N, M = len(preds), len(gts)
        C = np.array([self.cost(p, g) for p in preds for g in gts]).reshape(N, M)  ## N, M
        row_ids, col_ids = scipy.optimize.linear_sum_assignment(C)
        pairs = [(r,c) for r, c in zip(row_ids, col_ids)]
        pairs.sort(key=lambda x: x[1])
        row_ids = [x[0] for x in pairs]
        k = len(row_ids)
        for i in range(len(preds)):
            if i not in row_ids:
                row_ids.append(i)
        return k, row_ids

    def optimalOrder(self, preds, gts):
        k, idxs = self.match(preds, gts)
        return [preds[i] for i in idxs]


class SOREvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.metrics = Metrics(cfg.TEST.METRICS_OF_INTEREST)
        self.upper_bound = UpperBoundMatcher()

        self.coco_format = []
        self.results = []

    def reset(self):
        # self.metrics.from_config(cfg=self.cfg)
        self.coco_format = []
        self.results = []

    def process(self, inputs, outputs):
        """
        Params:
            @inputs: model input
            @outputs: model output (it should be in numpy format)
        """
        thres = self.cfg.TEST.THRESHOLD
        for inp, out in zip(inputs, outputs):
            image_name = inp["image_name"]
            preds = out["masks"]
            gts = list(inp["masks"].detach().cpu().numpy())

            ## UpperBound (debug only)
            # if self.cfg.TEST.UPPER_BOUND:
            #     preds = self.upper_bound.optimalOrder(preds, gts)

            ## EVAL
            self.results.append(self.metrics.process(preds=preds, gts=gts, thres=thres))

            ## COCO_FORMAT SAVE
            self.coco_format.append({
                "image_name": image_name,
                "masks": [encode( np.asfortranarray((x > 0.5).astype(np.uint8)) ) for x in preds],
                "bboxes": out.get("bboxes", []),
                "scores": out.get("scores", 0.0),
                "saliency": out.get("saliency", 0.0),
                "num": out.get("num", len(preds))
            })

            ## SAVE
            if self.cfg.TEST.EVAL_SAVE:
                out_path = os.path.join(self.cfg.OUTPUT_DIR, "results", self.dataset_name)
                os.makedirs(out_path, exist_ok=True)
                n = len(preds)

                uni = np.linspace(1.0, 0.5, n)
                pred_mask = np.zeros((inp["height"], inp["width"]), dtype=float)
                for i in range(n-1, -1, -1):
                    tmp = np.where(preds[i] > .5)
                    pred_mask[tmp] = uni[i]
                pred_mask = (pred_mask * 255).astype(np.uint8)
                Image.fromarray(pred_mask).save(
                    os.path.join(out_path, f"{image_name}_count_{n}.png")
                )

                # for i in range(n):
                #     Image.fromarray((preds[i]*255).astype(np.uint8)).save(
                #         os.path.join(out_path, f"{image_name}_top_{i+1}.png")
                #     )
                # for i in range(len(gts)):
                #     Image.fromarray((gts[i].numpy()*255).astype(np.uint8)).save(
                #         os.path.join(out_path, f"{image_name}_top_{i+1}_gt.png")
                #     )
    
    def evaluate(self):
        if self.cfg.SOLVER.NUM_GPUS > 1:
            comm.synchronize()
            
            results = comm.gather(self.results, dst=0)
            results = list(itertools.chain(*results))

            coco_format = comm.gather(self.coco_format, dst=0)
            coco_format = list(itertools.chain(*coco_format))

            if not comm.is_main_process():
                return {}
        else:
            results = self.results
            coco_format = self.coco_format

        coco_format_dict = dict( (x["image_name"], x) for x in coco_format)
        out_file = os.path.join(self.cfg.OUTPUT_DIR, self.dataset_name+"_coco_format.pth")
        torch.save(coco_format_dict, out_file)

        return self.metrics.aggregate(results)