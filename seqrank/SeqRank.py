import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .neck import build_neck_head
from .modules import build_gaze_shift_head, build_sis_head
from .component import PositionEmbeddingSine, PositionEmbeddingRandom
from .utils import calc_iou, debugDump, pad1d, mask2Boxes, xyhw2xyxy, xyxy2xyhw
from .loss import hungarianMatcherInPoints, batch_mask_loss_in_points, batch_bbox_loss

class LearnablePE(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.ape = nn.parameter.Parameter(torch.zeros((embed_dim, 25, 25)), requires_grad=True)
        nn.init.trunc_normal_(self.ape)

    def forward(self, x):
        """
        x: B, C, H, W
        return: B, C, H, W
        """
        ape = F.interpolate(self.ape.unsqueeze(0), size=x.shape[2::], mode="bilinear")  ## 1, C, H, W
        return ape.expand(len(x), -1, -1, -1)  ## B, C, H, W


@META_ARCH_REGISTRY.register()
class SeqRank(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.neck = build_neck_head(cfg)
        self.instance_seg = build_sis_head(cfg)
        self.gaze_shift = build_gaze_shift_head(cfg)

        self.pe_layer = {
            "SINE": PositionEmbeddingSine(cfg.MODEL.COMMON.EMBED_DIM // 2, normalize=True),
            "RANDOM": PositionEmbeddingRandom(cfg.MODEL.COMMON.EMBED_DIM // 2),
            "APE": LearnablePE(cfg.MODEL.COMMON.EMBED_DIM)
        }[cfg.MODEL.PE]

        self.cfg = cfg
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1, 3, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch_dict, *args, **argw):
        ## prepare image
        images = torch.stack([s["image"] for s in batch_dict], dim=0).to(self.device).contiguous()
        images = (images - self.pixel_mean) / self.pixel_std

        zs = self.backbone(images)
        zs = self.neck(zs)
        zs_pe = dict((k, self.pe_layer(zs[k])) for k in zs)
        q, qpe, out, auxs = self.instance_seg(
            feats=zs,
            feats_pe=zs_pe
        )

        pred_masks = out["masks"]  ## B, nq, H, W
        pred_bboxes = out["bboxes"].sigmoid()  ## B, nq, 4 [xyhw] in [0,1]
        pred_objs = out["scores"]  ## B, nq, 1
        gaze_shift_key = self.cfg.MODEL.GAZE_SHIFT_HEAD.KEY

        if self.training:
            ## Training
            masks = [x["masks"].to(self.device) for x in batch_dict]  ## list of k_i, Ht, Wt
            bboxes = [mask2Boxes(m) for m in masks]  ## list of k_i, 4[xyxy]
            n_max = max([len(x) for x in masks])
            gt_size = masks[0].shape[-2::]

            pred_masks = F.interpolate(pred_masks, size=gt_size, mode="bilinear")
            bi, qi, ti = hungarianMatcherInPoints(preds={"masks": pred_masks, "scores": pred_objs}, targets=masks, cfg=self.cfg)

            q_masks = torch.stack([pad1d(m, dim=0, num=n_max, value=0.0) for m in masks], dim=0)  ## B, n_max, H, W
            q_boxes = torch.stack([pad1d(bb, dim=0, num=n_max, value=0.0) for bb in bboxes], dim=0)  ## B, n_max, 4

            q_corresponse = torch.zeros_like(pred_objs)  ## B, nq, 1
            q_corresponse[bi, qi, 0] = (ti + 1).to(q_corresponse.dtype)  ## 1 to n_max

            mask_loss = batch_mask_loss_in_points(pred_masks[bi, qi], q_masks[bi, ti], cfg=self.cfg).mean()
            bbox_loss = batch_bbox_loss(xyhw2xyxy(pred_bboxes[bi, qi]), q_boxes[bi, ti], cfg=self.cfg).mean()
            
            obj_pos_weight = torch.tensor(self.cfg.LOSS.OBJ_POS, device=self.device)
            obj_neg_weight = torch.tensor(self.cfg.LOSS.OBJ_NEG, device=self.device)
            pos_mask = q_corresponse.gt(.5).float()
            neg_mask = 1.0 - pos_mask
            pos_obj_loss = F.binary_cross_entropy_with_logits(pred_objs, torch.ones_like(pred_objs), reduction="none")
            neg_obj_loss = F.binary_cross_entropy_with_logits(pred_objs, torch.zeros_like(pred_objs), reduction="none")
            obj_loss = (pos_obj_loss * obj_pos_weight * pos_mask + neg_obj_loss * obj_neg_weight * neg_mask).mean()
            
            # obj_loss = F.binary_cross_entropy_with_logits(pred_objs, q_corresponse.gt(.5).float(),
            #                                               pos_weight=obj_pos_weight) * obj_neg_weight

            if self.cfg.LOSS.AUX == "disable" or len(auxs) <= 0:
                aux_mask_loss = torch.zeros_like(mask_loss)
                aux_bbox_loss = torch.zeros_like(bbox_loss)
            else:
                aux_mask_loss = sum([
                    batch_mask_loss_in_points(
                        F.interpolate(aux["masks"], size=gt_size, mode="bilinear")[bi, qi],
                        q_masks[bi, ti],
                        cfg=self.cfg
                    ).mean()
                    for aux in auxs
                ])
                aux_bbox_loss = sum([
                    batch_bbox_loss(
                        xyhw2xyxy(torch.sigmoid(aux["bboxes"][bi, qi])),
                        q_boxes[bi, ti],
                        cfg=self.cfg
                    ).mean()
                    for aux in auxs
                ])

            sal_loss = torch.zeros_like(obj_loss).mean()  ## initialize as zero
            for i in range(n_max + 1):
                # q_vis_gt = q_corresponse.gt(i).float() * torch.rand_like(q_corresponse).le(0.15).float()
                q_vis = q_corresponse * q_corresponse.le(i).float()  # + q_vis_gt
                q_ans = q_corresponse.eq(i + 1).float()
                sal, _ = self.gaze_shift(
                    q=q,
                    z=zs[gaze_shift_key].flatten(2).transpose(-1, -2),
                    qpe=qpe,
                    zpe=zs_pe[gaze_shift_key].flatten(2).transpose(-1, -2),
                    q_vis=q_vis,
                    bbox=pred_bboxes,  ## xyhw
                    size=tuple(zs[gaze_shift_key].shape[2::])
                )
                sal_pos = F.binary_cross_entropy_with_logits(sal, torch.ones_like(sal), reduction="none")
                sal_neg = F.binary_cross_entropy_with_logits(sal, torch.zeros_like(sal), reduction="none")
                sal_ele_loss = q_ans * sal_pos + (1.0 - q_ans) * sal_neg
                sal_ele_loss = torch.where(torch.isnan(sal_ele_loss), torch.zeros_like(sal_ele_loss), sal_ele_loss)
                sal_loss += sal_ele_loss.mean()

            ## debugDump
            if np.random.rand() < 0.1:
                k = 5
                mm = pred_masks[bi, qi].sigmoid()[0:k].detach().cpu()  ## k, H, W
                tt = q_masks[bi, ti].cpu()[0:k]  ## k, H, W
                ss = pred_objs[bi, qi, 0].sigmoid()[0:k].detach().cpu().tolist()  ## k
                oo = [float(calc_iou(m, t)) for m, t in zip(mm, tt)]
                debugDump(
                    output_dir=self.cfg.OUTPUT_DIR,
                    image_name="latest",
                    texts=[ss, oo],
                    lsts=[list(mm), list(tt)],
                    data=None
                )

            return {
                "mask_loss": mask_loss,
                "bbox_loss": bbox_loss,
                "obj_loss": obj_loss * self.cfg.LOSS.CLS_COST,
                "sal_loss": sal_loss * self.cfg.LOSS.SAL_COST,
                "aux_mask_loss": aux_mask_loss * self.cfg.LOSS.AUX_WEIGHT,
                "aux_bbox_loss": aux_bbox_loss * self.cfg.LOSS.AUX_WEIGHT
            }
            ## end training
        else:
            ## inference
            size = tuple(zs[gaze_shift_key].shape[2::])
            z = zs[gaze_shift_key].flatten(2).transpose(-1, -2)
            zpe = zs_pe[gaze_shift_key].flatten(2).transpose(-1, -2)
            q_vis = torch.zeros_like(pred_objs)
            bs, nq, _ = q.shape
            bs_idx = torch.arange(bs, device=self.device, dtype=torch.long)

            results = [{
                "image_name": x.get("image_name", idx),
                "masks": [],
                "bboxes": [],
                "scores": [],
                "saliency": [],
                "num": 0
            } for idx, x in enumerate(batch_dict)]

            ends_batch = set()
            for i in range(nq):
                sal = self.gaze_shift(q=q, z=z, qpe=qpe, zpe=zpe, q_vis=q_vis, bbox=pred_bboxes, size=size)
                sal_max = torch.argmax(sal[:, :, 0], dim=1).long()  ##  B
                q_vis[bs_idx, sal_max, 0] = i + 1

                sal_scores = sal[bs_idx, sal_max, 0].sigmoid()  ## B
                obj_scores = pred_objs[bs_idx, sal_max, 0].sigmoid()  ## B
                the_masks = pred_masks[bs_idx, sal_max, :, :]  ## B, H, W
                the_bboxes = xyhw2xyxy(pred_bboxes[bs_idx, sal_max, :])  ## B, 4 [xyxy]

                ''' Comment:
                        We use fixed threshold for all benchmarks.
                        We adopt 0.001 as our threshold, since we want to rank as much salient objects as possible. 
                '''
                t_sal = 0.001
                t_obj = 0.001

                for idx in range(bs):
                    obj_score = obj_scores[idx]
                    sal_score = sal_scores[idx]
                    if obj_score < t_obj or sal_score < t_sal:
                        ends_batch.add(idx)
                    if idx in ends_batch: continue

                    hi, wi = batch_dict[idx]["height"], batch_dict[idx]["width"]
                    results[idx]["masks"].append(
                        F.interpolate(the_masks[idx:idx + 1, :, :].unsqueeze(1), size=(hi, wi), mode="bilinear")[
                            0, 0].sigmoid().detach().cpu().gt(.5).float().numpy()
                    )
                    results[idx]["bboxes"].append(
                        (the_bboxes[idx].detach().cpu() * torch.tensor([wi, hi, wi, hi])).tolist()
                    )
                    results[idx]["scores"].append(
                        float(obj_scores[idx].detach().cpu())
                    )
                    results[idx]["saliency"].append(
                        float(sal_scores[idx].detach().cpu())
                    )
                    results[idx]["num"] += 1
                if len(ends_batch) >= bs:
                    break
            return results
            # end inference
        # """ end if """
    # """ end forward """
# """ end class """
