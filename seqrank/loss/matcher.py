import scipy
import torch
from typing import Dict, List, Tuple
from .loss import batch_mask_loss, batch_mask_loss_in_points

@torch.no_grad()
def hungarianMatcher(preds: Dict, targets: List, cfg=None) -> Tuple:
    """
        Params:
            preds: a dict:
                "masks": torch.Tensor [B,nq,H,W] logit
                "scores": torch.Tensor B,nq,1 logit
            targets: list of targets with length=batch_size, each is a torch.Tensor
                in shape N,H,W (binary map indicates the foreground/background)
        Returns:
            indices: tuple [Tensor, Tensor, Tensor]
                each tensor is a 1D vector, repr index.
    """
    B = len(targets)
    indices = [[], [], []]
    for b in range(B):
        tgts = targets[b].unsqueeze(1)  ## N,1,H,W
        masks = preds["masks"][b].unsqueeze(1)  ## nq, 1, h, w
        N, _, H, W = tgts.shape
        nq = len(masks)

        mask_loss = batch_mask_loss(torch.repeat_interleave(masks, N, dim=0), tgts.repeat(nq, 1, 1, 1), cfg=cfg).reshape(nq, N)  ## nq, N
        cls_loss = -torch.sigmoid(preds["scores"][b]).repeat_interleave(N, dim=1)  ## nq, N
        cost_matrix = mask_loss + cls_loss  ## nq, N
        row_idxs, col_idxs = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        indices[0].append((torch.ones(len(row_idxs), device=cost_matrix.device) * b).long())  ## batch index
        indices[1].append(torch.tensor(row_idxs, device=cost_matrix.device, dtype=torch.long))  ## query index
        indices[2].append(torch.tensor(col_idxs, device=cost_matrix.device, dtype=torch.long))  ## target index
    return tuple(torch.cat(idx) for idx in indices)


@torch.no_grad()
def hungarianMatcherInPoints(preds: Dict, targets: List, cfg=None) -> Tuple:
    """
        Params:
            preds: a dict:
                "masks": torch.Tensor [B,nq,H,W] logit
                "scores": torch.Tensor B,nq,1 logit
            targets: list of targets with length=batch_size, each is a torch.Tensor
                in shape N,H,W (binary map indicates the foreground/background)
        Returns:
            indices: tuple [Tensor, Tensor, Tensor]
                each tensor is a 1D vector, repr index.
    """
    B = len(targets)
    indices = [[], [], []]
    for b in range(B):
        tgts = targets[b].unsqueeze(1)  ## N,1,H,W
        masks = preds["masks"][b].unsqueeze(1)  ## nq, 1, h, w
        N, _, H, W = tgts.shape
        nq = len(masks)

        mask_loss = batch_mask_loss_in_points(torch.repeat_interleave(masks, N, dim=0), tgts.repeat(nq, 1, 1, 1), cfg=cfg).reshape(nq, N)  ## nq, N
        cls_loss = -torch.sigmoid(preds["scores"][b]).repeat_interleave(N, dim=1)  ## nq, N
        cost_matrix = mask_loss + cls_loss  ## nq, N
        row_idxs, col_idxs = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        indices[0].append((torch.ones(len(row_idxs), device=cost_matrix.device) * b).long())  ## batch index
        indices[1].append(torch.tensor(row_idxs, device=cost_matrix.device, dtype=torch.long))  ## query index
        indices[2].append(torch.tensor(col_idxs, device=cost_matrix.device, dtype=torch.long))  ## target index
    return tuple(torch.cat(idx) for idx in indices)

