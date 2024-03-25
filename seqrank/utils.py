import os, cv2
import pickle

import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision.ops import masks_to_boxes

def calc_iou(p, t):
    mul = (p*t).sum()
    add = (p+t).sum()
    return mul / (add - mul + 1e-6)

def debugDump(output_dir, image_name, texts, lsts, size=(256, 256), data=None):
    """
    Args:
        texts: list of list of text
        lsts: list of list of torch.Tensor H, W
    """
    out_dir = os.path.join(output_dir, "debug")
    os.makedirs(out_dir, exist_ok=True)
    outs = []
    for txts, lst in zip(texts, lsts):
        lst = [cv2.resize((x.numpy()*255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR) for x in lst]
        lst = [Image.fromarray(x) for x in lst]
        for x, t in zip(lst, txts):
            ImageDraw.Draw(x).text((0, 0), str(t), fill="red")
        out = Image.fromarray(np.concatenate([np.array(x) for x in lst], axis=1))
        outs.append(np.array(out))
    out = Image.fromarray(np.concatenate(outs, axis=0))
    out.save(os.path.join(out_dir, image_name+".png"))

    if not isinstance(data, type(None)):
        try:
            with open(os.path.join(out_dir, "latest.pk"), "wb") as f:
                pickle.dump(data, f)
        except:
            pass


def pad1d(x, dim, num, value=0.0):
    """

    Args:
        pad a torch.Tensor along dim (at the end) to be dim=num
        x: any shape torch.Tensor
        dim: int
        repeats: int

    Returns:
        x: where x.shape[dim] = num
    """
    size = list(x.shape)
    size[dim] = num - size[dim]
    assert size[dim] >= 0, "{} < 0".format(size[dim])
    v = torch.ones(size, dtype=x.dtype, device=x.device) * value
    return torch.cat([x, v], dim=dim)

def mask2Boxes(masks):
    """

    Args:
        masks: n, H, W

    Returns:
        bbox: n, 4 [(x1,y1),(x2,y2)] \in [0,1]

    """
    n, H, W = masks.shape
    bbox = masks_to_boxes(masks)
    xi, yi, xa, ya = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    xi = xi / W
    yi = yi / H
    xa = xa / W
    ya = ya / H
    return torch.clamp(torch.stack([xi, yi, xa, ya], dim=1), 0.0, 1.0)

def xyhw2xyxy(bbox):
    """

    Args:
        bbox: N, 4 [0,1]

    Returns:
        bbox: N, 4
    """
    x, y, h, w = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1, y1, x2, y2 = x-w/2., y-h/2., x+w/2., y+h/2.
    return torch.stack([x1, y1, x2, y2], dim=-1)  ## N, 4

def xyxy2xyhw(bbox):
    """

    Args:
        bbox: N, 4 [0,1]

    Returns:
        bbox: N, 4
    """
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x, y, h, w = (x1+x2)/2., (y1+y2)/2., y2-y1, x2-x1
    return torch.stack([x, y, h, w], dim=-1)  ## N, 4
