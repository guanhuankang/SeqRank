import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from seqrank.component import init_weights_, LayerNorm2D

from .registry import NECK_HEAD

class UnFold(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.window_size = kernel_size[0]*kernel_size[1]
        self.h = lambda x: math.floor((x+2*padding-dilation*(kernel_size[0]-1)-1)/stride + 1)
        self.w = lambda x: math.floor((x+2*padding-dilation*(kernel_size[1]-1)-1)/stride + 1)
    def forward(self, x):
        b,c,h,w = x.shape
        h, w = self.h(h), self.w(w)
        return self.unfold(x).contiguous().reshape(b,c,self.window_size,h,w)  ## b,c,window_size,h,w

class FRC(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(d_in, d_out, 1), nn.GELU(), LayerNorm2D(d_out))
        self.conv2 = nn.Sequential(nn.Conv2d(d_out, d_out, 1), nn.GELU(), LayerNorm2D(d_out))
        self.unfold = UnFold(3, padding=1)
        init_weights_(self)

    def calcMask(self, low_feat):
        """

        Args:
            low_feat: B, C, H, W

        Returns:
            mask: b,c,w_s,h,w binary

        """
        x = self.unfold(low_feat) ## b,c,w_s,h,w
        thr = torch.mean(x, dim=2, keepdim=True)  ## b,c,1,h,w
        pos = (x - thr).gt(0.0).float()  ## binary, b,c,w_s,h,w
        idc = (torch.unsqueeze(low_feat, dim=2) - thr).gt(0.0).float()  ## binary, b,c,1,h,w
        mask = idc * pos + (1.0 - idc) * (1.0 - pos)  ## b,c,w_s,h,w
        return mask  ## b,c,w_s,h,w

    def forward(self, high_feat, low_feat):
        """
        fuse high-level feat to low-level feat.
        Args:
            high_feat: high-level features, B, d_in, h, w
            low_feat: low-level fetrues, B, d_out, H, W
            where h < H, w < W
        Returns:
            x: B, d_out, H, W
        """
        mask = self.calcMask(low_feat)  ## b,c,w_s,h,w
        x = self.unfold(F.interpolate(self.conv1(high_feat), size=low_feat.shape[2::], mode="bilinear"))  ## b,c,w_s,h,w
        x = torch.sum(x * mask, dim=2) / (torch.sum(mask, dim=2) + 1e-6)
        return self.conv2(x + low_feat)  ## B,d_out,H,W

@NECK_HEAD.register()
class FrcPN(nn.Module):
    '''
        Cross-Scale Feature Re-Coordinate Pyramid Network (CS-FrcPN)
    '''
    @configurable
    def __init__(self, dim=256, feat_dims=(128,256,512,1024), feat_keys=["res2","res3","res4","res5"]):
        super().__init__()
        self.feat_keys = ["res2", "res3", "res4", "res5"]
        self.lateral_conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(d_in, dim, 1),
            nn.GELU(),
            LayerNorm2D(dim)
        ) for d_in in feat_dims])
        self.frcs = nn.ModuleList([FRC(dim, dim) for _ in feat_dims[0:-1]])
        self.feat_keys = feat_keys


    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.MODEL.COMMON.EMBED_DIM,
            "feat_dims": cfg.MODEL.BACKBONE.NUM_FEATURES,
            "feat_keys": cfg.MODEL.BACKBONE.FEATURE_KEYS
        }

    def forward(self, feats):
        """

        Args:
            feats: dict with keys as self.feat_keys

        Returns:
            feats: dict with same keys and same shape values as input

        """
        feats = [layer(feats[k]) for layer, k in zip(self.lateral_conv, self.feat_keys)][::-1]  ## high->low after lateral_conv
        for i, frc in enumerate(self.frcs):
            feats[i+1] = frc(high_feat=feats[i], low_feat=feats[i+1])
        feats = dict((k, v) for k, v in zip(self.feat_keys, feats[::-1]))
        return feats
