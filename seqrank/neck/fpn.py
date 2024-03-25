import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from seqrank.component import init_weights_, LayerNorm2D
from .registry import NECK_HEAD

class FPNLayer(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.scale = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), LayerNorm2D(dim))
    
    def forward(self, high_feat, low_feat):
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2::], mode="bilinear")
        return self.conv(self.scale(high_feat) + low_feat)

@NECK_HEAD.register()
class FPN(nn.Module):
    @configurable
    def __init__(self, dim=256, feat_dims=(128,256,512,1024), feat_keys=["res2","res3","res4","res5"]):
        super().__init__()
        self.feat_keys = feat_keys
        self.lateral_conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(d_in, dim, 1),
            nn.GELU(),
            LayerNorm2D(dim)
        ) for d_in in feat_dims])
        self.fpn_layers = nn.ModuleList([
            FPNLayer(dim=dim)
            for _ in range(len(feat_dims)-1)
        ])
        self.feat_keys = feat_keys

        init_weights_(self.lateral_conv)

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
        for i, fpn in enumerate(self.fpn_layers):
            feats[i+1] = fpn(high_feat=feats[i], low_feat=feats[i+1])
        feats = dict((k, v) for k, v in zip(self.feat_keys, feats[::-1]))
        return feats
