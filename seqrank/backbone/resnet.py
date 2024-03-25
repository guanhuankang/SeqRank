import torch.nn as nn
import torchvision.models as models
from detectron2.config import configurable
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

class ResNet(nn.Module):
    def __init__(self, cfg, resnet_name="resnext101"):
        super().__init__()
        resnet = {
            "ResNeXt50": models.resnext50_32x4d,
            "ResNeXt101": models.resnext101_32x8d,
            "ResNet152": models.resnet152
        }[resnet_name](pretrained=True)

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.key_features = cfg.MODEL.BACKBONE.FEATURE_KEYS
        
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        
        self._out_feature_channels = dict(
            (k, n)
            for k,n in zip(cfg.MODEL.BACKBONE.FEATURE_KEYS, cfg.MODEL.BACKBONE.NUM_FEATURES)
        )

    def forward(self, x):
        out = {}
        out["res1"] = self.layer0(x)
        out["res2"] = self.layer1(out["res1"])
        out["res3"] = self.layer2(out["res2"])
        out["res4"] = self.layer3(out["res3"])
        out["res5"] = self.layer4(out["res4"])
        ret = dict((k, out[k]) for k in out if k in self.key_features)
        return ret

@BACKBONE_REGISTRY.register()
class ResNeXt50(ResNet, Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, resnet_name="ResNeXt50")
        
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.key_features
        }
    
@BACKBONE_REGISTRY.register()
class ResNeXt101(ResNet, Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, resnet_name="ResNeXt101")
        
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.key_features
        }
    
@BACKBONE_REGISTRY.register()
class ResNet152(ResNet, Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, resnet_name="ResNet152")
        
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self.key_features
        }
    