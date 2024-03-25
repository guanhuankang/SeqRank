from detectron2.utils.registry import Registry

SALIENCY_INSTANCE_SEG = Registry("SALIENCY_INSTANCE_SEG")
SALIENCY_INSTANCE_SEG.__doc__ = """
Saliency Instance Segmentation (SIS) head
"""

def build_sis_head(cfg):
    name = cfg.MODEL.SIS_HEAD.NAME
    return SALIENCY_INSTANCE_SEG.get(name)(cfg)


GAZE_SHIFT_HEAD = Registry("GAZE_SHIFT")
GAZE_SHIFT_HEAD.__doc__ = """
Gaze shift prediction head
"""

def build_gaze_shift_head(cfg):
    name = cfg.MODEL.GAZE_SHIFT_HEAD.NAME
    return GAZE_SHIFT_HEAD.get(name)(cfg)
