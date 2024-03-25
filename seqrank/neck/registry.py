from detectron2.utils.registry import Registry

NECK_HEAD = Registry("NECK_HEAD")
NECK_HEAD.__doc__ = """
neck head
"""

def build_neck_head(cfg):
    name = cfg.MODEL.NECK.NAME
    return NECK_HEAD.get(name)(cfg)
