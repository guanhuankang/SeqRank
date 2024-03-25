import sys
import torch
from collections import OrderedDict

def convert_backbone(src_pth_file, tgt_pth_file = None):
    if tgt_pth_file==None:
        tgt_pth_file = src_pth_file
    pth = torch.load(src_pth_file)
    assert "model" in pth, "{} (keys:{}) lacks 'model' key ".format(src_pth_file, pth.keys())

    updated_pth = {
        "model": OrderedDict([ ("backbone."+k,v) for k,v in pth["model"].items() if "backbone" not in k])
    }
    if len(updated_pth["model"]) == len(pth["model"]):
        for k in pth:
            if k!="model":
                updated_pth[k] = pth[k]
        torch.save(updated_pth, tgt_pth_file)
        print("success convertion: {} -> {}".format(src_pth_file, tgt_pth_file))
    else:
        print("fail! len_of {} != len_of {}".format(len(updated_pth["model"]), len(pth["model"])))

if __name__=="__main__":
    src_file = sys.argv[1]
    tgt_file = None if len(sys.argv)<=2 else sys.argv[2]
    print("converting {} -> {}".format(src_file, tgt_file))
    convert_backbone(
        src_pth_file=src_file,
        tgt_pth_file=tgt_file
    )
