import os, tqdm
import argparse
import numpy as np
import torch
import cv2
from pycocotools.mask import decode, encode

def visualize(masks):
    color_maps = [[61, 87, 234], [99, 192, 251], [188, 176, 100], [153, 102, 68]] ## Tian
    color_maps += [
        (np.array([119, 85, 8]).astype(float) * x).astype(np.uint8)
        for x in np.linspace(1., 0.2, 20)
    ]
    color_maps += [color_maps[-1]] * 100
    
    H, W = (480, 640)
    n = len(masks)
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(n):
        j = n-i-1
        m = decode(masks[j])
        xs, ys = np.where(m > 0.5)
        vis[xs, ys] = color_maps[j]
        # cx, cy = xs.mean(), ys.mean()
    return vis

def visualize2colormap(pthfile, output):
    os.makedirs(output, exist_ok=True)
    data = torch.load(pthfile)
    for name in tqdm.tqdm(data):
        colormap = visualize(data[name]["masks"])
        cv2.imwrite(os.path.join(output, name+"_colormap.png"), colormap)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='visualize2colormap')
    parser.add_argument("-p", "--pth-file", dest="pthfile", type=str, help="results in pth format")
    parser.add_argument("-o", "--output", dest="output", type=str, help="colormaps folder")
    args = parser.parse_args()
    
    visualize2colormap(pthfile=args.pthfile, output=args.output)
    