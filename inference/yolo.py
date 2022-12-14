import glob
import os
from pathlib import Path
import sys

import PIL
import cv2
from PIL import Image
import numpy as np
import torch

from yolov5.models.experimental import attempt_load
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.plots import Annotator

sys.path.append("yolov5")


Image.MAX_IMAGE_PIXELS = None


def resultYOLO(results, model, path, im, im0s, img, s):
    results = non_max_suppression(results[0], 0.25, 0.45, None, False, 1000)

    n_objects = 0
    seen = 0
    names = model.names
    X = []
    Y = []
    for i, det in enumerate(results):
        seen += 1
        p, im0, frame = path, im0s.copy(), getattr(img, 'frame', 0)
        p = Path(p)  # to Path
        # txt_path = str(save_dir + p.stem) + ('' if img.mode == 'image' else f'_{frame}')  # im.txt
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            n_objects = len(det)
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                X.append(xywh[0] * im.shape[2])
                Y.append(xywh[1] * im.shape[2])
    return n_objects, X, Y


def detectYolo(step):
    print("Running YOLOv5")
    weights = []
    for weight in glob.glob("*.pt"):
        weights.append(weight)
    print(weights)

    device = torch.device('cuda', 0)
    model = attempt_load(weights, device=device)
    model.conf = 0.53  # confidence value

    dim = 640  # 40 pixels => 20x20 meters
    h = dim
    w = dim
    slide = dim * step / 100
    dims = (dim, dim)

    xmin = 0
    xmax = dim
    ymin = 0
    ymax = dim

    for f in glob.glob("*.tif"):
        print("Load image")
        image = 0
        image = Image.open(f)
        width_im, height_im = image.size
        rows = round((height_im / dim) / (step / 100))
        columns = round((width_im / dim) / (step / 100))
        print("Columns:", columns, " Rows:", rows)

    # numpyarray = np.array(img_cropped)
    for column in range(columns):
        for row in range(rows):
            img_cropped = image.crop(xmin, ymin, xmax, ymax)
            break
        break
    # if (np.mean(numpyarray)) != 0:
    #     img = LoadImages("teste2.tif")
    #     for path, im, im0s, vid_cap, s in img:
    #         im = im
    #     im = torch.from_numpy(im).to(device)
    #     im.float()
    #     im = im / 255  # 0 - 255 to 0.0 - 1.0
    #     if len(im.shape) == 3:
    #         im = im[None]  # expand for batch dim
    #
    #     n, X, Y = resultYOLO(model(im), model, path, im, im0s, img, s)
    #     print("n", n, "x", X, "y", Y)
    #     image = cv2.imread("teste2.tif")
    #     for i in range(n):
    #         image = cv2.circle(image, (int(X[i]), int(Y[i])), 15, (255, 0, 0))
    #     cv2.imwrite("a.jpg", image)
detectYolo(20)
