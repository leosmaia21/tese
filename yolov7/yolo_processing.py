import glob
import sys

import cv2

sys.path.append("yolov7")

from yolov7.utils.datasets import LoadImages, letterbox

import os
from pathlib import Path
import PIL
from PIL import Image
import numpy as np
import torch
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords, xyxy2xywh
import torch
import torch.backends.cudnn as cudnn


def detectYolo(step):
    print("Running YOLOv5")
    weights = []
    for weight in glob.glob("*.pt"):
        weights.append(weight)
    print(weights)

    # half = device.type != 'cpu'  # half precision only supported on CUDA

    device = torch.device("cuda")
    # Load model
    model = attempt_load(weights[0], map_location=device)

    dim = 640  # 40 pixels => 20x20 meters
    h = dim
    w = dim
    slide = dim * step / 100
    dims = (dim, dim)

    xmin = 0
    xmax = dim
    ymin = 0
    ymax = dim
    x = 0
    Image.MAX_IMAGE_PIXELS = None
    # for f in glob.glob("*.tif"):
    #     image = Image.open(f)
    #     # image.show()
    #     width_im, height_im = image.size
    #     rows = round((height_im / dim) / (step / 100))
    #     columns = round((width_im / dim) / (step / 100))
    #     print("Columns:", columns, " Rows:", rows)
    #     for row in range(rows):
    #         for column in range(columns):
    #             img_cropped = image.crop((xmin, ymin, xmax, ymax))
    #             img_cropped.save('test_crop/' + str(row) + '_' + str(column) + '.jpg')
    #             xmin = xmax
    #             xmax = xmax + dim
    #         xmin = 0
    #         xmax = dim
    #         ymin = ymax
    #         ymax = ymax + dim
    x = []
    y = []
    img0 = Image.open('teste.tif')
    img0 = cv2.cvtColor(np.array(img0), cv2.COLOR_RGB2BGR)
    img = letterbox(img0)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # print(im)
    model.eval()
    pred = model(img)[0]
    pred = non_max_suppression(pred)
    for i, det in enumerate(pred):  # detections per image
        im0, frame = img0.copy(), getattr(img0, 'frame', 0)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                x.append(xywh[0] * img.shape[2])
                y.append(xywh[1] * img.shape[2])
    print(x, ", ", y)


detectYolo(100)
