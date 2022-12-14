import glob
import sys

sys.path.append("yolov7")
from torch._jit_internal import loader
from torchvision.transforms import transforms

from yolov7.yolov7.utils.datasets import LoadImages

import os
from pathlib import Path
import PIL
from PIL import Image
import numpy as np
import torch
import cv2
from yolov7.models.experimental import attempt_load
from yolov7.yolov7.utils.general import non_max_suppression, scale_coords, xyxy2xywh
import torch
import torch.backends.cudnn as cudnn


def detectYolo(step):
    print("Running YOLOv5")
    weights = []
    for weight in glob.glob("*.pt"):
        weights.append(weight)
    print(weights)

    # half = device.type != 'cpu'  # half precision only supported on CUDA

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    print("finish")
    x = []
    y = []
    img_test = Image.open('teste.tif')
    # img_test = img_test.convert('RGB')
    # img_test = np.array(img_test)
    # img_test = img_test.transpose(2,0,1)
    # if (np.mean(numpyarray)) != 0:
    img_test = LoadImages('teste.tif')
    for path, im, im0s, vid_cap in img_test:
        im = im
        im = torch.from_numpy(im).to(device)
        im = im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)
        # print(im)
        model.eval()
        pred = model(im)[0]
        pred = non_max_suppression(pred)
        for i, det in enumerate(pred):  # detections per image
            p, im0, frame = path, im0s.copy(), getattr(img_test, 'frame', 0)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    x.append(xywh[0] * im.shape[2])
                    y.append(xywh[1] * im.shape[2])
    print(x, ", ",y)
    #     if len(im.shape) == 3:
    #         im = im[None]  # expand for batch dim
    #
    #     n, X, Y = resultYOLO(model(im), model, path, im, im0s, img, s)
    #     print("n", n, "x", X, "y", Y)
    #     image = cv2.imread("teste2.tif")
    #     for i in range(n):
    #         image = cv2.circle(image, (int(X[i]), int(Y[i])), 15, (255, 0, 0))
    #     cv2.imwrite("a.jpg", image)


detectYolo(100)
