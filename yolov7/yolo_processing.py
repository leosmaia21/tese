import glob
from PIL import Image
import torch
import cv2
import numpy as np
import math
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression, scale_coords, xyxy2xywh


class Mamoa:
    def __init__(self, pX, pY):
        self.pX = pX
        self.pY = pY


def removeDuplicates(mamoas, offset):
    newList = []
    flag = True
    if mamoas:
        newList.append(mamoas[0])
        for i in mamoas:
            for j in newList:
                distance = math.sqrt(((i.pX - j.pX) ** 2) + ((i.pY - j.pY) ** 2))
                if distance < offset:
                    flag = False
                    break
            if flag:
                newList.append(i)
            flag = True
    return newList


def prepare_image(img_cropped, device):
    """This function receives the image(PIL) and the device as arguments, returns two images, one in cv2(BGR) format and
    the other ready to be used in the model """
    img0 = cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR)
    img = letterbox(img0)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img0, img


# def convertoPixeis2GeoCoord(x, y):


def resultYolo(img0, img, pred):
    x = []
    y = []
    pred = non_max_suppression(pred)
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                x.append(xywh[0] * img.shape[2])
                y.append(xywh[1] * img.shape[2])
    return len(x), x, y


def detectYolo(step):
    print("Running YOLOv5")
    mamoas = []
    weights = []
    for weight in glob.glob("*.pt"):
        weights.append(weight)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights, map_location=device)
    model = model.eval()

    dim = 640  # 40 pixels => 20x20 meters
    slide = round(dim * step / 100)
    xmin, ymin, xmax, ymax = 0, 0, dim, dim
    Image.MAX_IMAGE_PIXELS = None
    # for f in glob.glob("*.tif"):
    image = Image.open('cropped.tif')
    width_im, height_im = image.size
    rows = round((height_im / dim) / (step / 100))
    columns = round((width_im / dim) / (step / 100))
    print("Columns:", columns, " Rows:", rows)
    image_to_save = cv2.imread('cropped.tif')
    for row in range(rows):
        for column in range(columns):
            img_cropped = image.crop((xmin, ymin, xmax, ymax))
            img0, img = prepare_image(img_cropped, device)
            n, x, y = resultYolo(img0, img, model(img)[0])
            for i in range(n):
                mamoas.append(Mamoa(xmin + x[i], ymin + y[i]))
            image_to_save = cv2.rectangle(image_to_save, (xmin, ymin), (xmax, ymax), (255, 0, 0), 4)
            xmin += slide
            xmax += slide
        xmin = 0
        xmax = dim
        ymin += slide
        ymax += slide
    for m in mamoas:
        print(m.pX, m.pY)
        image_to_save = cv2.circle(image_to_save, (round(m.pX), round(m.pY)), 10, (0, 0, 255), 4)
    mamoas = removeDuplicates(mamoas, 15)
    print('oba')
    for m in mamoas:
        print(m.pX, m.pY)
        image_to_save = cv2.circle(image_to_save, (round(m.pX), round(m.pY)), 15, (255, 0, 0), 4)
    cv2.imwrite('50%.tif', image_to_save)


detectYolo(50)
