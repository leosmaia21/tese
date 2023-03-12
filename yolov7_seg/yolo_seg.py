from PIL import Image
import cv2
import numpy as np
import os
import torch
import pickle
import rasterio
import math
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from utils.segment.general import process_mask


class Mamoa:
    def __init__(self, pX, pY, bb, seg):
        self.pX = pX
        self.pY = pY
        self.prob = 0
        self.bb = bb
        self.bb_GeoCoord = []
        self.afterValidation = False
        self.seg = seg

    def convert2GeoCoord(self, tifGeoCoord, width_im, height_im):
        self.bb_GeoCoord.append(round(map(self.bb[0], 0, width_im, tifGeoCoord[0], tifGeoCoord[2])))
        self.bb_GeoCoord.append(round(map(self.bb[1], 0, height_im, tifGeoCoord[3], tifGeoCoord[1])))
        self.bb_GeoCoord.append(round(map(self.bb[2], 0, width_im, tifGeoCoord[0], tifGeoCoord[2])))
        self.bb_GeoCoord.append(round(map(self.bb[3], 0, height_im, tifGeoCoord[3], tifGeoCoord[1])))

        return self.bb_GeoCoord


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

def map(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2

# Convert masks(n,160,160) into segments(n,xy)
def masks2segments(masks, strategy='largest'):
    segments = []
    for x in masks.int().cpu().numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]	#type: ignore
        if strategy == 'concat':  # concatenate all segments
            c = np.concatenate([x.reshape(-1, 2) for x in c])
        elif strategy == 'largest':  # select largest segment
            c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        segments.append(c.astype('float32'))
    return segments

def resultYoloSeg(img_cropped, model, device):
    img0 = cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR) #type:ignore
    img = letterbox(img0)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255
    if len(img.shape) == 3:
        img = img[None]

    pred, out = model(img)
    proto = out[1]
    pred = non_max_suppression(pred, nm = 32)
    for i, det in enumerate(pred):
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample = True)  # HWC
            return masks2segments(masks)

def detectYoloSeg(filename, step=20):

    print("Running YOLOv7-segmentation")
    mamoas = []
    weights = 'best.pt'
    # validationModel = pickle.load(open("pointCloud.sav", "rb"))
    # pointClouds = "../LAS/"

    #load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights, device=device)
    # model = DetectMultiBackend(weights, device=device , fp16=False)
    # model = model.eval()
    dim = 640
    slide = round(dim * (step / 100))

    xmin, ymin, xmax, ymax = 0, 0, dim, dim

    geoRef = rasterio.open(filename)
    tifGeoCoord = (geoRef.bounds[0], geoRef.bounds[1], geoRef.bounds[2], geoRef.bounds[3])
    print("coordenadas reais:", tifGeoCoord)
    
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(filename).convert('RGB')
    width_im, height_im = image.size

    # xy = (523, 2140, 7500, 5363)
    # xy = (0, 43000, 1200, 43700)
    # xy = (4650, 3210, 6000, 4260)
    # xy = (0, 22622, 2660, 24700)
    xy = (0, 22622, 640, 22622 + 640)
    
    image = image.crop(xy)
    image.save("teste_ground.tif")

    x1 = map(xy[0], 0 , width_im, tifGeoCoord[0], tifGeoCoord[2])
    y1 = map(height_im - xy[3], 0 , height_im, tifGeoCoord[1], tifGeoCoord[3])
    x2 = map(xy[2], 0 , width_im, tifGeoCoord[0], tifGeoCoord[2])
    y2 = map(height_im - xy[1], 0 , height_im, tifGeoCoord[1], tifGeoCoord[3])
    tifGeoCoord = (x1, y1, x2, y2) 
    width_im, height_im = image.size
    rows = round((height_im / dim) / (step / 100))
    columns = round((width_im / dim) / (step / 100))

    im = cv2.imread("teste_ground.tif")
    count = 0
    for row in range(int(rows)):
        for column in range(int(columns)):
            print("\r", round((count / (rows * columns)) * 100, 1), "%", flush=True, end="")
            count += 1
            img_cropped = image.crop((xmin, ymin, xmax, ymax))
            res = resultYoloSeg(img_cropped, model, device)
            if res:
                for mamoa in res:
                    bb_aux = []
                    for point in mamoa:
                        point[0] += xmin
                        point[1] += ymin
                    bb_aux.append(round(min([mamoa[i][0] for i in range(len(mamoa))])))
                    bb_aux.append(round(min([mamoa[i][1] for i in range(len(mamoa))])))
                    bb_aux.append(round(max([mamoa[i][0] for i in range(len(mamoa))])))
                    bb_aux.append(round(max([mamoa[i][1] for i in range(len(mamoa))])))
                    pX = bb_aux[0] + (bb_aux[2] - bb_aux[0]) / 2
                    pY = bb_aux[1] + (bb_aux[3] - bb_aux[1]) / 2
                    mamoas.append(Mamoa(pX, pY, bb_aux, mamoa))
            xmin += slide
            xmax += slide
        xmin = 0
        xmax = dim
        ymin += slide
        ymax += slide

    mamoas = removeDuplicates(mamoas, 15)
    
    print("Mamoas encontradas:", len(mamoas))
    for mamoa in mamoas:
        for point in mamoa.seg:
            cv2.circle(im, (int(point[0]), int(point[1])), 1, (0, 0, 255), -1)


            # cv2.rectangle(im, (mamoa.bb[0], mamoa.bb[1]), (mamoa.bb[2], mamoa.bb[3]), (0, 255, 0), 2)
    cv2.imwrite("teste_ground.tif", im)

if __name__ == '__main__':
    detectYoloSeg('Arcos-lrm.tif', step=40)
