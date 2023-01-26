from PIL import Image
import os.path
from click import group
import torch
import cv2
import numpy as np
import math
import rasterio
import rasterio.transform
import laspy
import os
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from jakteristics import las_utils, compute_features, FEATURE_NAMES
import pickle
import csv


def convert_polygon_to_bb(file: str):
    mamoas = []
    final = []

    with open(file, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            mamoas.append(row[0].split())

    for m in mamoas:
        aux =  []
        aux.append(min(float(m[i]) for i in range(0, len(m), 2)))
        aux.append(min(float(m[i]) for i in range(1, len(m), 2)))
        aux.append(max(float(m[i]) for i in range(0, len(m), 2)))
        aux.append(max(float(m[i]) for i in range(1, len(m), 2)))
        aux.append(((aux[0] + aux[2]) / 2))
        aux.append(((aux[1] + aux[3]) / 2))
        final.append(aux.copy())
    return final 

def map(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2


class Mamoa:
    def __init__(self, pX, pY, prob, bb):
        self.pX = pX
        self.pY = pY
        self.prob = prob
        self.bb = bb
        self.bb_GeoCoord = []
        self.afterValidation = False

    def convert2GeoCoord(self, tifGeoCoord, width_im, height_im):
        self.bb_GeoCoord.append(round(map(self.bb[0], 0, width_im, tifGeoCoord[0], tifGeoCoord[2])))
        self.bb_GeoCoord.append(round(map(height_im - self.bb[3], 0, height_im, tifGeoCoord[1], tifGeoCoord[3])))
        self.bb_GeoCoord.append(round(map(self.bb[2], 0, width_im, tifGeoCoord[0], tifGeoCoord[2])))
        self.bb_GeoCoord.append(round(map(height_im - self.bb[1], 0, height_im, tifGeoCoord[1], tifGeoCoord[3])))

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


# Validates a detection using the Point Clouds
def pointCloud(validationModel, pointClouds, bb):
    if os.path.isfile("tmp.las"):
        os.remove("tmp.las")

    tmp = ""
    for cloud in os.listdir(pointClouds):
        tmp = pointClouds + "/" + cloud
        break

	# Creates empty .las file to later populate it with points
    with laspy.open(tmp) as f:
        w = laspy.open("tmp.las", mode="w", header = f.header)
        w.close()

    count = 0
	# Iterates over the point clouds
    with laspy.open("tmp.las", mode = "a") as w:
        for cloud in os.listdir(pointClouds):
            with laspy.open(pointClouds + "/" + cloud) as f:
				# Checks if there is an overlap with the cropped image and the point cloud
                if f.header.x_min <= bb[0] and f.header.y_min <= bb[1] and  f.header.x_max >= bb[2] and f.header.y_max >= bb[3]:
                # if bb[0] <= f.header.x_max and bb[1] >= f.header.x_min and bb[2] <= f.header.y_max and bb[3] >= f.header.y_min:
                    # Appends the points of the overlapping region to the previously created .las file
                    las = f.read()	#type: ignore
                    x, y = las.points.x.copy(), las.points.y.copy()
                    mask = (x >= bb[0]) & (x <= bb[2]) & (y >= bb[1]) & (y <= bb[3])
                    roi = las.points[mask]
                    w.append_points(roi)	#type: ignore
                    count += 1
    print("numeros de .las usados:", count)
	
	# If temporary las was populated with points
    if count > 0:
        xyz = las_utils.read_las_xyz("tmp.las")

		# Compute 3D features
        features = compute_features(xyz, search_radius=3)	#type: ignore
        if np.isnan(features).any() == False:

            stats = {}
            for i in FEATURE_NAMES:
                stats[i] = []
 
            for feature in features:
                for i in range(len(FEATURE_NAMES)):
                    stats[FEATURE_NAMES[i]].append(feature[i])

            # Each point contributes to 14 features which is too heavy, therefore calculate
            # the mean and standard deviation of of every feature for each point
            X = []
            for i in FEATURE_NAMES:		
                mean = np.mean(stats[i])
                stdev = np.std(stats[i])
                X += [mean,stdev]

			# Removes temporary las
            os.remove("tmp.las")
			
			# 1 is validated, -1 is not validated
            result = validationModel.predict([X]) 
            print("result: ", result)
            if result == -1:
                print("false")
                return False
            else:
                print("true")
                return True

	# Return -1 if there are no Point Clouds in this region
    print("erro")
    return -1

def resultYolo(img_cropped, model, device):
    x = []
    y = []
    bb = []
    prob = []
    img0 = cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR) #type:ignore
    img = letterbox(img0)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    # print("shape", img.shape)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]
    pred = non_max_suppression(pred)
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  #type: ignore 
                x.append(xywh[0] * img.shape[2])
                y.append(xywh[1] * img.shape[2])
                # print("valores: ", int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                prob.append(f'{conf:.2f}')
                bb.append(xyxy)
    return len(x), x, y, prob, bb


def detectYolov7(filename, step=20, offset=20):

    print("Running YOLOv7")
    mamoas = []
    weights = 'teste_canedo.pt'
    validationModel = pickle.load(open("pointCloud.sav", "rb"))
    pointClouds = "../LAS/"

    #load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights, map_location=device)
    model = model.eval()
    dim = 640
    slide = round(dim * step / 100)

    xmin, ymin, xmax, ymax = 0, 0, dim, dim

    geoRef = rasterio.open(filename)
    tifGeoCoord = (geoRef.bounds[0], geoRef.bounds[1], geoRef.bounds[2], geoRef.bounds[3])
    print("coordenadas reais:", tifGeoCoord)
    
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(filename).convert('RGB')
    width_im, height_im = image.size
    # xy = (523, 2140, 7500, 5363)
    xy = (0, 43000, 1200, 43700)
    # xy = (4650, 3210, 6000, 4260)
    # xy = (0, 22622, 2660, 24700)
    
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

    print("Columns:", columns, " Rows:", rows)
    for row in range(round(rows)):
        for column in range(round(columns)):
            img_cropped = image.crop((xmin, ymin, xmax, ymax))
            n, x, y, prob, bb = resultYolo(img_cropped, model, device)
            for i in range(n):
                bb_aux = []
                bb_aux.append(int(bb[i][0]) + xmin)
                bb_aux.append(int(bb[i][1]) + ymin)
                bb_aux.append(int(bb[i][2]) + xmin)
                bb_aux.append(int(bb[i][3]) + ymin)
                mamoas.append(Mamoa(xmin + x[i], ymin + y[i], prob[i], bb_aux.copy()))
            xmin += slide
            xmax += slide
        xmin = 0
        xmax = dim
        ymin += slide
        ymax += slide

    mamoas = removeDuplicates(mamoas, offset)


    # validation with points cloud
    for mamoa in mamoas:
         mamoa.convert2GeoCoord(tifGeoCoord, width_im, height_im)
         print(mamoa.bb_GeoCoord)
         # mamoa.afterValidation = pointCloud(validationModel, pointClouds, mamoa.bb_GeoCoord)

    image = cv2.imread("teste_ground.tif")	#type: ignore
    print("tamanho", len(mamoas))
    for m in mamoas:
        image = cv2.rectangle(image, (m.bb[0], m.bb[1]), (m.bb[2], m.bb[3]), (255, 0, 0), 2)	#type: ignore
        if m.afterValidation == True:
            image = cv2.rectangle(image, (m.bb[0], m.bb[1]), (m.bb[2], m.bb[3]), (0, 0, 255), 3)	#type: ignore

    ground_truth = convert_polygon_to_bb("anotacoes_arcos.csv")
    cv2.imwrite("teste_ground.tif", image)

    # with open("results.csv", "a") as f:
    #     writer = csv.writer(f)
    #     for mamoa in mamoas:
    #         aux = mamoa.bb_GeoCoord
    #         aux.append(mamoa.afterValidation)
    #         writer.writerow(aux)


    # cv2.imwrite("teste2_e6e.tif", image)

    # img = cv2.imread("cropped.tif")
    # for m in mamoas:
    #     print("valores: ", int(m.bb[0]), int(m.bb[1]), int(m.bb[2]), int(m.bb[3]))
    #     start = (int(m.bb[0]), int(m.bb[1]))
    #     end = (int(m.bb[2]), int(m.bb[3]))
    #     img = cv2.rectangle(img, start, end, (255,0,0), 3)
    # cv2.imwrite("bb.tif", img)
    # src = rasterio.open('Arcos-lrm.tif')
    # image = Image.open('Arcos-lrm.tif').convert('RGB')
    # width_im, height_im = image.size
    # # image = image.crop((int(width_im/2 - 1000), int(width_im/2 +1000), int(height_im/4 -1000), int(height_im/4 + 1000)))
    # width_im, height_im = image.size
    # raster_transform = src.transform

    # pixel_x = width_im / 2
    # pixel_y = height_im / 4


    # x,y = rasterio.transform.xy(transform = raster_transform, 
    #                             rows = pixel_y, 
    #                             cols = pixel_x)
    # print(mamoas[0].pX, mamoas[0].pY)
    # xMinImg = src.bounds[0]
    # xMaxImg = src.bounds[2]
    # yMinImg = src.bounds[1]
    # yMaxImg = src.bounds[3]
    # print(xMaxImg, yMaxImg)
    # for m in mamoas:
    #     print(m.pX, m.pY)
    # for m in mamoas
    #     image_to_save = cv2.circle(image_to_save, (round(m.pX), round(m.pY)), 15, (0, 0, 255), 6) #type: ignore
    # for m in mamoas:
    #     image_to_save = cv2.circle(image_to_save, (round(m.pX), round(m.pY)), 20, (255, 0, 0), 6)#type: ignore
    #     image_to_save = cv2.putText(image_to_save, str(m.prob), (round(m.pX) - 30, round(m.pY) - 30),#type: ignore
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)#type: ignore
    # cv2.imwrite('cropped_40%.tif', image_to_save)#type: ignore


if __name__ == '__main__':
    detectYolov7('Arcos-lrm.tif', step=40, offset=20)
