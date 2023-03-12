from PIL import Image
import pyqtree
import glob
import os.path
import sys
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
from shapely.geometry import Point, Polygon
from shapely.wkt import loads

csv.field_size_limit(sys.maxsize)

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
        self.validation = False

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


# Validates a detection using the Point Clouds
def pointCloud(spindex, validationModel, pointClouds, bb):
    tmp = ''
    for cloud in os.listdir(pointClouds):
        tmp = pointClouds + '/' + cloud
        break


    # Creates empty .las file to later populate it with points
    with laspy.open(tmp) as f:
        w = laspy.open('tmp.las', mode='w', header = f.header)
        w.close()

    count = 0
    # Checks if there is an overlap with the cropped image and the point cloud
    matches = spindex.intersect((bb[0], bb[2], bb[1], bb[3]))

    # Iterates over the matched point clouds
    with laspy.open('tmp.las', mode = 'a') as w:
        for match in matches:
            with laspy.open(match) as f:
                # Appends the points of the overlapping region to the previously created .las file
                las = f.read()          
                x, y = las.points[las.classification == 2].x.copy(), las.points[las.classification == 2].y.copy()
                mask = (x >= bb[0]) & (x <= bb[2]) & (y >= bb[1]) & (y <= bb[3])
                if True in mask:
                    roi = las.points[las.classification == 2][mask]
                    w.append_points(roi)
                    count += 1
        
    if count > 0:
        xyz = las_utils.read_las_xyz('tmp.las')
        #FEATURE_NAMES = ['planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality']
        features = compute_features(xyz, search_radius=3)#, feature_names = ['planarity', 'linearity', 'surface_variation', 'sphericity', 'verticality'])
        
        if np.isnan(features).any() == False:

            stats = {}
            for i in FEATURE_NAMES:
                stats[i] = []
            
            for feature in features:
                for i in range(len(FEATURE_NAMES)):
                    stats[FEATURE_NAMES[i]].append(feature[i])

            X = []
            for i in FEATURE_NAMES:        
                mean = np.mean(stats[i])
                stdev = np.std(stats[i])
                X += [mean,stdev]
                #print(i + ': ' + str(mean) + ' - ' + str(stdev))



            #X += list(np.max(xyz, axis=0)-np.min(xyz, axis=0))
            #X += [np.mean(xyz, axis=0)[2], np.std(xyz, axis=0)[2]]
            
            
            os.remove('tmp.las')
            if validationModel.predict([X]) == -1:
                return False
            else:
                return True

    
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

def LBRroi(polygons, bb):
    p = Polygon([(bb[0], bb[2]), (bb[0], bb[3]), (bb[1], bb[3]), (bb[1], bb[2])])
    intersection = []
    for polygon in polygons:
        if polygon.intersects(p):
            intersection.append(polygon.intersection(p))
    return intersection

def LBR(roi, bb):
    b = [bb[0], bb[2], bb[1], bb[3]]
    p = Polygon([(b[0], b[2]), (b[0], b[3]), (b[1], b[3]), (b[1], b[2])])
    for polygon in roi:
        if polygon.intersects(p):
            return True
    return False

def detectYolov7(filename, step=20, offset=20):

    print("Running YOLOv7")
    mamoas = []
    validationModel = pickle.load(open("pointCloud.sav", "rb"))
    pointClouds = "../LAS/"
    polygonsCsv = "Segmentation.csv"

    weights = []
    for w in glob.glob("weights_folds/folds/" + "*.pt"):
        weights.append(w)


    # spindex = pyqtree.Index(bbox=(0, 0, 100, 100))
    # for cloud in os.listdir(pointClouds):
    #     with laspy.open(pointClouds + '/' + cloud) as f:
    #         spindex.insert(pointClouds + '/' + cloud, (f.header.x_min, f.header.y_min, f.header.x_max, f.header.y_max))


    polygons = []
    with open(polygonsCsv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = loads(row['WKT'])
            polygons.append(p)

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
    # xy = (0, 43000, 1200, 43700)
    # xy = (4650, 3210, 6000, 4260)
    xy = (0, 22622, 2660, 24700)
    
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
            xMinGeo = map(xmin, 0 , width_im, tifGeoCoord[0], tifGeoCoord[2])
            yMinGeo = map(ymin, height_im , 0, tifGeoCoord[1], tifGeoCoord[3])
            xMaxGeo = map(xmax, 0 , width_im, tifGeoCoord[0], tifGeoCoord[2])
            yMaxGeo = map(ymax, height_im , 0, tifGeoCoord[1], tifGeoCoord[3])

            roi = LBRroi(polygons, (xMinGeo, xMaxGeo, yMinGeo, yMaxGeo))
            if len(roi) == 0:
                continue
            img_cropped = image.crop((xmin, ymin, xmax, ymax))
            n, x, y, prob, bb = resultYolo(img_cropped, model, device)
            for i in range(n):
                bb_aux = []
                bb_aux.append(int(bb[i][0]) + xmin)
                bb_aux.append(int(bb[i][1]) + ymin)
                bb_aux.append(int(bb[i][2]) + xmin)
                bb_aux.append(int(bb[i][3]) + ymin)
                m = Mamoa(xmin + x[i], ymin + y[i], prob[i], bb_aux.copy())
                m.convert2GeoCoord(tifGeoCoord, width_im, height_im)
                lbr = LBR(roi, m.bb_GeoCoord)
                if LBR(roi, m.bb_GeoCoord):
                    # m.validation = pointCloud(validationModel, pointClouds, m.bb_GeoCoord)
                    mamoas.append(m)
            xmin += slide
            xmax += slide
        xmin = 0
        xmax = dim
        ymin += slide
        ymax += slide

    mamoas = removeDuplicates(mamoas, offset)
    # for mamoa in mamoas:
    #     mamoa.validation = pointCloud(validationModel, pointClouds, mamoa.bb_GeoCoord)

    image = cv2.imread("teste_ground.tif")	#type: ignore
    print("tamanho", len(mamoas))
    for m in mamoas:
        image = cv2.rectangle(image, (m.bb[0], m.bb[1]), (m.bb[2], m.bb[3]), (255, 0, 0), 2)	#type: ignore
        print(m.bb_GeoCoord)
        if m.validation == True:
            image = cv2.rectangle(image, (m.bb[0], m.bb[1]), (m.bb[2], m.bb[3]), (0, 0, 255), 3)	#type: ignore

    # ground_truth = convert_polygon_to_bb("anotacoes_arcos.csv")
    # with open("results/not_aug/results_Arcos.csv", "r") as f:
    #     rows = csv.reader(f)
    # for g in mamoas:
    #     xmin = map(int(g.bb_GeoCoord[0]), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
    #     ymin = map(int(g.bb_GeoCoord[1]), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
    #     xmax = map(int(g.bb_GeoCoord[2]), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
    #     ymax = map(int(g.bb_GeoCoord[3]), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
    #     cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)	#type: ignore


    with open("anotacoes_arcos.csv") as c:
        rows = csv.reader(c)
        for row in rows:
            row = row[0].split()
            xmin = min([row[i] for i in range(0, len(row), 2)])
            ymin = min([row[i] for i in range(1, len(row), 2)])
            xmax = max([row[i] for i in range(0, len(row), 2)])
            ymax = max([row[i] for i in range(1, len(row), 2)])

            xmin = map(float(xmin), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
            ymin = map(float(ymin), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
            xmax = map(float(xmax), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
            ymax = map(float(ymax), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)	#type: ignore
    cv2.imwrite("teste_ground.tif", image)
    
    # with open("results_" + filename.split(".")[0] + ".csv", "a") as f:
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
