import glob
from PIL import Image
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



class Mamoa:
    def __init__(self, pX, pY, prob):
        self.pX = pX
        self.pY = pY
        self.prob = prob

    # def convert2GeoCoord(self,):
# TODO: fazer a funcao de converter pixeis para geo coordenadas


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
def pointCloud(validationModel, pointClouds, cropExtent, className, bb):
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
				if bb[0] <= f.header.x_max and bb[1] >= f.header.x_min and bb[2] <= f.header.y_max and bb[3] >= f.header.y_min:
					# Appends the points of the overlapping region to the previously created .las file
					las = f.read()#type: ignore
					x, y = las.points.x.copy(), las.points.y.copy()
					mask = (x >= bb[0]) & (x <= bb[1]) & (y >= bb[2]) & (y <= bb[3])
					roi = las.points[mask]
					w.append_points(roi)	#type: ignore
					count += 1
	
	# If temporary las was populated with points
	if count > 0:
		xyz = las_utils.read_las_xyz("tmp.las")

		# Compute 3D features
		features = compute_features(xyz, search_radius=3)#type: ignore
		
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
			if validationModel.predict([X]) == -1:
				return False
			else:
				return True

	# Return -1 if there are no Point Clouds in this region
	return -1

def resultYolo(img_cropped, model, device):
    x = []
    y = []
    prob = []
    img0 = cv2.cvtColor(np.array(img_cropped), cv2.COLOR_RGB2BGR) #type:ignore
    img = letterbox(img0)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
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
                prob.append(f'{conf:.2f}')
    return len(x), x, y, prob


def detectYolo(filename, step=20, offset=20):
    """Funtion to run yolov7, arguements are the filename, step and offset, step(default=20) is the percentage for the
    image slide, and the offset(default=20) is the number of pixeis i which duplicates will be removed"""

    print("Running YOLOv7")
    mamoas = []
    weights = []

    for weight in glob.glob("*.pt"):
        weights.append(weight)

    #load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights, map_location=device)
    model = model.eval()

    dim = 640  # 40 pixels => 20x20 meters
    slide = round(dim * step / 100)

    xmin, ymin, xmax, ymax = 0, 0, dim, dim

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(filename).convert('RGB')
    width_im, height_im = image.size

    rows = round((height_im / dim) / (step / 100))
    columns = round((width_im / dim) / (step / 100))

    print("Columns:", columns, " Rows:", rows)

    for row in range(round(rows)):
        for column in range(round(columns)):
            img_cropped = image.crop((xmin, ymin, xmax, ymax))
            n, x, y, prob = resultYolo(img_cropped, model, device)
            for i in range(n):
                mamoas.append(Mamoa(xmin + x[i], ymin + y[i], prob[i]))
            xmin += slide
            xmax += slide
        xmin = 0
        xmax = dim
        ymin += slide
        ymax += slide


    mamoas = removeDuplicates(mamoas, offset)
    src = rasterio.open('Arcos-lrm.tif')
    image = Image.open('Arcos-lrm.tif').convert('RGB')
    width_im, height_im = image.size
    # image = image.crop((int(width_im/2 - 1000), int(width_im/2 +1000), int(height_im/4 -1000), int(height_im/4 + 1000)))
    width_im, height_im = image.size
    raster_transform = src.transform

    pixel_x = width_im / 2
    pixel_y = height_im / 4


    x,y = rasterio.transform.xy(transform = raster_transform, 
                                rows = pixel_y, 
                                cols = pixel_x)
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
    detectYolo('cropped.tif', step=40, offset=20)
