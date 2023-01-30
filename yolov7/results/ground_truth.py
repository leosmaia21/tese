import os
import csv
import numpy as np
import rasterio
from PIL import Image
import cv2

def check_duplicates(ground_mamoas, mamoas, threshold):
    matched_polygons = []
    for gt in ground_mamoas:
        for new in mamoas:
            #calculate euclidean distance between the center of the bounding boxes
            dist = np.linalg.norm(np.array([gt[0]+gt[2]/2, gt[1]+gt[3]/2]) - np.array([new[0]+new[2]/2, new[1]+new[3]/2]))
            if dist <= threshold:
                matched_polygons.append(new)
                break
    return matched_polygons
def map(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2

# def draw():

#     geoRef = rasterio.open('Arcos-lrm.tif')
#     tifGeoCoord = (geoRef.bounds[0], geoRef.bounds[1], geoRef.bounds[2], geoRef.bounds[3])
#     Image.MAX_IMAGE_PIXELS = None
#     image = Image.open('Arcos-lrm.tif').convert('RGB')
#     width_im, height_im = image.size
#     image = cv2.imread('Arcos-lrm.tif')

#     with open("results/not_aug/results_Arcos.csv", "r") as f:
#         rows = csv.reader(f)
#         for g in rows:
#             xmin = map(int(g[0]), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
#             ymin = map(int(g[1]), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
#             xmax = map(int(g[2]), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
#             ymax = map(int(g[3]), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
#             cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)	#type: ignore
    
#     cv2.imwrite('teste_bb.png', image)

def ground_truth(folder):
    print("Ground truth (", folder, ")")
    files = os.listdir(folder)

    ground_files = [file for file in os.listdir() if file.endswith(".csv")]

    ground_mamoas = []

    for file in ground_files:
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                aux = []
                wkt = row["WKT"]
                wkt = wkt.split("(((")[1].split(")))")[0]
                wkt = wkt.replace(",", " ")
                wkt = wkt.split(" ")
                aux.append(min([float(x) for x in wkt[::2]]))
                aux.append(min([float(y) for y in wkt[1::2]]))
                aux.append(max([float(x) for x in wkt[::2]]))
                aux.append(max([float(y) for y in wkt[1::2]]))
                ground_mamoas.append(aux.copy())

    mamoas = []
    for file in files:
        with open(folder + file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                row[0] = float(row[0])
                row[1] = float(row[1])
                row[2] = float(row[2])
                row[3] = float(row[3])
                mamoas.append(row)
    results = check_duplicates(ground_mamoas, mamoas, 10)
    print("Number of mamoas detected: ", len(mamoas))
    print("Number of ground truth: ", len(results))
    print("Number of trues: ", len([x for x in results if x[4] == 'True']))

    # filename = "../Arcos-lrm.tif"
    # geoRef = rasterio.open(filename)
    # tifGeoCoord = (geoRef.bounds[0], geoRef.bounds[1], geoRef.bounds[2], geoRef.bounds[3])
    # Image.MAX_IMAGE_PIXELS = None
    # image = Image.open(filename).convert('RGB')
    # width_im, height_im = image.size
    # image = cv2.imread(filename)

    # for g in mamoas:
    #     xmin = map(int(g[0]), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
    #     ymin = map(int(g[1]), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
    #     xmax = map(int(g[2]), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
    #     ymax = map(int(g[3]), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
    #     cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)	#type: ignore
    
    # cv2.imwrite('bb_all.png', image)


if __name__ == "__main__":
    ground_truth("not_aug_250_fulldata/")
    ground_truth("aug_250_fulldata/")



