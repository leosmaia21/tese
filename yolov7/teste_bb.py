import cv2
import csv
import rasterio
from PIL import Image



def map(value, min1, max1, min2, max2):
	return (((value - min1) * (max2 - min2)) / (max1 - min1)) + min2

def draw():

    geoRef = rasterio.open('Arcos-lrm.tif')
    tifGeoCoord = (geoRef.bounds[0], geoRef.bounds[1], geoRef.bounds[2], geoRef.bounds[3])
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open('Arcos-lrm.tif').convert('RGB')
    width_im, height_im = image.size
    image = cv2.imread('Arcos-lrm.tif')

    with open("results/not_aug/results_Arcos.csv", "r") as f:
        rows = csv.reader(f)
        for g in rows:
            xmin = map(int(g[0]), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
            ymin = map(int(g[1]), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
            xmax = map(int(g[2]), tifGeoCoord[0], tifGeoCoord[2], 0, width_im)
            ymax = map(int(g[3]), tifGeoCoord[1], tifGeoCoord[3], height_im, 0)
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)	#type: ignore
    
    cv2.imwrite('teste_bb.png', image)
if __name__ == '__main__':
    draw()
