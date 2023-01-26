import os
import csv
from os.path import isfile
import cv2

files = os.listdir()

images = [file for file in files if file.endswith('.png')]

for image in images:
    label = image.split(".")[0] + '.txt'
    im = cv2.imread(image)
    if isfile(label):
        with open(label, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data = row[0].split(" ")[1:]
                x = [float(data[i]) for i in range(0, len(data), 2)]
                y = [float(data[i]) for i in range(1, len(data), 2)]
                for i in range(len(x)):
                    cv2.circle(im, (int(x[i]), int(y[i])), 2, (0, 0, 255), -1)
    cv2.imwrite("final/" + image, im)
