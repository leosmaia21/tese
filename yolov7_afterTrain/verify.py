import cv2
import os
import glob
import re

resolution = 640
first = True
for folder in next(os.walk('.'))[1]:
    for image in glob.glob(folder+ "/*.jpg"):
        im = cv2.imread(image)
        str_found = image.split('/')[1].split('.')[0]
        label = open(folder + "/labels/"+ str_found+ ".txt")
        label_data = label.readlines()
        size = int(folder.split('_')[0])
        for line in label_data:
            info = line.split()
            start = (int(resolution * float(info[1]) - size), int(resolution * float(info[2]) - size))
            end = (int(resolution * float(info[1]) + size), int(resolution * float(info[2]) + size))
            im = cv2.rectangle(im, start, end, (0,0,255), 2)
        cv2.imwrite(image,im)

