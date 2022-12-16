import sys

import torch
import cv2
import time
sys.path.append("yolov7")

from yolov7.utils import letterbox

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('best6.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()