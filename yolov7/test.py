import sys

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import time
sys.path.append("yolov7")

from torchvision import transforms
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint, plot_skeleton_kpts

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weigths = torch.load('best6.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()