from estimate_pose import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from m2bk import *
import sys
import argparse
import open3d as o3d
import time

np.random.seed(1)
dataset_handler = DatasetHandler(2)
image1 = dataset_handler.images_rgb[0]
depth1 = dataset_handler.depth_maps[0]
image2 = dataset_handler.images_rgb[1]
#TODO get the pose matrices using two RGBD images

#TODO print homogenous pose matrix (4x4)
