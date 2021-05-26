from estimate_pose import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from m2bk import *
import sys
import argparse
import time

np.random.seed(1)
#TODO Create dataset handeler with 50 frames
dataset_handler = #

# Initialize trajectory and the initial pose
trajectory = np.zeros((3, dataset_handler.num_frames))
robot_pose = np.eye(4)
for i in range(dataset_handler.num_frames-1):
    """
    TODO:
    1- Loop over all the dataset: 
        1.1- Load consecutifs RGBD images
        1.2- Get the pose 
        1.3- Get the current position of the robot (vehicle)  hint: the  4th column of the robot pose  
        1.4- store the position of the robot in trajectory
    """
# TODO display the trajectory of the robot using visualize_trajectory