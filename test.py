from estimate_pose import estimate_motion, match_features, extract_features, visualize_features, filter_matches_distance, get_pose
import cv2
import numpy as np
from matplotlib import pyplot as plt
from m2bk import *
import sys
import argparse
import open3d as o3d
import time

np.random.seed(1)
dataset_handler = DatasetHandler(50)

trajectory = np.zeros((3, dataset_handler.num_frames))

# Initialize camera pose
robot_pose = np.eye(4)
for i in range(dataset_handler.num_frames-1):
    image1 = dataset_handler.images_rgb[i]
    depth1 = dataset_handler.depth_maps[i]

    image2 = dataset_handler.images_rgb[i+1]

    rmat, tvec = get_pose(image1, image2, depth1, dataset_handler.k, display=False)
    current_pose = np.eye(4)
    current_pose[0:3, 0:3] = rmat
    current_pose[0:3, 3] = tvec.T
    # Build the robot's pose from the initial position by multiplying previous and current poses
    robot_pose = robot_pose @ np.linalg.inv(current_pose)
    # Calculate current camera position from origin
    position = robot_pose @ np.array([0., 0., 0., 1.])
    print(position)
    trajectory[:, i + 1] = position[0:3]
print(np.array(trajectory).shape)
visualize_trajectory(np.array(trajectory))
