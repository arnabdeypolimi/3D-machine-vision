from estimate_pose import estimate_motion, match_features, extract_features, visualize_features, filter_matches_distance, get_pose
import cv2
import numpy as np
from matplotlib import pyplot as plt
from m2bk import *
import sys
import argparse
import open3d as o3d
import time

class AzureKinect():
    """
    class for azure kinect related functions
    """

    def __init__(self, device_config_path):
        self.config = o3d.io.read_azure_kinect_sensor_config("/home/houssem/PhD/codes/3D-machine-vision/default_config.json")
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, 608.9779052734375, 608.77679443359375,
                                                           636.66748046875,
                                                           367.427490234375)
        self.device = 0
        self.align_depth_to_color = 1

    def start(self):
        self.sensor = o3d.io.AzureKinectSensor(self.config)
        if not self.sensor.connect(self.device):
            raise RuntimeError('Failed to connect to sensor')

    def get_frame(self, wait= True):
        while 1:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue
            if wait:
                color_image = cv2.cvtColor(np.asarray(rgbd.color).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imshow('bgr', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            rgbdtopcd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbd.color, rgbd.depth,
                                                                      convert_rgb_to_intensity=False,
                                                                      depth_trunc=15.0)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbdtopcd, intrinsic=self.intrinsic)

        return cv2.cvtColor(np.asarray(rgbd.color).astype(np.uint8), cv2.COLOR_RGB2BGR), np.asarray(rgbd.depth), pcd

if __name__ == "__main__":
    # create object of point cloud class
    parser = argparse.ArgumentParser(description='Azure kinect display RGBD image')
    parser.add_argument('--device_config_path', default="./default_config.json", type=str, help='input json kinect config')
    opt = parser.parse_args()

    sensor = AzureKinect(opt)
    sensor.start()
    trajectory = []
    pointclouds = []
    for i in range(60):
        start = time.time()
        image1, depth1, pcd1 = sensor.get_frame()
        image2, depth2, pcd2 = sensor.get_frame()
        robot_pose = np.eye(4)
        #try:
        rmat, tvec =  get_pose(image1, image2, depth1, sensor.intrinsic.intrinsic_matrix)
        """
        except:
            print("skpied")
            rmat = np.eye(3)
            tvec = np.array([0, 0, 0])
        """
        current_pose = np.eye(4)
        current_pose[0:3, 0:3] = rmat
        current_pose[0:3, 3] = tvec.T *0.001
        pcd2 = pcd2.transform(current_pose)
        pcd3 = pcd1 + pcd2
        o3d.visualization.draw_geometries([pcd1])
        o3d.visualization.draw_geometries([pcd2])
        o3d.visualization.draw_geometries([pcd3])
        # Build the robot's pose from the initial position by multiplying previous and current poses
        robot_pose= robot_pose @ np.linalg.inv(current_pose)
        # Calculate current camera position from origin
        position = robot_pose @ np.array([0., 0., 0., 1.])
        trajectory.append(position)

    plt.plot(trajectory[:][0], trajectory[:][1])
    plt.gca().set_aspect("equal")
    plt.show()

