import numpy as np
import cv2
import open3d as o3d
import argparse
from scipy.linalg import expm
import copy


def skew(vect):
    # TODO
    return np.eye(3)

def SE3_exp(linear_velocity, angular_velocity):
    # TODO
    return np.eye(4)


def transform_pointcloud(pointcloud, pose_T):
    #TODO
    return pointcloud

def filter_pointcloud(pointcloud, distance):
    # TODO
    return pointcloud

class AzureKinect():
    """
    class for azure kinect related functions
    """

    def __init__(self, device_config_path):
        self.config = o3d.io.read_azure_kinect_sensor_config(device_config_path)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, 608.9779052734375, 608.77679443359375,
                                                           636.66748046875,
                                                           367.427490234375)
        self.device = 0
        self.align_depth_to_color = 1

    def start(self):
        self.sensor = o3d.io.AzureKinectSensor(self.config)
        if not self.sensor.connect(self.device):
            raise RuntimeError('Failed to connect to sensor')

    def get_frame(self):
        while 1:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue
            return rgbd.color, rgbd.depth


class Create_point_cloud():
    """
    Class contain functions to generate and save point clouds
    """

    def __init__(self, opt):
        self.output_file = opt.output_file
        self.cam = AzureKinect(opt.device_config_path)
        self.cam.start()

    def create_point_cloud(self):
        while True:
            color_frame, depth_frame = self.cam.get_frame()
            # display color image
            color_image = cv2.cvtColor(np.asarray(color_frame).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow('bgr', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # generate RGBD image by combining depth and rgb frames
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_frame, depth_frame,
                                                                  convert_rgb_to_intensity=False,
                                                                  depth_trunc=15.0)

        # generate point cloud from RGBD and write it
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.cam.intrinsic)
        o3d.io.write_point_cloud(self.output_file, pcd)

        pcd_filtred = filter_pointcloud(copy.copy(pcd), 1.0)
        new_filename = self.output_file.split(".")[0] + "_filtred." + self.output_file.split(".")[1]
        o3d.io.write_point_cloud(new_filename, pcd_filtred)

        # TODO : define properly the twist vector
        transform = SE3_exp([0, 0, 0], [0, 0, 0])
        pcd_trasnformed = transform_pointcloud(copy.copy(pcd), transform)

        new_filename = self.output_file.split(".")[0] + "_transformed." + self.output_file.split(".")[1]
        o3d.io.write_point_cloud(new_filename, pcd_trasnformed)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # create object of point cloud class
    parser = argparse.ArgumentParser(description='Azure kinect display RGBD image')
    parser.add_argument('--device_config_path', type=str, help='input json kinect config')
    parser.add_argument('--output_file', type=str, help='The path of output point cloud file')
    opt = parser.parse_args()

    pc = Create_point_cloud(opt)
    pc.create_point_cloud()
