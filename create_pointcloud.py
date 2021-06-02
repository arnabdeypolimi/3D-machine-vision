import numpy as np
import cv2
import open3d as o3d
import argparse
import copy
from detectron import Detectron


def filter_pointcloud(pointcloud, distance):
    mask = np.asarray(pointcloud.points)[:, 2] < distance
    new_pts = np.asarray(pointcloud.points)[mask]
    new_color = np.asarray(pointcloud.colors)[mask]
    pointcloud.points = o3d.utility.Vector3dVector(new_pts)
    pointcloud.colors = o3d.utility.Vector3dVector(new_color)
    return pointcloud

def filter_mask(image, depth, mask):
    #TODO
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
        self.det = Detectron()

    def get_segmentation(self, rgb_image, label = "person"):
        """
        generate segmentation mask from rgb frame using Detectron2
        rgb_image: rgb_image frame
        return: segmentation mask
        """
        seg_image = self.det.predict(rgb_image, label)
        return seg_image

    def create_point_cloud(self):
        while True:
            color_frame, depth_frame = self.cam.get_frame()
            # display color image
            color_image = cv2.cvtColor(np.asarray(color_frame).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow('bgr', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        #segmentation
        depth_image = np.asanyarray(np.asarray(depth_frame))
        color_image = np.asanyarray(np.asarray(color_frame))

        seg_image = self.get_segmentation(color_image)
        color_image, depth_image = #TODO filter using the instance segmentation

        img_depth = o3d.geometry.Image(depth_image)
        img_color = o3d.geometry.Image(color_image)


        # generate RGBD image by combining depth and rgb frames
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False,
                                                              depth_trunc=15.0)

        # generate point cloud from RGBD
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.cam.intrinsic)

        #pcd_filtred = filter_pointcloud(copy.copy(pcd), 1)
        o3d.visualization.draw_geometries([pcd])
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # create object of point cloud class
    parser = argparse.ArgumentParser(description='Azure kinect display RGBD image')
    parser.add_argument('--device_config_path', type=str, help='input json kinect config')
    parser.add_argument('--output_file', type=str, default="test.ply", help='The path of output point cloud file')
    opt = parser.parse_args()

    pc = Create_point_cloud(opt)
    pc.create_point_cloud()

