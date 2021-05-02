import numpy as np
import cv2
import open3d as o3d
import sys, getopt

class AzureKinect:
    """
    class for azure kinect related functions
    """
    def __init__(self):
        ##TODO
        #self.config = <read azure kinect config file>

        self.device = 0
        self.align_depth_to_color = 1

    def start(self):
        self.sensor = o3d.io.AzureKinectSensor(self.config)
        if not self.sensor.connect(self.device):
            raise RuntimeError('Failed to connect to sensor')

    def frames(self):
        while 1:
            rgbd = self.sensor.capture_frame(self.align_depth_to_color)
            if rgbd is None:
                continue
            color, depth = np.asarray(rgbd.color).astype(np.uint8),np.asarray(rgbd.depth).astype(np.float32)
            return color, depth

class Create_point_cloud():
    """
    Class contain functions to generate and save point clouds
    """
    def __init__(self):
        self.count=0
        self.timer=3
        self.outputfile=''
        self.cam = AzureKinect()
        self.cam.start()


    def create_point_cloud(self, argv):
        """
        this function generate a ply file containing point clouds
        """
        ## parsing command line arguments
        try:
            opts, args = getopt.getopt(argv, "pointcloud:o:t:d:", ["ofile=", "timer=", "device="])
        except getopt.GetoptError:
            print('python save_pointcloud.py -o <outputfile> -t <timer> -d <device>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('test.py -o <outputfile> -t <timer> -d <device>')
                sys.exit()
            elif opt in ("-o", "--output_file"):
                self.outputfile = arg
            elif opt in ("-t", "--timer"):
                self.timer = int(arg)
            elif opt in ("-d", "--device"):
                self.device = arg


        while self.count<self.timer:
            self.count+=1


            color_frame, depth_frame = self.cam.frames()

            depth_image=depth_frame.astype(np.uint16)
            color_image=color_frame
            # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            img_depth = o3d.geometry.Image(depth_image)
            img_color = o3d.geometry.Image(color_image)

            # generate RGBD image by combining depth and rgb frames
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False,
                                                                      depth_trunc=15.0)

            # default camera intrinsic parameters for azure kinect sensor
            intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, 608.9779052734375, 608.77679443359375,
                                                          636.66748046875,
                                                          367.427490234375)

            ##TODO
            #generate point cloud from RGBD image using Open3d library
            pcd =


            # transform the point cloud
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            #display color image
            cv2.imshow('bgr', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        ##TODO
        #write pointcloud as ply file using open3d


        cv2.destroyAllWindows()


# starting point of the program
if __name__=="__main__":
    # create object of point cloud class
    pc=Create_point_cloud()
    pc.create_point_cloud(sys.argv[1:])
