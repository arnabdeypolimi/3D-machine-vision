import argparse
import open3d as o3d


class ViewerWithCallback:
    def __init__(self, config, device, align_depth_to_color):
        """
        create object of this class
        :param config: config
        :param device: device id
        :param align_depth_to_color: depth and color frame alignment(True or False)
        """
        self.flag_exit = False
        self.align_depth_to_color = align_depth_to_color

        self.sensor = o3d.io.AzureKinectSensor(config)
        if not self.sensor.connect(device):
            raise RuntimeError('Failed to connect to sensor')

    # escape callback
    def escape_callback(self, vis):
        self.flag_exit = True
        return False

    # run rgbd visualizer
    def run(self):
        glfw_key_escape = 256
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(glfw_key_escape, self.escape_callback)
        vis.create_window('viewer', 1920, 540)
        print("Sensor initialized. Press [ESC] to exit.")

        vis_geometry_added = False
        while not self.flag_exit:
            ##TODO
            #capture aligned rgbd frame



            if rgbd is None:
                continue

            # add first rgbd frame in visualizer
            if not vis_geometry_added:
                vis.add_geometry(rgbd)
                vis_geometry_added = True

            ##TODO
            # update the visualizer with current rgbd frame


            vis.poll_events()
            # update the renderer
            vis.update_renderer()


## starting point of the program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Azure kinect display RGBD image')
    parser.add_argument('--config', type=str, help='input json kinect config')
    parser.add_argument('-a',
                        '--align_depth_to_color',
                        action='store_true',
                        help='enable align depth image to color')
    args = parser.parse_args()

    # device id
    device = 0

    # creating config
    if args.config is not None:
        config = o3d.io.read_azure_kinect_sensor_config(args.config)
    else:
        config = o3d.io.AzureKinectSensorConfig()

    ##TODO
    # create object of ViewerWithCallback class


    ##TODO
    # run the viewer using the run method