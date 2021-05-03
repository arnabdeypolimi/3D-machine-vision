# 3D-machine-vision
The companion github repository of 3D machine vision course (Polytech Nice Sophia University, France)
##Install d
``` 
sudo apt install curl git 
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
sudo apt-get update 

sudo apt install libk4a1.4*
sudo apt install k4a-tools
sudo cp 99-k4a.rules /etc/udev/rules.d/
```
##Test kinect 
```
k4aviewer
```
##ROS installation
Run roscore and check the version 
```
roscore
```
If there ROS is not installed run the following command lines

```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop-full
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo apt install python-rosdep
sudo rosdep init
rosdep update
```
##Setup local ROS workspace 

```
cd ~
mdkir catkin_ws/src -p 
cd catkin_ws/src
git clone https://github.com/microsoft/Azure_Kinect_ROS_Driver.git
cd ..
catkin_make --force-cmake
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
##Test azure_kinect_ros : 
```
roslaunch azure_kinect_ros_driver driver.launch
rosrun rviz rviz
```
add PointCloud2 and chose the right topic and modify the topic to display the point cloud









