## ALexnet_ROS_Cpp
This code use AlexNet on Cudnn-GPU, ROS C++ using Nvidia Xavier

+ Step 1: Download AlexNet bin file from [here](https://www.mediafire.com/file/x2qw5q7ldu0z573/lib.zip/file)
+ Step 2: Then extract and copy to src/codegen/lib

+ Step 3: catkin_make
+ Step 4: run camera_node, here using zed 2 camera
+ Run : rosrun alex_net_gpu alex_detector
+ cmd: rqt_image_view, and select image topic: /Alex_detector

Enjoin