//
// Created by dinhnambkhn on 21. 12. 10..
//
#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32MultiArray.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include "myAlexNetGPU.h"
#include "myAlexNetGPU_types.h"

#include <chrono>
#include <fstream>

// Declare the output data (defined in codegen/lib/line_detect/examples/main.cu)
int I_size[3];
ros::Publisher img_pub;

std::vector<std::string> class_vectors;

//Callback for new image message data
void callbackImg(const sensor_msgs::Image::ConstPtr& inmsg)
{
    //Declare the input data
    cv::Mat in_image = cv_bridge::toCvShare(inmsg, sensor_msgs::image_encodings::BGR8)->image;

    int width = in_image.cols;
    int height = in_image.rows;
    I_size[0] = 3;
    I_size[1] = width;
    I_size[2] = height;

    // Call the entry-point 'line_detection'.
    //line_detection(in_image.data, I_size, predictedPosNorm, Iout_data, Iout_size);
    //show the image size
    std::cout << "Image size: " << width << "x" << height << std::endl;
    //check img empty
    if(in_image.empty())
    {
        std::cout << "Image empty" << std::endl;
    }
    else
    {
        //begin counting time
        auto start = std::chrono::high_resolution_clock::now();

        //show the image
        auto img_index = myAlexNetGPU(in_image.data, I_size);
        //std::cout << "img_index "    << img_index << std::endl;

        auto class_name = class_vectors[int(img_index)];

        ROS_INFO("img_index %d and name %s", int(img_index), class_name.c_str());
        //end counting time
        auto finish = std::chrono::high_resolution_clock::now();
        //time in milliseconds
        std::chrono::duration<double, std::milli> elapsed = finish - start;
        std::cout << "Time: " << elapsed.count() << " ms" << std::endl;

        //create the output image
        cv::Mat out_image(height, width, CV_8UC3, cv::Scalar(0,0,0));
        //copy the input image to output image
        in_image.copyTo(out_image(cv::Rect(0,0,width,height)));
        //write string to image
        cv::putText(out_image, std::to_string(img_index), cv::Point(10,50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255), 2);
        //write time to image red color
        cv::putText(out_image, std::to_string(elapsed.count()) + " ms", cv::Point(20,100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);

        //write class name to image green color
        cv::putText(out_image, class_name, cv::Point(20,150), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0,255,0), 2);


        //publish the image
        sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_image).toImageMsg();
        img_pub.publish(out_msg);
        //show the image
        cv::imshow("Image", out_image);

    }
    //ROS_INFO

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "gpunet");
    ros::NodeHandle nh;
    ros::Rate loop_rate(10);

    //read myDataFile.csv
    class_vectors.reserve(1001);

    std::ifstream myfile("/home/dinhnambkhn/GPU_coder_cpp/Code_AlexNet/src/alex_net_gpu/src/myDataFile.csv");
    std::string line;
    std::string delimiter = ",";
    std::string token;
    int i = 0;
    while (std::getline(myfile, line))
    {
        std::istringstream iss(line);
        while (std::getline(iss, token, delimiter[0]))
        {
            class_vectors.push_back(token);
        }
        i++;
    }
    myfile.close();

    //print the data
    for(int p = 0; p < 10; p++)
    {
        std::cout << class_vectors[p] << std::endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////////

    //ROS_INFO "Loading network model..."
    ROS_INFO("Loading network model...");
    // Initialization
    cv::Mat in_image = cv::Mat::zeros(cv::Size(320,180), CV_8UC3);
    I_size[0] = 3;
    I_size[1] = 227;
    I_size[2] = 227;

    auto img_index = myAlexNetGPU(in_image.data, I_size);
    std::cout << "img_index "    << img_index << std::endl;

    //subscriber image topic
    ros::Subscriber sub = nh.subscribe("/zed2/zed_node/stereo/image_rect_color", 1, &callbackImg);
    ROS_INFO("Node Started Successfully");

    std::string img_topic_name = "Alex_detector";
    img_pub = nh.advertise<sensor_msgs::Image>(img_topic_name, 1);


    ros::spin();
    return 0;
}
