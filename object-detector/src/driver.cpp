#include <iostream>
#include <opencv2/opencv.hpp>
#include "ortclient.h"

int main(int argc, char* argv[]) {
    OrtClient ort_client("../../assets/models/yolov4/yolov4.onnx", "../../assets/models/yolov4/labels.txt");
    auto res = ort_client.init();
    std::cout << (bool) res << std::endl;
    
    cv::Mat input_image = cv::imread("../../assets/images/bike1.png");
    std::vector<int> dims{input_image.rows, input_image.cols};
    cv::Mat trans = cv::Mat(dims, CV_8UC3, input_image.data);
    cv::imwrite("../../assets/images/testconvert.png", trans);

    auto output = ort_client.runModel(input_image.data, input_image.cols, input_image.rows);

    std::cout << "Done" << std::endl;
    return 0;
}