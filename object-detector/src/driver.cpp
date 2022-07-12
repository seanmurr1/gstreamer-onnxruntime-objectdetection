#include <iostream>
#include <opencv2/opencv.hpp>
#include "ortclient.h"

int main(int argc, char* argv[]) {
    std::string model_path = "../../assets/models/yolov4/yolov4.onnx";
    std::string label_path = "../../assets/models/yolov4/labels.txt";

    OrtClient ort_client;
    auto res = ort_client.init(model_path, label_path);
    
    cv::Mat input_image = cv::imread("../../assets/images/bike1.png");
    cv::Mat input_image2 = cv::imread("../../assets/images/dog.jpeg");
    cv::Mat input_image3 = cv::imread("../../assets/images/dog2.jpg");

    auto output = ort_client.runModel(input_image.data, input_image.cols, input_image.rows);
    auto output2 = ort_client.runModel(input_image2.data, input_image2.cols, input_image2.rows);
    auto output3 = ort_client.runModel(input_image3.data, input_image3.cols, input_image3.rows);

    return 0;
}