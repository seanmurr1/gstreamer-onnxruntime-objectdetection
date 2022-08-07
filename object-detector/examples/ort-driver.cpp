#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/ortclient.h"

int main(int argc, char* argv[]) {
    std::string model_path = "../../assets/models/yolov4/yolov4.onnx";
    std::string label_path = "../../assets/models/yolov4/labels.txt";

    OrtClient ort_client;
    auto res = ort_client.Init(model_path, label_path, GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED, GST_ORT_EXECUTION_PROVIDER_CPU);
    assert(res);

    cv::Mat input_image = cv::imread("../../assets/images/bike1.png");
    cv::Mat input_image2 = cv::imread("../../assets/images/dog.jpeg");
    cv::Mat input_image3 = cv::imread("../../assets/images/dog2.jpg");

    // Imread reads in BGR format
    ort_client.RunModel(input_image.data, input_image.cols, input_image.rows, false);
    ort_client.RunModel(input_image2.data, input_image2.cols, input_image2.rows, false);
    ort_client.RunModel(input_image3.data, input_image3.cols, input_image3.rows, false);

    cv::imwrite("../../assets/images/output1.jpeg", input_image);
    cv::imwrite("../../assets/images/output2.jpeg", input_image2);
    cv::imwrite("../../assets/images/output3.jpeg", input_image3);

    return 0;
}