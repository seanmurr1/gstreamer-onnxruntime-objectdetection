#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/ortclient.h"

/**
 * Sample driver program to test ORT functionality without using the full plugin.
 * 
 * Usage: ./ort-driver <path to ONNX model file> <path to label file for model> <path to input image> <path to desired output image location> <optional execution provider>
 * where the execution provider may be CPU or CUDA (default is CPU). 
 * 
 * Most common image formats should work, e.g. PNG, JPG, etc. as long as OpenCV supports it.
 * 
 * Currently, this driver only uses YOLOv4. However, as more object detection algorithms are implemented,
 * this will be a configurable CL arg as well.
 * 
 * Object detection will be run on input image, and the output image will essentially be a copy of the 
 * input image with bounding box information and accuracy scores written to it.
 */
int main(int argc, char* argv[]) {
  if (argc != 5 && argc != 6) {
    std::cout << "Usage: " << argv[0]  << " <model-file> <label-file> <input-image> <output-location> <execution-provider>" << std::endl;
    std::cout << "Note: <execution-provider> is optional and defaults to CPU. Options are CPU, CUDA" << std::endl;
    return -1;
  }
  std::string model_path = argv[1];
  std::string label_path = argv[2];
  OrtClient ort_client;
  bool res;
  if (argc == 6) {
    std::string exec_provider = argv[5];
    if (exec_provider == "CPU") {
      res = ort_client.Init(model_path, label_path, GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED, GST_ORT_EXECUTION_PROVIDER_CPU);
    } else if (exec_provider == "CUDA") {
      res = ort_client.Init(model_path, label_path, GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED, GST_ORT_EXECUTION_PROVIDER_CUDA);
    } else {
      std::cout << "Unable to recognize execution provider!" << std::endl;
      return -1;
    }
  } else {
    res = ort_client.Init(model_path, label_path, GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED, GST_ORT_EXECUTION_PROVIDER_CPU);
  }
  assert(res);
  cv::Mat input_image = cv::imread(argv[3]);
  // Imread reads in BGR format
  ort_client.RunModel(input_image.data, input_image.cols, input_image.rows, false);
  cv::imwrite(argv[4], input_image);
  return 0;
}