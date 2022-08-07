#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/ortclient.h"

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