#ifndef __ORT_CLIENT_H__
#define __ORT_CLIENT_H__

#include <onnxruntime_cxx_api.h>
#include "objectdetectionmodel.h"
#include "gstortelement.h"

class OrtClient {
    private:
        std::unique_ptr<ObjectDetectionModel> model;
        std::string onnx_model_path;
        std::string class_labels_path;
        std::vector<std::string> labels;

        size_t input_tensor_size;
        size_t num_input_nodes;
        std::vector<const char*> input_node_names;
        std::vector<std::vector<int64_t>> input_node_dims;
        size_t num_output_nodes;
        std::vector<const char*> output_node_names;
        std::vector<std::vector<int64_t>> output_node_dims;

        Ort::SessionOptions session_options;
        Ort::AllocatorWithDefaultOptions allocator;
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::Env> env;

        bool is_init;

        bool loadClassLabels();
        void setModelInputOutput();
        bool createSession(GstOrtOptimizationLevel opti_level, GstOrtExecutionProvider provider); 

    public:
        OrtClient() = default;
        ~OrtClient();
        bool init(std::string const& model_path, std::string const& label_path, GstOrtOptimizationLevel = GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED, GstOrtExecutionProvider = GST_ORT_EXECUTION_PROVIDER_CPU, GstOrtDetectionModel = GST_ORT_DETECTION_MODEL_YOLOV4);
        bool isInitialized();
        void runModel(uint8_t *const data, int width, int height, bool is_rgb, float = 0.25, float = 0.213);
        void runModel(uint8_t *const data, GstVideoMeta *vmeta, float = 0.25, float = 0.213);
};

#endif