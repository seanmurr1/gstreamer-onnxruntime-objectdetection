#ifndef ORTCLIENT_H
#define ORTCLIENT_H

#include <onnxruntime_cxx_api.h>
#include "objectdetectionmodel.h"

class OrtClient {
    private:
        ObjectDetectionModel* model;
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
        //Ort::MemoryInfo memory_info;
        std::unique_ptr<Ort::Session> session;
        std::unique_ptr<Ort::Env> env;

        bool check_init;

        bool loadClassLabels();
        void setOrtEnv();
        void setModelInputOutput();
        void createSession(); // TODO add optimization level and exeuction provider

    public:
        ~OrtClient();
        bool init(std::string model_path, std::string label_path);
        bool isInitialized();
        uint8_t* runModel(uint8_t *data, int width, int height);
};

#endif