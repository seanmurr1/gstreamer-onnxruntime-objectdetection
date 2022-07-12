#include <fstream>
#include "ortclient.h"
#include "yolov4.h"

OrtClient::~OrtClient() {
    for (const char* node_name : input_node_names) {
        allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));
    }
    for (const char* node_name : output_node_names) {
        allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));
    }
    
    delete model;
}

bool OrtClient::isInitialized() {
    return check_init;
}

/**
 * @brief Set up ONNX Runtime environment and set session options (e.g. graph optimaztion level).
 */
void OrtClient::setOrtEnv() {
    env = std::make_unique<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));

    // TODO: add customizability
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

/**
 * @brief Parses ONNX model input/output information.
 */
void OrtClient::setModelInputOutput() {
    num_input_nodes = session->GetInputCount();
    input_node_names = std::vector<const char*>(num_input_nodes);
    input_node_dims = std::vector<std::vector<int64_t>>(num_input_nodes);

    num_output_nodes = session->GetOutputCount();
    output_node_names = std::vector<const char*>(num_output_nodes);
    output_node_dims = std::vector<std::vector<int64_t>>(num_output_nodes);

    // Input nodes
    for (size_t i = 0; i < num_input_nodes; i++) {
        char *input_name = session->GetInputName(i, allocator);
        input_node_names[i] = input_name;
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_dims[i] = tensor_info.GetShape();
    }
    // Object detection model should only take in one input node
    assert(input_node_dims.size() == 1);
    // Fix variable batch size, as we process a single frame at a time
    if (input_node_dims[0][0] == -1) {
        input_node_dims[0][0] = 1;
    }
    // Output nodes
    for (size_t i = 0; i < num_output_nodes; i++) {
        char *output_name = session->GetOutputName(i, allocator);
        output_node_names[i] = output_name;
        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        output_node_dims[i] = tensor_info.GetShape();
    }
}

/**
 * @brief Creates ONNX Runtime session from existing environment.
 * Uses environment and session options set up in `setOrtEnv`.
 */
void OrtClient::createSession() {
    session = std::make_unique<Ort::Session>(Ort::Session(*env, onnx_model_path.c_str(), session_options));
}

bool OrtClient::loadClassLabels() {
    size_t num_classes = model->getNumClasses();
    labels = std::vector<std::string>(num_classes);
    std::ifstream input(class_labels_path);
    std::string line;
    
    for (size_t i = 0; i < num_classes; i++) {
        if (!getline(input, line)) {
            // Malformed label file
            return false;
        }
        labels[i] = line;
    }
    return true;
}

/**
 * @brief Initializes YOLOv4 for object detection. 
 * Creates ONNX Runtime environment, session, parses input/output, etc.
 * 
 * @return true if setup was successful.
 * @return false if setup failed.
 */
bool OrtClient::init(std::string model_path, std::string label_path) {
    onnx_model_path = model_path;
    class_labels_path = label_path;
    // TODO: add ability to change model
    model = new YOLOv4();
    setOrtEnv();
    createSession();
    setModelInputOutput();
    //memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensor_size = model->getInputTensorSize();

    if (!loadClassLabels()) {
        std::cout << "Malformed label file" << std::endl;
        return false;
        // TODO: cleanup here?
    }

    check_init = true;
    return true;
}

/**
 * @brief Runs object detection model on input data.
 */
uint8_t* OrtClient::runModel(uint8_t *data, int width, int height) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> &input_tensor_values = model->preprocess(data, width, height);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims[0].data(), input_node_dims[0].size());
    assert(input_tensor.IsTensor());

    std::vector<Ort::Value> model_output = session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, num_input_nodes, output_node_names.data(), num_output_nodes);
    //input_tensor.release(); // ?
    uint8_t *processed_data = model->postprocess(model_output, labels);
    for (size_t i = 0; i < model_output.size(); i++) {
        //model_output[i].release();
    }
    return processed_data;
}