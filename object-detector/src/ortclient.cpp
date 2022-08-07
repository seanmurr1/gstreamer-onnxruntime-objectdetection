#include <fstream>
#include "ortclient.h"
#include "yolov4.h"

#include <providers/cpu/cpu_provider_factory.h>
#ifdef GST_ML_ONNX_RUNTIME_HAVE_CUDA
#include <providers/cuda/cuda_provider_factory.h>
#endif

OrtClient::OrtClient() {
    try {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    } catch (Ort::Exception& e) {
        GST_ERROR ("%s\n", e.what());
    }
}

// TODO: use GetInputNameAllocated to prevent needed to free node names
OrtClient::~OrtClient() {
    for (const char* node_name : input_node_names) {
        allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));
    }
    for (const char* node_name : output_node_names) {
        allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));
    }
}

bool OrtClient::IsInitialized() {
    return is_init;
}

/**
 * @brief Set up ONNX Runtime environment, session options, and create new session.
 * 
 * @param opti_level ORT optimization level.
 * @param provider ORT execution provider.
 * @return true if setup succeeded.
 * @return false if setup failed.
 */
bool OrtClient::CreateSession(GstOrtOptimizationLevel opti_level, GstOrtExecutionProvider provider, int device_id) {
    try {
        switch (opti_level) {
            case GST_ORT_OPTIMIZATION_LEVEL_DISABLE_ALL:
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
                break;
            case GST_ORT_OPTIMIZATION_LEVEL_ENABLE_BASIC:
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
                break;
            case GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED:
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
                break;
            case GST_ORT_OPTIMIZATION_LEVEL_ENABLE_ALL:
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                break;
            default:
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
                break;
        }
        switch (provider) {
            case GST_ORT_EXECUTION_PROVIDER_CUDA:
#ifdef GST_ML_ONNX_RUNTIME_HAVE_CUDA
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id));
#else 
                GST_ERROR ("Unable to setup CUDA execution provider!");
                return false;
#endif
                break;
            case GST_ORT_EXECUTION_PROVIDER_CPU:
                break;
            default:
                break;
        }
        session = Ort::Session(env, onnx_model_path.c_str(), session_options);
        return true;
    } catch (Ort::Exception& e) {
        GST_ERROR ("%s\n", e.what());
        return false;
    }
}

/**
 * @brief Parses ONNX model input/output information.
 */
bool OrtClient::SetModelInputOutput() {
    try {
        num_input_nodes = session.GetInputCount();
        input_node_names = std::vector<const char*>(num_input_nodes);
        input_node_dims = std::vector<std::vector<int64_t>>(num_input_nodes);
        num_output_nodes = session.GetOutputCount();
        output_node_names = std::vector<const char*>(num_output_nodes);
        output_node_dims = std::vector<std::vector<int64_t>>(num_output_nodes);
        // Input nodes
        for (size_t i = 0; i < num_input_nodes; i++) {
            char *input_name = session.GetInputName(i, allocator);
            input_node_names[i] = input_name;
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_node_dims[i] = tensor_info.GetShape();
        }
        // Object detection model should only take in one input node (e.g. an image)
        assert(input_node_dims.size() == 1);
        // Fix variable batch size (batch size of -1), as we process a single frame at a time
        if (input_node_dims[0][0] == -1) {
            input_node_dims[0][0] = 1;
        }
        // Output nodes
        for (size_t i = 0; i < num_output_nodes; i++) {
            char *output_name = session.GetOutputName(i, allocator);
            output_node_names[i] = output_name;
            Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            output_node_dims[i] = tensor_info.GetShape();
        }
        return true;
    } catch (Ort::Exception& e) {
        GST_ERROR ("%s\n", e.what());
        return false;
    }
}

/**
 * @brief Loads class labels from label file.
 * 
 * @return true if class labels are properly formatted.
 * @return false if label file was malformed.
 */
bool OrtClient::LoadClassLabels() {
    size_t num_classes = model->GetNumClasses();
    labels = std::vector<std::string>(num_classes);
    std::ifstream input(class_labels_path);
    if (!input.good()) {
        GST_ERROR ("Unable to open label file!");
        return false;
    }
    std::string line;
    
    for (size_t i = 0; i < num_classes; i++) {
        if (!getline(input, line)) {
            GST_ERROR ("Malformed label file!");
            return false;
        }
        labels[i] = line;
    }
    return true;
}

/**
 * @brief Initializes ORT client for object detection.
 * Creates ORT environment, session, parses input/output, etc.
 * 
 * @param model_path path to model file.
 * @param label_path path to class labels file.
 * @param opti_level ORT optimization level.
 * @param provider ORT execution provider.
 * @param detection_model object detection model to use.
 * @param device_id device ID for hardware acceleration.
 * @return true if setup was successful.
 * @return false if setup failed.
 */
bool OrtClient::Init(std::string const& model_path, std::string const& label_path, GstOrtOptimizationLevel opti_level, GstOrtExecutionProvider provider, GstOrtDetectionModel detection_model, int device_id) {
    onnx_model_path = model_path;
    class_labels_path = label_path;
    // Setup object detection model
    switch (detection_model) {
        case GST_ORT_DETECTION_MODEL_YOLOV4:
            model = std::unique_ptr<ObjectDetectionModel>(new YOLOv4());
            break;
        default: 
            // Default model is YOLOv4
            model = std::unique_ptr<ObjectDetectionModel>(new YOLOv4());
            break;
    }
    input_tensor_size = model->GetInputTensorSize();
    // Set up internal tensor value vector (acts as a cache)
    input_tensor_values = std::vector<float>(input_tensor_size);
    if (!CreateSession(opti_level, provider, device_id) || !SetModelInputOutput() || !LoadClassLabels()) {
        is_init = false;
        return false;
    }
    is_init = true;
    return true;
}

/**
 * @brief Runs object detection model on input data.
 * Input data is modified in-place.
 * 
 * @param data input image data.
 * @param width image width.
 * @param height image height.
 * @param rgb is image RGB or BGR format.
 * @param score_threshold score threshold when filtering bounding boxes.
 * @param nms_threshold threshold for non-maximal suppression and IOU.
 */
void OrtClient::RunModel(uint8_t *const data, int width, int height, bool is_rgb, float score_threshold, float nms_threshold) {
    if (!is_init) {
        GST_ERROR ("Unable to run inference when ORT client has not been initialized!");
        return;
    }
    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        model->Preprocess(data, input_tensor_values, width, height, is_rgb);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims[0].data(), input_node_dims[0].size());
        assert(input_tensor.IsTensor());
        std::vector<Ort::Value> model_output = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, num_input_nodes, output_node_names.data(), num_output_nodes);
        model->Postprocess(model_output, labels, score_threshold, nms_threshold);
    } catch (Ort::Exception& e) {
        GST_ERROR ("%s\n", e.what());
    }
}

/**
 * @brief Runs object detection model on input data.
 * Input data is modified in-place.
 * 
 * @param data input image data.
 * @param vmeta GST video meta.
 * @param score_threshold score threshold when filtering bounding boxes.
 * @param nms_threshold threshold for non-maximal suppression and IOU.
 */
void OrtClient::RunModel(uint8_t *const data, GstVideoMeta *vmeta, float score_threshold, float nms_threshold) {
    switch (vmeta->format) {
        case GST_VIDEO_FORMAT_RGB:
            RunModel(data, vmeta->width, vmeta->height, true, score_threshold, nms_threshold);
            break;
        case GST_VIDEO_FORMAT_BGR:
            RunModel(data, vmeta->width, vmeta->height, false, score_threshold, nms_threshold);
            break; 
        default:
            GST_ERROR ("Unable to recognize color format!");
            break;
    }
}