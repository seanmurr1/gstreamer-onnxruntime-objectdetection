#ifndef __OBJECT_DETECTION_MODEL_H__
#define __OBJECT_DETECTION_MODEL_H__

#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include <gst/video/video.h>

class ObjectDetectionModel {
    public:
        virtual ~ObjectDetectionModel() = 0;
        virtual size_t getNumClasses() = 0;
        virtual size_t getInputTensorSize() = 0;
        virtual std::vector<float> &preprocess(uint8_t *const data, int width, int height, bool rgb) = 0;
        virtual void postprocess(std::vector<Ort::Value> const &model_output, std::vector<std::string> const &class_labels, float score_threshold, float nms_threshold) = 0;
};

#endif
