#ifndef OBJECTDETECTIONMODEL_H
#define OBJECTDETECTIONMODEL_H

#include <cstdint>
#include <onnxruntime_cxx_api.h>

class ObjectDetectionModel {
    public:
        virtual ~ObjectDetectionModel() = 0;
        virtual size_t getNumClasses() = 0;
        virtual size_t getInputTensorSize() = 0;
        virtual std::vector<float> preprocess(uint8_t* data, int width, int height) = 0;
        virtual uint8_t* postprocess(std::vector<Ort::Value> &model_output, std::vector<std::string> class_labels) = 0;
};

#endif
