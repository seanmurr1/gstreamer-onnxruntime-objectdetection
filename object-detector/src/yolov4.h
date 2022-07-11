#ifndef YOLOV4_H
#define YOLOV4_H

#include <opencv2/opencv.hpp>
#include "objectdetectionmodel.h"

// Representation of bounding box
typedef struct _BoundingBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int class_index;

    _BoundingBox(float xmin, float ymin, float xmax, float ymax, float score, int class_index) : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax), score(score), class_index(class_index) {}

} BoundingBox;

class YOLOv4 : public ObjectDetectionModel {
    private:
        // Model information
        const int NUM_CLASSES = 80;
        const int INPUT_HEIGHT = 416;
        const int INPUT_WIDTH = 416;
        const int INPUT_CHANNELS = 3;

        // Original image information
        int org_image_w;
        int org_image_h;
        cv::Mat org_image;
        float resize_ratio;

        std::vector<cv::Scalar> class_colors;
        std::vector<float> input_tensor_values;

        void loadClassColors();
        cv::Mat padImage(cv::Mat const &image);
        std::vector<BoundingBox*> getBoundingBoxes(std::vector<Ort::Value> &model_output, std::vector<float> &anchors, std::vector<float> &strides, std::vector<float> &xyscale, float threshold);
        float bbox_iou(BoundingBox *bbox1, BoundingBox *bbox2);
        std::vector<BoundingBox*> nms(std::vector<BoundingBox*> bboxes, float threshold);
        void writeBoundingBoxes(std::vector<BoundingBox*> bboxes, std::vector<std::string> class_names);

    public:
        ~YOLOv4();
        YOLOv4();
        size_t getNumClasses();
        size_t getInputTensorSize();
        std::vector<float> &preprocess(uint8_t *const data, int width, int height);
        uint8_t* postprocess(std::vector<Ort::Value> &model_output, std::vector<std::string> class_labels);
};

#endif