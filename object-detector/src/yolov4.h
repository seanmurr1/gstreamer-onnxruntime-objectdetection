#ifndef __YOLOV4_H__
#define __YOLOV4_H__

#include <opencv2/opencv.hpp>
#include "objectdetectionmodel.h"

// Representation of bounding box
struct BoundingBox {
  float xmin;
  float ymin;
  float xmax;
  float ymax;
  float score;
  int class_index;

  BoundingBox(float xmin, float ymin, float xmax, float ymax, float score, int class_index) : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax), score(score), class_index(class_index) {}
};

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
    cv::Mat padded_image;
    float resize_ratio;
    float dw;
    float dh;
    bool is_rgb;
        
    std::vector<cv::Scalar> class_colors;
    std::vector<float> anchors;
    std::vector<float> strides;
    std::vector<float> xyscale;

    std::vector<std::list<std::unique_ptr<BoundingBox>>> class_boxes;
    std::vector<std::unique_ptr<BoundingBox>> filtered_boxes;

    void LoadClassColors();
    void PadImage(cv::Mat const& image);
    std::pair<int, float> FindMaxClass(float const *layer_output, long offset);
    bool TransformCoordinates(std::vector<float>& coords, int layer, int row, int col, int anchor);
    void GetBoundingBoxes(std::vector<Ort::Value> const& model_output, float threshold);
    float BboxIOU(std::unique_ptr<BoundingBox> const& bbox1, std::unique_ptr<BoundingBox> const& bbox2);
    void Nms(float threshold);
    void WriteBoundingBoxes(std::vector<std::string> const& class_names);

  public:
    YOLOv4();
    ~YOLOv4() = default;
    size_t GetNumClasses();
    size_t GetInputTensorSize();
    void Preprocess(uint8_t *const data, std::vector<float>& input_tensor_values, int width, int height, bool is_rgb);
    void Postprocess(std::vector<Ort::Value> const& model_output, std::vector<std::string> const& class_labels, float score_threshold, float nms_threshold);
};

#endif