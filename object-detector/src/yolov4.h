/*
 * GStreamer
 * Copyright (C) 2006 Stefan Kost <ensonic@users.sf.net>
 * Copyright (C) 2022  <<user@hostname.org>>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

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

/**
 * @brief YOLOv4 object detection model. Performs pre/post-processing steps.
 */
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

    // NOTE: std::list is used here over std::vector for random index deletion
    // See YOLOv4::Nms function
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