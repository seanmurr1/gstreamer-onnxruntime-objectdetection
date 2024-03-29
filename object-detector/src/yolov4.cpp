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

#include "yolov4.h"

/**
 * @brief Construct a new YOLOv4 object.
 */
YOLOv4::YOLOv4() {
  LoadClassColors();
  class_boxes = std::vector<std::list<std::unique_ptr<BoundingBox>>>(NUM_CLASSES);
  anchors = std::vector<float>{12.f,16.f, 19.f,36.f, 40.f,28.f, 36.f,75.f, 76.f,55.f, 72.f,146.f, 142.f,110.f, 192.f,243.f, 459.f,401.f};
  strides = std::vector<float>{8.f, 16.f, 32.f};
  xyscale = std::vector<float>{1.2, 1.1, 1.05};
  padded_image = cv::Mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(128, 128, 128));
}

// Need to implement virutal destructor for ObjectDetectionModel interface.
// YOLOv4 has no explicit resources to deal with here.
ObjectDetectionModel::~ObjectDetectionModel() { }

size_t YOLOv4::GetNumClasses() {
  return NUM_CLASSES;
}

size_t YOLOv4::GetInputTensorSize() {
  return INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
}

/**
 * @brief Pads image to YOLOv4 input specifications.
 * Preserves aspect ratio. Pad with grey (128, 128, 128) pixels.
 * Stores padded iamge internally in a cache (cv::Mat).
 * 
 * @param image image to pad.
 */
void YOLOv4::PadImage(cv::Mat const& image) {
  resize_ratio = std::min(INPUT_WIDTH / (org_image_w * 1.0f), INPUT_HEIGHT / (org_image_h * 1.0f));
  // New dimensions to preserve aspect ratio
  int nw = resize_ratio * org_image_w;
  int nh = resize_ratio * org_image_h;
  // Padding on either side
  dw = (INPUT_WIDTH - nw) / 2.0f;
  dh = (INPUT_HEIGHT - nh) / 2.0f;
  // Reset padded image (padded_image acts as a cache)
  padded_image = cv::Scalar(128, 128, 128);
  // Resize original image into padded image
  cv::resize(image, padded_image(cv::Rect(dw, dh, std::floor(nw), std::floor(nh))), cv::Size(std::floor(nw), std::floor(nh)));
}

/**
 * @brief Preprocesses input data to comply with specifications of YOLOv4 algorithm.
 * 
 * @param data data to process.
 * @param input_tensor_values out-param to store preprocessed tensor values. Has sufficient size for input tensor.
 * @param width image width.
 * @param height image height.
 * @param is_rgb is image RGB or BGR format.
 */
void YOLOv4::Preprocess(uint8_t *const data, std::vector<float>& input_tensor_values, int width, int height, bool is_rgb) {
  org_image_h = height;
  org_image_w = width;
  this->is_rgb = is_rgb;
  
  std::vector<int> image_size{org_image_h, org_image_w};
  // Wrap opencv mat image
  // NOTE: this does not copy data, simply wraps
  org_image = cv::Mat(image_size, CV_8UC3, data);
  // Pad image 
  PadImage(org_image);
  // Change from BGR to RGB ordering if needed
  if (!is_rgb) {
    cv::cvtColor(padded_image, padded_image, cv::COLOR_BGR2RGB);
  }
  // Assign mat values to tensor data vector out-param and scale (COPIES DATA)
  for (size_t i = 0; i < GetInputTensorSize(); i++) {
    input_tensor_values[i] = (float) padded_image.data[i] / 255.f;
  }
}

// Apply sigmoid function to a value; returns a number between 0 and 1
float sigmoid(float value) {
  float k = (float) exp(-1.0f * value);
  return 1.0f / (1.0f + k);
}

/**
 * @brief Transforms YOLOv4 output coordinates into xmin, ymin, xmax, ymax 
 * coordinates relative to original input image. Transforms coordinates
 * in place into `coords` parameter.
 * 
 * @param coords raw YOLOv4 output coordinates of bounding box {x, y, w, h}. 
 * @param layer current layer.
 * @param row row of grid cell.
 * @param col column of grid cell.
 * @param anchor anchor index in grid cell.
 * @return true if transformed coordinates are valid.
 * @return false if transformed coordinates are invalid.
 */
bool YOLOv4::TransformCoordinates(std::vector<float>& coords, int layer, int row, int col, int anchor) {
  float x = coords[0];
  float y = coords[1];
  float w = coords[2];
  float h = coords[3];
  // Transform coordinates
  x = ((sigmoid(x) * xyscale[layer]) - 0.5f * (xyscale[layer] - 1.0f) + col) * strides[layer];
  y = ((sigmoid(y) * xyscale[layer]) - 0.5f * (xyscale[layer] - 1.0f) + row) * strides[layer];
  h = exp(h) * anchors[(layer * 6) + (anchor * 2) + 1]; 
  w = exp(w) * anchors[(layer * 6) + (anchor * 2)];   
  // Convert (x, y, w, h) => (xmin, ymin, xmax, ymax)
  float xmin = x - w * 0.5f;
  float ymin = y - h * 0.5f;
  float xmax = x + w * 0.5f;
  float ymax = y + h * 0.5f;
  // Convert (xmin, ymin, xmax, ymax) => (xmin_org, ymin_org, xmax_org, ymax_org), relative to original image
  float xmin_org = 1.0f * (xmin - dw) / resize_ratio;
  float ymin_org = 1.0f * (ymin - dh) / resize_ratio;
  float xmax_org = 1.0f * (xmax - dw) / resize_ratio;
  float ymax_org = 1.0f * (ymax - dh) / resize_ratio;
  // Disregard clipped boxes
  if (xmin_org > xmax_org || ymin_org > ymax_org) {
    return false;
  }
  // Disregard boxes with invalid size/area
  auto area = (xmax_org - xmin_org) * (ymax_org - ymin_org);
  if (area <= 0) {
    return false;
  }
  // Update vector
  coords[0] = xmin_org;
  coords[1] = ymin_org;
  coords[2] = xmax_org;
  coords[3] = ymax_org;
  return true;
}

/**
 * @brief Finds class index with highest probabilty for a given bounding box.
 * Returns index corresponding to max probability along with the max probability.
 * 
 * @param layer_output current layer of model output.
 * @param offset offset to beginning of bounding box data.
 * @return std::pair<int, float> (index, probability).
 */
std::pair<int, float> YOLOv4::FindMaxClass(float const *layer_output, long offset) {
  int max_class = -1;
  float max_prob;
  for (int i = 0; i < NUM_CLASSES; i++) {
    if (max_class == -1 || layer_output[offset + 5 + i] > max_prob) {
      max_class = i;
      max_prob = layer_output[offset + 5 + i];
    }
  }
  return std::pair<int, float>(max_class, max_prob);
}

/**
 * @brief Parses model output to extract bounding boxes. Filters bounding boxes and converts coordinates
 * to be respective to original image. Stores filtered bounding boxes internally.
 * 
 * @param model_output YOLOv4 inferencing output.
 * @param threshold threshold to filter boxes based on confidence/score.
 */
void YOLOv4::GetBoundingBoxes(std::vector<Ort::Value> const& model_output, float threshold) {
  // Iterate through output layers
  for (size_t layer = 0; layer < model_output.size(); layer++) {
    // Layer data
    float const *layer_output = model_output[layer].GetTensorData<float>();
    auto layer_shape = model_output[layer].GetTensorTypeAndShapeInfo().GetShape();
    auto grid_size = layer_shape[1];
    auto anchors_per_cell = layer_shape[3];
    auto features_per_anchor = layer_shape[4];
    // Iterate through grid cells in current layer, and anchors in each grid cell
    for (auto row = 0; row < grid_size; row++) {
      for (auto col = 0; col < grid_size; col++) {
        for (auto anchor = 0; anchor < anchors_per_cell; anchor++) {
          // Calculate offset for current grid cell and anchor
          long offset = (row * grid_size * anchors_per_cell * features_per_anchor) + (col * anchors_per_cell * features_per_anchor) + (anchor * features_per_anchor);
          // Extract data
          float x = layer_output[offset + 0];
          float y = layer_output[offset + 1];
          float w = layer_output[offset + 2];
          float h = layer_output[offset + 3]; 
          float conf = layer_output[offset + 4];
          if (conf < threshold) {
            continue;
          }
          // Convert coordinates
          std::vector<float> coords{x, y, w, h};
          if (!TransformCoordinates(coords, layer, row, col, anchor)) {
            continue;
          }
          // Find class with highest probability
          std::pair<int, float> max_class_prob = FindMaxClass(layer_output, offset);
          // Calculate score and compare against threshold
          float score = conf * max_class_prob.second;
          if (score < threshold) {
            continue;
          }
          // Create bounding box and add to vector
          auto bbox = std::make_unique<BoundingBox>(BoundingBox(coords[0], coords[1], coords[2], coords[3], score, max_class_prob.first));
          class_boxes[max_class_prob.first].push_back(move(bbox));
        }
      }
    }
  }
}

/**
 * @brief Calculate the intersection over union (IOU) of two bounding boxes.
 * 
 * @param bbox1 first bounding box.
 * @param bbox2 second bounding box.
 * @return float IOU of the two boxes.
 */
float YOLOv4::BboxIOU(std::unique_ptr<BoundingBox> const& bbox1, std::unique_ptr<BoundingBox> const& bbox2) {
  float area1 = (bbox1->xmax - bbox1->xmin) * (bbox1->ymax - bbox1->ymin);
  float area2 = (bbox2->xmax - bbox2->xmin) * (bbox2->ymax - bbox2->ymin);
  // Coords of intersection box
  float left = std::max(bbox1->xmin, bbox2->xmin);
  float right = std::min(bbox1->xmax, bbox2->xmax);
  float top = std::max(bbox1->ymin, bbox2->ymin);
  float bottom = std::min(bbox1->ymax, bbox2->ymax);

  float intersection_area;
  if (left > right || top > bottom) {
    intersection_area = 0;
  } else {
    intersection_area = (right - left) * (bottom - top);
  }
  float union_area = area1 + area2 - intersection_area;
  return intersection_area / union_area;
}

// Compares score of two bounding boxes. Used to sort vectors of bounding boxes.
bool CompareBoxScore(std::unique_ptr<BoundingBox>& b1, std::unique_ptr<BoundingBox>& b2) {
  return b2->score < b1->score;
}

/**
 * @brief Perform non-maximal suppression (nms) on found/filtered bounding boxes.
 * NOTE: this version computes nms per class.
 * NOTE: std::list is used instead of std::vector for O(1) removal while iterating.
 * Using std::vector requires a sacrifice in either space or time complexity 
 * for this algorithm.
 * 
 * Stores filtered bounding boxes internally.
 * 
 * @param threshold IOU threshold for nms.
 */
void YOLOv4::Nms(float threshold) {
  for (auto i = 0; i < NUM_CLASSES; i++) {
    std::list<std::unique_ptr<BoundingBox>>& boxes = class_boxes[i];
    if (boxes.empty()) {
      continue;
    }
    boxes.sort(CompareBoxScore);
    while (!boxes.empty()) {
      filtered_boxes.push_back(move(boxes.front()));
      boxes.pop_front();
      std::unique_ptr<BoundingBox>& accepted_box = filtered_boxes.back();
      std::list<std::unique_ptr<BoundingBox>>::iterator itr = boxes.begin();
      while (itr != boxes.end()) {
        std::unique_ptr<BoundingBox>& test_box = *itr;
        if (BboxIOU(accepted_box, test_box) > threshold) {
          // Performance: free memory early when possible
          test_box.reset();
          itr = boxes.erase(itr);
        } else {
          itr++;
        }
      }
    }
    class_boxes[i].clear();
  }
}

// Create unique, constant color for each class id.
void YOLOv4::LoadClassColors() {
  // Convert HSV to RGB
  // Saturation and Value will always be 1
  // Hue depends on class index
  class_colors = std::vector<cv::Scalar>(NUM_CLASSES);
  for (int i = 0; i < NUM_CLASSES; i++) {
    float h = ((1.0f * i) / NUM_CLASSES) * 360;
    float x = 1.0f - std::abs(std::fmod(h / 60, 2) - 1);
    float r, g, b;
    if (h >= 0 && h < 60) {
      r = 255, g = x * 255, b = 0;
    } else if (h >= 60 && h < 120) {
      r = x * 255, g = 255, b = 0;
    } else if (h >= 120 && h < 180) {
      r = 0, g = 255, b = x * 255;
    } else if (h >= 180 && h < 240) {
      r = 0, g = x * 255, b = 255;
    } else if (h >= 240 && h < 300) {
      r = x * 255, g = 0, b = 255;
    } else {
      r = 255, g = 0, b = x * 255;
    }
    class_colors[i] = cv::Scalar(r, g, b);
  }
}

/**
 * @brief Write filtered bounding boxes and class labels/scores to original image.
 * 
 * @param class_names vector of class names.
 */
void YOLOv4::WriteBoundingBoxes(std::vector<std::string> const& class_names) {
  float font_scale = 0.5f;
  int bbox_thick = (int) (0.6f * (org_image_h + org_image_w) / 600.f);

  for (size_t i = 0; i < filtered_boxes.size(); i++) {
    // Bounding box information
    std::unique_ptr<BoundingBox>& bbox = filtered_boxes[i];
    std::string const& class_name = class_names[bbox->class_index];
    float score = bbox->score;
    auto c1 = cv::Point(bbox->xmin, bbox->ymin);
    auto c2 = cv::Point(bbox->xmax, bbox->ymax);

    cv::Scalar color = class_colors[bbox->class_index];
    // Shift color for consistency if image is BGR
    if (!is_rgb) {
      auto temp = color[0];
      color[0] = color[2];
      color[2] = temp;
    }
    // Place rectangle around bounding box
    cv::Rect rect = cv::Rect(c1, c2);
    cv::rectangle(org_image, rect, color, bbox_thick);

    std::stringstream msg;
    msg << class_name << ": " << roundf(score * 100) / 100;
    int base_line = 0;
    auto t_size = cv::getTextSize(msg.str(), 0, font_scale, bbox_thick / 2, &base_line);
    // Place rectangle for class label & score message
    cv::rectangle(org_image, c1, cv::Point(c1.x + t_size.width, c1.y - t_size.height - 3), color, -1);
    // Place message
    cv::putText(org_image, msg.str(), cv::Point(bbox->xmin, bbox->ymin - 2), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), bbox_thick / 2);
    // Performance: free memory early
    // bbox.reset();
  }
  filtered_boxes.clear();
}

/**
 * @brief Postprocess ORT model output with YOLOv4 bounding box information.
 * Write filtered bounding boxes to original image data using internally
 * stored reference set from Preprocess method.
 * 
 * @param model_output ORT output.
 * @param class_labels YOLOv4 class labels.
 * @param score_threshold threshold for bounding box scores.
 * @param nms_threshold threshold for computing non-maximal suppression.
 */
void YOLOv4::Postprocess(std::vector<Ort::Value> const& model_output, std::vector<std::string> const& class_labels, float score_threshold, float nms_threshold) {
  GetBoundingBoxes(model_output, score_threshold);
  Nms(nms_threshold);
  WriteBoundingBoxes(class_labels);
}
