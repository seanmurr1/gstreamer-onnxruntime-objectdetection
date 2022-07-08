#include "yolov4.h"

ObjectDetectionModel::~ObjectDetectionModel() { }

YOLOv4::~YOLOv4() { }

size_t YOLOv4::getNumClasses() {
    return NUM_CLASSES;
}

size_t YOLOv4::getInputTensorSize() {
    return INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
}

/**
 * @brief Pads inputted image to YOLOv4 input specifications.
 * Preserves aspect ratio. Pad with grey (128, 128, 128) pixels.
 * 
 * @param image image to pad.
 * @return cv::Mat padded image.
 */
cv::Mat YOLOv4::padImage(cv::Mat image) {
    resize_ratio = std::min(INPUT_WIDTH / (org_image_w * 1.0f), INPUT_HEIGHT / (org_image_h * 1.0f));
    // New dimensions to preserve aspect ratio
    int nw = resize_ratio * org_image_w;
    int nh = resize_ratio * org_image_h;

    // Padding on either side
    float dw = std::floor((INPUT_WIDTH - nw) / 2.0f);
    float dh = std::floor((INPUT_HEIGHT - nh) / 2.0f);

    cv::Mat out(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh));
    resized.copyTo(out(cv::Rect(dw, dh, resized.cols, resized.rows)));
    return out;
}

/**
 * @brief Preprocesses input data to comply with input specifications of YOLOv4 algorithm.
 * 
 * @param data data to process.
 * @param width image width.
 * @param height image height.
 */
std::vector<float> YOLOv4::preprocess(uint8_t* data, int width, int height) {
    org_image_h = height;
    org_image_w = width;
    // Create opencv mat image
    std::vector<int> image_size{height, width};
    
    // NOTE: this does not copy data, simply wraps
    org_image = cv::Mat(image_size, CV_8UC3, data);

    // TODO: check if its width, height or reversed 
    cv::Mat padded = padImage(org_image);

    // TODO: check about swapping RGB or HWC / transpose
    // Swap to RGB order
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

    cv::Mat image_float;
    // Convert data to floats
    //padded.convertTo(image_float, CV_32FC3);
    //cv::imwrite("../../assets/images/paddedpleaseplease.png", image_float);
    //std::cout << "Post:float: " << (float) image_float.data[2003] << std::endl;

    std::vector<float> input_tensor_values;
    input_tensor_values.assign(padded.data, padded.data + padded.total() * padded.channels());
    //std::cout << "before scale: " << input_tensor_values[2003] << std::endl;
    for (size_t i = 0; i < input_tensor_values.size(); i++) {
        input_tensor_values[i] = input_tensor_values[i] / 255.f;
    }
    //std::cout << "after scale: " << input_tensor_values[2003] << std::endl;

    return input_tensor_values;
}

// Apply sigmoid function to a value; returns a number between 0 and 1
float sigmoid(float value) {
    float k = (float) exp(-1.0f * value);
    return 1.0f / (1.0f + k);
}

/**
 * @brief Parses model output to extract bounding boxes. Filters bounding boxes and converts coordinates
 * to be respective to original image.
 * 
 * @param model_output YOLOv4 inferencing output.
 * @param anchors vector of YOLOv4 anchors value (for each anchor index of each layer).
 * @param strides vector of strides for each output layer.
 * @param xyscale vector of xyscale for each output layer.
 * @param threshold threshold to filter boxes based on confidence/score.
 * @return std::vector<BoundingBox*> filtered bounding boxes from model output (all layers).
 */
std::vector<BoundingBox*> YOLOv4::getBoundingBoxes(std::vector<Ort::Value> &model_output, std::vector<float> anchors, std::vector<float> strides, std::vector<float> xyscale, float threshold) {
    std::vector<BoundingBox*> bboxes;
    auto dw = (INPUT_WIDTH - resize_ratio * org_image_w) / 2;
    auto dh = (INPUT_HEIGHT - resize_ratio * org_image_h) / 2;

    size_t num_output_nodes = 3;

    // Iterate through output layers
    for (size_t layer = 0; layer < num_output_nodes; layer++) {
        float *layer_output = model_output[layer].GetTensorMutableData<float>();
        auto layer_shape = model_output[layer].GetTensorTypeAndShapeInfo().GetShape();
        
        // Layer data
        auto grid_size = layer_shape[1];
        auto anchors_per_cell = layer_shape[3];
        auto features_per_anchor = layer_shape[4];

        // Iterate through grid cells in current layer, and anchors in each grid cell
        for (auto row = 0; row < grid_size; row++) {
            for (auto col = 0; col < grid_size; col++) {
                for (auto anchor = 0; anchor < anchors_per_cell; anchor++) {
                    // Calculate offset for current grid cell and anchor
                    auto offset = (row * grid_size * anchors_per_cell * features_per_anchor) + (col * anchors_per_cell * features_per_anchor) + (anchor * features_per_anchor);
                    // Extract data
                    auto x = layer_output[offset + 0];
                    auto y = layer_output[offset + 1];
                    auto h = layer_output[offset + 3]; 
                    auto w = layer_output[offset + 2];
                    auto conf = layer_output[offset + 4];

                    if (conf < threshold) {
                        //continue;
                    }

                    // Transform coordinates
                    x = ((sigmoid(x) * xyscale[layer]) - 0.5f * (xyscale[layer] - 1.0f) + col) * strides[layer];
                    y = ((sigmoid(y) * xyscale[layer]) - 0.5f * (xyscale[layer] - 1.0f) + row) * strides[layer];
                    h = exp(h) * anchors[(layer * 6) + (anchor * 2) + 1]; 
                    w = exp(w) * anchors[(layer * 6) + (anchor * 2)];   
                    // Convert (x, y, w, h) => (xmin, ymin, xmax, ymax)
                    auto xmin = x - w * 0.5f;
                    auto ymin = y - h * 0.5f;
                    auto xmax = x + w * 0.5f;
                    auto ymax = y + h * 0.5f;
                    // Convert (xmin, ymin, xmax, ymax) => (xmin_org, ymin_org, xmax_org, ymax_org), relative to original image
                    auto xmin_org = 1.0f * (xmin - dw) / resize_ratio;
                    auto ymin_org = 1.0f * (ymin - dh) / resize_ratio;
                    auto xmax_org = 1.0f * (xmax - dw) / resize_ratio;
                    auto ymax_org = 1.0f * (ymax - dh) / resize_ratio;

                    // Disregard clipped boxes
                    if (xmin_org > xmax_org || ymin_org > ymax_org) {
                        continue;
                    }
                    // Disregard boxes with invalid size/area
                    auto area = (xmax_org - xmin_org) * (ymax_org - ymin_org);
                    if (area <= 0 || isnan(area) || !isfinite(area)) {
                        continue;
                    }

                    // Find class with highest probability
                    int max_class = -1;
                    float max_prob;
                    for (int i = 0; i < NUM_CLASSES; i++) {
                        if (max_class == -1 || layer_output[offset + 5 + i] > max_prob) {
                            max_class = i;
                            max_prob = layer_output[offset + 5 + i];
                        }
                    }
                    // Calculate score and compare against threshold
                    float score = conf * max_prob;
                    if (score <= threshold) {
                        continue;
                    }
                    // Create bounding box and add to vector
                    BoundingBox *bbox = new BoundingBox(xmin_org, ymin_org, xmax_org, ymax_org, score, max_class);
                    bboxes.push_back(bbox);
                }
            }
        }
    }
    return bboxes;
}

/**
 * @brief Calculate the intersection over union (IOU) of two bounding boxes.
 * 
 * @param bbox1 first bounding box.
 * @param bbox2 second bounding box.
 * @return float IOU of the two boxes.
 */
float YOLOv4::bbox_iou(BoundingBox *bbox1, BoundingBox *bbox2) {
    float area1 = (bbox1->xmax - bbox1->xmin) * (bbox1->ymax - bbox1->ymin);
    float area2 = (bbox2->xmax - bbox2->xmin) * (bbox2->ymax - bbox2->ymin);

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
bool compareBoxScore(BoundingBox* b1, BoundingBox* b2) {
    return b2->score < b1->score;
}

/**
 * @brief Perform non-maximal suppression (nms) on vector of bounding boxes.
 * 
 * @param bboxes vector of bounding boxes to perform nms on.
 * @param threshold IOU threshold for nms.
 * @return std::vector<BoundingBox*> filtered boxes after applying nms.
 */
std::vector<BoundingBox*> YOLOv4::nms(std::vector<BoundingBox*> bboxes, float threshold) {
    // Organize boxes by class 
    std::unordered_map<int, std::vector<BoundingBox*>> class_map;
    for (size_t i = 0; i < bboxes.size(); i++) {
        class_map[bboxes[i]->class_index].push_back(bboxes[i]);        
    }
    //num_classes_detected = class_map.size();

    std::vector<BoundingBox*> filtered_boxes;

    // Iterate through each class detected
    for (auto &pair : class_map) {
        std::vector<BoundingBox*> boxes = pair.second;
        // Sort class specific boxes by score in decreasing order 
        std::sort(boxes.begin(), boxes.end(), compareBoxScore);

        while (boxes.size() > 0) {
            // Extract box with highest score
            BoundingBox *accepted_box = boxes[0];
            filtered_boxes.push_back(accepted_box);
            boxes.erase(boxes.begin());

            std::vector<BoundingBox*> safe_boxes;
            // Compare extracted box with all remaining class boxes
            for (size_t i = 0; i < boxes.size(); i++) {
                BoundingBox *test_box = boxes[i];
                if (bbox_iou(accepted_box, test_box) <= threshold) {
                    safe_boxes.push_back(test_box);
                } else {
                    delete test_box;
                }
            }
            // Update class boxes
            boxes = safe_boxes;
        }
    }
    return filtered_boxes;
}

/**
 * @brief Write bounding boxes and class labels/scores to original image.
 * 
 * @param bboxes filtered bounding boxes.
 * @param class_names vector of class names.
 */
void YOLOv4::writeBoundingBoxes(std::vector<BoundingBox*> bboxes, std::vector<std::string> class_names) {
    float font_scale = 0.5f;
    int bbox_thick = (int) (0.6f * (org_image_h + org_image_w) / 600.f);
    std::srand(5);

    std::unordered_map<int, cv::Scalar> class_colors;

    for (size_t i = 0; i < bboxes.size(); i++) {
        // Bounding box information
        BoundingBox *bbox = bboxes[i];
        std::string class_name = class_names[bbox->class_index];
        float score = bbox->score;
        auto c1 = cv::Point(bbox->xmin, bbox->ymin);
        auto c2 = cv::Point(bbox->xmax, bbox->ymax);

        std::cout << "Found " << class_name << std::endl;

        // Get color for class
        cv::Scalar color;
        if (class_colors.find(bbox->class_index) != class_colors.end()) {
            color = class_colors.at(bbox->class_index);
        } else {
            color = cv::Scalar((std::rand()%256), std::rand()%256, std::rand()%256);
            class_colors[bbox->class_index] = color;
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
        delete bbox;
    }
}

uint8_t* YOLOv4::postprocess(std::vector<Ort::Value> &model_output, std::vector<std::string> class_labels) {
    std::vector<float> anchors{12.f,16.f, 19.f,36.f, 40.f,28.f, 36.f,75.f, 76.f,55.f, 72.f,146.f, 142.f,110.f, 192.f,243.f, 459.f,401.f};
    std::vector<float> strides{8.f, 16.f, 32.f};
    std::vector<float> xyscale{1.2, 1.1, 1.05};

    std::vector<BoundingBox*> bboxes = getBoundingBoxes(model_output, anchors, strides, xyscale, 0.25);
    bboxes = nms(bboxes, 0.213);
    writeBoundingBoxes(bboxes, class_labels);
    cv::imwrite("../../assets/images/pleasepleaseplease.png", org_image);

    return org_image.data;
}
