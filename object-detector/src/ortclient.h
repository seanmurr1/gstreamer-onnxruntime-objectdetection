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

#ifndef __ORT_CLIENT_H__
#define __ORT_CLIENT_H__

#include <onnxruntime_cxx_api.h>
#include "objectdetectionmodel.h"
#include "gstortelement.h"

/**
 * @brief ONNX Runtime client. Able to run object-detection
 * inferencing sessions with an object detection model.
 */
class OrtClient {
  private:
    Ort::Env env;
    Ort::Session session{nullptr};
    //Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::unique_ptr<ObjectDetectionModel> model;
    std::string onnx_model_path;
    std::string class_labels_path;
    std::vector<std::string> labels;

    size_t input_tensor_size;
    std::vector<Ort::AllocatedStringPtr> stored_names; // Needed to make sure unique_ptrs don't go out of scope
    size_t num_input_nodes;
    std::vector<const char*> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims;
    size_t num_output_nodes;
    std::vector<const char*> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims;

    std::vector<float> input_tensor_values;

    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    bool is_init;

    bool LoadClassLabels();
    bool SetModelInputOutput();
    bool CreateSession(GstOrtOptimizationLevel opti_level, GstOrtExecutionProvider provider, int device_id); 

  public:
    OrtClient();
    ~OrtClient() = default;
    bool Init(std::string const& model_path, std::string const& label_path, GstOrtOptimizationLevel = GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED, GstOrtExecutionProvider = GST_ORT_EXECUTION_PROVIDER_CPU, GstOrtDetectionModel = GST_ORT_DETECTION_MODEL_YOLOV4, int = 0);
    bool IsInitialized();
    void RunModel(uint8_t *const data, int width, int height, bool is_rgb, float = 0.25, float = 0.213);
    void RunModel(uint8_t *const data, GstVideoMeta *vmeta, float = 0.25, float = 0.213);
};

#endif