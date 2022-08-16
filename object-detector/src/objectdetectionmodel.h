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

#ifndef __OBJECT_DETECTION_MODEL_H__
#define __OBJECT_DETECTION_MODEL_H__

#include <cstdint>
#include <onnxruntime_cxx_api.h>
#include <gst/video/video.h>

/**
 * @brief Interface for an ML object detection model.
 * Includes pre/post-processing steps and model information.
 */
class ObjectDetectionModel {
  public:
    virtual ~ObjectDetectionModel() = 0;
    virtual size_t GetNumClasses() = 0;
    virtual size_t GetInputTensorSize() = 0;
    virtual void Preprocess(uint8_t* const data, std::vector<float>& input_tensor_values, int width, int height, bool is_rgb) = 0;
    virtual void Postprocess(std::vector<Ort::Value> const& model_output, std::vector<std::string> const& class_labels, float score_threshold, float nms_threshold) = 0;
};

#endif
