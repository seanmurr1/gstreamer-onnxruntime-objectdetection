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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstortelement.h"

GType
gst_ort_optimization_level_get_type (void)
{
  static GType ort_optimization_type = 0;

  if (g_once_init_enter (&ort_optimization_type)) {
    static GEnumValue optimization_level_types[] = {
      {GST_ORT_OPTIMIZATION_LEVEL_DISABLE_ALL, "Disable all optimization",
          "disable-all"},
      {GST_ORT_OPTIMIZATION_LEVEL_ENABLE_BASIC,
            "Enable basic optimizations (redundant node removals))", "enable-basic"},
      {GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED,
            "Enable extended optimizations (redundant node removals + node fusions)", "enable-extended"},
      {GST_ORT_OPTIMIZATION_LEVEL_ENABLE_ALL,
          "Enable all possible optimizations", "enable-all"},
      {0, NULL, NULL},
    };

    GType temp = g_enum_register_static ("GstOrtOptimizationLevel",
        optimization_level_types);

    g_once_init_leave (&ort_optimization_type, temp);
  }

  return ort_optimization_type;
}

GType
gst_ort_execution_provider_get_type (void)
{
  static GType ort_execution_type = 0;

  if (g_once_init_enter (&ort_execution_type)) {
    static GEnumValue execution_provider_types[] = {
      {GST_ORT_EXECUTION_PROVIDER_CPU, "CPU execution provider", "cpu"},
      {GST_ORT_EXECUTION_PROVIDER_CUDA, "CUDA execution provider", "cuda"},
      {0, NULL, NULL},
    };

    GType temp = g_enum_register_static ("GstOrtExecutionProvider",
        execution_provider_types);

    g_once_init_leave (&ort_execution_type, temp);
  }

  return ort_execution_type;
}

GType
gst_ort_detection_model_get_type (void)
{
  static GType ort_model_type = 0;

  if (g_once_init_enter (&ort_model_type)) {
    static GEnumValue model_types[] = {
      {GST_ORT_DETECTION_MODEL_YOLOV4, "YOLOv4 object detection model", "yolov4"},
      {0, NULL, NULL},
    };

    GType temp = g_enum_register_static ("GstOrtModel",
        model_types);

    g_once_init_leave (&ort_model_type, temp);
  }

  return ort_model_type;
}