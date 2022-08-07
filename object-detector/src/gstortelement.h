#ifndef __GST_ORT_ELEMENT_H__
#define __GST_ORT_ELEMENT_H__

#include <gst/gst.h>

typedef enum {
  GST_ORT_OPTIMIZATION_LEVEL_DISABLE_ALL,
  GST_ORT_OPTIMIZATION_LEVEL_ENABLE_BASIC,
  GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED,
  GST_ORT_OPTIMIZATION_LEVEL_ENABLE_ALL
} GstOrtOptimizationLevel;

typedef enum {
  GST_ORT_EXECUTION_PROVIDER_CPU,
  GST_ORT_EXECUTION_PROVIDER_CUDA
} GstOrtExecutionProvider;

typedef enum {
  GST_ORT_DETECTION_MODEL_YOLOV4
} GstOrtDetectionModel;

G_BEGIN_DECLS

GType gst_ort_optimization_level_get_type (void);
#define GST_TYPE_ORT_OPTIMIZATION_LEVEL (gst_ort_optimization_level_get_type ())

GType gst_ort_execution_provider_get_type (void);
#define GST_TYPE_ORT_EXECUTION_PROVIDER (gst_ort_execution_provider_get_type ())

GType gst_ort_detection_model_get_type (void);
#define GST_TYPE_ORT_DETECTION_MODEL (gst_ort_detection_model_get_type ())

G_END_DECLS

#endif