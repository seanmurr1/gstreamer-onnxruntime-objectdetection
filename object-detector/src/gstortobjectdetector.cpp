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

/**
 * SECTION:element-ortobjectdetector
 * @short_description: Detect objects in each video frame.
 *
 * ortobjectdetector is a GStreamer plugin that allows users to run
 * ONNX Runtime (ORT) object detection inference sessions on a pipeline of 
 * video data. Users can utilize any supported ONNX model (e.g. YOLOv4).
 * 
 * Please see the README for installation instructions.
 * 
 * Users may control the specific object detection model used, optimization level,
 * execution provider, filtering thresholds, and hardware acceleration device.
 * 
 * The plugin supports either RGB or BGR video data in GST's video/x-raw format.
 * It outputs the same data format.
 *
 * ## Example pipeline:
 * 
 * ```
 * gst-launch-1.0 filesrc location=video1.mp4 ! \
 * qtdemux name=demux  demux.audio_0 ! \
 * queue ! \
 * decodebin ! \
 * audioconvert ! \
 * audioresample ! \
 * autoaudiosink  
 * demux.video_0 ! \
 * queue ! \
 * decodebin ! \
 * videoconvert ! \
 * ortobjectdetector \
 * model-file=yolov4.onnx \
 * label-file=labels.txt \
 * score-threshold=0.25 \ 
 * nms-threshold=0.213 \ 
 * optimization-level=enable-extended \
 * execution-provider=cpu \
 * detection-model=yolov4 ! \
 * videoconvert ! \
 * fpsdisplaysink
 * ```
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/base/base.h>
#include <gst/controller/controller.h>
#include <gst/video/video.h>
#include <gst/video/gstvideometa.h>

#include "gstortobjectdetector.h"
#include "ortclient.h"

GST_DEBUG_CATEGORY_STATIC (gst_ortobjectdetector_debug);
#define GST_CAT_DEFAULT gst_ortobjectdetector_debug

enum
{
  PROP_0,
  PROP_MODEL_FILE,
  PROP_LABEL_FILE,
  PROP_OPTIMIZATION_LEVEL,
  PROP_EXECUTION_PROVIDER,
  PROP_SCORE_THRESHOLD,
  PROP_NMS_THRESHOLD,
  PROP_DETECTION_MODEL,
  PROP_DEVICE_ID
};

// Default prop values
#define DEFAULT_SCORE_THRESHOLD 0.25f
#define DEFAULT_NMS_THRESHOLD 0.213f
#define DEFAULT_EXECUTION_PROVIDER GST_ORT_EXECUTION_PROVIDER_CPU
#define DEFAULT_OPTIMIZATION_LEVEL GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED
#define DEFAULT_DETECTION_MODEL GST_ORT_DETECTION_MODEL_YOLOV4
#define DEFAULT_DEVICE_ID 0

/* the capabilities of the inputs and outputs.
 *
 * FIXME:describe the real formats here.
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE("{RGB,BGR}"))
    );

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE("{RGB,BGR}"))
    );

#define gst_ortobjectdetector_parent_class parent_class
G_DEFINE_TYPE (Gstortobjectdetector, gst_ortobjectdetector, GST_TYPE_BASE_TRANSFORM);
GST_ELEMENT_REGISTER_DEFINE (ortobjectdetector, "ortobjectdetector", GST_RANK_NONE,
    GST_TYPE_ORTOBJECTDETECTOR);

static void gst_ortobjectdetector_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_ortobjectdetector_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

static GstFlowReturn gst_ortobjectdetector_transform_ip (GstBaseTransform *
    base, GstBuffer * outbuf);

static void gst_ortobjectdetector_finalize (GObject * object);


/* GObject vmethod implementations */

/* initialize the ortobjectdetector's class */
static void
gst_ortobjectdetector_class_init (GstortobjectdetectorClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_ortobjectdetector_set_property;
  gobject_class->get_property = gst_ortobjectdetector_get_property;
  gobject_class->finalize = gst_ortobjectdetector_finalize;

  g_object_class_install_property (gobject_class, PROP_MODEL_FILE,
      g_param_spec_string ("model-file", "ONNX model file", "Path to ONNX model file",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_LABEL_FILE,
      g_param_spec_string ("label-file", "Class label file", "Path to class label file for ONNX model",
          NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SCORE_THRESHOLD,
      g_param_spec_float ("score-threshold", "Score threshold", "Threshold for filtering bounding boxes by score",
          0.0, 1.0, DEFAULT_SCORE_THRESHOLD, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_NMS_THRESHOLD,
      g_param_spec_float ("nms-threshold", "NMS threshold", "Threshold for filtering bounding boxes during non-maximal suppresion",
          0.0, 1.0, DEFAULT_NMS_THRESHOLD, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_OPTIMIZATION_LEVEL,
      g_param_spec_enum ("optimization-level", "Optimization level", "ORT optimization level",
          GST_TYPE_ORT_OPTIMIZATION_LEVEL, DEFAULT_OPTIMIZATION_LEVEL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_EXECUTION_PROVIDER,
      g_param_spec_enum ("execution-provider", "Execution provider", "ORT execution provider",
          GST_TYPE_ORT_EXECUTION_PROVIDER, DEFAULT_EXECUTION_PROVIDER, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  
  g_object_class_install_property (gobject_class, PROP_DETECTION_MODEL,
      g_param_spec_enum ("detection-model", "Detection model", "Object detection model",
          GST_TYPE_ORT_DETECTION_MODEL, DEFAULT_DETECTION_MODEL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_DEVICE_ID,
      g_param_spec_int ("device-id", "Device ID", "Device ID for hardware acceleration",
        0, G_MAXINT, 0, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_details_simple (gstelement_class,
      "ortobjectdetector",
      "Generic/Filter",
      "FIXME:Generic Template Filter", " <<user@hostname.org>>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_template));

  GST_BASE_TRANSFORM_CLASS (klass)->transform_ip =
      GST_DEBUG_FUNCPTR (gst_ortobjectdetector_transform_ip);

  /* debug category for fltering log messages */
  GST_DEBUG_CATEGORY_INIT (gst_ortobjectdetector_debug, "ortobjectdetector", 0,
      "ortobjectdetector debug info");
}

/* initialize the new element
 * initialize instance structure
 */
static void
gst_ortobjectdetector_init (Gstortobjectdetector * self)
{
  self->ort_client = std::unique_ptr<OrtClient>(new OrtClient());
  self->score_threshold = DEFAULT_SCORE_THRESHOLD;
  self->nms_threshold = DEFAULT_NMS_THRESHOLD;
  self->optimization_level = DEFAULT_OPTIMIZATION_LEVEL;
  self->execution_provider = DEFAULT_EXECUTION_PROVIDER;
  self->detection_model = DEFAULT_DETECTION_MODEL;
  self->device_id = DEFAULT_DEVICE_ID;
}

static void
gst_ortobjectdetector_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  Gstortobjectdetector *self = GST_ORTOBJECTDETECTOR (object);
  const gchar *filename;

  switch (prop_id) {
    case PROP_MODEL_FILE:
      filename = g_value_get_string(value);
      if (filename && g_file_test(filename, (GFileTest) (G_FILE_TEST_EXISTS | G_FILE_TEST_IS_REGULAR))) {
        if (self->model_file) 
          g_free(self->model_file);
        self->model_file = g_strdup(filename);
      } else {
        GST_WARNING_OBJECT (self, "Model file '%s' not found!", filename);
        gst_base_transform_set_passthrough(GST_BASE_TRANSFORM (self), TRUE);
      }
      break;
    case PROP_LABEL_FILE:
      filename = g_value_get_string(value);
      if (filename && g_file_test(filename, (GFileTest) (G_FILE_TEST_EXISTS | G_FILE_TEST_IS_REGULAR))) {
        if (self->label_file) 
          g_free(self->label_file);
        self->label_file = g_strdup(filename);
      } else {
        GST_WARNING_OBJECT (self, "Label file '%s' not found!", filename);
        gst_base_transform_set_passthrough(GST_BASE_TRANSFORM (self), TRUE);
      }
      break;
    case PROP_SCORE_THRESHOLD:
      self->score_threshold = g_value_get_float(value);
      break;
    case PROP_NMS_THRESHOLD:
      self->nms_threshold = g_value_get_float(value);
      break;
    case PROP_OPTIMIZATION_LEVEL:
      self->optimization_level = (GstOrtOptimizationLevel) g_value_get_enum (value);
      break;
    case PROP_EXECUTION_PROVIDER:
      self->execution_provider = (GstOrtExecutionProvider) g_value_get_enum (value);
      break;
    case PROP_DETECTION_MODEL:
      self->detection_model = (GstOrtDetectionModel) g_value_get_enum (value);
      break;
    case PROP_DEVICE_ID:
      self->device_id = g_value_get_int(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_ortobjectdetector_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  Gstortobjectdetector *self = GST_ORTOBJECTDETECTOR (object);

  switch (prop_id) {
    case PROP_MODEL_FILE:
      g_value_set_string(value, self->model_file);
      break;
    case PROP_LABEL_FILE:
      g_value_set_string(value, self->label_file);
      break;
    case PROP_SCORE_THRESHOLD:
      g_value_set_float(value, self->score_threshold);
      break;
    case PROP_NMS_THRESHOLD:
      g_value_set_float(value, self->nms_threshold);
      break;
    case PROP_OPTIMIZATION_LEVEL:
      g_value_set_enum(value, self->optimization_level);
      break;
    case PROP_EXECUTION_PROVIDER:
      g_value_set_enum(value, self->execution_provider);
      break;
    case PROP_DETECTION_MODEL:
      g_value_set_enum(value, self->detection_model);
      break;
    case PROP_DEVICE_ID:
      g_value_set_int(value, self->device_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_ortobjectdetector_finalize (GObject * object)
{
  Gstortobjectdetector *self = GST_ORTOBJECTDETECTOR (object);
  g_free (self->model_file);
  g_free (self->label_file);
  G_OBJECT_CLASS (gst_ortobjectdetector_parent_class)->finalize (object);
}

static gboolean
gst_ortobjectdetector_ort_setup (GstBaseTransform *base) {
  Gstortobjectdetector *self = GST_ORTOBJECTDETECTOR (base);
  std::unique_ptr<OrtClient>& ort_client = self->ort_client;
  
  GST_OBJECT_LOCK (self);
  if (ort_client->IsInitialized()) {
    GST_OBJECT_UNLOCK (self);
    return TRUE;
  }

  if (!self->model_file || !self->label_file) {
    GST_OBJECT_UNLOCK (self);
    GST_ERROR_OBJECT (self, "Unable to initialize ORT client without model and/or label file!");
    return FALSE;
  }

  GST_INFO_OBJECT (self, "model-file: %s\n", self->model_file);
  GST_INFO_OBJECT (self, "label-file: %s\n", self->label_file);
  GST_INFO_OBJECT (self, "score-threshold: %f\n", self->score_threshold);
  GST_INFO_OBJECT (self, "nms-threshold: %f\n", self->nms_threshold);
  GST_INFO_OBJECT (self, "optimization-level: %d\n", self->optimization_level);
  GST_INFO_OBJECT (self, "execution-provider: %d\n", self->execution_provider);
  GST_INFO_OBJECT (self, "detection-model: %d\n", self->detection_model);
  GST_INFO_OBJECT (self, "device-id: %d\n", self->device_id);
  GST_INFO_OBJECT (self, "Initializing ORT client...\n");
  gboolean res = ort_client->Init(self->model_file, self->label_file, self->optimization_level, self->execution_provider, self->detection_model, self->device_id);
  GST_INFO_OBJECT (self, "Initialized: %s\n", res ? "true" : "false");
  GST_OBJECT_UNLOCK (self);
  return res;
}

/* GstBaseTransform vmethod implementations */

/* this function does the actual processing (IP = in place)
 */
static GstFlowReturn
gst_ortobjectdetector_transform_ip (GstBaseTransform * base, GstBuffer * outbuf)
{
  Gstortobjectdetector *self = GST_ORTOBJECTDETECTOR (base);
  std::unique_ptr<OrtClient>& ort_client = self->ort_client;

  if (!gst_ortobjectdetector_ort_setup(base)) {
    return GST_FLOW_ERROR;
  }

  if (GST_CLOCK_TIME_IS_VALID (GST_BUFFER_TIMESTAMP (outbuf)))
    gst_object_sync_values (GST_OBJECT (self), GST_BUFFER_TIMESTAMP (outbuf));

  if (gst_base_transform_is_passthrough(base)) {
    return GST_FLOW_OK;
  }

  GstMapInfo info;
  GstVideoMeta *vmeta = gst_buffer_get_video_meta(outbuf);

  if (!vmeta) {
    GST_WARNING_OBJECT (base, "missing video meta");
    return GST_FLOW_ERROR;
  }

  if (gst_buffer_map(outbuf, &info, GST_MAP_READWRITE)) {
    // Modify frame in place
    ort_client->RunModel(info.data, vmeta, self->score_threshold, self->nms_threshold);
    gst_buffer_unmap (outbuf, &info);
  }

  return GST_FLOW_OK;
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
ortobjectdetector_init (GstPlugin * ortobjectdetector)
{
  return GST_ELEMENT_REGISTER (ortobjectdetector, ortobjectdetector);
}

// Needed for C++ template rather than C 
#ifndef PACKAGE
#define PACKAGE "ortobjectdetector"
#define PACKAGE_VERSION "1.19.0.1"
#define GST_LICENSE "LGPL"
#define GST_PACKAGE_NAME "ortobjectdetector"
#define GST_PACKAGE_ORIGIN "https://gstreamer.freedesktop.org"
#endif

/* gstreamer looks for this structure to register ortobjectdetectors */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    ortobjectdetector,
    "ortobjectdetector",
    ortobjectdetector_init,
    PACKAGE_VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
