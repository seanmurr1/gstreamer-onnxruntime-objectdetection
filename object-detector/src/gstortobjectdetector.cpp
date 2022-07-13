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
 *
 * FIXME:Describe ortobjectdetector here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! ortobjectdetector ! fakesink silent=TRUE
 * ]|
 * </refsect2>
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
  PROP_DETECTION_MODEL
  // TODO: add image format?
};

// Default prop values
#define DEFAULT_SCORE_THRESHOLD 0.25f
#define DEFAULT_NMS_THRESHOLD 0.213f
#define DEFAULT_EXECUTION_PROVIDER GST_ORT_EXECUTION_PROVIDER_CPU
#define DEFAULT_OPTIMIZATION_LEVEL GST_ORT_OPTIMIZATION_LEVEL_ENABLE_EXTENDED
#define DEFAULT_MODEL GST_ORT_DETECTION_MODEL_YOLOV4

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
          GST_TYPE_ORT_DETECTION_MODEL, DEFAULT_MODEL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

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

  /* debug category for fltering log messages
   *
   * FIXME:exchange the string 'Template ortobjectdetector' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_ortobjectdetector_debug, "ortobjectdetector", 0,
      "Template ortobjectdetector");
}

/*****************************************************
// TODO: link transform caps in class init?
// TODO: finalize function
*****************************************************/

/* initialize the new element
 * initialize instance structure
 */
static void
gst_ortobjectdetector_init (Gstortobjectdetector * self)
{
  self->ort_client = new OrtClient();
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
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_ortobjectdetector_ort_setup (GstBaseTransform *base) {
  Gstortobjectdetector *self = GST_ORTOBJECTDETECTOR (base);
  auto ort_client = (OrtClient*) self->ort_client;

  GST_OBJECT_LOCK (self);
  if (ort_client->isInitialized()) {
    GST_OBJECT_UNLOCK (self);
    return TRUE;
  }

  if (!self->model_file || !self->label_file) {
    GST_OBJECT_UNLOCK (self);
    GST_ERROR_OBJECT (self, "Unable to initialize ORT client without model and/or label file.");
    return FALSE;
  }

  g_print ("model-file: %s\n", self->model_file);
  g_print ("label-file: %s\n", self->label_file);
  g_print ("Initializing...\n");
  auto res = ort_client->init(self->model_file, self->label_file);
  g_print ("Initialized: %s\n", res ? "true" : "false");
  GST_OBJECT_UNLOCK (self);
  return res;
}

// Optional. Given the pad in this direction and the given caps, what caps are allowed on the other pad in this element ?
static GstCaps *
gst_ortobjectdetector_transform_caps (GstBaseTransform *base, GstPadDirection direction, GstCaps *caps, GstCaps *filter_caps) {
  Gstortobjectdetector *self = GST_ORTOBJECTDETECTOR (base);
  auto ort_client = (OrtClient*) self->ort_client;

  if (!gst_ortobjectdetector_ort_setup(base)) {
    return NULL;
  }

  GST_LOG_OBJECT (self, "Transforming caps %" GST_PTR_FORMAT, caps);

  if (gst_base_transform_is_passthrough(base)) {
    //return gst_caps_ref (caps);
    return caps;
  }

  return caps;
}


/* GstBaseTransform vmethod implementations */

/* this function does the actual processing (IP = in place)
 */
static GstFlowReturn
gst_ortobjectdetector_transform_ip (GstBaseTransform * base, GstBuffer * outbuf)
{
  Gstortobjectdetector *self = GST_ORTOBJECTDETECTOR (base);
  auto ort_client = (OrtClient*) self->ort_client;

  if (!gst_ortobjectdetector_ort_setup(base)) {
    return GST_FLOW_ERROR;
  }

  if (GST_CLOCK_TIME_IS_VALID (GST_BUFFER_TIMESTAMP (outbuf)))
    gst_object_sync_values (GST_OBJECT (self), GST_BUFFER_TIMESTAMP (outbuf));

  /* FIXME: do something interesting here.  This simply copies the source
   * to the destination. */

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
    // This should modify data in place?
    auto res = ort_client->runModel(info.data, vmeta->width, vmeta->height);
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

#ifndef PACKAGE
#define PACKAGE "ortobjectdetector"
#define PACKAGE_VERSION "1.19.0.1"
#define GST_LICENSE "LGPL"
#define GST_PACKAGE_NAME "ortobjectdetector"
#define GST_PACKAGE_ORIGIN "https://gstreamer.freedesktop.org"
#endif

/* gstreamer looks for this structure to register ortobjectdetectors
 *
 * FIXME:exchange the string 'Template ortobjectdetector' with you ortobjectdetector description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    ortobjectdetector,
    "ortobjectdetector",
    ortobjectdetector_init,
    PACKAGE_VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
