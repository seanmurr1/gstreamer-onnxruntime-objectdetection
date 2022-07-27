#include <gst/gst.h>
#include <gst/check/gstcheck.h>

/* Structure to contain all our information, so we can pass it to callbacks */
typedef struct _PipelineData {
  GstElement *pipeline;
  GstElement *filesrc;
  GstElement *qtdemux; 
  GstElement *decodebin;
  GstElement *convert1;
  GstElement *capsfilter;
  GstElement *object_detector;
  GstElement *convert2;
  GstElement *sink;
} PipelineData;

/* Link dynamic qtdemux video src pad to decodebin sink pad */
static void 
qtdemux_pad_added_handler (GstElement *src, GstPad *new_pad, PipelineData *data) {
  GstPad *sink_pad = gst_element_get_static_pad (data->decodebin, "sink");
  GstPadLinkReturn ret; 
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;

  g_print ("Received new pad '%s' from '%s':\n", GST_PAD_NAME (new_pad), GST_ELEMENT_NAME (src));

  /* If our pad is already linked, we have nothing to do here */
  if (gst_pad_is_linked (sink_pad)) {
    g_print ("We are already linked. Ignoring.\n");
    goto exit;
  }

  /* Check the new pad's type */
  new_pad_caps = gst_pad_get_current_caps (new_pad);
  new_pad_struct = gst_caps_get_structure (new_pad_caps, 0);
  new_pad_type = gst_structure_get_name (new_pad_struct);
  if (!g_str_has_prefix (new_pad_type, "video/x-h264")) {
    g_print ("It has type '%s' which is not video/x-h264. Ignoring.\n", new_pad_type);
    goto exit;
  }

  /* Attempt the link */
  ret = gst_pad_link (new_pad, sink_pad);
  if (GST_PAD_LINK_FAILED (ret)) {
    g_print ("Type is '%s' but link failed.\n", new_pad_type);
  } else {
    g_print ("Link succeeded (type '%s').\n", new_pad_type);
  }

exit:
  /* Unreference the new pad's caps, if we got them */
  if (new_pad_caps != NULL)
    gst_caps_unref (new_pad_caps);

  /* Unreference the sink pad */
  gst_object_unref (sink_pad);
}

/* Link dynamic decodebin video src pad to videoconver1 sink pad */
static void 
decodebin_pad_added_handler (GstElement *src, GstPad *new_pad, PipelineData *data) {
  GstPad *sink_pad = gst_element_get_static_pad (data->convert1, "sink");
  GstPadLinkReturn ret; 
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;

  g_print ("Received new pad '%s' from '%s':\n", GST_PAD_NAME (new_pad), GST_ELEMENT_NAME (src));

  /* If our pad is already linked, we have nothing to do here */
  if (gst_pad_is_linked (sink_pad)) {
    g_print ("We are already linked. Ignoring.\n");
    goto exit;
  }

  /* Check the new pad's type */
  new_pad_caps = gst_pad_get_current_caps (new_pad);
  new_pad_struct = gst_caps_get_structure (new_pad_caps, 0);
  new_pad_type = gst_structure_get_name (new_pad_struct);
  if (!g_str_has_prefix (new_pad_type, "video/x-raw")) {
    g_print ("It has type '%s' which is not raw video. Ignoring.\n", new_pad_type);
    goto exit;
  }

  /* Attempt the link */
  ret = gst_pad_link (new_pad, sink_pad);
  if (GST_PAD_LINK_FAILED (ret)) {
    g_print ("Type is '%s' but link failed.\n", new_pad_type);
  } else {
    g_print ("Link succeeded (type '%s').\n", new_pad_type);
  }

exit:
  /* Unreference the new pad's caps, if we got them */
  if (new_pad_caps != NULL)
    gst_caps_unref (new_pad_caps);

  /* Unreference the sink pad */
  gst_object_unref (sink_pad);
}

static gboolean
bus_call (GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = data;

  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End-of-stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR: {
      gchar *debug = NULL;
      GError *err = NULL;

      gst_message_parse_error (msg, &err, &debug);

      g_print ("Error: %s\n", err->message);
      g_error_free (err);

      if (debug) {
        g_print ("Debug details: %s\n", debug);
        g_free (debug);
      }

      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }

  return TRUE;
}

void
test_supported_format (GstCaps *caps)
{
  PipelineData data;
  GstStateChangeReturn ret;
  GMainLoop *loop;
  GstBus *bus;
  guint watch_id;

  /* initialization */
  gst_init (NULL, NULL);
  loop = g_main_loop_new (NULL, FALSE);

  /* create elements */
  data.pipeline = gst_pipeline_new ("my_pipeline");

  /* watch for messages on the pipeline's bus (note that this will only
   * work like this when a GLib main loop is running) */
  bus = gst_pipeline_get_bus (GST_PIPELINE (data.pipeline));
  watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  data.filesrc = gst_element_factory_make ("filesrc", "my_filesource");
  data.qtdemux = gst_element_factory_make ("qtdemux", "my_qtdemux");
  data.decodebin = gst_element_factory_make ("decodebin", "my_decodebin");
  data.convert1 = gst_element_factory_make ("videoconvert", "videoconvert1");
  data.capsfilter = gst_element_factory_make ("capsfilter", "capsfilter");
  data.object_detector = gst_element_factory_make ("ortobjectdetector", "ortobjectdetector");
  data.convert2 = gst_element_factory_make ("videoconvert", "audioconvert2");
  data.sink = gst_element_factory_make ("autovideosink", "videosink");

  if (!data.qtdemux || !data.decodebin || !data.sink || !data.filesrc) {
    ck_abort_msg ("Decodebin or qtdemux or filesrc or output could not be found - check your install\n");
  } else if (!data.convert1 || !data.convert2 || !data.capsfilter) {
    ck_abort_msg ("Could not create videoconvert element(s) or capsfilter, check your installation\n");
  } else if (!data.object_detector) {
    ck_abort_msg ("ORT object detector plugin could not be found. Make sure it "
             "is installed correctly in $(libdir)/gstreamer-1.0/ or "
             "~/.gstreamer-1.0/plugins/ and that gst-inspect-1.0 lists it. "
             "If it doesn't, check with 'GST_DEBUG=*:2 gst-inspect-1.0' for "
             "the reason why it is not being loaded.");
  }

  g_object_set (G_OBJECT (data.filesrc), "location", "../../assets/videos/car_video.mp4", NULL);
  g_object_set (G_OBJECT (data.object_detector), "model-file", "../../assets/models/yolov4/yolov4.onnx", NULL);
  g_object_set (G_OBJECT (data.object_detector), "label-file", "../../assets/models/yolov4/labels.txt", NULL);

  g_object_set (G_OBJECT (data.capsfilter), "caps", caps, NULL);

  // SET OTHER PROPS IF NEEDED FOR VARIOUS TESTS

  gst_bin_add_many (GST_BIN (data.pipeline), data.filesrc, data.qtdemux, data.decodebin, data.convert1, data.capsfilter, data.object_detector, data.convert2, data.sink, NULL);

  /* Link pipeline after demuxer first */
  if (!gst_element_link_many (data.convert1, data.capsfilter, data.object_detector, data.convert2, data.sink, NULL)) {
    ck_abort_msg ("Failed to link pipeline beyond demuxer!\n");
  }
  /* Link pipline demuxer portion */
  if (!gst_element_link_many (data.filesrc, data.qtdemux, NULL)) {
    ck_abort_msg ("Unable to link filesrc to qtdemux!\n");
  }

  g_signal_connect (data.qtdemux, "pad-added", G_CALLBACK (qtdemux_pad_added_handler), &data);
  g_signal_connect (data.decodebin, "pad-added", G_CALLBACK (decodebin_pad_added_handler), &data);

  /* run */
  ret = gst_element_set_state (data.pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    GstMessage *msg;

    g_print ("Failed to start up pipeline!\n");

    /* check if there is an error message with details on the bus */
    msg = gst_bus_poll (bus, GST_MESSAGE_ERROR, 0);
    if (msg) {
      GError *err = NULL;

      gst_message_parse_error (msg, &err, NULL);
      g_print ("ERROR: %s\n", err->message);
      g_error_free (err);
      gst_message_unref (msg);
    }
    ck_abort();
  }

  g_timeout_add_seconds(2, g_main_loop_quit, loop);
  g_main_loop_run (loop);

  /* clean up */
  gst_element_set_state (data.pipeline, GST_STATE_NULL);
  gst_object_unref (data.pipeline);
  g_source_remove (watch_id);
  g_main_loop_unref (loop);
}

GST_START_TEST(test_supported_format_video_rgb)
{
  test_supported_format(gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGB", NULL));
}
GST_END_TEST;

Suite *gst_ortobjectdetector_suite(void) {
  Suite *s = suite_create("GstOrtObjectDetector");
  TCase *supported_formats = tcase_create("Supported Formats");
  guint timeout;
  timeout = 10;

#ifdef HAVE_VALGRIND
  {
    if (RUNNING_ON_VALGRIND)
      timeout *= 4;
  }
#endif 

  tcase_set_timeout(supported_formats, timeout);
  tcase_add_test(supported_formats, test_supported_format_video_rgb);

  suite_add_tcase(s, supported_formats);
  return s;
}

// Run tests
GST_CHECK_MAIN(gst_ortobjectdetector);