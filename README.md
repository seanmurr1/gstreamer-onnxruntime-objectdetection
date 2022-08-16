# GStreamer ONNX Runtime Object Detection Plugin

## Background
GStreamer is a multimedia framework that allows users to create data pipelines
that link various components/plugins to create workflows and/or streaming applications. 
See <https://gstreamer.freedesktop.org/> for more information.

ONNX Runtime is a machine learning inferencing engine/framework that allows users to 
optimize inferencing with ONNX models. See <https://onnxruntime.ai/> for more information.

## The Plugin
The ORT Object Detection plugin is a GStreamer plugin that allows users to perform 
inferencing with an object detection model in real time with a GST pipeline.

Users may create a GST pipeline that includes some form of video data, pass that 
video data through the ortobjectdetection plugin, and receive processed video 
frames that include the bounding boxes and accuracy scores from the object detection
inferencing session.

The plugin includes a variety of options in which users may customize. These include:
- ONNX model file path
- label file path
- score threshold
- nms threshold
- optimization level
- execution provider
- hardware acceleration device
- object detection model

Currently, the plugin supports only one object detection model, YOLOv4, and two
execution providers, CPU (default) and CUDA.

## License TODO
This code is provided under a MIT license [MIT], which basically means "do
with it as you wish, but don't blame us if it doesn't work". You can use
this code for any project as you wish, under any license as you wish. We
recommend the use of the LGPL [LGPL] license for applications and plugins,
given the minefield of patents the multimedia is nowadays. See our website
for details [Licensing].

## Requirements/Dependencies
- cmake >= 3.22.5 <https://cmake.org/>
- meson build system >= 0.54.0 <https://mesonbuild.com/>
- ninja >= 1.10.0 <https://ninja-build.org/>
- GStreamer >= 1.19 <https://gstreamer.freedesktop.org/>
- OpenCV >= 4.6.0 <https://opencv.org/>
- ONNX Runtime >= 1.12 <https://onnxruntime.ai/>

NOTE: if you wish to utilize hardware acceleration, please install the corresponding 
version of ORT (e.g. with CUDA support).

## Usage
Configure and build all targets (driver, plugin, tests) as such:

    meson builddir
    ninja -C builddir

### Targets

#### libgstortobjectdetector.so
This is the built plugin. You can check if it has been built correctly with:

    gst-inspect-1.0 builddir/objectdetector/libgstortobjectdetector.so

You may need to update GStreamer's plugin path such that gst-inspect can properly 
pick it up without an absolute path. This can be done with 
    export GST_PLUGIN_PATH=<path to plugin>

Now you can run `gst-inspect-1.0 ortobjectdetector` to see information regarding the plugin.
Please see the `gstortobjectdetector.cpp` file for sample pipelines/usage of the plugin.

#### ortobjectdetector-test
This is a test file for the plugin. Make sure that you update
the GST_PLUGIN_PATH first. This tests various supported/unsupported formats of the plugin.

#### ort-driver
This is a sample driver program to allow users to test the ORT functionality of this repo
without using the full plugin. User's can input a few CL arguments to run object detection 
on a single image. Please see the `ort-driver.cpp` file for more details.

## Future Work
- Add support for more object detection models (e.g. FasterRCNN)
- Add support for more ORT execution providers (e.g. TensorRT)
- Optimize pre/post-processing steps
  - remove OpenCV dependency
  - manually parse images
- Add more plugin tests