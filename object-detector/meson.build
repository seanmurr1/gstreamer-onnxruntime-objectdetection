plugin_c_args = ['-DHAVE_CONFIG_H']

cdata = configuration_data()
cdata.set_quoted('PACKAGE_VERSION', gst_version)
cdata.set_quoted('PACKAGE', 'gst-ort-plugins')
cdata.set_quoted('GST_LICENSE', 'LGPL')
cdata.set_quoted('GST_API_VERSION', api_version)
cdata.set_quoted('GST_PACKAGE_NAME', 'GStreamer Onnx-Runtime Plug-ins')
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://gstreamer.freedesktop.org')
configure_file(output : 'config.h', configuration : cdata)

# Audio is probably not needed for now
gstaudio_dep = dependency('gstreamer-audio-1.0',
    fallback: ['gst-plugins-base', 'audio_dep'])

gstvideo_dep = dependency('gstreamer-video-1.0', version : '>=1.19',
    fallback : ['gst-plugins-base', 'video_dep'])

opencv_dep = dependency('opencv4')
onnxrt_dep = dependency('libonnxruntime')

# The ortobjectdetector Plugin

if onnxrt_dep.found()
	onnxrt_include_root = onnxrt_dep.get_pkgconfig_variable('includedir')
	onnxrt_includes = [onnxrt_include_root / 'core/session', onnxrt_include_root / 'core']
	onnxrt_dep_args = []
	
	compiler = meson.get_compiler('cpp')
	if compiler.has_header(onnxrt_include_root / 'core/providers/cuda/cuda_provider_factory.h')
	  onnxrt_dep_args = ['-DGST_ML_ONNX_RUNTIME_HAVE_CUDA']
	endif

  gstortobjectdetector_sources = [
    'src/gstortobjectdetector.cpp',
    ]

  driver_sources = [
    'src/yolov4.cpp',
    'src/ortclient.cpp',
    'src/driver.cpp'
  ]

  gstortobjectdetector = library('gstortobjectdetector',
    gstortobjectdetector_sources,
    c_args : plugin_c_args,
    cpp_args : onnxrt_dep_args,
    link_args : [],
    include_directories : [onnxrt_includes],
    dependencies : [gst_dep, gstbase_dep, gstvideo_dep, onnxrt_dep, opencv_dep],
    install : true,
    install_dir : plugins_install_dir,
    )

  executable('driver',
    driver_sources,
    c_args : [],
    cpp_args : onnxrt_dep_args,
    link_args : [],
    include_directories : [onnxrt_includes],
    dependencies : [onnxrt_dep, opencv_dep]
  )

    # pkgconfig.generate(gstortobjectdetector, install_dir : plugins_pkgconfig_install_dir)
 endif