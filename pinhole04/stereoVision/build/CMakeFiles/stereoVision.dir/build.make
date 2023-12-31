# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/michael/Michael/SLAMBook/pinhole04/stereoVision

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/michael/Michael/SLAMBook/pinhole04/stereoVision/build

# Include any dependencies generated for this target.
include CMakeFiles/stereoVision.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/stereoVision.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/stereoVision.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stereoVision.dir/flags.make

CMakeFiles/stereoVision.dir/main.cpp.o: CMakeFiles/stereoVision.dir/flags.make
CMakeFiles/stereoVision.dir/main.cpp.o: ../main.cpp
CMakeFiles/stereoVision.dir/main.cpp.o: CMakeFiles/stereoVision.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/michael/Michael/SLAMBook/pinhole04/stereoVision/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stereoVision.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stereoVision.dir/main.cpp.o -MF CMakeFiles/stereoVision.dir/main.cpp.o.d -o CMakeFiles/stereoVision.dir/main.cpp.o -c /home/michael/Michael/SLAMBook/pinhole04/stereoVision/main.cpp

CMakeFiles/stereoVision.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stereoVision.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/michael/Michael/SLAMBook/pinhole04/stereoVision/main.cpp > CMakeFiles/stereoVision.dir/main.cpp.i

CMakeFiles/stereoVision.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stereoVision.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/michael/Michael/SLAMBook/pinhole04/stereoVision/main.cpp -o CMakeFiles/stereoVision.dir/main.cpp.s

# Object files for target stereoVision
stereoVision_OBJECTS = \
"CMakeFiles/stereoVision.dir/main.cpp.o"

# External object files for target stereoVision
stereoVision_EXTERNAL_OBJECTS =

stereoVision: CMakeFiles/stereoVision.dir/main.cpp.o
stereoVision: CMakeFiles/stereoVision.dir/build.make
stereoVision: /usr/local/lib/libopencv_gapi.so.4.8.0
stereoVision: /usr/local/lib/libopencv_stitching.so.4.8.0
stereoVision: /usr/local/lib/libopencv_alphamat.so.4.8.0
stereoVision: /usr/local/lib/libopencv_aruco.so.4.8.0
stereoVision: /usr/local/lib/libopencv_bgsegm.so.4.8.0
stereoVision: /usr/local/lib/libopencv_bioinspired.so.4.8.0
stereoVision: /usr/local/lib/libopencv_ccalib.so.4.8.0
stereoVision: /usr/local/lib/libopencv_cvv.so.4.8.0
stereoVision: /usr/local/lib/libopencv_dnn_objdetect.so.4.8.0
stereoVision: /usr/local/lib/libopencv_dnn_superres.so.4.8.0
stereoVision: /usr/local/lib/libopencv_dpm.so.4.8.0
stereoVision: /usr/local/lib/libopencv_face.so.4.8.0
stereoVision: /usr/local/lib/libopencv_freetype.so.4.8.0
stereoVision: /usr/local/lib/libopencv_fuzzy.so.4.8.0
stereoVision: /usr/local/lib/libopencv_hdf.so.4.8.0
stereoVision: /usr/local/lib/libopencv_hfs.so.4.8.0
stereoVision: /usr/local/lib/libopencv_img_hash.so.4.8.0
stereoVision: /usr/local/lib/libopencv_intensity_transform.so.4.8.0
stereoVision: /usr/local/lib/libopencv_line_descriptor.so.4.8.0
stereoVision: /usr/local/lib/libopencv_mcc.so.4.8.0
stereoVision: /usr/local/lib/libopencv_quality.so.4.8.0
stereoVision: /usr/local/lib/libopencv_rapid.so.4.8.0
stereoVision: /usr/local/lib/libopencv_reg.so.4.8.0
stereoVision: /usr/local/lib/libopencv_rgbd.so.4.8.0
stereoVision: /usr/local/lib/libopencv_saliency.so.4.8.0
stereoVision: /usr/local/lib/libopencv_sfm.so.4.8.0
stereoVision: /usr/local/lib/libopencv_stereo.so.4.8.0
stereoVision: /usr/local/lib/libopencv_structured_light.so.4.8.0
stereoVision: /usr/local/lib/libopencv_superres.so.4.8.0
stereoVision: /usr/local/lib/libopencv_surface_matching.so.4.8.0
stereoVision: /usr/local/lib/libopencv_tracking.so.4.8.0
stereoVision: /usr/local/lib/libopencv_videostab.so.4.8.0
stereoVision: /usr/local/lib/libopencv_viz.so.4.8.0
stereoVision: /usr/local/lib/libopencv_wechat_qrcode.so.4.8.0
stereoVision: /usr/local/lib/libopencv_xfeatures2d.so.4.8.0
stereoVision: /usr/local/lib/libopencv_xobjdetect.so.4.8.0
stereoVision: /usr/local/lib/libopencv_xphoto.so.4.8.0
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_people.so
stereoVision: /usr/lib/libOpenNI.so
stereoVision: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
stereoVision: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
stereoVision: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
stereoVision: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
stereoVision: /usr/local/lib/libopencv_shape.so.4.8.0
stereoVision: /usr/local/lib/libopencv_highgui.so.4.8.0
stereoVision: /usr/local/lib/libopencv_datasets.so.4.8.0
stereoVision: /usr/local/lib/libopencv_plot.so.4.8.0
stereoVision: /usr/local/lib/libopencv_text.so.4.8.0
stereoVision: /usr/local/lib/libopencv_ml.so.4.8.0
stereoVision: /usr/local/lib/libopencv_phase_unwrapping.so.4.8.0
stereoVision: /usr/local/lib/libopencv_optflow.so.4.8.0
stereoVision: /usr/local/lib/libopencv_ximgproc.so.4.8.0
stereoVision: /usr/local/lib/libopencv_video.so.4.8.0
stereoVision: /usr/local/lib/libopencv_videoio.so.4.8.0
stereoVision: /usr/local/lib/libopencv_imgcodecs.so.4.8.0
stereoVision: /usr/local/lib/libopencv_objdetect.so.4.8.0
stereoVision: /usr/local/lib/libopencv_calib3d.so.4.8.0
stereoVision: /usr/local/lib/libopencv_dnn.so.4.8.0
stereoVision: /usr/local/lib/libopencv_features2d.so.4.8.0
stereoVision: /usr/local/lib/libopencv_flann.so.4.8.0
stereoVision: /usr/local/lib/libopencv_photo.so.4.8.0
stereoVision: /usr/local/lib/libopencv_imgproc.so.4.8.0
stereoVision: /usr/local/lib/libopencv_core.so.4.8.0
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_features.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_search.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_io.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
stereoVision: /usr/lib/x86_64-linux-gnu/libpng.so
stereoVision: /usr/lib/x86_64-linux-gnu/libz.so
stereoVision: /usr/lib/libOpenNI.so
stereoVision: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
stereoVision: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkIOCore-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libfreetype.so
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkIOImage-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkRenderingUI-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkkissfft-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libGLEW.so
stereoVision: /usr/lib/x86_64-linux-gnu/libX11.so
stereoVision: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.15.3
stereoVision: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.15.3
stereoVision: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.15.3
stereoVision: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.15.3
stereoVision: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libtbb.so.12.5
stereoVision: /usr/lib/x86_64-linux-gnu/libvtksys-9.1.so.9.1.0
stereoVision: /usr/lib/x86_64-linux-gnu/libpcl_common.so
stereoVision: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
stereoVision: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
stereoVision: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.74.0
stereoVision: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so.1.74.0
stereoVision: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
stereoVision: /usr/lib/x86_64-linux-gnu/libqhull_r.so.8.0.2
stereoVision: CMakeFiles/stereoVision.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/michael/Michael/SLAMBook/pinhole04/stereoVision/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable stereoVision"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stereoVision.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stereoVision.dir/build: stereoVision
.PHONY : CMakeFiles/stereoVision.dir/build

CMakeFiles/stereoVision.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stereoVision.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stereoVision.dir/clean

CMakeFiles/stereoVision.dir/depend:
	cd /home/michael/Michael/SLAMBook/pinhole04/stereoVision/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/michael/Michael/SLAMBook/pinhole04/stereoVision /home/michael/Michael/SLAMBook/pinhole04/stereoVision /home/michael/Michael/SLAMBook/pinhole04/stereoVision/build /home/michael/Michael/SLAMBook/pinhole04/stereoVision/build /home/michael/Michael/SLAMBook/pinhole04/stereoVision/build/CMakeFiles/stereoVision.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stereoVision.dir/depend

