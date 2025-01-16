# CMakeList.txt : CMake projects in app_hrnet_yolox_cam
cmake_minimum_required (VERSION 3.8)
project (app_hrnet_yolox_cam)
set(PROJECT_NAME app_hrnet_yolox_cam)
set(POKY_PATH $ENV{SDKTARGETSYSROOT})

# Add the source to the executable of this project.
add_executable (${PROJECT_NAME}
	main.cpp
	camera.cpp
	drpai_ctl.cpp
	image.cpp
	wayland.cpp
	box.cpp)

add_compile_options(${PROJECT_NAME}  ${CXX} ${LDFLAGS} -l -g m -lrt -Ofast -DNDEBUG -Wall)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

link_directories(${POKY_PATH}/usr/lib/)
link_directories(${POKY_PATH}/usr/lib64/)
include_directories("include")
include_directories("package")
include_directories("${POKY_PATH}/usr/include/gstreamer-1.0/")
include_directories("${POKY_PATH}/usr/include/glib-2.0/")
include_directories("${POKY_PATH}/usr/lib64/glib-2.0/include")
include_directories("${POKY_PATH}/usr/local/include")
include_directories("${POKY_PATH}/usr/include/opencv4")
include_directories(SYSTEM $ENV{OECORE_NATIVE_SYSROOT} ${POKY_PATH}/usr/include )
target_link_libraries(${PROJECT_NAME} 
	pthread glib-2.0 mmngr gobject-2.0 gstreamer-1.0  gstbase-1.0 gstapp-1.0 
	opencv_imgcodecs opencv_imgproc opencv_core opencv_highgui
	jpeg webp tiff z tbb gtk-3 png16 gdk-3 cairo
	lzma rt cairo-gobject
	xkbcommon wayland-cursor wayland-egl wayland-client epoxy
	fribidi harfbuzz fontconfig
	glib-2.0 gobject-2.0 gdk_pixbuf-2.0 gmodule-2.0 pangocairo-1.0
	atk-1.0 gio-2.0 pango-1.0 freetype pixman-1 uuid pcre
	mount resolv expat pangoft2-1.0 blkid
	EGL GLESv2 mmngrbuf)
