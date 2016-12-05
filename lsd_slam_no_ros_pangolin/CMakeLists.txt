set(PROJECT_NAME lsd_slam_core)
cmake_minimum_required(VERSION 2.6)
project(${PROJECT_NAME})

set(BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_MODULE_PATH   ${PROJECT_SOURCE_DIR}/cmake)

FIND_PACKAGE(G2O)
FIND_PACKAGE(CSparse)
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} )

find_package(Eigen3 REQUIRED)
find_package(SuiteParse REQUIRED) # Apparently needed by g2o
find_package(Boost COMPONENTS system thread)
find_package(Pangolin 0.2 QUIET)
find_package(OpenCV REQUIRED)

#flags
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O3 -g -std=c++0x -fno-omit-frame-pointer"
) 

# Set source files
set(lsd_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/DataStructures/Frame.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/FramePoseStruct.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/FrameMemory.cpp
  ${PROJECT_SOURCE_DIR}/src/SlamSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/LiveSLAMWrapper.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMap.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMapPixelHypothesis.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalFuncs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/SophusUtil.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistorter.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/Sim3Tracker.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/Relocalizer.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/SE3Tracker.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/TrackingReference.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Timestamp.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/FabMap.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/KeyFrameGraph.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/g2oTypeSim3Sophus.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/TrackableKeyFrameSearch.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
)

# decide if we have Pangolin
if(Pangolin_FOUND)
  message("--- found PANGOLIN, compiling lsd_pangolin library.")
  include_directories( ${Pangolin_INCLUDE_DIRS} ) 
  set(lsd_pangolin_SOURCE_FILES 
    ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/KeyFrameDisplay.cpp
    ${PROJECT_SOURCE_DIR}/src/IOWrapper/Pangolin/PangolinOutput3DWrapper.cpp)
  set(HAS_PANGOLIN 1)
else()
  message("--- could not find PANGOLIN, not compiling lsd_pangolin library.")
  message("    this means there will be no 3D display / GUI available for lsd_dataset.")
  set(lsd_pangolin_SOURCE_FILES )
  set(HAS_PANGOLIN 0)
endif ()

# decide if we have openCV
if (OpenCV_FOUND)
	message("--- found OpenCV, compiling dso_opencv library.")
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	set(HAS_OPENCV 1)
else ()
	message("--- could not find OpenCV.")
	set(HAS_OPENCV 0)
endif ()


# build shared library.
add_library(${PROJECT_NAME} SHARED ${lsd_SOURCE_FILES} ${lsd_pangolin_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} g2o_core g2o_stuff csparse cxsparse g2o_solver_csparse g2o_csparse_extension g2o_types_sim3 g2o_types_sba boost_thread)

# build main executable (only if we have both OpenCV and Pangolin)
if (HAS_OPENCV AND HAS_PANGOLIN)
  message("--- compiling lsd_dataset and lsd_camera.")
  add_executable(live_lsd_slam src/main_live_camera.cpp)
  target_link_libraries(live_lsd_slam ${PROJECT_NAME} boost_system boost_thread csparse cxsparse ${Pangolin_LIBRARIES} ${OpenCV_LIBS})

  add_executable(dataset_lsd_slam src/main_on_images.cpp)
  target_link_libraries(dataset_lsd_slam ${PROJECT_NAME} boost_system boost_thread csparse cxsparse ${Pangolin_LIBRARIES} ${OpenCV_LIBS})
else()
  message("--- not building lsd_dataset, since either don't have openCV or Pangolin.")
endif()